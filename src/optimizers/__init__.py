from typing import *
from itertools import accumulate

from src.optimizers.utils import update_svd, batch_update_svd, unpack

import functools
import tensorflow as tf

from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


class BaseSVDOptimizer(tf.keras.optimizers.Optimizer):
    """Base optimizer class for SVD optimization.

    Notes
    -----
    Needs model for initialization such that architecture can be unpacked.
    Different acceleration schemes can be imposed via '_apply_acceleration'.
    Transforming the gradients is identical for all SVD optimizations.
    """

    def __init__(self, model: tf.keras.Model, context_model: Optional[tf.keras.Model] = None, 
                 learning_rate: float = 10e-4, nu: Optional[float] = None,
                 epsilon: float = 10e-8, method: str = 'chi', k: Optional[int] = None, name: Optional[str] = None):
        """Initialize base optimizer.

        Parameters
        ----------
        model: tf.keras.Model
            Model accompanied by optimizer. Needed for architecture unpacking
        learning_rate: float
            Learning rate for optimizer
            (Default is 10e-4)
        nu: float
            Learning rate for cayley transform. If None it is set equal to learning rate
            (Default is None)
        method: str
            Method used to calculate cayley transform
            (default is 'chi')
        k: Optional[int]
            Iterations for fixed point calculation of cayley transform.
            When None the full inverse is used.
            (default is None)
        name: Optional[str]
            Name of optimizer
            (default is None)
        """
        super(BaseSVDOptimizer, self).__init__(name=name)
        # Set parameters
        self.learning_rate = learning_rate
        self.nu = nu if nu is not None else learning_rate
        self.model = model   
        self.context_model = context_model
        self.k = k
        self.method = method
        self.epsilon = epsilon
        # Unpack model
        self.layer_names, self.layers = zip(*unpack([self.model]))
        # Unpack variable starting indices per layer
        self.layer_indices = [0] + list(accumulate([len(layer.trainable_variables) for layer in self.layers]))[:-1]
        # Indices of svd layers
        # Layer is built such that first three variables are u, s ,v
        self.svd_layer_indices = [
            idx for idx, name in enumerate(self.layer_names) if 'svd' in name]
        # Indices of additive context layers
        # Layer is built such that first four variables are u, s, v and W
        self.add_layer_indices = [
            idx for idx, layer in enumerate(self.layers) if ('ca_dense' in layer.name) & ('add' in layer.name)]
        # Indices of multiplicative context layers
        # Layer is built such that first five variables are u, s, v and W, b
        self.mul_layer_indices = [
            idx for idx, layer in enumerate(self.layers) if ('ca_dense' in layer.name) & ('mul' in layer.name)]
        # Generate context input names for context layers
        # self.context_input_names = [layer.input[-1].name for layer in self.layers if 'ca_dense' in layer.name]
#         for layer in self.layers:
#             if 'ca_dense' in layer.name:
#                 print(layer.input[-1].name)
        
    def _create_slots(self, var_list: List[tf.Variable]):
        """Create slots for optimizer

        Parameters
        ----------
        var_list: List[tf.Variable]
            List of variables for which slots are made
        """
        raise NotImplementedError("Implement necessary slots for acceleration.")

    def _apply_acceleration(self, grad, var):
        """Apply modified acceleration.

        Notes
        -----
        This application implements both adaptive learning rate and momentum into the gradient calculation.
        Subsequently this reduces the gradient update to a regular SGD update.

        Parameters
        ----------
        grad: tf.Tensor
            Gradient tensor
        var: tf.Variable
            Variable

        Returns
        -------
        Updated gradient corresponding to acceleration application.
        """
        raise NotImplementedError("Implement acceleration")
        
    def _transform_gradients(self, grads_and_vars: List, c: Union[tf.Tensor, Dict[str, tf.Tensor]]):
        """Transform gradients before application.

        Notes
        -----
        This function is called before application in 'apply_gradients'.

        Parameters
        ----------
        grads_and_vars: Iterable
            Gradients and variables
            
        Returns
        -------
            Gradients and variables with updated gradients
        """
        # Make gradients and variable subscriptable
        grads_and_vars = list(grads_and_vars)
        # Get list of all variable indices
        indices = list(range(len(grads_and_vars)))
        # Calculate SVD variables per layer
        for idx in self.svd_layer_indices:
            # Variable indices for svd layer
            starting_idx = self.layer_indices[idx]
            variable_idx = slice(starting_idx, starting_idx + 3)
            # Get gradients and variables for components
            (du, ds, dv), (u, s, v) = zip(*grads_and_vars[variable_idx])
            # Modify gradients
            du, ds, dv = [self._apply_acceleration(dx, x) for (dx, x) in [(du, u), (ds, s), (dv, v)]]
            # Update svd layer with modified gradients
            du, ds, dv = update_svd(u, s, v, du, ds, dv,
                                    self.nu, self.learning_rate, self.nu, self.epsilon, self.method, self.k)
            # Re-add updated gradients to grads & vars
            grads_and_vars[variable_idx] = [(du, u), (ds, s), (dv, v)]
            # Delete svd indices
            del indices[variable_idx]

        # Check whether dictionary or tensor is passed
        if tf.is_tensor(c):
            context = c
        if isinstance(c, dict):
            context = c[list(c.keys())[0]]
        else:
            context = c
        
        # Calculate additive CA-SVD variables per layer
        for idx in self.add_layer_indices:
            # Variable indices for svd layer
            starting_idx = self.layer_indices[idx]
            variable_idx = slice(starting_idx, starting_idx + 4)
            # Get gradients and variables for components
            (du, ds, dv, dw), (u, s, v, w) = zip(*grads_and_vars[variable_idx])
            # Modify gradients
            du, ds, dv, dw = [self._apply_acceleration(dx, x) for (dx, x) in [(du, u), (ds, s), (dv, v), (dw, w)]]
            # Get context vector
            context_ = context
            # Embed with context model
            if self.context_model is not None:
                context_ = self.context_model(context_)
            # Reshape context
            context_ = tf.reshape(context_, [-1, context_.shape[-1]]) 
            # Add context to singular values
            s_ = s + context_ @ w
            # Update svd layer with modified gradients using batch assembled gradient
            du, ds, dv = batch_update_svd(u, s_, v, du, ds, dv,
                                    self.nu, self.learning_rate, self.nu, self.epsilon, self.method, self.k)
            # Add learning rate and sign to dw
            dw = -self.learning_rate * dw
            # Remove context aware part from ds
            ds = ds + context_ @ dw
            # Reduce over batch dimension
            ds = tf.reduce_mean(ds, axis=0)
            # Re-add updated gradients to grads & vars
            grads_and_vars[variable_idx] = [(du, u), (ds, s), (dv, v), (dw, w)]
            # Delete svd indices
            del indices[variable_idx]

        # Calculate multiplicative CA-SVD variables per layer
        for idx in self.mul_layer_indices:
            # Get context vector
            context_ = context
            # Embed with context model
            if self.context_model is not None:
                context_ = self.context_model(context_)
            # Reshape context
            context_ = tf.reshape(context_, [-1, context_.shape[-1]])
            # Variable indices for svd layer
            starting_idx = self.layer_indices[idx]
            variable_idx = slice(starting_idx, starting_idx + 5)
            # Get gradients and variables for components
            (du, ds, dv, dw, db), (u, s, v, w, b) = zip(*grads_and_vars[variable_idx])
             # Modify gradients
            du, ds, dv, dw, db = [self._apply_acceleration(dx, x) for (dx, x) in [(du, u), (ds, s), (dv, v), (dw, w), (db, b)]]
            # Get context activation
            context_activation = self.layers[idx].context_activation
            # Remove context aware part from gradients for S
            chi = context_activation(context_ @ w + b) + self.epsilon
            ds = ds / chi
            s_ = chi * s
            # Update svd layer with modified gradients using batch assembled gradient
            du, ds, dv = batch_update_svd(u, s_, v, du, ds, dv,
                                    self.nu, self.learning_rate, self.nu, self.epsilon, self.method, self.k)
            # Add learning rate and sign to dw and db
            dw = -self.learning_rate * dw
            db = -self.learning_rate * db
            # Remove context aware part from ds with updated context weights
            ds = s_ + ds
            chi = context_activation(context_ @ (w + dw) + (b + db)) + self.epsilon
            ds = s - ds / chi
            # Reduce over batch dimension
            ds = tf.reduce_mean(ds, axis=0)
            # Re-add updated gradients to grads & vars
            grads_and_vars[variable_idx] = [(du, u), (ds, s), (dv, v), (dw, w), (db, b)]
            # Delete svd indices
            del indices[variable_idx]

        # Iterate over normal weights
        for idx, (grad, var) in enumerate(grads_and_vars):
            if idx in indices:
                # Apply adam
                grad = self._apply_acceleration(grad, var)
                # Scale with learning rate
                grads_and_vars[idx] = (-self.learning_rate * grad, var)
        # Remove NANs
        return [(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad), var) for grad, var in grads_and_vars]
    
    def apply_gradients(self,
                      grads_and_vars, context, 
                      name=None,
                      experimental_aggregate_gradients=True):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        The method sums gradients from all replicas in the presence of
        `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
        passing `experimental_aggregate_gradients=False`.
        Example:
        ```python
        grads = tape.gradient(loss, vars)
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
        # Processing aggregated gradients.
        optimizer.apply_gradients(zip(grads, vars),
            experimental_aggregate_gradients=False)
        ```
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
          name: Optional name for the returned operation. Default to the name passed
            to the `Optimizer` constructor.
          experimental_aggregate_gradients: Whether to sum gradients from different
            replicas in the presense of `tf.distribute.Strategy`. If False, it's
            user responsibility to aggregate the gradients. Default to True.
        Returns:
          An `Operation` that applies the specified gradients. The `iterations`
          will be automatically increased by 1.
        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
          RuntimeError: If called in a cross-replica context.
        """
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        var_list = [v for (_, v) in grads_and_vars]

        with ops.name_scope_v2(self._name):
            # Create iteration if necessary.
            with ops.init_scope():
                self._create_all_weights(var_list)
        if not grads_and_vars:
            # Distribution strategy does not support reducing an empty list of
            # gradients
            return control_flow_ops.no_op()
        if distribute_ctx.in_cross_replica_context():
            raise RuntimeError(
                "`apply_gradients() cannot be called in cross-replica context. "
                "Use `tf.distribute.Strategy.run` to enter replica "
                "context.")

        strategy = distribute_ctx.get_strategy()
        if (not experimental_aggregate_gradients and strategy and
            isinstance(strategy,
                         (parameter_server_strategy.ParameterServerStrategyV1,
                          parameter_server_strategy_v2.ParameterServerStrategyV2,
                          central_storage_strategy.CentralStorageStrategy,
                          central_storage_strategy.CentralStorageStrategyV1))):
            raise NotImplementedError(
                "`experimental_aggregate_gradients=False is not supported for "
                "ParameterServerStrategy and CentralStorageStrategy")
            
        apply_state = self._prepare(var_list)
        if experimental_aggregate_gradients:
            grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
            grads_and_vars = self._aggregate_gradients(grads_and_vars)
        
        # Transform using context
        grads_and_vars = self._transform_gradients(grads_and_vars , context)
        
        if optimizer_utils.strategy_supports_no_merge_call():
            return self._distributed_apply(strategy, grads_and_vars, name,
                                           apply_state)
        else:
            return distribute_ctx.get_replica_context().merge_call(
                functools.partial(self._distributed_apply, apply_state=apply_state),
                args=(grads_and_vars,),
                kwargs={
                    "name": name,
                })

    def _resource_apply_dense(self, grad: tf.Tensor, handle: tf.Variable, apply_state: dict):
        """Application of gradients for dense tensors.

        Notes
        -----
        This application function just does a addition of the gradient.

        Parameters
        ----------
        grad: tf.Tensor
            Gradient for application
        handle: tf.Variable
            Variable on which to apply gradient
        apply_state: dict
            State of application

        Returns
        -------
        Updated variable
        """
        return handle.assign_add(grad)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        """Application of gradients for sparse tensors.

        Notes
        -----
        This application function just does a addition of the gradient.

        Parameters
        ----------
        grad: tf.Tensor
            Gradient for application
        handle: tf.Variable
            Variable on which to apply gradient
        indices: tf.Tensor
            Indices of sparse tensor for which to apply gradients
        apply_state: dict
            State of application

        Returns
        -------
        Updated variable
        """
        return handle.assign_add(grad)

    def get_config(self):
        """Get configuration.

        Returns
        -------
        Serialized configuration
        """
        return super().get_config()


class SVDSGD(BaseSVDOptimizer):
    """Optimizer function for SVD based architectures with keras optimizer compatibility"""

    def __init__(self, model: tf.keras.Model, context_model: Optional[tf.keras.Model] = None, 
                 learning_rate: float = 10e-4, nu: Optional[float] = None,
                 epsilon: float = 10e-8, beta: float = 0.9, method: str = 'chi', k: Optional[int] = None,
                 name: Optional[str] = None):
        """Initialize optimizer

        Parameters
        ----------
        model: tf.keras.Model
            Model accompanied by optimizer. Needed for architecture unpacking
        learning_rate: float
            Learning rate for optimizer
            (Default is 10e-4)
        nu: float
            Learning rate for cayley transform. If None it is set equal to learning rate
            (Default is None)
        beta: float
            Momentum parameter
            (Default is 0.9)
        method: str
            Method used to calculate cayley transform
            (default is 'chi')
        k: Optional[int]
            Iterations for fixed point calculation of cayley transform.
            When None the full inverse is used.
            (default is None)
        name: Optional[str]
            Name of optimizer
            (default is None)
        """
        super(SVDSGD, self).__init__(
            name=name, learning_rate=learning_rate, nu=nu, 
            model=model, context_model=context_model, k=k, method=method, epsilon=epsilon)
        # Set acceleration parameters
        self.beta = beta

    def _create_slots(self, var_list: List[tf.Variable]):
        """Create slots for optimizer

        Parameters
        ----------
        var_list: List[tf.Variable]
            List of variables for which slots are made
        """
        # Create slots for momentum and velocity
        for variable in var_list:  
            self.add_slot(tf.Variable(variable, aggregation=tf.VariableAggregation.MEAN), "momentum")

    def _apply_acceleration(self, grad: tf.Tensor, var: tf.Tensor):
        """Apply modified SGD with momentum.

        Notes
        -----
        This application implements both adaptive learning rate and momentum into the gradient calculation.
        Subsequently this reduces the gradient update to a regular SGD update.

        Parameters
        ----------
        grad: tf.Tensor
            Gradient tensor
        var: tf.Variable
            Variable

        Returns
        -------
        Updated gradient corresponding to SGD application.
        """
        # Get slots
        momentum = self.get_slot(var, "momentum")
        # Calculate updated variables
        momentum.assign(tf.scalar_mul(self.beta,  momentum) + grad)
        # Apply iteration scaling
        return momentum


class SVDAdam(BaseSVDOptimizer):
    """Adam Optimizer function for SVD based architectures with keras optimizer compatibility"""

    def __init__(self, model: tf.keras.Model, context_model: Optional[tf.keras.Model] = None,
                 learning_rate: float = 10e-4, nu: Optional[float] = None,
                 epsilon: float = 10e-8, beta: float = 0.9, gamma: float = 0.999,
                 method: str = 'chi', k: Optional[int] = None,  name: Optional[str] = None):
        """Initialize optimizer

        Parameters
        ----------
        model: tf.keras.Model
            Model accompanied by optimizer. Needed for architecture unpacking
        learning_rate: float
            Learning rate for optimizer
            (Default is 10e-4)
        nu: float
            Learning rate for cayley transform. If None it is set equal to learning rate
            (Default is None)
        beta: float
            Momentum parameter
            (Default is 0.9)
        gamma: float
            Velocity parameter
            (default is 0.999)
        method: str
            Method used to calculate cayley transform
            (default is 'chi')
        k: Optional[int]
            Iterations for fixed point calculation of cayley transform.
            When None the full inverse is used.
            (default is None)
        name: Optional[str]
            Name of optimizer
            (default is None)
        """
        super(SVDAdam, self).__init__(
            name=name, learning_rate=learning_rate, nu=nu, 
            model=model, context_model=context_model, k=k, method=method, epsilon=epsilon)
        # Set acceleration parameters
        self.beta = beta
        self.gamma = gamma

    def _create_slots(self, var_list: List[tf.Variable]):
        """Create slots for optimizer

        Parameters
        ----------
        var_list: List[tf.Variable]
            List of variables for which slots are made
        """
        # Create slots for momentum and velocity
        for variable in var_list:
            self.add_slot(variable, "momentum")
            self.add_slot(variable, "velocity")

    def _apply_acceleration(self, grad, var):
        """Apply modified adam.

        Notes
        -----
        This application implements both adaptive learning rate and momentum into the gradient calculation.
        Subsequently this reduces the gradient update to a regular SGD update.

        Parameters
        ----------
        grad: tf.Tensor
            Gradient tensor
        var: tf.Variable
            Variable

        Returns
        -------
        Updated gradient corresponding to Adam application.
        """
        # Get slots
        momentum = self.get_slot(var, "momentum")
        velocity = self.get_slot(var, "velocity")
        # Calculate updated variables
        momentum.assign(self.beta * momentum + tf.multiply(1. - self.beta, grad))
        velocity.assign(self.gamma * velocity + tf.multiply(1. - self.gamma, tf.math.pow(grad, 2)))
        # Apply iteration scaling
        momentum_ = momentum / (1. - tf.math.pow(self.beta, tf.cast(self.iterations + 1, momentum.dtype)))
        velocity_ = velocity / (1. - tf.math.pow(self.gamma, tf.cast(self.iterations + 1, velocity.dtype)))
        # Return adam scaled gradients
        return tf.sqrt(velocity_ + self.epsilon) ** (-1) * momentum_