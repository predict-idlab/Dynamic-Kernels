import tensorflow as tf

from typing import *
from src.initializers import SingularValueInitializer


class CADenseMul(tf.keras.layers.Layer):
    """src based densely connected layer."""

    def __init__(self,
                 units: int,
                 rank: int,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 context_activation: Optional[callable] = None
                 ):
        """Initialise layer.

        Parameters
        ----------
        units: int
            number of nodes
        rank: int
            rank of decomposition (<= units)
        activation: str
            Activation function name
        use_bias: bool
            Whether to add bias
        context_activation
            Context activation function
        """
        super(CADenseMul, self).__init__()
        # initialise parameters
        self.units = units
        self.rank = rank
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.context_activation = context_activation if context_activation is not None else tf.nn.sigmoid

        # initialise variables
        self._u = None
        self._s = None
        self._v = None
        self._bias = None
        self._context_bias = None
        self._context_kernel = None

    def build(self, input_shapes: List[tf.TensorShape]):
        # unpack input shapes
        assert len(input_shapes) == 2
        input_shape, context_shape = input_shapes
        # define shapes
        u_shape = tf.TensorShape([input_shape[-1], self.rank])
        s_shape = tf.TensorShape([self.rank])
        v_shape = tf.TensorShape([self.units, self.rank])
        w_shape = tf.TensorShape([context_shape[-1], self.rank])
        b_shape = tf.TensorShape([self.rank])
        bias_shape = tf.TensorShape([self.units])
        # define initializers
        o_initializer = tf.keras.initializers.Orthogonal()
        s_initializer = SingularValueInitializer(input_shape[-1], self.units)
        z_initializer = tf.keras.initializers.get('Zeros')
        # define variables
        self._u = self.add_weight(
            "U", shape=u_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._s = self.add_weight(
            "S", shape=s_shape, dtype=tf.float32, initializer=s_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._v = self.add_weight(
            "V", shape=v_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._context_kernel = self.add_weight(
            "W", shape=w_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._context_bias = self.add_weight(
            "B", shape=b_shape, dtype=tf.float32, initializer=z_initializer, aggregation=tf.VariableAggregation.MEAN)
        if self.use_bias:
            self._bias = self.add_weight(
                "bias", shape=bias_shape, dtype=tf.float32, initializer=z_initializer, aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs):
        # Unpack inputs
        inputs, context = inputs
        # Build context embedding
        h = context @ self._context_kernel + self._context_bias
        # Build context activation
        chi = self.context_activation(h)
        # Modulate singular values
        s = self._s * chi
        # build kernel with batch dimension
        temp = tf.einsum('nr, br->bnr', self._u, s)
        kernel = tf.einsum('bnr,mr->bnm', temp, self._v)
        outputs = tf.einsum('bn, bnm->bm', inputs, kernel)
        # Add bias
        if self.use_bias:
            outputs += self._bias
        # Activate
        return self.activation(outputs)


class CADenseAdd(tf.keras.layers.Layer):
    """SVD based densely connected layer."""

    def __init__(self,
                 units: int,
                 rank: int,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 name: str = ''
                 ):
        super(CADenseAdd, self).__init__()
        # initialise parameters
        self.units = units
        self.rank = rank
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        # initialise variables
        self._u = None
        self._s = None
        self._v = None
        self._context_kernel = None
        self._bias = None

    def build(self, input_shapes):
        # unpack input shapes
        assert len(input_shapes) == 2
        input_shape, context_shape = input_shapes
        # define shapes
        u_shape = tf.TensorShape([input_shape[-1], self.rank])
        s_shape = tf.TensorShape([self.rank])
        v_shape = tf.TensorShape([self.units, self.rank])
        w_shape = tf.TensorShape([context_shape[-1], self.rank])
        bias_shape = tf.TensorShape([self.units])
        # define initializers
        o_initializer = tf.keras.initializers.Orthogonal()
        s_initializer = SingularValueInitializer(input_shape[-1], self.units)
        z_initializer = tf.keras.initializers.get('Zeros')
        # define variables
        self._u = self.add_weight(
            "U", shape=u_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._s = self.add_weight(
            "S", shape=s_shape, dtype=tf.float32, initializer=s_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._v = self.add_weight(
            "V", shape=v_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        self._context_kernel = self.add_weight(
            "W", shape=w_shape, dtype=tf.float32, initializer=o_initializer, aggregation=tf.VariableAggregation.MEAN)
        if self.use_bias:
            self._bias = self.add_weight(
                "bias", shape=bias_shape, dtype=tf.float32, initializer=z_initializer, aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs):
        # Unpack inputs
        inputs, context = inputs
        # Build context embedding
        h = context @ self._context_kernel
        # Modulate singular values
        s = self._s + h
        # build kernel with batch dimension
        temp = tf.einsum('nr, br->bnr', self._u, s)
        kernel = tf.einsum('bnr,mr->bnm', temp, self._v)
        outputs = tf.einsum('bn, bnm->bm', inputs, kernel)
        # Add bias
        if self.use_bias:
            outputs += self._bias
        # Activate
        return self.activation(outputs)