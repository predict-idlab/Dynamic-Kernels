import collections
import copy
import csv
import json
import os
import re
import sys
import time

import numpy as np
from keras import backend
from keras.distribute import distributed_file_utils
from keras.distribute import worker_training_state
from keras.optimizer_v2 import learning_rate_schedule
from keras.utils import generic_utils
from keras.utils import tf_utils
from keras.utils import version_utils
from keras.utils.data_utils import Sequence
from keras.utils.generic_utils import Progbar
from keras.utils.io_utils import path_to_string
from keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    ######
    This class overwrites keras callback and allows tracking of multiple learning rates within optimizer
    ######
    
    Example:
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Args:
      monitor: quantity to be monitored.
      attributes: attributes of optimizer to monitor and reduce on plateau
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                 monitor='val_loss',
                 attributes=['learning_rate',],
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
              f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.attributes = attributes
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                          'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.learning_rate)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    for attribute in self.attributes:
                        old_lr = backend.get_value(getattr(self.model.optimizer, attribute)) 
                        if old_lr > np.float32(self.min_lr):
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            setattr(self.model.optimizer, attribute, new_lr)
                            if self.verbose > 0:
                                print(f'\nEpoch %05d: ReduceLROnPlateau reducing {attribute} to %s.' % (epoch + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0