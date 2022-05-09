import tensorflow as tf
import numpy as np

import types

from os.path import join

@tf.function
def centralization(grad):
    if len(list(grad.shape)) == 2:
        scaled = grad - tf.reduce_mean(grad, axis=-1, keepdims=True)
        return scaled
    else:
        return grad
    
@tf.function
def normalize(grad):
    return grad / (tf.linalg.norm(grad, axis=-1, keepdims=True) + 10e-8)


def context_train_step(self, data):
    """Training step"""
    # Unpack
    (x, c), y = data
    # Get gradients
    with tf.GradientTape() as tape:
        y_pred = self((x, c), training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
#         gradients = [centralization(g) for g in gradients]

    # Make gradients and variables zip
    grads_and_vars = zip(gradients, trainable_variables)
    # Optimize variables
    self.optimizer.apply_gradients(grads_and_vars, c)
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

def wrap_model(model):
    model.train_step = types.MethodType(context_train_step, model)
    return model


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def unpack(packed):
    unpacked = []
    names = []
    for elements in packed:
        if hasattr(elements, 'layers'):
            for name, element in unpack(elements.layers):
                name = join(elements.name, name)
                unpacked.append(element)
                names.append(name)
        else:
            unpacked.append(elements)
            names.append(elements.name)
    return list(zip(names, unpacked))

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2.)) / d_model)
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(encoding, dtype=tf.float32)
