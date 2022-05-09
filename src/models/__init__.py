from typing import *

import tensorflow as tf

from numpy import delete

from .utils import *
from src.layers import CADenseMul as CADense



"""
Context aware code
"""


class CAEncoder(tf.keras.models.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int, dff: int, dff_rank: int,
                 maximum_position_encoding: int, input_vocab_size: Optional[int] = None,
                 context_vocab_size: Optional[int] = None, rate: float = 0.1):
        """Build context-aware encoder.

        Parameters
        ----------
        num_layers: int
            Number of layers in encoder
        d_model: int
            Depth of model
        d_model_rank
            Rank of model
        num_heads: int
            Number of heads
        dff: int
            Width of feedforward layers
        dff_rank: int
            Rank of feedforward layers
        maximum_position_encoding: int
            Maximal positioning encoding size
        input_vocab_size: Optional[int]
            Input vocabulary size
        context_vocab_size: Optional[int]
            Context vocabulary size
        rate: float
            dropout rate (default 0.1)
        """
        super(CAEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank

        self.input_vocab_size = input_vocab_size
        self.context_vocab_size = context_vocab_size

        # Embedding layer for input and context if needed
        if input_vocab_size is not None:
            self.input_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        if context_vocab_size is not None:
            self.context_embedding = tf.keras.layers.Embedding(context_vocab_size, d_model)

        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.encoder_layers = [
            CAEncoderLayer(d_model, d_model_rank, num_heads, dff, dff_rank, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, context, training, mask):
#         assert inputs.shape[0, 1] == context.shape[0, 1]
        attention_weights = {}
        seq_len = inputs.shape[1]

        # adding embedding and position encoding.
        if self.input_vocab_size is not None:
            inputs = self.input_embedding(inputs)  # (batch_size, input_seq_len, d_model)
        if self.context_vocab_size is not None:
            context = self.context_embedding(context)  # (batch_size, input_seq_len, d_model)

        inputs = self.positional_encoding(inputs, seq_len)
        inputs = self.dropout(inputs, training=training)

        for idx, layer in enumerate(self.encoder_layers):
            inputs, weights = layer(inputs, context, training, mask)
            attention_weights[f'encoder_layer_{idx}'] = weights
        return inputs, context, attention_weights  # (batch_size, input_seq_len, d_model)


class CADecoder(tf.keras.models.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int, dff: int, dff_rank: int,
                 maximum_position_encoding: int, input_vocab_size: Optional[int] = None,
                 context_vocab_size: Optional[int] = None, rate: float = 0.1):
        """Build context-aware decoder.

        Parameters
        ----------
        num_layers: int
            Number of layers in encoder
        d_model: int
            Depth of model
        d_model_rank
            Rank of model
        num_heads: int
            Number of heads
        dff: int
            Width of feedforward layers
        dff_rank: int
            Rank of feedforward layers
        maximum_position_encoding: int
            Maximal positioning encoding size
        input_vocab_size: Optional[int]
            Input vocabulary size
        context_vocab_size: Optional[int]
            Context vocabulary size
        rate: float
            dropout rate (default 0.1)
        """
        super(CADecoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank

        self.input_vocab_size = input_vocab_size
        self.context_vocab_size = context_vocab_size

        # Embedding layer for input and context if needed
        if input_vocab_size is not None:
            self.input_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        if context_vocab_size is not None:
            self.context_embedding = tf.keras.layers.Embedding(context_vocab_size, d_model)

        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.decoder_layers = [
            CADecoderLayer(d_model, d_model_rank, num_heads, dff, dff_rank, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, input_context, enc_output, output_context, training, look_ahead_mask, padding_mask):
#         assert tf.shape(inputs)[0, 1] == tf.shape(input_context)[0, 1]
        attention_weights = {}
        seq_len = inputs.shape[1]

        # adding embedding and position encoding.
        if self.input_vocab_size is not None:
            inputs = self.input_embedding(inputs)  # (batch_size, input_seq_len, d_model)
        if self.context_vocab_size is not None:
            input_context = self.context_embedding(input_context)  # (batch_size, input_seq_len, d_model)

        inputs = self.positional_encoding(inputs, seq_len)

        inputs = self.dropout(inputs, training=training)

        for idx, layer in enumerate(self.decoder_layers):
            inputs, weights = layer(inputs, input_context, enc_output, output_context,
                                    training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer_{idx}'] = weights
        return inputs, attention_weights  # (batch_size, input_seq_len, d_model)


class CATransformer(tf.keras.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int,
                 dff: int, dff_rank: int, input_vocab_size: Optional[int], target_vocab_size: Optional[int],
                 input_maximum_position_encoding: int, target_maximum_position_encoding: int,
                 input_context_vocab_size: int = None, target_context_vocab_size: int = None, rate: float = 0.1):
        """Build transformer with context aware kernels.

        Parameters
        ----------
        num_layers: int
            Number of encoder and decoder layers
        d_model: int
            Depth of model for attention heads
        d_model_rank: int
            Rank of model for attention heads
        num_heads: int
            Number of attention heads
        dff: int
            Units for feedforward layers
        dff_rank: int
            Rank for feedforward layers
        input_vocab_size: int
            Input vocabulary size for embedding
        target_vocab_size: int
            Target vocabulary size for embedding
        input_maximum_position_encoding: int
            Positional encoding size for input
        target_maximum_position_encoding: int
            Positional encoding size for target
        input_context_vocab_size: int
            Input vocabulary size for context embedding
        target_context_vocab_size: int
            Target vocabulary size for context embedding
        rate: float
            Dropout rate (default 0.1)
        """
        super(CATransformer, self).__init__()

        # Model parameters
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.num_heads = num_heads
        self.dff = dff
        self.dff_rank = dff_rank

        # Vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.input_context_vocab_size = input_context_vocab_size
        self.target_context_vocab_size = target_context_vocab_size

        # Positional encoding sizes
        self.input_maximum_position_encoding = input_maximum_position_encoding
        self.target_maximum_position_encoding = target_maximum_position_encoding

        # Dropout rate
        self.rate = rate

        # Encoder & decoder
        self.encoder = CAEncoder(num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
                                 input_maximum_position_encoding,  input_vocab_size, input_context_vocab_size, rate)
        self.decoder = CADecoder(num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
                                 target_maximum_position_encoding, target_vocab_size, target_context_vocab_size, rate)

        # Projection layer
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inputs, input_context, targets, target_context,
                 training, enc_padding_mask, look_ahead_mask, dec_padding_mask, return_weights: bool = True):
        """Call transformer.

        Parameters
        ----------
        inputs: tf.Tensor[int]
            Input data (batch_size, seq_len_input,)
        input_context: tf.Tensor[int] or tf.Tensor([float]
            Input context (batch_size, seq_len_input, ) or (batch_size, seq_len_input, d_model)
        targets: tf.Tensor[int]
            Target data (batch_size, seq_len_target, )
        target_context: tf.Tensor[int] or tf.Tensor([float]
            Target context (batch_size, seq_len_target, ) or (batch_size, seq_len_target, d_model)
        training: bool
            Training indicator
        enc_padding_mask: tf.Tensor[bool]
            Padding for encoder
        look_ahead_mask: tf.Tensor[bool]
            Look ahead mask
        dec_padding_mask: tf.Tensor[bool]
            Padding mask for decoder
        return_weights: bool
            Whether to return weights (default True)
        Returns
        -------
        Output: tf.Tensor[int]
            Output of transformer
        Encoder & decoder weights: dict[str->tf.Tensor[float]]
            Weights for encoder and decoder
        """
        # (batch_size, inp_seq_len, d_model)
        encoded_inputs, encoded_context, encoder_weights = self.encoder(inputs, input_context, training,
                                                                        enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        decoded_inputs, decoder_weights = self.decoder(targets, target_context, encoded_inputs, encoded_context,
                                                       training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(decoded_inputs)
        weights = [encoder_weights, decoder_weights]
        if return_weights:
            return final_output, weights
        else:
            return final_output

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    @tf.function
    def train_step(self, inputs, input_context, targets, target_context):
        # Select input targets and real targets
        target_inputs = targets[:, :-1]
        target_context_inputs = target_context[:, :-1]
        targets_real = targets[:, 1:]
        # Make padding masks
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inputs, target_inputs)

        with tf.GradientTape() as tape:
            # Predictions and loss w.r.t target inputs and real targets
            predictions, _ = self(inputs, input_context, target_inputs, target_context_inputs,
                                  True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.compiled_loss(targets_real, predictions)
            # Variables
            names, variables = list(self.unpacked.keys()), list(self.unpacked.values())
            # Gradients
            ### variables some order as self.trainable_variables for now. Need to fix disjoint code!!!!
            gradients = tape.gradient(loss, variables)
        # Indices of svd variables
        slices = [slice(idx, idx + 4) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
        length = range(len(variables))
        svd_indices = [idx for indices in slices for idx in length[indices]]
        for indices in slices:
            # Get gradients and variables for components
            u, s, v, w = variables[indices]
            du, ds, dv, dw = gradients[indices]
            # Calculate orthogonal update
            chi_u = chi(u, du, self.optimizer.learning_rate)
            chi_v = chi(v, dv, self.optimizer.learning_rate)
            u_update = u + chi_u @ u
            v_update = v + chi_v @ v
            # Context updated coefficients depending on encoder or decoder
            position = names[indices][1]
            if (('decoder' in position) and ('values' in position)) or (('decoder' in position) and ('keys' in position)):
                context = target_context # if 'decoder' and ('values' or 'keys') in layer.name else input_context
                if self.target_context_vocab_size is not None:
                    context = self.decoder.context_embedding(context)
            else:
                context = input_context
                if self.input_context_vocab_size is not None:
                    context = self.encoder.context_embedding(context)
            context = tf.reshape(context, (-1, context.shape[-1]))
            s_ = s + context @ w
            # calculate assembled gradient
            dk = batch_assembled_gradient(u, s_, v, du, ds, dv)
            # Calculate singular value updates
            psi_u = tf.transpose(u) @ chi_u @ u
            psi_v = tf.transpose(v) @ chi_v @ v
            s_matrix = tf.linalg.diag(s_)
            s_update_matrix = psi_u @ s_matrix + (s_matrix + psi_u @ s_matrix) @ tf.transpose(
                psi_v) - self.optimizer.learning_rate * (
                                      tf.transpose(u_update) @ dk @ v_update + tf.linalg.diag(context @ dw)
                              )
            # Update orthogonal matrices
            u.assign_add(chi_u @ u)
            v.assign_add(chi_v @ v)
            # Update singular values
            s.assign_add(tf.reduce_mean(tf.linalg.diag_part(s_update_matrix), axis=0))
            # regular updates
            w.assign_sub(self.optimizer.learning_rate * dw)
            # Optimize other variables
        remainder = zip(delete(gradients, svd_indices), delete(variables, svd_indices))
        self.optimizer.apply_gradients(remainder)
        self.epoch_loss.update_state(tf.reduce_mean(loss))

    def train(self, train_data, epochs, validation_data=None,
              train_metrics=None, validation_metrics=None, verbose: bool = True, save_ckpt: int = 5,
              ckpt_path: Optional[str] = None):
        # Checkpoints manager
        if ckpt_path is not None:
            _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
            _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)

        self.epoch_loss = tf.keras.metrics.Mean()
        self.unpacked = {join(name, var.name): var for name, layer in unpack([self]) for var in layer.trainable_variables}

        # Training
        for epoch in range(epochs):
            self.epoch_loss.reset_states()
            # training step
            for (batch, (inputs, input_context, targets, target_context)) in enumerate(train_data):
                if batch % 100 == 0:
                    print(f'Batch: {batch}, Loss: {self.epoch_loss.result()}')
                self.train_step(inputs, input_context, targets, target_context)

            if verbose:
                print(f'Epoch {epoch}, Loss: {self.epoch_loss.result()}')
            #                 print(f'Epoch {epoch}, train metrics: {train_metrics:.4f}' + ', '.join(['train_metrics']))
            #                 print(f'Epoch {epoch}, validation metrics: {validation_metrics:.4f}' + ', '.join(['validation_metrics']))

            if ((epoch + 1) % save_ckpt == 0) & (ckpt_path is not None):
                ckpt_save_path = _ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    def restore(self, ckpt_path):
        _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if _ckpt_manager.latest_checkpoint:
            _ckpt.restore(_ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def predict(self, data, max_length: int = 40, return_weights: bool = False):
        # Iterate over batches
        for encoder_input in data:
            output = self.predict_batch(encoder_input, max_length, return_weights)
            yield output

    def predict_batch(self, encoder_input, encoder_context, decoder_context, max_length: int = 40,
                      return_weights: bool = False):
        # Batch size and all indices
        batch_size = encoder_input.shape[0]
        indices = tf.range(batch_size)
        # Starting tokens
        start_tokens = tf.tile([[self.target_vocab_size - 2]], [batch_size, 1])
        # Empty output
        empty = tf.zeros((batch_size, max_length), dtype=tf.int32)
        output = tf.concat([start_tokens, empty], axis=-1)
        # Iterate over maximum length
        for i in range(max_length):
            n_indices = indices.shape[0]
            # Select indices that haven't finished
            encoder_input = tf.gather(encoder_input, indices, axis=0)
            decoder_input = tf.slice(tf.gather(output, indices, axis=0), [0, 0], [n_indices, i + 1])
            # Masking
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, decoder_input)
            # Predictions
            predictions, attention_weights = self(encoder_input,
                                                  encoder_context,
                                                  decoder_input,
                                                  decoder_context,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

            # Select last tokens from sequence dimension
            predictions = tf.squeeze(predictions[:, -1:, :])  # (batch_size, vocab_size)
            # Predict most likely token
            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # Get indices where sentence has not ended
            mask = tf.math.not_equal(predicted_ids, self.target_vocab_size - 1)
            indices = tf.boolean_mask(indices, mask)
            # Get position in sequence
            positions = tf.repeat([i + 1], indices.shape)
            indices_nd_positions = tf.stack([indices, positions], axis=-1)
            # Add predictions to output
            output = tf.tensor_scatter_nd_update(output, indices_nd_positions, tf.boolean_mask(predicted_ids, mask))
            if n_indices == 0: break
        return output[:, 1:], attention_weights if return_weights else output[:, 1:]


"""
Context aware supplementary layers 
"""


class CAMultiHeadAttention(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, num_heads: int):
        """Multi-head attention with context aware kernels

        Parameters
        ----------
        d_model
            Depth of model
        d_model_rank
            Rank of model
        num_heads
            Number of heads to split attention over
        """
        super(CAMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_model_rank = d_model_rank

        tf.Assert(d_model % num_heads == 0, [d_model, num_heads])

        self.depth = d_model // num_heads

        self.d_model_rank = d_model_rank

        self.query_weights = CADense(d_model, d_model_rank, use_bias=False)
        self.key_weights = CADense(d_model, d_model_rank, use_bias=False)
        self.value_weights = CADense(d_model, d_model_rank, use_bias=False)

        self.dense = CADense(d_model, d_model_rank, use_bias=False)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, values, keys, queries, values_context, keys_context, queries_context, mask):
#         assert values.shape[0, 1] == values_context.shape[0, 1]
#         assert keys.shape[0, 1] == keys_context.shape[0, 1]
#         assert queries.shape[0, 1] == queries_context.shape[0, 1]
        # Necessary parameters
        batch_size, seq_len_q, depth_q = tf.unstack(tf.shape(queries))
        _, seq_len_k, depth_k = tf.unstack(tf.shape(keys))
        _, seq_len_v, depth_v = tf.unstack(tf.shape(values))
        
        depth_q_context = tf.shape(queries_context)[-1]
        depth_k_context = tf.shape(keys_context)[-1]
        depth_v_context = tf.shape(values_context)[-1]
                
        # Reshape and transform
        queries = tf.reshape(queries, (-1, depth_q))
        keys = tf.reshape(keys, (-1, depth_k))
        values = tf.reshape(values, (-1, depth_v))
        
        queries_context = tf.reshape(queries_context, (-1, depth_q_context))
        keys_context = tf.reshape(keys_context, (-1, depth_k_context))
        values_context = tf.reshape(values_context, (-1, depth_v_context))

        queries = self.query_weights([queries, queries_context])  # (batch_size * seq_len_q, d_model)
        keys = self.key_weights([keys, keys_context])  # (batch_size * seq_len_k, d_model)
        values = self.value_weights([values, values_context])  # (batch_size * seq_len_v, d_model)

        queries = tf.reshape(queries, (batch_size, seq_len_q, self.d_model))  # (batch_size, seq_len_q, d_model)
        keys = tf.reshape(keys, (batch_size, seq_len_k, self.d_model))  # (batch_size, seq_len_k, d_model)
        values = tf.reshape(values, (batch_size, seq_len_v, self.d_model))  # (batch_size, seq_len_v, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size * seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (-1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense([concat_attention, queries_context])
        output = tf.reshape(output, (batch_size, seq_len_q, self.d_model))
        return output, attention_weights


class CAAttentionBlock(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, num_heads: int, rate: float):
        super(CAAttentionBlock, self).__init__()
        self.attention = CAMultiHeadAttention(d_model, d_model_rank, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)

    def call(self, values, keys, queries,
                 values_context, keys_context, queries_context,
                 look_ahead_mask, training):
        outputs, weights = self.attention(
            values, keys, queries,  # inputs
            values_context, keys_context, queries_context,  # context
            look_ahead_mask  # extra args
        )
        outputs = self.dropout(outputs, training=training)
        outputs = self.normalization(queries + outputs)
        return outputs, weights


class CAPointWiseFeedForward(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, width, rank, activation: str = 'relu'):
        super(CAPointWiseFeedForward, self).__init__()
        self.width = width
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.rank = rank
        self.dense = CADense(width, rank, activation=activation, use_bias=False)
        self.linear = CADense(d_model, d_model_rank, use_bias=False)

    def call(self, inputs: tf.Tensor, context: tf.Tensor):
#         assert inputs.shape[0, 1] == context.shape[0, 1]
        batch_size, seq_len, depth = tf.unstack(tf.shape(inputs))
        depth_context = tf.shape(context)[-1]

        # Reshape
        inputs = tf.reshape(inputs, (-1, depth))
        context = tf.reshape(context, (-1, depth_context))
        # Transform
        inputs = self.dense([inputs, context])  # (batch_size * seq_len, width)
        output = self.linear([inputs, context])  # (batch_size * seq_len, d_model)
        # Reshape
        return tf.reshape(output, (batch_size, seq_len, self.d_model))


class CAPointWiseFeedForwardBlock(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, width: int, rank: int, rate: float):
        super(CAPointWiseFeedForwardBlock, self).__init__()
        self.feed_forward = CAPointWiseFeedForward(d_model, d_model_rank, width, rank)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)

    def call(self, inputs, context, training):
        outputs = self.feed_forward(inputs, context)
        outputs = self.dropout(outputs, training=training)
        outputs = self.normalization(inputs + outputs)
        return outputs


class CADecoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, num_heads, dff, dff_rank, rate: float = 0.1):
        super(CADecoderLayer, self).__init__()
        self.attention_block_one = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.attention_block_two = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.feedforward_block = CAPointWiseFeedForwardBlock(d_model, d_model_rank, dff, dff_rank, rate)

    def call(self, inputs, input_context, encoder_output, output_context,
                 training, look_ahead_mask, padding_mask):
        attention_weights = {}

        inputs, weights = self.attention_block_one(
            inputs, inputs, inputs,
            input_context, input_context, input_context,
            look_ahead_mask, training)
        attention_weights['block_one'] = weights
        inputs, weights = self.attention_block_two(
            encoder_output, encoder_output, inputs,
            output_context, output_context, input_context,
            padding_mask, training)
        attention_weights['block_two'] = weights

        outputs = self.feedforward_block(inputs, input_context, training)  # (batch_size, target_seq_len, d_model)
        return outputs, attention_weights


class CAEncoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, num_heads, dff, dff_rank, rate=0.1):
        super(CAEncoderLayer, self).__init__()

        self.attention_block = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.feedforward_block = CAPointWiseFeedForwardBlock(d_model, d_model_rank, dff, dff_rank, rate)

    def call(self, inputs, context, training, mask):
        attention_weights = {}

        inputs, weights = self.attention_block(
            inputs, inputs, inputs,
            context, context, context,
            mask, training)  # (batch_size, input_seq_len, d_model)
        attention_weights['block_one'] = weights
        outputs = self.feedforward_block(inputs, context, training)  # (batch_size, input_seq_len, d_model)
        return outputs, attention_weights
    
    
"""
Regular code
"""

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, position: int):
        """ Initialise layer.

        Parameters
        ----------
        d_model: int
            Depth of embeddings
        position: int
            maximal position index
        """
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.position = position
        # Build positional encoding
        self.positional_encoding = positional_encoding(position, d_model)

    def call(self, inputs: tf.Tensor):
        """Call layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Input data (batch size x sequence length x model depth)

        Returns
        -------
        tf.Tensor: Positional encoded input
        """
        # get sequence length
        seq_len = tf.shape(inputs)[1]
        # scale inputs
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add encoding
        return inputs + self.positional_encoding[:, :seq_len, :]


class MultiHeadAttention(tf.keras.models.Model):
    """Multi-head attention layer."""
    def __init__(self, d_model: int, d_model_rank: Optional[int], num_heads: int):
        """Initialize layer.

        Parameters
        ----------
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Rank of SVD approximation.
            If None regular matrices are used.
        num_heads: int
            Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_model_rank = d_model_rank

        tf.Assert(d_model % num_heads == 0, [d_model, num_heads])

        self.depth = d_model // num_heads

        if d_model_rank is None:
            self.query_weights = tf.keras.layers.Dense(d_model)
            self.key_weights = tf.keras.layers.Dense(d_model)
            self.value_weights = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)

        else:
            self.query_weights = SVDDense(d_model, d_model_rank)
            self.key_weights = SVDDense(d_model, d_model_rank)
            self.value_weights = SVDDense(d_model, d_model_rank)

            self.dense = SVDDense(d_model, d_model_rank)

    def split_heads(self, x: tf.Tensor, batch_size: int):
        """Split the last dimension into (num_heads, depth).

        Notes
        -----
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        Parameters
        ----------
        x: tf.Tensor
            Input to split
        batch_size: int
            Batch size

        Returns
        -------
        tf.Tensor:
            Split inputs
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, values: tf.Tensor, keys: tf.Tensor, queries: tf.Tensor, mask: tf.Tensor):
        """Call multi-head attention layer.

        Parameters
        ----------
        values: tf.Tensor
            Values for attention
        keys: tf.Tensor
            Keys for attention
        queries: tf.Tensor
            Queries for tensor
        mask: tf.Tensor
            Masking tensor
        Returns
        -------
        tf.Tensor, tf.Tensor
            Attended values & attention weights
        """
        # Batch size
        batch_size = tf.shape(queries)[0]

        queries = self.query_weights(queries)  # (batch_size, seq_len, d_model)
        keys = self.key_weights(keys)  # (batch_size, seq_len, d_model)
        values = self.value_weights(values)  # (batch_size, seq_len, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


class AttentionBlock(tf.keras.models.Model):
    """Attention block."""
    def __init__(self, d_model: int, d_model_rank: Optional[int], num_heads: int, rate):
        """Initialize attention block

        Parameters
        ----------
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Rank of SVD approximation.
            If None regular matrices are used.
        num_heads: int
            Number of attention heads
        rate: float
            Dropout rate
        """
        super(AttentionBlock, self).__init__()
        # MHA
        self.attention = MultiHeadAttention(d_model, d_model_rank, num_heads)
        # Dropout
        self.dropout = tf.keras.layers.Dropout(rate)
        # Layer normalization
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, values, keys, queries, look_ahead_mask, training):
        """Call attention block.

        Parameters
        ----------
        values: tf.Tensor
            Values for attention
        keys: tf.Tensor
            Keys for attention
        queries: tf.Tensor
            Queries for tensor
        look_ahead_mask: tf.Tensor
            Masking tensor
        training: bool
            Training indicator
        Returns
        -------
        tf.Tensor, tf.Tensor
            Outputs & weights
        """
        # Attention
        outputs, weights = self.attention(values, keys, queries, look_ahead_mask)
        # Dropout
        outputs = self.dropout(outputs, training=training)
        # Normalization
        outputs = self.normalization(queries + outputs)
        return outputs, weights


class PointWiseFeedForward(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, width, width_rank, activation: str = 'relu'):
        super(PointWiseFeedForward, self).__init__()
        # (batch_size, seq_len, dff)
        if width_rank is None:
            self.dense = tf.keras.layers.Dense(width, activation=activation)
        else:
            self.dense = SVDDense(width, width_rank, activation=activation)
        # (batch_size, seq_len, d_model)
        if d_model_rank is None:
            self.linear = tf.keras.layers.Dense(d_model)
        else:
            self.linear = SVDDense(d_model, d_model_rank, activation='linear')

    def __call__(self, inputs):
        """Call point wise feedforward layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Inputs
        Returns
        -------
        tf.Tensor:
            Outputs
        """
        inputs = self.dense(inputs)
        return self.linear(inputs)


class DecoderLayer(tf.keras.models.Model):
    """Decoder layer."""
    def __init__(self, d_model: int, d_model_rank: Optional[int], num_heads: int,
                 dff: int, dff_rank: Optional[int], rate: float = 0.1):
        """Initialize decoder layer.

        Parameters
        ----------
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Model depth rank. If None regular layers are used otherwise SVD layer.
        num_heads: int
            Number of attention heads
        dff: int
            Width for point wise feedforward
        dff_rank: Optional[int]
            Width rank for point wise feedforward. Idem as d_model_rank.
        rate: float
            Dropout rate
            (default is 0.1)
        """
        super(DecoderLayer, self).__init__()
        self.attention_block_one = AttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.attention_block_two = AttentionBlock(d_model, d_model_rank, num_heads, rate)

        self.final_feedforward = PointWiseFeedForward(d_model, d_model_rank, dff, dff_rank)
        self.final_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, encoder_output, training, look_ahead_mask, padding_mask):
        """Call decoder layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Inputs
        encoder_output: tf.Tensor
            Encoder output
        training: bool
            Training indicator
        look_ahead_mask: tf.Tensor
            Look ahead mask
        padding_mask: tf.Tensor
            Padding mask

        Returns
        -------
        tf.Tensor, dict:
            Outputs & attention weights in dictionary
        """
        # Initialize weights
        attention_weights = {}

        # Attention block one
        inputs, weights = self.attention_block_one(
            inputs, inputs, inputs,
            look_ahead_mask, training)
        # Store block one weights
        attention_weights['block_one'] = weights
        # Attention block two
        inputs, weights = self.attention_block_two(
            encoder_output, encoder_output, inputs,
            padding_mask, training)
        # Store block two weights
        attention_weights['block_two'] = weights

        # Feedforward
        outputs = self.final_feedforward(inputs)  # (batch_size, input_seq_len, d_model)
        # Dropout
        outputs = self.final_dropout(outputs, training=training)
        # Normalization
        outputs = self.final_normalization(inputs + outputs)  # (batch_size, input_seq_len, d_model)
        return outputs, attention_weights


class EncoderLayer(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: Optional[int], num_heads: int,
                 dff: int, dff_rank: Optional[int], rate: float = 0.1):
        """Initialize decoder layer.

        Parameters
        ----------
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Model depth rank. If None regular layers are used otherwise SVD layer.
        num_heads: int
            Number of attention heads
        dff: int
            Width for point wise feedforward
        dff_rank: Optional[int]
            Width rank for point wise feedforward. Idem as d_model_rank.
        rate: float
            Dropout rate
            (default is 0.1)
        """
        super(EncoderLayer, self).__init__()
        # Attention block
        self.attention_block = AttentionBlock(d_model, d_model_rank, num_heads, rate)
        # Feedforward
        self.final_feedforward = PointWiseFeedForward(d_model, d_model_rank, dff, dff_rank)
        # Normalization
        self.final_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout
        self.final_dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, training, mask):
        """Call encoder layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Inputs
        training: bool
            Training indicator
        mask: tf.Tensor
            Mask

        Returns
        -------
        tf.Tensor, tf.Tensor:
            Outputs, weights
        """
        # Attention block
        # (batch_size, input_seq_len, d_model)
        inputs, weights = self.attention_block(inputs, inputs, inputs, mask, training)

        # Feedforward
        outputs = self.final_feedforward(inputs)  # (batch_size, input_seq_len, d_model)
        # Dropout
        outputs = self.final_dropout(outputs, training=training)
        # Normalization
        outputs = self.final_normalization(inputs + outputs)  # (batch_size, input_seq_len, d_model)
        return outputs, weights