# Code re-implementing transformer from TF 2.0 into Keras API of older TF version
# Try install 1.2.1 - if that fails isntall 1.13
import os
import re

import numpy as np

import keras
import keras.backend as K
import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = keras.layers.Dense(units=d_model)
        self.key_dense = keras.layers.Dense(units=d_model)
        self.value_dense = keras.layers.Dense(units=d_model)

        self.dense = keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # query, key, value, mask = inputs['query'], inputs['key'], inputs[
        #     'value'], inputs['mask']
        query, key, value, mask = inputs
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class LayerNormalization(keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")

    # attention = MultiHeadAttention(
    #     d_model, num_heads, name="attention")({
    #         'query': inputs,
    #         'key': inputs,
    #         'value': inputs,
    #         'mask': padding_mask
    #     })
    attention = MultiHeadAttention(d_model, num_heads, name="attention")([inputs, inputs, inputs, padding_mask])

    attention = keras.layers.Dropout(rate=dropout)(attention)
    # attention = keras.layers.LayerNormalization(
    #     epsilon=1e-6)(inputs + attention)
    attention = LayerNormalization()(inputs + attention)

    outputs = keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = keras.layers.Dense(units=d_model)(outputs)
    outputs = keras.layers.Dropout(rate=dropout)(outputs)
    # outputs = keras.layers.LayerNormalization(
    #     epsilon=1e-6)(attention + outputs)
    outputs = LayerNormalization()(attention + outputs)

    print(type(inputs), type(padding_mask), type(outputs))

    return keras.models.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(embedding_layer, vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings = embedding_layer(inputs)
    embeddings *= tf.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def decoder(vocab_size, num_layers, units,
            d_model, num_heads, dropout,
            embedding_layer, name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings = embedding_layer(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def loss_function(y_true, y_pred):
    max_len = 100
    # y_true = tf.reshape(y_true, shape=(-1, max_len - 1))
    y_true = tf.reshape(y_true, shape=(-1, max_len))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

def loss_func(y_true, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), tf.float32)
    loss_ = loss_object(y_true, y_pred)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()

#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)

#         self.warmup_steps = warmup_steps
#         self.initial_learning_rate = 0.0001

#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps**-1.5)

#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # def get_config(self):
    #     return {
    #         "initial_learning_rate": self.initial_learning_rate,
    #         "name": self.name
    #     }

class Trainer(object):
    # TODO:
    # 1. Refactor so data_generator is defined inside this class
    # 2. Figure out max-len issue
    def __init__(self, d_model:int, units:int, vocab_size:int, num_layers:int,
                 num_heads:int, dropout:float, epochs:int, batch_size:int, data_generator):
        self.d_model = d_model
        self.units = units
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_generator = data_generator
        self._get_train_valid_instances()
        self.build_transformer()

    def _get_train_valid_instances(self):
        self.train_cnt, self.valid_cnt = 0, 0
        with open(self.data_generator.train_file, mode='r') as infile:
            for line in infile:
                self.train_cnt += 1

        with open(self.data_generator.valid_file, mode='r') as infile:
            for line in infile:
                self.valid_cnt += 1

        self.n_train_iters = self.train_cnt // self.batch_size
        self.n_valid_iters = self.valid_cnt // self.batch_size

    def build_transformer(self, name="transformer"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.d_model)

        enc_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = encoder(
            embedding_layer=embedding_layer,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(inputs=[inputs, enc_padding_mask])

        dec_outputs = decoder(
            embedding_layer=embedding_layer,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs")(dec_outputs)
        transformer_model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

        learning_rate = CustomSchedule(self.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        transformer_model.compile(loss=loss_function, optimizer=optimizer)
        self.model = transformer_model
        self.model.summary()

    def train(self):
        np.random.seed(7)
        
        model_name = 'bpe_transformer_cornell_movie_weights_epoch{:02d}_loss{:.3f}.h5'
        train_datagen = self.data_generator.batch_generator(mode='train')
        valid_datagen = self.data_generator.batch_generator(mode='valid')

        for e in range(self.epochs):
            hist = self.model.fit_generator(train_datagen, steps_per_epoch=20, epochs=1,
                                    verbose=1, validation_data=valid_datagen, validation_steps=self.n_valid_iters)
            val_loss = sum(hist.history['val_loss']) / len(hist.history['val_loss'])

            # self.model.save_weights(model_name.format(e+1, val_loss))

            for s in ["Where have you been ?", "It's a trap !", "I'm not crazy , my mother had me tested ."]:
                self.predict(s)

        print('DONE TRAINING')

    def evaluate(self, sentence):
        START_TOKEN = self.data_generator.bos
        END_TOKEN = self.data_generator.eos
        sentence = tf.expand_dims(
            START_TOKEN + self.data_generator.tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        for i in range(100):
            predictions = self.model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        prediction = self.evaluate(sentence)

        predicted_sentence = self.data_generator.tokenizer.decode([i for i in prediction if i < self.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    d_model = 128
    units = 512
    vocab_size = 8192
    num_layers = 4
    num_heads = 4
    dropout = 0.3

    # sample_transformer = transformer(
    #     vocab_size=vocab_size,
    #     num_layers=num_layers,
    #     units=units,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     dropout=dropout,
    #     name="sample_transformer")

    embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model)
    sample_transformer = encoder(embedding_layer, vocab_size,
                                num_layers,
                                units,
                                d_model,
                                num_heads,
                                dropout,
                                name="encoder")

    sample_transformer.summary()

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    sample_transformer.compile(loss=loss_function, optimizer=optimizer)
    print('Model compiled!')

    # sample_transformer.save_weights('../test_transformer_weights.h5')
    # print('Weights saved...')

    # sample_transformer.load_weights('../test_transformer_weights.h5')
    # print('Weights loaded!')
