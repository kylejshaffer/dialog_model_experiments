import keras.backend as K
import numpy as np
import tensorflow as tf


def get_pad_mask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask

def get_sub_mask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j//2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb) for pos in range(max_len)
    ])

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc
