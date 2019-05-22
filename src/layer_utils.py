import keras.backend as K
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Wrapper
from keras.layers import *
from keras.initializers import *


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
            initializer=Ones(), trainable=True)
        
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
            initializer=Zeros(), trainable=True)

        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention(object):
    def __init__(self, d_model, attn_dropout=0.1):
        ''' Constructor
        
        Parameters
        ----------
        d_model : int
            Dimension of the hidden representations of the model

        attn_dropout : float, optional
            Dropout rate applied to attention states (the default is 0.1)
        
        '''

        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        ''' Define forward-pass of dot-product attention
        
        Parameters
        ----------
        q : tf.tensor
            Query-vector used for self-attention

        k : tf.tensor
            Key-vector used for self-attention

        v : tf.tensor
            Value-vector used for self-attention

        mask : tf.tensor
            Masking vector to mask padding
        
        Returns
        -------
        output : tf.tensor
            Vector representation after computing attention at 
            given timestep

        attn : tf.tensor
            Vector representation of attention states

        '''

        # Compute attention score of how much each non-focus word impacts the encoding
        # of a given focus word
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(1e+10) * (1-x))(mask)
            attn = Add()([attn, mmask])
        # Normalize attention scores
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        # Combine attention scores with value-vector
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention(object):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        if mode == 0:
            self.qs_layer = Dense(units=n_head * d_k, use_bias=False)
            self.ks_layer = Dense(units=n_head * d_k, use_bias=False)
            self.vs_layer = Dense(units=n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers, self.ks_layers, self.vs_layers = [], [], []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(units=d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(units=d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(units=d_k, use_bias=False)))
        
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        ''' Define forward-pass of multi-head attention
        
        Parameters
        ----------
        q : tf.tensor
            Query-vector used for self-attention

        k : tf.tensor
            Key-vector used for self-attention

        v : tf.tensor
            Value-vector used for self-attention

        mask : tf.tensor
            Masking vector to mask padding
        
        Returns
        -------
        output : tf.tensor
            Vector representation after computing attention at 
            given timestep

        attn : tf.tensor
            Vector representation of attention states
        '''

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])
                return x

            head = Lambda(reshape2)(head)

        elif self.mode == 1:
            heads, attns = [], []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm:
            return outputs, attn
        
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward(object):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        ''' Feed-forward network portion of encoder/decoder stacks
        
        Parameters
        ----------
        d_hid : int
            Number of units in final output feed-forward layer

        d_inner_hid : int
            Number of units in hidden feed-forward layer

        dropout : float, optional
            Dropout applied after final feed-forward layer (the default is 0.1)
        
        '''

        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        ''' Forward-pass of feed-forward section of encoder/decoder stack
        
        Parameters
        ----------
        x : tf.tensor
            Input representation (likely the output of the attention layer)
        
        Returns
        -------
        output : tf.tensor
            Output representation of feed-forward stack

        '''

        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        # Residual connection
        output = Add()([output, x])
        return self.layer_norm(output)

class SampledSoftmax(Layer):
    def __init__(self, num_sampled, num_classes,
                projection, bias, hidden_size):
        self.weights_reshaped = tf.transpose(projection)
        self.bias = bias
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.hidden_size = hidden_size

    def __call__(self, y_true, input):
        """ reshaping of y_true and input to make them fit each other """
        input = tf.reshape(input, (-1, self.hidden_size))
        y_true = tf.reshape(y_true, (-1, 1))

        return tf.nn.sampled_softmax_loss(
            weights=self.weights_reshaped,
            biases=self.bias,
            labels=y_true,
            inputs=input,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes,
            partition_strategy='div')
