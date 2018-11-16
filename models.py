import keras.backend as K
import data_utils

from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Add, Lambda

from layer_utils import *
from process_utils import *

# Encoder and decoder layers
class EncoderLayer(object):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, self_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, self_attn

class DecoderLayer(object):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
        output, self_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return output, self_attn, enc_attn


# Encoder and decoder modules
class Encoder(object):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        x = self.emb_layer(src_seq)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        
        if return_att:
            attns = []

        mask = Lambda(lambda x: get_pad_mask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att:
                attns.append(att)
        return (x, attns) if return_att else x

class Decoder(object):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
        dec_emb = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec_emb, pos])

        self_pad_mask = Lambda(lambda x: get_pad_mask(x, x))(tgt_seq)
        self_sub_mask = Lambda(get_sub_mask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])

        enc_mask = Lambda(lambda x: get_pad_mask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att:
            self_attns, enc_attns = [], []

        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_attns.append(self_att)
                enc_attns.append(enc_att)
        
        return (x, self_attns, enc_attns) if return_att else x

# Put everything together in Transformer module
class Transformer(object):
    def __init__(self, i_tokens, o_tokens, len_limit, d_model=256, \
                d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, \
                share_word_emb=False):
        self.i_tokens = i_tokens
        self.o_tokens = o_tokens
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[get_pos_encoding_matrix(len_limit, d_emb)])
        i_word_emb = Embedding(len(i_tokens), d_emb) # Embedding(i_tokens.num(), d_emb)
        if share_word_emb:
            # assert i_tokens.num() == o_tokens.num()
            assert len(i_tokens) == len(o_tokens)
            o_word_emb = i_word_emb

        else:
            o_word_emb = Embedding(len(o_tokens), d_emb)
        
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                        word_emb=i_word_emb, pos_emb=pos_emb)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                        word_emb=o_word_emb, pos_emb=pos_emb)
        self.target_layer = TimeDistributed(Dense(len(o_tokens), use_bias=False))

        self.build_graph()
        self.model.summary()

    def get_loss(self, args):
        y_pred, y_true = args
        y_true = tf.cast(y_true, 'int32')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
        loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
        loss = K.mean(loss)
        return loss

    def get_accu(self, args):
        y_pred, y_true = args
        mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
        corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
        corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
        return K.mean(corr)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def build_graph(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')

        src_seq = src_seq_input
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info:
            src_pos = None

        enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
        final_output = self.target_layer(dec_output)

        loss = Lambda(self.get_loss)([final_output, tgt_true])
        self.prpl = Lambda(K.exp)(loss)
        self.accu = Lambda(self.get_accu)([final_output, tgt_true])

        self.model = Model([src_seq_input, tgt_seq_input], loss)
        self.model.add_loss([loss])
        self.output_model = Model([src_seq_input, tgt_seq_input], final_output)

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('prpl')
        self.model.metrics_tensors.append(self.prpl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def decode_sequence(self, input_seq, delimiter=''):
        src_seq = input_seq
        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens
        return

    def train_generator(self):
        return


if __name__ == '__main__':
    # Test compiling Transformer model
    import string

    i_tokens = set(string.printable)
    o_tokens = i_tokens
    len_limit = 100
    transformer = Transformer(i_tokens=i_tokens, o_tokens=o_tokens, len_limit=len_limit,  d_model=256, \
                d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, \
                share_word_emb=True)

    
