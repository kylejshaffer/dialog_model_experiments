import argparse
import keras.backend as K
import data_utils
import math
import os
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

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
    def __init__(self, args, vocab:dict, len_limit:int=300, d_model:int=256, \
                 d_inner_hid:int=512, n_head:int=4, d_k:int=64, d_v:int=64, layers:int=2, dropout:float=0.1, \
                 embedding_dropout:float=0.2, share_word_emb:bool=True):
        # self.i_tokens = i_tokens
        # self.o_tokens = o_tokens
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.decode_model = None
        self.model_name = args.model_name
        self.n_train_examples = args.n_train_examples
        self.n_valid_examples = args.n_valid_examples
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.i_tokens = list(self.vocab.keys())

        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[get_pos_encoding_matrix(len_limit, d_emb)])
        i_word_emb = Embedding(len(self.i_tokens), d_emb) # Embedding(i_tokens.num(), d_emb)
        if share_word_emb:
            # assert i_tokens.num() == o_tokens.num()
            # assert len(i_tokens) == len(o_tokens)
            o_word_emb = i_word_emb
        else:
            o_word_emb = Embedding(len(o_tokens), d_emb)

        # if embedding_dropout > 0:
        #     i_word_emb = Dropout(0.2)(i_word_emb)
        
        self.encoder = Encoder(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, d_k=d_k,
                               d_v=d_v, layers=layers, dropout=dropout, word_emb=i_word_emb, pos_emb=pos_emb)
        # self.word_encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
        #                             word_emb=i_word_emb, pos_emb=pos_emb)
        # self.sent_encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
        #                             word_emb=i_word_emb, pos_emb=pos_emb)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                               word_emb=o_word_emb, pos_emb=pos_emb)
        # self.target_layer = TimeDistributed(Dense(units=len(o_tokens), use_bias=False))
        self.target_layer = TimeDistributed(Dense(units=len(self.vocab), use_bias=True))

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

    def get_perplexity(self, args):
        # cross_entropy = K.categorical_crossentropy(y_true, y_pred)
        cross_entropy = self.get_loss(args)
        perplexity = K.pow(2.0, cross_entropy)
        return perplexity

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def build_graph(self, optimizer='adam', active_layers=999):
        self.src_seq_input = Input(shape=(None,), dtype='int32')
        self.tgt_seq_input = Input(shape=(None,), dtype='int32')
        self.tgt_seq_true = Input(shape=(None,), dtype='int32')

        src_seq = self.src_seq_input
        tgt_seq = Lambda(lambda x: x[:, :-1])(self.tgt_seq_input)
        tgt_true = Lambda(lambda x: x[:, 1:])(self.tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info:
            src_pos = None

        enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
        self.final_output = self.target_layer(dec_output)

        loss = Lambda(self.get_loss)([self.final_output, tgt_true])
        # self.prpl = Lambda(K.exp)(loss)
        self.prpl = Lambda(self.get_perplexity)([self.final_output, tgt_true])
        self.accu = Lambda(self.get_accu)([self.final_output, tgt_true])

        self.model = Model([self.src_seq_input, self.tgt_seq_input], loss)
        self.model.add_loss([loss])
        self.output_model = Model([self.src_seq_input, self.tgt_seq_input], self.final_output)

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('prpl')
        self.model.metrics_tensors.append(self.prpl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def load_model(self, model_weight_path:str):
        assert self.model is not None, "You must build the model architecture before loading in weights!"
        self.model.load_weights(model_weight_path)
        self.output_model = Model([self.src_seq_input, self.tgt_seq_input], self.final_output)

    def decode_sequence(self, input_seq:list, delimiter=''):
        stop_tok = self.vocab['</s>']
        len_limit = 100

        # Prep input for feeding to model
        input_seq.insert(0, '<s>')
        src_seq = np.asarray([self.vocab[w] if w in self.vocab.keys() else self.vocab['<UNK>'] for w in input_seq])
        src_seq = np.reshape(a=src_seq, newshape=(1, len(src_seq)))

        # Set up decoder input data
        decoded_tokens = []
        target_seq = np.zeros((1, len_limit), dtype='int32')
        target_seq[0, 0] = self.vocab['<s>']

        # Loop through and generate decoder tokens
        print('Generating output...')
        for i in range(len_limit - 1):
            output = self.output_model.predict_on_batch([src_seq, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            if sampled_index == stop_tok:
                break
            decoded_tokens.append(self.inverse_vocab[int(sampled_index)])
            target_seq[0, i+1] = sampled_index

        return ' '.join(decoded_tokens)

    def train_generator(self):
        np.random.seed(7)

        ckpt_filename = '/data/users/kyle.shaffer/chat_models/' + self.model_name + '-epoch{:02d}-ppl{:2f}.h5'

        n_train_iters = math.ceil(self.n_train_examples / self.batch_size)
        n_valid_iters = math.ceil(self.n_valid_examples / self.batch_size)

        s2s_processor = data_utils.S2SProcessing(train_file=self.train_file, valid_file=self.valid_file, vocab=self.vocab,
                                                 batch_size=self.batch_size, encoder_max_len=300, decoder_max_len=100,
                                                 shuffle_batch=True)

        for e in range(self.n_epochs):
            train_datagen = s2s_processor.generate_s2s_batches(mode='train')
            valid_datagen = s2s_processor.generate_s2s_batches(mode='valid')

            self.model.fit_generator(train_datagen, steps_per_epoch=n_train_iters)
            
            # Manually iterating over batches to train from generator
            # loss_tracker, ppl_tracker, acc_tracker = 0., 0., 0.
            # batch_tracker = 0
            # for batch_iter in range(n_train_iters):
            #     enc_batch, dec_batch = next(train_datagen)
            #     hist = self.model.fit([enc_batch, dec_batch], verbose=0)
            #     loss_tracker += hist.history['loss'][0]
            #     ppl_tracker += hist.history['prpl'][0]
            #     acc_tracker += hist.history['accu'][0]
            #     batch_tracker += 1
            #     sys.stdout.write('\r loss={} | prpl={} | acc={}'.format((loss_tracker / batch_tracker),
            #                                                             (ppl_tracker / batch_tracker), (acc_tracker / batch_tracker)))

            valid_loss, valid_ppl, valid_acc = self.validate(valid_datagen, n_valid_iters)
            print('EPOCH {} SUMMARY'.format(e + 1))
            print('=' * 50)
            print('Loss:', valid_loss)
            print('Perplexity:', valid_ppl)
            print('Accuracy:', valid_acc)
            print()

            self.model.save_weights(filepath=ckpt_filename.format(e+1, valid_ppl))

        return

    def validate(self, valid_datagen, n_valid_iters):
        # valid_loss, valid_ppl, valid_acc = 0., 0., 0.
        # n_iters = 0
        # for _ in range(n_valid_iters):
        #     x_batch, y_batch, _ = next(valid_datagen)
        #     step_loss, step_ppl, step_acc = self.model.evaluate([x_batch, y_batch])
        #     # Update our counters
        #     n_iters += 1
        #     valid_loss += step_loss
        #     valid_ppl += step_ppl
        #     valid_acc += step_acc

        # valid_loss /= n_iters
        # valid_ppl /= n_iters
        # valid_acc /= n_iters

        valid_loss, valid_ppl, valid_acc = self.model.evaluate_generator(generator=valid_datagen, steps=n_valid_iters)

        return valid_loss, valid_ppl, valid_acc

class TransformerClassifier(object):
    def __init__(self, args, len_limit, d_model=256, d_inner_hid=512, \
                 n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, \
                 embedding_dropout=0.2):
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.vocab_size = args.vocab_size
        self.input_length = len_limit
        self.log_file = open('log.txt', 'w')
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.decode_model = None
        self.num_classes = 8
        self.final_activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
        self.loss_fn = 'binary_crossentropy' if self.final_activation == 'sigmoid' else 'sparse_categorical_crossentropy'
        d_emb = d_model
        self.tokenizer = Tokenizer(num_words=self.vocab_size+1, filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n',
                                   lower=True, split=' ', oov_token="<UNK>")

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[get_pos_encoding_matrix(len_limit, d_emb)])
        i_word_emb = Embedding(self.vocab_size, d_emb)

        if embedding_dropout > 0:
            dropout_layer = Dropout(embedding_dropout)
            # i_word_emb = Dropout(0.2)(i_word_emb)
            i_word_emb = dropout_layer(i_word_emb)

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                               word_emb=i_word_emb, pos_emb=pos_emb)

        self.out_layer = Dense(units=self.num_classes, activation=self.final_activation)
        self.build_graph()

    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.input_length,
                             value=0, padding='post', truncating='post')

    def _report_metrics(self, epoch, y_test, y_hat, y_score=None):
        report_str = """
        EPOCH {} METRICS
        =====================
        Accuracy: {}
        ROC-AUC: {}
        Avg Precision: {}
        Precision: {}
        Recall: {}
        F1: {}
        """

        if y_score is not None:
            roc_score = roc_auc_score(y_test, y_score)
            ap_score = average_precision_score(y_test, y_score)
        else:
            roc_score = 'NA'
            ap_score = 'NA'

        report_str_populated = report_str.format(
            epoch,
            accuracy_score(y_test, y_hat),
            roc_score,
            ap_score,
            precision_score(y_test, y_hat, average='macro'),
            recall_score(y_test, y_hat, average='macro'),
            f1_score(y_test, y_hat, average='macro')
        )

        print(report_str_populated)
        self.log_file.write(report_str_populated)

    def build_graph(self, active_layers=999):
        src_seq = Input(shape=(None,), dtype='int32')
        src_pos = Lambda(self.get_pos_seq)(src_seq)
        enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
        final_output = GlobalMaxPooling1D()(enc_output)
        outputs = self.out_layer(final_output)

        self.model = Model(inputs=[src_seq], outputs=outputs)
        self.model.summary()

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

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

    def train(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        np.random.seed(7)

        ckpt_filename = "transformer_model-epoch{:02d}-{:.2f}.h5"
        self.model.compile('adam', loss=self.loss_fn, metrics=['acc'])
        self.tokenizer.fit_on_texts(X_train)

        X_train = self._get_sequences(X_train)
        if X_val is not None:
            X_val = self._get_sequences(X_val)
            # Manually write the epoch loop so we can get P/R reports
            for e in range(self.n_epochs):
                hist = self.model.fit(X_train, y_train, validation_data=[X_val, y_val],
                                      batch_size=self.batch_size, epochs=1, class_weight=class_weight)
                # y_score = self.model.predict(X_val, batch_size=self.batch_size).flatten()
                y_score = self.model.predict(X_val, batch_size=self.batch_size)
                # y_hat = np.asarray([1 if score > 0.5 else 0 for score in y_score])
                y_hat = np.argmax(y_score, axis=1).squeeze()
                self._report_metrics(epoch=e + 1, y_test=y_val, y_hat=y_hat, y_score=None)
                self.model.save(filepath=ckpt_filename.format(e + 1, np.mean(hist.history['val_acc'])))
        else:
            self.model.fit(X_train, y_train, batch_size=self.batch_size,
                           epochs=self.n_epochs, validation_split=0.1)
        
        self.log_file.close()
    