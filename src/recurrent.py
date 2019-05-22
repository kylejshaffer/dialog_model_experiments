import argparse
import keras.backend as K
import data_utils
import math
import os
import sys

import keras.backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Embedding, GlobalMaxPooling1D, GRU, LSTM
from keras.layers import Activation, Dot, Add, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adagrad, Adam, RMSprop, SGD

from transformer import Transformer
from attention import AttLayer
from data_utils import *


class RNNSeq2Seq(Transformer):
    def __init__(self, args, rnn_params, vocab):
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)
        self.train_from = args.train_from
        self.opt_string = args.optimizer
        self.rec_cell_selector = {'lstm': LSTM, 'gru': GRU}
        self.rec_cell = self.rec_cell_selector[rnn_params['rec_cell']]
        self.embedding_dim = args.embedding_dim
        self.encoder_dim = args.encoder_dim
        self.decoder_dim = args.decoder_dim
        self.num_encoder_layers = rnn_params['num_encoder_layers']
        self.num_decoder_layers = rnn_params['num_decoder_layers']
        self.model_name = 'lstm_att_movie_transfer_chatbot_epoch{:02d}_loss{:.3f}.h5' # args.model_name
        self.n_train_examples = args.n_train_examples
        self.n_valid_examples = args.n_valid_examples
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.num_sampled = 20000
        self.eval_thresh = 500000

        self._choose_optimizer()
        # self.build_model()
        if (self.train_from == '') or (self.train_from is None):
            self.build_simple_model()
        else:
            self.model = load_model(self.train_from, custom_objects={'sparse_loss': lambda x, y: K.sparse_categorical_crossentropy(x, y, True)})
            print('Model loaded from {}...'.format(self.train_from))
            print('Current learning rate: {}'.format(K.get_value(self.model.optimizer.lr)))
            # print('Resetting learning rate...')
            # K.set_value(self.model.optimizer.lr, 0.01)
            # print('New starting learning rate:', K.get_value(self.model.optimizer.lr))
            orig_opt = self.model.optimizer
            # Optionally rebuild model with new trainable attention weights
            if not('_att_' in self.train_from):
                # self._rebuild_model()
                self._rebuild_with_attention()
            decoder_target = tf.placeholder(dtype='int32', shape=[None, None])
            self.model.summary()
            self.model.compile(loss=self.sparse_loss, optimizer=orig_opt, target_tensors=[decoder_target])
        # self.compile()
        # self.saver = tf.train.Saver()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adam', 'sgd', 'momentum', 'rmsprop'}, 'Please select valid optimizer!'
        learning_rate = 0.001

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = Adagrad(lr=0.01)
        elif self.opt_string == 'adam':
            self.optimizer = Adam(lr=0.001)
        elif self.opt_string == 'rmsprop':
            self.optimizer = RMSprop(lr=learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = SGD(lr=learning_rate)
        elif self.opt_string == 'momentum':
            self.optimizer = SGD(lr=learning_rate, momentum=0.99, nesterov=True)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def _init_layers(self):
        # Sequence placeholders
        self.encoder_in_layer = Input(shape=(None,), dtype='int32', name='encoder_input')
        self.decoder_in_layer = Input(shape=(None,), dtype='int32', name='decoder_input')
        self.decoder_output_seq = Input(shape=(None,), dtype='float32', name='decoder_output')
        self.length_ph = Input(batch_shape=(None,), dtype='int32', name='length_ph')

        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)
        self.encoder_layers = [Bidirectional(self.rec_cell(units=self.encoder_dim, return_state=True, return_sequences=True, dropout=0.0)) for _ in \
                                range(self.num_encoder_layers)]
        if self.num_decoder_layers > 1:
            self.decoder_layers = [self.rec_cell(units=self.decoder_dim * 2, return_sequences=True) for _ in \
                                   range(self.num_decoder_layers - 1)]
        else:
            self.decoder_layers = []

    def _rebuild_model(self):
        print('Resetting embedding layer using new vocab...')
        emb_layer_name = [l.name for l in self.model.layers if 'embedding' in l.name][0]
        W_emb = self.model.get_layer(emb_layer_name).get_weights()[0]
        diff = self.vocab_size - W_emb.shape[0]
        W_emb_new = np.random.normal(size=(diff, W_emb.shape[1]))
        W_combined = np.vstack((W_emb, W_emb_new))
        del W_emb
        del W_emb_new

        # New embedding layer
        new_embed_layer = Embedding(input_dim=W_combined.shape[0], output_dim=W_combined.shape[1],
                                 embeddings_initializer=Constant(W_combined), mask_zero=True, trainable=True)
        
        # Reconstruct Encoder
        encoder_embed = new_embed_layer(self.model.layers[1].output)
        encoder_embed = self.model.layers[3](encoder_embed)
        encoder_outputs, _, _ = self.model.layers[4](encoder_embed)
        encoder_outputs2, state_h, state_c = self.model.layers[6](encoder_outputs)
        encoder_states = [state_h, state_c]

        # Reconstruct Decoder
        decoder_embed = new_embed_layer(self.model.layers[0].output)
        decoder_embed = self.model.layers[5](decoder_embed)
        decoder_outputs = self.model.layers[7](decoder_embed, initial_state=encoder_states)

        logits = Dense(units=self.vocab_size, activation='linear', name='logits')
        logits_out = logits(decoder_outputs)

        # Wire new model together
        model = Model(inputs=[self.model.layers[1].output, self.model.layers[0].output], outputs=logits_out)
        model.summary()
        self.model = model

    def _rebuild_with_attention(self):
        print('Rebuilding model with trainable attention weights...')
        # Reconstruct Encoder
        emb_layer_name = [l.name for l in self.model.layers if 'embedding' in l.name][0]
        emb_layer = self.model.get_layer(emb_layer_name)
        encoder_embed = emb_layer(self.model.layers[1].output)
        encoder_embed = self.model.layers[3](encoder_embed)
        encoder_outputs, _, _ = self.model.layers[4](encoder_embed)
        encoder_outputs2, state_h, state_c = self.model.layers[6](encoder_outputs)
        encoder_states = [state_h, state_c]

        # Reconstruct Decoder
        decoder_embed = emb_layer(self.model.layers[0].output)
        decoder_embed = self.model.layers[5](decoder_embed)
        decoder_outputs = self.model.layers[7](decoder_embed, initial_state=encoder_states)

        # Attention
        attention = Dot(axes=[2, 2], name='context_dot')([decoder_outputs, encoder_outputs])
        attention = Activation('softmax', name='attention_probs')(attention)
        context = Dot(axes=[2, 1], name='attention_dot')([attention, encoder_outputs])
        decoder_combined_context = Concatenate()([context, decoder_outputs])
        decoder_combined_context = Dropout(0.2, name='attention_dropout')(decoder_combined_context)
        logits_out = Dense(units=self.vocab_size, activation='linear', name='logits')(decoder_combined_context)

        model = Model(inputs=[self.model.layers[1].output, self.model.layers[0].output], outputs=logits_out)
        model.summary()
        self.model = model

    def compile(self):
        self.sess.run(tf.global_variables_initializer())

    def build_simple_model(self):
        # from keras.utils import multi_gpu_model

        encoder_in_layer = Input(shape=(None,), dtype='int32', name='encoder_input')
        decoder_in_layer = Input(shape=(None,), dtype='int32', name='decoder_input')
        # decoder_out_layer = Input(shape=(None,), dtype='int32', name='decoder_output')
        decoder_target = tf.placeholder(dtype='int32', shape=[None, None])
        
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)
        encoder_embedding = embedding_layer(encoder_in_layer)
        encoder_embedding = Dropout(0.3)(encoder_embedding)
        encoder = LSTM(units=self.encoder_dim, return_state=True, return_sequences=True)
        encoder2 = LSTM(units=self.encoder_dim, return_state=True, return_sequences=True)
        encoder_outputs, _, _ = encoder(encoder_embedding)
        encoder_outputs, state_h, state_c = encoder2(encoder_outputs)

        encoder_states = [state_h, state_c]

        decoder_embedding = embedding_layer(decoder_in_layer)
        decoder_embedding = Dropout(0.3)(decoder_embedding)
        decoder = LSTM(units=self.encoder_dim, return_sequences=True)
        decoder_outputs = decoder(decoder_embedding, initial_state=encoder_states)

        # Attention
        attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
        attention = Activation('softmax', name='attention_probs')(attention)
        context = Dot(axes=[2, 1])([attention, encoder_outputs])
        decoder_combined_context = Concatenate()([context, decoder_outputs])
        decoder_combined_context = Dropout(0.2)(decoder_combined_context)

        decoder_logits = Dense(units=self.vocab_size, activation='linear')(decoder_outputs)

        opt = Adagrad()
        self.model = Model(inputs=[encoder_in_layer, decoder_in_layer], outputs=decoder_logits)
        # self.model = multi_gpu_model(model, gpus=2)
        self.model.compile(loss=self.sparse_loss, optimizer=opt, target_tensors=[decoder_target])
        self.model.summary()
        
    def build_model(self):
        from pprint import pprint

        self._init_layers()
        decoder_target = tf.placeholder(dtype='int32', shape=[None, None])

        # Encoder
        encoder_embeddings = self.embedding_layer(self.encoder_in_layer)
        # encoder_embeddings = Dropout(0.1)(encoder_embeddings)

        for layer_ix, layer in enumerate(self.encoder_layers):
            if layer_ix == 0:
                encoder_out, state_h_fw, state_h_bw, state_c_fw, state_c_bw = layer(encoder_embeddings)
                pprint(encoder_out)
            else:
                encoder_out, state_h_fw, state_h_bw, state_c_fw, state_c_bw
                pprint(encoder_out)
        # encoder_last = encoder[:, -1, :]
        state_h = Concatenate()([state_h_fw, state_h_bw])
        state_c = Concatenate()([state_c_fw, state_c_bw])
        
        # Decoder
        decoder_embeddings = self.embedding_layer(self.decoder_in_layer)
        # decoder_embeddings = Dropout(0.3)(decoder_embeddings)
        decoder = self.rec_cell(units=self.decoder_dim * 2, return_sequences=True)(decoder_embeddings, initial_state=[state_h, state_c])
        if len(self.decoder_layers) > 0:
            for dec_layer in self.decoder_layers:
                decoder = dec_layer(decoder)

        # Attention
        attention = Dot(axes=[2, 2])([decoder, encoder_out])
        attention = Activation('softmax', name='attention_probs')(attention)
        context = Dot(axes=[2, 1])([attention, encoder_out])
        decoder_combined_context = Concatenate()([context, decoder])
        decoder_combined_context = Dropout(0.2)(decoder_combined_context)
        hidden_projection = Dense(units=100, activation='relu')(decoder_combined_context)
        logits_out = Dense(units=self.vocab_size, activation='linear', name='logits_out')(hidden_projection) # self.logits_layer(decoder_combined_context)
        # self.W, self.b = self.logits_layer.weights[0], self.logits_layer.weights[-1]

        # Reshaped input for sampled-softmax
        # inputs_reshaped = tf.reshape(decoder_combined_context, [-1, int(decoder_combined_context.get_shape()[2])])
        # weights_reshaped = tf.transpose(self.W)
        # labels_reshaped = tf.reshape(self.decoder_output_seq, [-1, 1])

        # self.train_step_loss = tf.nn.sampled_softmax_loss(weights=weights_reshaped, biases=self.b, inputs=inputs_reshaped,
        #                                              labels=labels_reshaped, num_sampled=self.num_sampled, num_classes=self.vocab_size)

        # mask = tf.sequence_mask(self.length_ph)
        # masked_output = tf.cast(mask, dtype='int32') * self.decoder_output_seq
        # train_step_loss = tf.contrib.seq2seq.sequence_loss(targets=self.decoder_output_seq, logits=self.logits_out,
        #                                                    weights=tf.to_float(mask))

        # Set up eval loss
        # self.valid_step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_output_seq,
        #                                                                       logits=self.logits_out)
        # self.train_loss = tf.reduce_mean(train_step_loss)
        # self.valid_loss = tf.reduce_mean(self.valid_step_loss)

        # self.train_op = self.optimizer.minimize(self.train_loss)

        # Let's see if we can combine layers in a Keras model for easier read/write
        opt = RMSprop() # SGD(lr=1.0, momentum=0.9)
        self.model = Model(inputs=[self.encoder_in_layer, self.decoder_in_layer], outputs=[logits_out])
        self.model.compile(loss=self.sparse_loss, optimizer=optimizer, target_tensors=[decoder_target])
        self.model.summary()

    def sparse_loss(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, name='sparse_loss')

    def _train_on_batch(self, x_batch, y_in_batch, y_out_batch, length_batch):
        _, loss_ = self.sess.run([self.train_op, self.train_loss],
                                  feed_dict={self.encoder_in_layer: x_batch,
                                            self.decoder_in_layer: y_in_batch,
                                            self.decoder_output_seq: y_out_batch,
                                            self.length_ph: length_batch})
        
        return loss_

    def _eval_on_batch(self, x, y_in, y_out, length_batch, normalize=False):
        # currently running train_loss - may need to be changed later
        valid_loss_ = self.sess.run(self.train_loss, feed_dict={self.encoder_in_layer: x,
                                                                self.decoder_in_layer: y_in,
                                                                self.decoder_output_seq: y_out,
                                                                self.length_ph: length_batch})
        return valid_loss_

    def train(self):
        np.random.seed(7)

        model_dir = '/data/users/kyle.shaffer/chat_models/'

        n_train_iters = math.ceil(self.n_train_examples / self.batch_size)
        n_valid_iters = math.ceil(self.n_valid_examples / self.batch_size)

        s2s_processor = data_utils.S2SProcessing(train_file=self.train_file, valid_file=self.valid_file, vocab=self.vocab,
                                                 batch_size=self.batch_size, encoder_max_len=100, decoder_max_len=100,
                                                 shuffle_batch=True, model_type='recurrent')

        for e in range(self.n_epochs):
            train_datagen = s2s_processor.generate_s2s_batches(mode='train')
            valid_datagen = s2s_processor.generate_s2s_batches(mode='valid')
            valid_datagen_check = s2s_processor.generate_s2s_batches(mode='valid')
            
            all_train_loss = 0
            batch_cnt = 0
            samples_cnt = 0
            for train_iter in range(n_train_iters):
                x_batch, y_in_batch, y_out_batch, length_batch = next(train_datagen)
                if (samples_cnt % self.eval_thresh == 0) and (batch_cnt > 0):
                    self.evaluate(valid_generator=valid_datagen_check)
                    self.model.save(os.path.join(model_dir, 'lstm_chatbot_ckpt.h5'))
                
                loss_ = self._train_on_batch(x_batch, y_in_batch, y_out_batch, length_batch)

                batch_cnt += 1
                all_train_loss += loss_
                samples_cnt += len(x_batch)
                update_loss = all_train_loss / batch_cnt
                sys.stdout.write('\r num_samples_trained: {} \t|\t loss : {:8.3f} \t|\t prpl : {:8.3f}'.format((samples_cnt),
                                update_loss, np.exp(update_loss)))

            # Epoch summary metrics
            print()
            print('\n\nEPOCH {} METRICS'.format(e+1))
            print('=' * 60)
            report_loss = self.evaluate(valid_generator=valid_datagen, num_eval_examples=self.n_valid_examples)
            print('\n\n')
            # self.save(save_path=model_dir, ckpt_name='{}_epoch{}.ckpt'.format(self.model_name, e+1))
            # self.model.save(os.path.join(model_dir, 'lstm_chatbot_epoch_{}_loss{}.h5'.format(e+1, report_loss)))

    def train_keras(self):
        np.random.seed(7)
        debug = False

        test_sents = ["Hey , how are you doing ?",
                      "i'm just looking for someone to talk to .",
                      "You should give me your credit card number and then we can get this thing rolling !"]

        model_dir = '/data/users/kyle.shaffer/chat_models'

        n_train_iters = math.ceil(self.n_train_examples / self.batch_size)
        n_valid_iters = math.ceil(self.n_valid_examples / self.batch_size)

        s2s_processor = data_utils.S2SProcessing(train_file=self.train_file, valid_file=self.valid_file, vocab=self.vocab,
                                                 batch_size=self.batch_size, encoder_max_len=250, decoder_max_len=250,
                                                 shuffle_batch=True, model_type='recurrent')

        if debug:
            print('TRAINING')
            for i in range(n_train_iters):
                sys.stdout.write('\r {}'.format(i))
                x, y = next(train_datagen)


            print('VALIDATION')

            for i in range(n_valid_iters):
                sys.stdout.write('\r {}'.format(i))
                x, y = next(valid_datagen)

        min_lr = 1e-8
        lr_scale = 0.7
        best_loss = 99.
        for e in range(self.n_epochs):
            train_datagen = s2s_processor.generate_s2s_batches(mode='train')
            valid_datagen = s2s_processor.generate_s2s_batches(mode='valid')
            # Train and validate for an epoch
            hist = self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters, validation_data=valid_datagen,
                                validation_steps=n_valid_iters, epochs=1, shuffle=False)
            # Optionally change learning rate if model does not improve
            val_loss = sum(hist.history['val_loss']) / len(hist.history['val_loss'])
            if val_loss < best_loss:
                best_loss = val_loss
            elif K.get_value(self.model.optimizer.lr) <= min_lr:
                print('Minimum LR reached - exiting training!')
                break
            else:
                print('Annealing learning rate...')
                curr_lr = K.get_value(self.model.optimizer.lr)
                new_lr = curr_lr * lr_scale
                print('Updating LR to:', new_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
            
            # Save out model
            self.model.save(os.path.join(model_dir, self.model_name.format(e+1, val_loss)))

            # Look at qualitative output
            print('Testing input sentences...')
            for sent in test_sents:
                print('==>', sent)
                response = self.decode_sequence(input_seq=sent.split())
                print('==>', response)
                print()

        print('DONE TRAINING')
        return

    def evaluate(self, valid_generator, num_eval_examples=10000):
        n_batch_iters = num_eval_examples // self.batch_size
        total_val_loss = 0
        val_batch_cntr = 0
        print('\n\nvalidating')
        for i in range(n_batch_iters):
            print('=', end='', flush=True)
            val_batch_cntr += 1
            x_val_batch, y_in_val_batch, y_out_val_batch, length_batch = next(valid_generator)
            valid_loss_ = self._eval_on_batch(x=x_val_batch, y_in=y_in_val_batch, y_out=y_out_val_batch, length_batch=length_batch)
            total_val_loss += valid_loss_

        report_loss = total_val_loss / val_batch_cntr
        print('\nValidation metrics - loss: {:8.3f} | prpl: {:8.3f}\n'.format(report_loss, np.exp(report_loss)))
        return report_loss

    def decode_sequence(self, input_seq:list, delimiter='', model_type='rnn'):
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

        # Optionally set up positions for CNN-based model
        if model_type == 'cnn':
            src_pos = np.expand_dims(np.array(range(len(input_seq))), axis=0)
            target_pos = np.expand_dims(np.array(range(len_limit)), axis=0)

        # Loop through and generate decoder tokens
        print('Generating output...')
        for i in range(len_limit - 1):
            print('=', end='', flush=True)
            if model_type == 'rnn':
                output = self.model.predict_on_batch([src_seq, target_seq])
            elif model_type == 'cnn':
                output = self.model.predict_on_batch([src_seq, src_pos, target_seq, target_pos])
            sampled_index = np.argmax(output[0, i, :])
            if sampled_index == stop_tok:
                break
            decoded_tokens.append(self.inverse_vocab[int(sampled_index)])
            target_seq[0, i+1] = sampled_index

        return ' '.join(decoded_tokens)

    def save(self, save_path='./', ckpt_name='model.ckpt'):
        self.saver.save(self.sess, save_path + ckpt_name)
        print('Model saved to file:', save_path)

    def load(self, save_path='./', ckpt_name='model.ckpt'):
        self.saver.restore(self.sess, save_path + ckpt_name)
        print('Model restored...')

class HanRnnSeq2Seq(RNNSeq2Seq):
    def __init__(self, args, vocab):
        self.train_from = args.train_from
        self.opt_string = args.optimizer
        self.rec_cell_selector = {'lstm': LSTM, 'gru': GRU}
        self.rec_cell = self.rec_cell_selector[args.rec_cell]
        self.embedding_dim = args.embedding_dim
        self.encoder_dim = args.encoder_dim
        self.decoder_dim = args.decoder_dim
        self.num_encoder_layers = args.num_encoder_layers
        self.num_decoder_layers = args.num_decoder_layers
        self.model_name = 'lstm_context_chatbot_epoch{:02d}_loss{:.3f}.h5' # args.model_name
        self.n_train_examples = args.n_train_examples
        self.n_valid_examples = args.n_valid_examples
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.num_sampled = 20000
        self.eval_thresh = 500000
        self._choose_optimizer()
        self.build_model()

    def build_model(self):
        context_input = Input(shape=(None,), name='context_input')
        current_input = Input(shape=(None,), name='current_input')
        response_input = Input(shape=(None,), name='response_input')
        decoder_target = tf.placeholder(shape=[None, None], dtype='int32')

        embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True, name='embedding')
        context_embed = embed_layer(context_input)
        current_embed = embed_layer(current_input)

        # ENCODER
        bidi_encoder1 = Bidirectional(self.rec_cell(units=self.encoder_dim // 2, return_sequences=True, name='encoder1'))
        bidi_encoder2 = Bidirectional(self.rec_cell(units=self.encoder_dim // 2, return_sequences=True, return_state=True, name='encoder2'))
        # att_layer = AttLayer(attention_dim=self.encoder_dim, name='word_level_attention')

        # Encode context
        context_encoded = bidi_encoder1(context_embed)
        context_encoded, context_forward_h, context_forward_c, context_backward_h, context_backward_c = bidi_encoder2(context_encoded)
        # context_encoded_att = att_layer(context_encoded)

        # Encode current utterance to respond to
        current_encoded = bidi_encoder1(current_embed)
        current_encoded, current_forward_h, current_forward_c, current_backward_h, current_backward_c = bidi_encoder2(current_encoded)
        # current_encoded_att = att_layer(current_encoded)

        encoded_concat = Concatenate(axis=1)([context_encoded, current_encoded])
        encoder_output = self.rec_cell(units=self.encoder_dim, return_sequences=True, name='top_level_encoder')(encoded_concat)
        # encoder_output = Add()([context_encoded_att, current_encoded_att])
        
        # Aggregate hidden states to pass to decoder
        forward_h = Add()([context_forward_h, current_forward_h])
        backward_h = Add()([context_backward_h, current_backward_h])
        forward_c = Add()([context_forward_c, current_forward_c])
        backward_c = Add()([context_backward_c, current_backward_c])

        state_h = Concatenate(axis=1)([forward_h, backward_h])
        state_c = Concatenate(axis=1)([forward_c, backward_c])

        # DECODER
        rnn_decoder = self.rec_cell(units=self.decoder_dim, return_sequences=True, name='decoder1')

        decoder_embed = embed_layer(response_input)
        decoder_embed = Dropout(0.2)(decoder_embed)
        decoder_output = rnn_decoder(decoder_embed, initial_state=[state_h, state_c])

        # Attention
        attention = Dot(axes=[2, 2])([decoder_output, encoder_output])
        attention = Activation('softmax', name='attention_probs')(attention)
        context = Dot(axes=[2, 1])([attention, encoder_output])
        decoder_combined_context = Concatenate()([context, decoder_output])

        logits_out = Dense(units=self.vocab_size, activation='linear', name='logits')(decoder_combined_context)

        self.model = Model(inputs=[context_input, current_input, response_input], outputs=logits_out)
        self.model.compile(loss=self.sparse_loss, optimizer=self.optimizer, target_tensors=[decoder_target])
        self.model.summary()

    def train(self):
        np.random.seed(7)

        test_sents = ["What is your name ?\t My name is John ."
                      "Hey , how are you doing ?\tPretty good , how are you ?",
                      "I think I might be the victim of a spamming attack .\tPlease consider chaning your password .",
                      "Yeah , that sounds like a good plan .\tYou should give me your credit card number and then we can get this thing rolling !",
                      "I 'm not sure I want to do this anymore .\tYou should get me your login so I can take care of it ."]

        model_dir = '/data/users/kyle.shaffer/chat_models'

        n_train_iters = math.ceil(self.n_train_examples / self.batch_size)
        n_valid_iters = math.ceil(self.n_valid_examples / self.batch_size)

        han_s2s_processing = HanS2SProcessing(train_file=self.train_file, valid_file=self.valid_file, vocab=self.vocab,
                                              batch_size=self.batch_size, encoder_max_len=250, decoder_max_len=250,
                                              shuffle_batch=True, model_type='recurrent')

        min_lr = 1e-8
        lr_scale = 0.7
        best_loss = 99.
        for e in range(self.n_epochs):
            train_datagen = han_s2s_processing.generate_s2s_batches(mode='train')
            valid_datagen = han_s2s_processing.generate_s2s_batches(mode='valid')
            # Train and validate for an epoch
            hist = self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters, validation_data=valid_datagen,
                                validation_steps=n_valid_iters, epochs=1, shuffle=False)
            # Optionally change learning rate if model does not improve
            val_loss = sum(hist.history['val_loss']) / len(hist.history['val_loss'])
            if val_loss < best_loss:
                best_loss = val_loss
            elif K.get_value(self.model.optimizer.lr) <= min_lr:
                print('Minimum LR reached - exiting training!')
                break
            else:
                print('Annealing learning rate...')
                curr_lr = K.get_value(self.model.optimizer.lr)
                new_lr = curr_lr * lr_scale
                print('Updating LR to:', new_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
            
            # Save out model
            self.model.save(os.path.join(model_dir, self.model_name.format(e+1, val_loss)))

            # Look at qualitative output
            print('Testing input sentences...')
            for sent in test_sents:
                print('==>', sent)
                response = self.decode_sequence(input_seq=sent.split())
                print('==>', response)
                print()

        print('DONE TRAINING')
        return
    
    def decode_sequence(self, input_seq:str, delimiter='', model_type='rnn'):
        stop_tok = self.vocab['</s>']
        len_limit = 200

        # Prep input for feeding to model
        context, current = input_seq.split('\t')
        context.insert(0, '<s>')
        current.insert(0, '<s>')
        src_context = np.asarray([self.vocab[w] if w in self.vocab.keys() else self.vocab['<UNK>'] for w in context])
        src_current = np.asarray([self.vocab[w] if w in self.vocab.keys() else self.vocab['<UNK>'] for w in current])
        src_context = np.reshape(a=src_context, newshape=(1, len(src_context)))
        src_current = np.reshape(a=src_current, newshape=(1, len(src_current)))

        # Set up decoder input data
        decoded_tokens = []
        target_seq = np.zeros((1, len_limit), dtype='int32')
        target_seq[0, 0] = self.vocab['<s>']

        # Optionally set up positions for CNN-based model
        if model_type == 'cnn':
            src_pos = np.expand_dims(np.array(range(len(input_seq))), axis=0)
            target_pos = np.expand_dims(np.array(range(len_limit)), axis=0)

        # Loop through and generate decoder tokens
        print('Generating output...')
        for i in range(len_limit - 1):
            print('=', end='', flush=True)
            if model_type == 'rnn':
                output = self.model.predict_on_batch([src_context, src_current, target_seq])
            elif model_type == 'cnn':
                output = self.model.predict_on_batch([src_seq, src_pos, target_seq, target_pos])
            sampled_index = np.argmax(output[0, i, :])
            if sampled_index == stop_tok:
                break
            decoded_tokens.append(self.inverse_vocab[int(sampled_index)])
            target_seq[0, i+1] = sampled_index

        return ' '.join(decoded_tokens)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=False, default='/data/users/kyle.shaffer/ased_data/combined_multilabel_train.jl')
    parser.add_argument('--valid_file', type=str, required=False, default='/data/users/kyle.shaffer/ased_data/combined_multilabel_valid.jl')
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--vocab_size', type=int, required=False, default=50000)
    parser.add_argument('--train_from', type=str, required=False, default='')
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--rec_cell', type=str, required=False, default='lstm')
    parser.add_argument('--embedding_dim', type=int, required=False, default=300)
    parser.add_argument('--encoder_dim', type=int, required=False, default=256)
    parser.add_argument('--decoder_dim', type=int, required=False, default=256)
    parser.add_argument('--num_encoder_layers', type=int, required=False, default=2)
    parser.add_argument('--num_decoder_layers', type=int, required=False, default=1)
    parser.add_argument('--n_train_examples', type=int, required=False, default=100000)
    parser.add_argument('--n_valid_examples', type=int, required=False, default=50000)

    args = parser.parse_args()

    dummy_vocab = {'the': 0, 'cat': 1, 'dog': 2, 'mongoose': 3, 'this': 4}

    han_rnn_s2s = HanRnnSeq2Seq(args=args, vocab=dummy_vocab) 
    print('Successfully built HAN model!')
