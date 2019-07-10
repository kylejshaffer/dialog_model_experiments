import argparse
import math
import numpy as np
import os

import sys
import tensorflow as tf
import utils

import keras
from keras.layers import Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Activation, Input, Dropout, Embedding, GRU, LSTM, Dense, Lambda
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed


class BidirectionalLM(object):
    def __init__(self, args, vocab, train_file, valid_file, model_name):
        self.vocab = vocab
        self.train_file = train_file
        self.valid_file = valid_file
        self.model_name = model_name
        # TF session setup
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)
        self.num_train_examples = args.num_train_examples
        self.num_val_examples = args.num_val_examples
        self.batch_size = args.batch_size
        self.valid_batch_size = 160
        self.epochs = args.epochs
        self.eval_thresh = self._set_eval_thresh()
        self.seq_len = args.seq_len
        self.vocab_size = len(self.vocab)
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_hidden_layers = args.num_hidden_layers
        self.num_sampled = args.num_sampled
        self.embedding_dropout = args.embedding_dropout_rate
        self.learning_rate = args.learning_rate
        self.opt_string = args.optimizer
        # self.visualize_gradients = args.visualize_gradients
        self._choose_optimizer()
        self.build_graph()
        self.compile()
        # self._tboard_setup()
        self.saver = tf.train.Saver()

    def _set_eval_thresh(self):
        n_train_iters = self.num_train_examples // self.batch_size
        eval_every = n_train_iters // 20
        return eval_every

    def _init_layers(self):
        # Sequence placeholders
        self.input_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.output_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_output')
        self.output_seq_bw = tf.reverse(self.output_seq, axis=[1])

        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                        embeddings_initializer='glorot_uniform', name='embedding',
                                        mask_zero=True)
        self.embedding_dropout_layer = Dropout(self.embedding_dropout, name='embedding_dropout')
        self.fwd_layers = [LSTM(units=self.hidden_dim, return_sequences=True, go_backwards=False) for _ in range(self.num_hidden_layers)]
        self.bwd_layers = [LSTM(units=self.hidden_dim, return_sequences=True, go_backwards=True) for _ in range(self.num_hidden_layers)]
        self.proj_layer = Dense(units=512, activation='relu', name='down_projection')
        self.logits_layer = Dense(units=self.vocab_size, activation='linear', name='logits')

    def sparse_loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    def build_graph(self):
        self._init_layers()
        embedded = self.embedding_layer(self.input_seq)
        embedded = self.embedding_dropout_layer(embedded)
        for layer_idx in list(range(len(self.fwd_layers)))[:-1]:
            fw_layer = self.fwd_layers[layer_idx]
            bw_layer = self.bwd_layers[layer_idx]
            if layer_idx == 0:
                fw_encoded = fw_layer(embedded)
                bw_encoded = bw_layer(embedded)
            else:
                fw_encoded = fw_layer(fw_encoded)
                bw_encoded = bw_layer(bw_encoded)

        final_fw_layer = self.fwd_layers[-1]
        final_bw_layer = self.bwd_layers[-1]
        final_fw = final_fw_layer(fw_encoded)
        final_bw = final_bw_layer(bw_encoded)
        # final_fw_context = Lambda(lambda x: x[:, :-2, :])(final_fw)
        # final_bw_context = Lambda(lambda x: x[:, 2:, :])(final_bw)
        # final_encoded_context = Concatenate(axis=-1)([final_fw_context, final_bw_context])

        fw_encoded_projection = self.proj_layer(final_fw)
        bw_encoded_projection = self.proj_layer(final_bw)

        logits_fw = self.logits_layer(fw_encoded_projection)
        logits_bw = self.logits_layer(bw_encoded_projection)

        self.W, self.b = self.logits_layer.weights[0], self.logits_layer.weights[-1]

        # Reshape input for sampled-softmax
        # inputs_reshaped = tf.reshape(final_encoded_context, [-1, int(final_encoded_context.get_shape()[2])])
        inputs_reshaped_fw = tf.reshape(fw_encoded_projection, [-1, int(fw_encoded_projection.get_shape()[2])])
        inputs_reshaped_bw = tf.reshape(bw_encoded_projection, [-1, int(bw_encoded_projection.get_shape()[2])])
        weights_reshaped = tf.transpose(self.W)
        labels_reshaped_fw = tf.reshape(self.output_seq, [-1, 1])
        labels_reshaped_bw = tf.reshape(self.output_seq_bw, [-1, 1])

        # Set up training loss
        # self.train_step_loss = tf.nn.sampled_softmax_loss(weights=weights_reshaped, biases=self.b, inputs=inputs_reshaped,
        #                                                   labels=labels_reshaped, num_sampled=self.num_sampled, num_classes=self.vocab_size)
        self.train_step_loss_fw = tf.nn.sampled_softmax_loss(weights=weights_reshaped, biases=self.b, inputs=inputs_reshaped_fw,
                                                             labels=labels_reshaped_fw, num_sampled=self.num_sampled, num_classes=self.vocab_size)
        self.train_step_loss_bw = tf.nn.sampled_softmax_loss(weights=weights_reshaped, biases=self.b, inputs=inputs_reshaped_bw,
                                                             labels=labels_reshaped_bw, num_sampled=self.num_sampled, num_classes=self.vocab_size)
        self.train_loss_fw = tf.reduce_mean(self.train_step_loss_fw)
        self.train_loss_bw = tf.reduce_mean(self.train_step_loss_bw)
        self.train_loss_joint = self.train_loss_fw + self.train_loss_bw
        self.train_op = self.optimizer.minimize(self.train_loss_joint)


        # self.valid_step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_seq, logits=logits)
        self.valid_step_loss_fw = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_seq, logits=logits_fw)
        self.valid_step_loss_bw = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_seq_bw, logits=logits_bw)
        self.valid_loss_fw = tf.reduce_mean(self.valid_step_loss_fw)
        self.valid_loss_bw = tf.reduce_mean(self.valid_step_loss_bw)
        self.valid_loss_joint = self.train_loss_fw + self.train_loss_bw

        # self.train_loss = tf.reduce_mean(self.train_step_loss)
        # self.valid_loss = tf.reduce_mean(self.valid_step_loss)

        # self.train_op = self.optimizer.minimize(self.train_loss)

        # self.model = Model(inputs=self.input_seq, outputs=logits)
        # self.model.compile(loss=self.sparse_loss, optimizer=self.opt_string, target_tensors=[self.output_seq])
        # self.model.summary()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adadelta', 'adam', 'sgd', 'momentum', 'rmsprop'}, 'Please select valid optimizer!'

        learning_rate = self.learning_rate

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.opt_string == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer()
        elif self.opt_string == 'adam':
            self.optimizer = tf.train.AdamOptimizer()
        elif self.opt_string == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def compile(self):
        self.sess.run(tf.global_variables_initializer())

    def _train_on_batch(self, x_batch, y_batch):
        _, loss_ = self.sess.run([self.train_op, self.train_loss_joint],
                                feed_dict={self.input_seq: x_batch,
                                self.output_seq: y_batch})
        return loss_

    def _eval_on_batch(self, x, y, normalize=False):
        valid_loss_ = self.sess.run(self.valid_loss_joint, feed_dict={self.input_seq: x,
                                                                self.output_seq: y})
        return valid_loss_

    def train(self):
        np.random.seed(7)

        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.valid_batch_size

        train_data = utils.LanguageModelData(data_file=self.train_file, vocab=self.vocab,
                                             max_seq_len=self.seq_len, batch_size=self.batch_size)
        valid_data = utils.LanguageModelData(data_file=self.valid_file, vocab=self.vocab,
                                             max_seq_len=self.seq_len, batch_size=self.valid_batch_size)

        # ckpt_fname = 'bidi_lm_{epoch:02d}-{val_loss:.2f}.h5'
        # ckpt = ModelCheckpoint(ckpt_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters,
        #                         epochs=self.epochs, validation_data=valid_datagen, validation_steps=n_valid_iters,
        #                         callbacks=[ckpt])

        for e in range(self.epochs):
            train_datagen = train_data.generate_batches(mask=False)
            valid_datagen = valid_data.generate_batches(mask=False)

            all_train_loss = 0
            batch_cnt = 0
            samples_cnt = 0

            for train_iter in range(n_train_iters):
                x_batch, y_batch = next(train_datagen)
                if (batch_cnt % self.eval_thresh == 0) and (batch_cnt > 0):
                    self.evaluate(valid_generator=valid_datagen)
                    self.save()

                loss_ = self._train_on_batch(x_batch, y_batch)

                batch_cnt += 1
                all_train_loss += loss_
                samples_cnt += len(x_batch)
                update_loss = all_train_loss / batch_cnt
                sys.stdout.write('\r num_samples_trained: {} \t|\t loss : {:8.3f} \t|\t prpl : {:8.3f}'.format((samples_cnt),
                                  (update_loss / 2), (np.exp(update_loss / 2))))

            # Epoch summary metrics
            print('\n\nEPOCH {} METRICS'.format(e+1))
            print('=' * 60)
            self.evaluate(valid_generator=valid_datagen, num_eval_examples=self.num_val_examples)
            print('\n\n')
            self.save(ckpt_name='model_epoch{}.ckpt'.format(e+1))

    def evaluate(self, valid_generator, num_eval_examples=10000):
        n_batch_iters = num_eval_examples // self.valid_batch_size
        total_val_loss = 0
        val_batch_cntr = 0
        print('\n\nvalidating...')
        for i in range(n_batch_iters):
            print('=', end='', flush=True)
            val_batch_cntr += 1
            x_val_batch, y_val_batch = next(valid_generator)
            valid_loss_ = self._eval_on_batch(x=x_val_batch, y=y_val_batch)
            total_val_loss += valid_loss_

        report_loss = total_val_loss / val_batch_cntr
        print('\nValidation metrics - loss: {:8.3f} | prpl: {:8.3f}\n'.format((report_loss / 2), (np.exp(report_loss / 2))))

    def save(self, save_path='./', ckpt_name='model.ckpt'):
        self.saver.save(self.sess, save_path + ckpt_name)
        print("Model saved to file:", save_path)

    def load(self, load_path='./', ckpt_name='model.ckpt'):
        self.saver.restore(self.sess, save_path + ckpt_name)
        print('Model restored.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=10000000)
    parser.add_argument('--num_val_examples', type=int, required=False, default=60000)
    parser.add_argument('--eval_thresh', type=int, required=False, default=1836000)
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--seq_len', type=int, required=False, default=30)
    parser.add_argument('--vocab_size', type=int, required=False, default=200002)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--hidden_dim', type=int, required=False, default=512)
    parser.add_argument('--kernel_width', type=int, required=False, default=3)
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=3)
    parser.add_argument('--optimizer', type=str, required=False, default='adagrad')
    parser.add_argument('--num_sampled', type=int, required=False, default=5000)
    parser.add_argument('--embedding_dropout', type=float, required=False, default=0.2)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--visualize_gradients', type=bool, required=False, default=False)

    args = parser.parse_args()

    lm = BidirectionalLM(args)
    print('Model instantiated')
    # print('Training...\n\n')