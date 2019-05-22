import keras.backend as K
import data_utils
import math
import os
import sys

import keras.backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Add, Lambda, GlobalMaxPooling1D, GRU, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adagrad, Adam, RMSprop, SGD


class ConvSeq2Seq(RNNSeq2Seq):
    def __init__(self, args, vocab):
        self.opt_string = args.optimizer
        self.embedding_dim = args.embedding_dim
        self.max_len = 100
        self.pos_embedding_dim = 100
        self.encoder_dim = 512
        self.decoder_dim = 512
        self.num_encoder_layers = 10
        self.num_decoder_layers = 10
        self.kernel_width = 5
        self.model_name = args.model_name
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
        self.eval_thresh = 50000

        self._choose_optimizer()
        self.build_model()
        self.saver = tf.train.Saver()

    def conv_block(self, x_input, block_type, residual=False):
        if block_type == 'encoder':
            cut_size = int(self.encoder_dim / 2)
            padding = 'same'
        elif block_type == 'decoder':
            cut_size = int(self.decoder_dim / 2)
            padding = 'causal'
        
        conv_op = Conv1D(filters=self.encoder_dim, kernel_size=self.kernel_width, padding=padding)
        conv_out = conv_op(x_input)
        linear_output = conv_out[:, :, :cut_size]
        gated_output = conv_out[:, :, cut_size:]
        final_output = linear_output * K.sigmoid(gated_output)

        if residual:
            final_output = final_output + x_input

        return final_output

    def _init_layers(self):
        self.encoder_words_input = Input(shape=(None,), dtype='int32', name='encoder_words_input')
        self.decoder_words_input = Input(shape=(None,), dtype='int32', name='decoder_words_input')
        self.encoder_pos_input = Input(shape=(None,), dtype='int32', name='encoder_pos_input')
        self.decoder_pos_input = Input(shape=(None,), dtype='int32', name='decoder_pos_input')
        # self.decoder_target = Input(shape=(None,), dtype='int32', name='decoder_target')
        self.decoder_target = tf.placeholder(dtype='int32', shape=[None, None])

        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=False)
        self.position_embedding = Embedding(input_dim=self.max_len, output_dim=self.pos_embedding_dim)
        # self.encoder_layers = [GatedConvBlock(Conv1D(filters=self.encoder_dim, kernel_size=self.kernel_width, padding='same')) for _ in range(self.num_encoder_layers)]
        # self.decoder_layers = [GatedConvBlock(Conv1D(filters=self.decoder_dim, kernel_size=self.kernel_width, padding='causal')) for _ in range(self.num_decoder_layers)]

    def build_model(self):
        self._init_layers()

        # Encoder
        encoder_word_embed = self.embedding_layer(self.encoder_words_input)
        encoder_pos_embed = self.position_embedding(self.encoder_pos_input)
        encoder_word_embed = Dropout(0.3)(encoder_word_embed)
        encoder_pos_embed = Dropout(0.2)(encoder_pos_embed)
        encoder_embed = Concatenate(axis=-1)([encoder_word_embed, encoder_pos_embed])

        for i in range(self.num_encoder_layers):
            if i == 0:
                encoder_output = Lambda(lambda x: self.conv_block(x, block_type='encoder'))(encoder_embed)
            else:
                encoder_output = Lambda(lambda x: self.conv_block(x, block_type='encoder'))(encoder_output)
        encoder_last = encoder_output[:,-1,:]
        encoder_states = [encoder_last, encoder_last]

        # Decoder
        decoder_word_embed = self.embedding_layer(self.decoder_words_input)
        decoder_word_embed = Dropout(0.3)(decoder_word_embed)
        # decoder_pos_embed = self.position_embedding(self.decoder_pos_input)
        # decoder_embed = Concatenate(axis=-1)([decoder_word_embed, decoder_pos_embed])

        # for i in range(self.num_decoder_layers):
        #     if i == 0:
        #         decoder_output = Lambda(lambda x: self.conv_block(x, block_type='decoder'))(decoder_embed)
        #     else:
        #         decoder_output = Lambda(lambda x: self.conv_block(x, block_type='decoder'))(decoder_output)
        decoder = LSTM(units=self.decoder_dim, return_sequences=True)
        decoder_output = decoder(decoder_embedding, initial_state=encoder_states)

        # Attention
        attention = Dot(axes=[2, 2])([decoder_output, encoder_output])
        attention = Activation('softmax', name='attention_probs')(attention)
        context = Dot(axes=[2, 1])([attention, encoder_output])
        decoder_combined_context = Concatenate()([context, decoder_output])
        decoder_combined_context = Dropout(0.2)(decoder_combined_context)
        hidden_projection = Dense(units=100, activation='relu')(decoder_combined_context)
        logits_out = Dense(units=self.vocab_size, activation='linear', name='logits_out')(hidden_projection)

        self.model = Model(inputs=[self.encoder_words_input, self.encoder_pos_input, 
                                   self.decoder_words_input, self.decoder_pos_input], outputs=[logits_out])

        opt = Adagrad()
        self.model.compile(loss=self.sparse_loss, optimizer=opt, target_tensors=[self.decoder_target])
        self.model.summary()

        return

    def train_keras(self):
        np.random.seed(7)
        debug = False

        test_sents = ["Hey , how are you doing ?",
                      "i'm just looking for someone to talk to .",
                      "You should give me your credit card number and then we can get this thing rolling !"]

        model_dir = '/data/users/kyle.shaffer/chat_models'
        tboard = TensorBoard(log_dir='conv_tboard_logs', write_graph=False, histogram_freq=0, write_grads=False, update_freq='batch')

        n_train_iters = math.ceil(self.n_train_examples / self.batch_size)
        n_valid_iters = math.ceil(self.n_valid_examples / self.batch_size)

        s2s_processor = data_utils.S2SProcessing(train_file=self.train_file, valid_file=self.valid_file, vocab=self.vocab,
                                                 batch_size=self.batch_size, encoder_max_len=100, decoder_max_len=100,
                                                 shuffle_batch=True, model_type='cnn')

        if debug:
            print('TRAINING')
            for i in range(n_train_iters):
                sys.stdout.write('\r {}'.format(i))
                x, y = next(train_datagen)


            print('VALIDATION')

            for i in range(n_valid_iters):
                sys.stdout.write('\r {}'.format(i))
                x, y = next(valid_datagen)

        for e in range(self.n_epochs):
            train_datagen = s2s_processor.generate_s2s_batches(mode='train')
            valid_datagen = s2s_processor.generate_s2s_batches(mode='valid')
            hist = self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters, validation_data=valid_datagen,
                                validation_steps=n_valid_iters, epochs=1, shuffle=False, callbacks=[tboard])

            # Look at qualitative output
            print('Testing input sentences...')
            for sent in test_sents:
                print('==>', sent)
                response = self.decode_sequence(input_seq=sent.split(), model_type='cnn')
                print('==>', response)
                print()

        print('DONE TRAINING')
        return
        