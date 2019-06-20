import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def get_bpe_tokenizer(input_text:list, tgt_vocab_size:int, save_filename:str=None):
    print('Getting BPE vocab...')
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                            input_text, target_vocab_size=tgt_vocab_size)
    print("{} BPE tokens found in text...".format(tokenizer.vocab_size))

    if save_filename is not None:
        tokenizer.save_to_file(save_filename)
        print("BPE tokenizer saved!")

    return tokenizer

class DataProcessor(object):
    def __init__(self, max_len:int, tokenizer, train_file:str, valid_file:str, batch_size:int):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.valid_file = valid_file
        self.bos = self.tokenizer.vocab_size
        self.eos = self.tokenizer.vocab_size + 1
        self.vocab_size = self.tokenizer.vocab_size + 2
        self.batch_size = batch_size
        
    def pad_batch(self, encoder_batch, decoder_batch):
        max_enc_length = self.max_len # max([len(s) for s in encoder_batch])
        max_dec_length = self.max_len # max([len(s) for s in decoder_batch])

        if max_enc_length > self.max_len:
            max_enc_length = self.max_len
        if max_dec_length > self.max_len:
            meax_dec_length = self.max_len

        enc_container, dec_in_container, dec_out_container = [], [], []
        for enc_seq, dec_seq in zip(encoder_batch, decoder_batch):
            if len(enc_seq) >= max_enc_length:
                enc_seq = enc_seq[:max_enc_length]
            enc_seq.insert(0, self.bos)
            enc_seq.append(self.eos)
            enc_container.append(enc_seq)

            if len(dec_seq) >= max_dec_length:
                dec_seq = dec_seq[:max_dec_length]
            dec_out_seq = dec_seq[:]
            dec_out_seq.append(self.eos)
            dec_seq.insert(0, self.bos)

            dec_in_container.append(dec_seq)
            dec_out_container.append(dec_out_seq)

        enc_padded = tf.keras.preprocessing.sequence.pad_sequences(enc_container, padding='post', maxlen=self.max_len)
        dec_in_padded = tf.keras.preprocessing.sequence.pad_sequences(dec_in_container, padding='post', maxlen=self.max_len)
        dec_out_padded = tf.keras.preprocessing.sequence.pad_sequences(dec_out_container, padding='post', maxlen=self.max_len)

        return enc_padded, dec_in_padded, dec_out_padded

    def get_line(self, data_file):
        with open(data_file, mode='r') as infile:
            for line in infile:
                context, response, _ = line.strip().split('\t')
                context_bpe, response_bpe = self.tokenizer.encode(context), self.tokenizer.encode(response)
                
                yield context_bpe, response_bpe

    def batch_generator(self, mode:str='train'):
        assert mode in {'train', 'valid'}, "Please select as valid mode from: {train, valid}!"
        data_file = self.train_file if mode == 'train' else self.valid_file
        
        while True:
            encoder_batch, decoder_batch = [], []
            for context, resp in self.get_line(data_file):
                encoder_batch.append(context)
                decoder_batch.append(resp)
                if len(encoder_batch) == self.batch_size:
                    enc_padded, dec_in_padded, dec_out_padded = self.pad_batch(encoder_batch, decoder_batch)
                    yield [enc_padded, dec_in_padded], dec_out_padded

                    encoder_batch, decoder_batch = [], []

            # Check for non-empty batches
            if len(encoder_batch) > 0:
                enc_padded, dec_in_padded, dec_out_padded = self.pad_batch(encoder_batch, decoder_batch)
                yield [enc_padded, dec_in_padded], dec_out_padded


# Run test as sanity check
if __name__ == '__main__':
    def load_movie_text(input_file:str):
        movie_lines = []
        with open(input_file, 'r') as infile:
            for line in infile:
                movie_lines.append(line.strip())

        return movie_lines

    default_movie_file = '/Users/kyleshaffer/Documents/cornell movie-dialogs corpus/dialogs_text.txt'
    train_file = "/Users/kyleshaffer/Documents/cornell movie-dialogs corpus/cornell_movie_dialog_no_context_train.txt"
    valid_file = "/Users/kyleshaffer/Documents/cornell movie-dialogs corpus/cornell_movie_dialog_no_context_valid.txt"

    print(os.path.exists(train_file))

    movie_lines = load_movie_text(default_movie_file)

    bpe_tok = get_bpe_tokenizer(movie_lines, 15000)

    data_processor = DataProcessor(max_len=100, tokenizer=bpe_tok, train_file=train_file,
                                   valid_file=valid_file, batch_size=20)

    train_datagen = data_processor.batch_generator(mode='train')
    for _ in range(100):
        x, y = next(train_datagen)
        x_in, y_in = x
        print(x_in.shape, y_in.shape, y.shape)
