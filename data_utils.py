import numpy as np
import tensorflow as tf


def get_bpe_vocab(vocab_file):
    special_toks = ['<s>', '</s>', '<UNK>', '<PAD>']
    c = {}
    word_idx = 1
    with tf.gfile.GFile(vocab_file, 'r') as infile:
        for line in infile:
            if line.strip() in special_toks:
                continue
            c[line.strip()] = word_idx
            word_idx += 1
        for st in special_toks:
            if st not in c.keys():
                if st == '<PAD>':
                    c[st] = 0
                else:
                    c[st] = max(c.values()) + 1
    print('VOCAB SIZE = {}'.format(len(c)))
    return c

class S2SProcessing(object):
    def __init__(self, data_file, vocab, max_seq_len, batch_size=256, shuffle_batch=False):
        self.data_file = data_file
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch

    def get_tokid(self, word):
        if word in self.vocab.keys():
            tokid = self.vocab[word]
        else:
            tokid = self.vocab['<UNK>']
        return tokid

    def get_orig_word(self, tokid):
        orig_word = self.inv_vocab[int(tokid)]
        return orig_word

    def get_line(self):
        bos = '<s>'
        eos = '</s>'
        with tf.gfile.GFile(self.data_file, 'r') as infile:
            for line in infile:
                encoder_words, decoder_words = line.strip().split('\t')
                encoder_words, decoder_in_words = encoder_words.strip().split(), decoder_words.strip().split()
                decoder_out_words = decoder_in_words[:]
                # Truncate sentences that go past `max_seq_len`
                if len(encoder_words) > self.max_seq_len:
                    encoder_words = encoder_words[: self.max_seq_len]
                if len(decoder_in_words) > self.max_seq_len:
                    decoder_in_words = decoder_in_words[: self.max_seq_len]
                    decoder_out_words = decoder_out_words[: self.max_seq_len]
                # Insert special BOS/EOS tokens
                encoder_words.insert(0, bos)
                decoder_in_words.insert(0, bos)
                decoder_out_words.append(eos)
                # Lookup IDs
                encoder_toks = [self.get_tokid(w) for w in encoder_words]
                decoder_in_toks = [self.get_tokid(w) for w in decoder_in_words]
                decoder_out_toks = [self.get_tokid(w) for w in decoder_out_words]
                yield encoder_toks, decoder_in_toks, decoder_out_toks

    def s2s_padding(self, encoder_batch, decoder_in_batch, decoder_out_batch,
                    encoder_batch_lengths, decoder_batch_lengths):
        pad = self.vocab['<PAD>']
        max_encoder_length = max(encoder_batch_lengths)
        max_decoder_length = max(decoder_batch_lengths)

        for idx, batch in enumerate(encoder_batch):
            if len(batch) < max_encoder_length:
                diff = max_encoder_length - len(batch)
                encoder_padding = [pad] * diff
                encoder_batch_padded = batch + encoder_padding
                encoder_batch[idx] = encoder_batch_padded
            if len(decoder_in_batch[idx]) < max_decoder_length:
                decoder_in_seq = decoder_in_batch[idx]
                decoder_out_seq = decoder_out_batch[idx]
                diff = max_decoder_length - len(decoder_in_seq)
                decoder_padding = [pad] * diff
                decoder_in_seq_padded = decoder_in_seq + decoder_padding
                decoder_in_batch[idx] = decoder_in_seq_padded
                decoder_out_seq_padded = decoder_out_seq + decoder_padding
                decoder_out_batch[idx] = decoder_out_seq_padded

        return encoder_batch, decoder_in_batch, decoder_out_batch

    def generate_s2s_batches(self):
        encoder_batch, decoder_in_batch, decoder_out_batch  = [], [], []
        encoder_batch_lengths, decoder_batch_lengths = [], []
        for encoder_toks, decoder_in_toks, decoder_out_toks in self.get_line():
            # Build up batches
            encoder_batch.append(encoder_toks)
            decoder_in_batch.append(decoder_in_toks)
            decoder_out_batch.append(decoder_out_toks)
            # Store lengths
            encoder_batch_lengths.append(len(encoder_toks))
            decoder_batch_lengths.append(len(decoder_in_toks))
            if len(encoder_batch) == self.batch_size:
                encoder_batch, decoder_in_batch, decoder_out_batch = self.s2s_padding(encoder_batch,
                                                                                decoder_in_batch,
                                                                                decoder_out_batch,
                                                                                encoder_batch_lengths=encoder_batch_lengths,
                                                                                decoder_batch_lengths=decoder_batch_lengths)
                yield np.asarray(encoder_batch), np.asarray(decoder_in_batch), np.asarray(decoder_out_batch)
                # Reset batch containers
                encoder_batch, decoder_in_batch, decoder_out_batch  = [], [], []
                encoder_batch_lengths, decoder_batch_lengths = [], []
        if len(encoder_batch) > 0:
            encoder_batch, decoder_in_batch, decoder_out_batch = self.s2s_padding(encoder_batch,
                                                                            decoder_in_batch,
                                                                            decoder_out_batch,
                                                                            encoder_batch_lengths=encoder_batch_lengths,
                                                                            decoder_batch_lengths=decoder_batch_lengths)
            yield np.asarray(encoder_batch), np.asarray(decoder_in_batch), np.asarray(decoder_out_batch)
            