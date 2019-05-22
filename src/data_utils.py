import numpy as np
import tensorflow as tf


def get_vocab(vocab_file, min_freq:int=3):
    special_toks = ['<UNK>', '<PAD>', '<s>', '</s>']
    c = {}
    word_idx = 1
    with tf.gfile.GFile(vocab_file, 'r') as infile:
        for line in infile:
            w, count = line.strip().split('\t')
            count = int(count)
            if (w in special_toks) or (count < min_freq):
                continue
            c[w.strip()] = word_idx
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
    def __init__(self, train_file, valid_file, vocab, encoder_max_len=300, decoder_max_len=100,
                 batch_size=256, shuffle_batch=False, model_type='transformer'):
        self.train_file = train_file
        self.valid_file = valid_file
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.model_type = model_type

    def get_tokid(self, word):
        if word in self.vocab.keys():
            tokid = self.vocab[word]
        else:
            tokid = self.vocab['<UNK>']
        return tokid

    def get_orig_word(self, tokid):
        orig_word = self.inv_vocab[int(tokid)]
        return orig_word

    def get_line(self, data_file):
        bos = '<s>'
        eos = '</s>'
        with tf.gfile.GFile(data_file, 'r') as infile:
            for line_ix, line in enumerate(infile):
                line_split = line.strip().split('\t')
                # if len(line_split) != 2:
                #     print(line_ix)
                #     print(line)
                #     print(line_split)
                encoder_words, decoder_words, _ = line.strip().split('\t')
                encoder_words = encoder_words.replace('<SOD>', '')
                decoder_words = decoder_words.replace('<EOD>', '')
                encoder_words, decoder_in_words = encoder_words.strip().split(), decoder_words.strip().split()
                decoder_out_words = decoder_in_words[:]
                # Truncate sentences that go past `max_seq_len`
                if len(encoder_words) > self.encoder_max_len:
                    encoder_words = encoder_words[: self.encoder_max_len]
                if len(decoder_in_words) > self.decoder_max_len:
                    decoder_in_words = decoder_in_words[: self.decoder_max_len]
                    decoder_out_words = decoder_out_words[: self.decoder_max_len]
                # Insert special BOS/EOS tokens
                if encoder_words[0] == '<SOD>':
                    encoder_words.insert(1, bos)
                else:
                    encoder_words.insert(0, bos)
                decoder_in_words.insert(0, bos)

                if decoder_out_words[-1] == '<EOD>':
                    decoder_out_words.insert(-1, eos)
                else:
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

    def generate_s2s_batches(self, mode='train'):
        assert mode in {'train', 'valid'}, 'Supply a mode that is either `train` or `valid`!'
        data_file = self.train_file if mode == 'train' else self.valid_file

        while True:
            encoder_batch, decoder_in_batch, decoder_out_batch  = [], [], []
            encoder_batch_lengths, decoder_batch_lengths = [], []
            for encoder_toks, decoder_in_toks, decoder_out_toks in self.get_line(data_file=data_file):
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
                    # print(np.asarray(encoder_batch).shape, np.asarray(decoder_in_batch).shape)
                    if self.model_type == 'transformer':
                        yield [np.asarray(encoder_batch), np.asarray(decoder_in_batch)], None # , np.asarray(decoder_out_batch)
                    elif self.model_type == 'recurrent':
                        yield [np.asarray(encoder_batch), np.asarray(decoder_in_batch)], np.asarray(decoder_out_batch)# ,\
                            # np.asarray(decoder_batch_lengths)
                    elif self.model_type == 'cnn':
                        encoder_in, decoder_in, decoder_out = np.asarray(encoder_batch), np.asarray(decoder_in_batch), np.asarray(decoder_out_batch)
                        encoder_positions = np.asarray([list(range(encoder_in.shape[1])) for _ in range(encoder_in.shape[0])])
                        decoder_positions = np.asarray([list(range(decoder_in.shape[1])) for _ in range(decoder_in.shape[0])])
                        yield [encoder_in, encoder_positions, decoder_in, decoder_positions], decoder_out
                    # Reset batch containers
                    encoder_batch, decoder_in_batch, decoder_out_batch  = [], [], []
                    encoder_batch_lengths, decoder_batch_lengths = [], []

            if len(encoder_batch) > 0:
                encoder_batch, decoder_in_batch, decoder_out_batch = self.s2s_padding(encoder_batch,
                                                                                decoder_in_batch,
                                                                                decoder_out_batch,
                                                                                encoder_batch_lengths=encoder_batch_lengths,
                                                                                decoder_batch_lengths=decoder_batch_lengths)
                
                # print(np.asarray(encoder_batch).shape, np.asarray(decoder_in_batch).shape)
                if self.model_type == 'transformer':
                    yield [np.asarray(encoder_batch), np.asarray(decoder_in_batch)], None # , np.asarray(decoder_out_batch)
                elif self.model_type == 'recurrent':
                    yield [np.asarray(encoder_batch), np.asarray(decoder_in_batch)], np.asarray(decoder_out_batch)# ,\
                        # np.asarray(decoder_batch_lengths)
                elif self.model_type == 'cnn':
                    encoder_in, decoder_in, decoder_out = np.asarray(encoder_batch), np.asarray(decoder_in_batch), np.asarray(decoder_out_batch)
                    encoder_positions = np.asarray([list(range(encoder_in.shape[1])) for _ in range(encoder_in.shape[0])])
                    decoder_positions = np.asarray([list(range(decoder_in.shape[1])) for _ in range(decoder_in.shape[0])])
                    yield [encoder_in, encoder_positions, decoder_in, decoder_positions], decoder_out

class HanS2SProcessing(S2SProcessing):
    def __init__(self, train_file, valid_file, vocab, encoder_max_len=300, decoder_max_len=100,
                 batch_size=256, shuffle_batch=False, model_type='transformer'):
        super(HanS2SProcessing, self).__init__(train_file, valid_file, vocab, encoder_max_len=300, decoder_max_len=100,
                                               batch_size=256, shuffle_batch=False, model_type='transformer')

    def get_line(self, data_file):
        bos = '<s>'
        eos = '</s>'
        with tf.gfile.GFile(data_file, 'r') as infile:
            for line_ix, line in enumerate(infile):
                context_words, current_words, decoder_words = line.strip().split('\t')
                # encoder_words = encoder_words.replace('<SOD>', '')
                # decoder_words = decoder_words.replace('<EOD>', '')
                context_words = context_words.strip().split()
                current_words = current_words.strip().split()
                decoder_in_words = decoder_words.strip().split()
                decoder_out_words = decoder_in_words[:]
                # Truncate sentences that go past `max_seq_len`
                if len(context_words) > self.encoder_max_len:
                    context_words = context_words[: self.encoder_max_len]
                if len(current_words) > self.encoder_max_len:
                    current_words = current_words[: self.encoder_max_len]
                if len(decoder_in_words) > self.decoder_max_len:
                    decoder_in_words = decoder_in_words[: self.decoder_max_len]
                    decoder_out_words = decoder_out_words[: self.decoder_max_len]
                # Insert special BOS/EOS tokens
                if context_words[0] == '<SOD>':
                    context_words.insert(1, bos)
                else:
                    context_words.insert(0, bos)
                current_words.insert(0, bos)
                decoder_in_words.insert(0, bos)

                if decoder_out_words[-1] == '<EOD>':
                    decoder_out_words.insert(-1, eos)
                else:
                    decoder_out_words.append(eos)
                # Lookup IDs
                context_toks = [self.get_tokid(w) for w in context_words]
                current_toks = [self.get_tokid(w) for w in current_words]
                decoder_in_toks = [self.get_tokid(w) for w in decoder_in_words]
                decoder_out_toks = [self.get_tokid(w) for w in decoder_out_words]
                assert len(decoder_in_toks) == len(decoder_out_toks), "Mismatch in decoder tokens!"
                yield context_toks, current_toks, decoder_in_toks, decoder_out_toks

    def s2s_padding(self, context_batch, current_batch, decoder_in_batch, decoder_out_batch,
                    context_batch_lengths, current_batch_lengths, decoder_batch_lengths):
        pad = self.vocab['<PAD>']
        max_context_length = max(context_batch_lengths)
        max_current_length = max(current_batch_lengths)
        max_decoder_length = max(decoder_batch_lengths)

        for idx, batch in enumerate(current_batch):
            # Pad current input
            if len(batch) < max_current_length:
                diff = max_current_length - len(batch)
                current_padding = [pad] * diff
                current_batch_padded = batch + current_padding
                current_batch[idx] = current_batch_padded
            # Pad previous context input
            if len(context_batch[idx]) < max_context_length:
                context_batch_ = context_batch[idx]
                diff = max_context_length - len(context_batch_)
                context_padding = [pad] * diff
                context_batch_padded = context_batch_ + context_padding
                context_batch[idx] = context_batch_padded
            # Pad decoder input
            if len(decoder_in_batch[idx]) < max_decoder_length:
                decoder_in_seq = decoder_in_batch[idx]
                decoder_out_seq = decoder_out_batch[idx]
                diff = max_decoder_length - len(decoder_in_seq)
                decoder_padding = [pad] * diff
                decoder_in_seq_padded = decoder_in_seq + decoder_padding
                decoder_in_batch[idx] = decoder_in_seq_padded
                decoder_out_seq_padded = decoder_out_seq + decoder_padding
                decoder_out_batch[idx] = decoder_out_seq_padded

        return context_batch, current_batch, decoder_in_batch, decoder_out_batch

    def generate_s2s_batches(self, mode='train'):
        assert mode in {'train', 'valid'}, 'Supply a mode that is either `train` or `valid`!'
        data_file = self.train_file if mode == 'train' else self.valid_file

        while True:
            context_batch, current_batch, decoder_in_batch, decoder_out_batch  = [], [], [], []
            context_batch_lengths, current_batch_lengths, decoder_batch_lengths = [], [], []
            for context_toks, current_toks, decoder_in_toks, decoder_out_toks in self.get_line(data_file=data_file):
                # Build up batches
                context_batch.append(context_toks)
                current_batch.append(current_toks)
                decoder_in_batch.append(decoder_in_toks)
                decoder_out_batch.append(decoder_out_toks)
                # Store lengths
                context_batch_lengths.append(len(context_toks))
                current_batch_lengths.append(len(current_toks))
                decoder_batch_lengths.append(len(decoder_in_toks))
                if len(current_batch) == self.batch_size:
                    context_batch_pad, current_batch_pad, decoder_in_batch_pad, decoder_out_batch_pad = self.s2s_padding(context_batch,
                                                                                    current_batch,
                                                                                    decoder_in_batch,
                                                                                    decoder_out_batch,
                                                                                    context_batch_lengths=context_batch_lengths,
                                                                                    current_batch_lengths=current_batch_lengths,
                                                                                    decoder_batch_lengths=decoder_batch_lengths)
                    # print(np.asarray(encoder_batch).shape, np.asarray(decoder_in_batch).shape)
                    yield [np.asarray(context_batch_pad), np.asarray(current_batch_pad), np.asarray(decoder_in_batch_pad)], np.asarray(decoder_out_batch_pad)
                    # Reset batch containers
                    encoder_batch, decoder_in_batch, decoder_out_batch  = [], [], []
                    encoder_batch_lengths, decoder_batch_lengths = [], []

            if len(encoder_batch) > 0:
                context_batch_pad, current_batch_pad, decoder_in_batch_pad, decoder_out_batch_pad = self.s2s_padding(context_batch,
                                                                                    current_batch,
                                                                                    decoder_in_batch,
                                                                                    decoder_out_batch,
                                                                                    context_batch_lengths=context_batch_lengths,
                                                                                    current_batch_lengths=current_batch_lengths,
                                                                                    decoder_batch_lengths=decoder_batch_lengths)
                
                # print(np.asarray(encoder_batch).shape, np.asarray(decoder_in_batch).shape)
                yield [np.asarray(context_batch_pad), np.asarray(current_batch_pad), np.asarray(decoder_in_batch_pad)], np.asarray(decoder_out_batch_pad)
