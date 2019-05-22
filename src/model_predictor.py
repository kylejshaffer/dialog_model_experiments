import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import utils

from nltk.tokenize import word_tokenize

def load_graph(graph_path, name):
    with tf.gfile.GFile(graph_path, 'rb') as infile:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(infile.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=name)

    return graph

class Vocab(object):
    def __init__(self, vocab_file):
        self.map = utils.get_vocab(vocab_file)
        self.inv_map = {v: k for k, v in self.map.items()}
        self.bos = self.map['<s>']
        self.eos = self.map['</s>']
        self.sod = self.map['<SOD>']
        self.unk = self.map['<UNK>']
        self.pad = self.map['<PAD>']

    def get_id(self, token):
        if token in self.map:
            return self.map[token]
        return self.unk

class ModelPredictor(object):
    def __init__(self, args):
        self.decode_len_limit = 100
        self.vocab = Vocab(args.vocab_file)
        self.graph = load_graph(args.graph_file, 'lstm_chatbot')
        self.x_ph = self.graph.get_tensor_by_name('lstm_chatbot/x_input:0')
        self.predictions= self.graph.get_tensor_by_name('lstm_chatbot/logits/MatMul:0')
        self.sess = tf.Session(graph=self.graph)

    def tokenize(self, input_sent, use_nltk=False):
        if use_nltk:
            toks = word_tokenize(input_sent)
        else:
            toks = input_sent.strip().split()
        tok_ids = [self.vocab.get_id(w) for w in toks]
        tok_ids.insert(0, self.vocab.bos)
        tok_ids.insert(0, self.vocab.sod)
        return tok_ids

    def predict(self, x_input, use_nltk=False):
        x_input = self.tokenize(x_input)

        src_seq = np.asarray(x_input)
        src_seq = np.reshape(a=src_seq, newshape=(1, len(x_input)))

        decoded_tokens = []
        target_seq = np.zeros((1, self.decode_len_limit), dtype='int32')
        target_seq[0, 0] = self.vocab.get_id('<s>')

        print('Generating output...')
        for i in range(self.decode_len_limit):
            print('=', end='', flush=True)
            output = self.sess.run(self.predictions, feed_dict={self.x_ph: src_seq})
        return

    def score_sent(self, sent, use_nltk=False, normalize_with_length=True):
        sent_ids = self.tokenize(sent, use_nltk=use_nltk)
        prediction_sent_ids = np.reshape(a=np.array(sent_ids), newshape=(1, len(sent_ids)))

        logprobs_fw = self.sess.run(self.predictions_fw, feed_dict={self.x_ph: prediction_sent_ids})
        logprobs_bw = np.flip(self.sess.run(self.predictions_bw, feed_dict={self.x_ph: prediction_sent_ids}), axis=0)
        print(logprobs_bw.shape)

        output_fw, output_bw = [], []
        for idx, word_id in enumerate(sent_ids):
            fw_score = logprobs_fw[idx, word_id]
            bw_score = logprobs_bw[idx, word_id]
            output_fw.append(fw_score)
            output_bw.append(bw_score)

        final_scores = [((fw_score + bw_score) / 2) for fw_score, bw_score in zip(output_fw, output_bw)]
        
        if normalize_with_length:
            sent_score = sum(final_scores) / len(final_scores)
        else:
            sent_score = sum(final_scores)
        return sent_score

    def rank_slot_entries(self, sent, word_idx, use_nltk=False, n_best=100, return_fwd_bwd_separate=False):
        sent_ids = self.tokenize(sent, use_nltk=use_nltk)
        prediction_sent_ids = np.reshape(a=np.array(sent_ids), newshape=(1, len(sent_ids)))

        logprobs_fw = self.sess.run(self.predictions_fw, feed_dict={self.x_ph: prediction_sent_ids})
        logprobs_bw = np.flip(self.sess.run(self.predictions_bw, feed_dict={self.x_ph: prediction_sent_ids}), axis=0)

        word_logprobs_fw = logprobs_fw[word_idx, :]
        word_logprobs_bw = logprobs_bw[word_idx, :]

        # Return forward and backward log-probs separately
        if return_fwd_bwd_separate:
            fwd_word_ids_ranked = sorted(list(zip(range(len(word_logprobs_fw)), word_logprobs_fw)), key=lambda x: x[-1], reverse=True)[:n_best]
            bwd_word_ids_ranked = sorted(list(zip(range(len(word_logprobs_bw)), word_logprobs_bw)), key=lambda x: x[-1], reverse=True)[:n_best]
            fw_words = [self.vocab.inv_map[w_id] for w_id, w_score in fwd_word_ids_ranked]
            bw_words = [self.vocab.inv_map[w_id] for w_id, w_score in bwd_word_ids_ranked]
            return fw_words, bw_words

        # Take the unweighted mean of the fowrard log-probabilities and backward log-probabilities
        word_slot_scores = [((fw_score + bw_score) / 2) for fw_score, bw_score in zip(word_logprobs_fw, word_logprobs_bw)]
        scored_words = sorted(list(zip(range(len(word_slot_scores)), word_slot_scores)), key=lambda x: x[-1], reverse=True)[:n_best]
        ranked_word_ids = [scored_word[0] for scored_word in scored_words]
        ranked_words = [self.vocab.inv_map[w_id] for w_id in ranked_word_ids]

        return ranked_words

# Test some predictions and print out qualitative results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file')
    parser.add_argument('--vocab_file')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_predictor = ModelPredictor(args)
    print('Model instantiated!')

    # Test inference
    sent1 = 'Trump told White House counsel that he wanted to order the Justice Department to prosecute his political adversaries.'
    print(sent1)
    output = model_predictor.score_sent(sent1, use_nltk=True)
    print('Log-prob of dummy sentence 1:')
    print(output)
    print()

    sent2 = "best site for facebook likes , twitter and youtube likes . register here : it's free . :-)"
    print(sent2)
    output2 = model_predictor.score_sent(sent2, use_nltk=False)
    print('Log-prob of dummy sentence 2:')
    print(output2)
    print()

    sent3 = "sweet 16 Emoji_75 gonna be awesome ♥"
    print(sent3)
    output3 = model_predictor.score_sent(sent3, use_nltk=False)
    print('Log-prob of dummy sentence 3:')
    print(output3)
    print()

    sent4 = "sweet 16 Emoji_75 gonna be weird ♥"
    print(sent4)
    output4 = model_predictor.score_sent(sent4, use_nltk=False)
    print('Log-prob of dummy sentence 4:')
    print(output4)
    print()

    # Get ranked output for a slot in sentence
    # Here, we're getting the top-suggested words for 'awesome' in sentence 3 above
    slot_words = model_predictor.rank_slot_entries(sent=sent3, word_idx=5, n_best=20)
    print('Top-ranked words at this position:')
    print(slot_words)
    print()

    # Get forward and backward words separately
    word_idx = 5
    fw_words, bw_words = model_predictor.rank_slot_entries(sent=sent3, word_idx=word_idx, n_best=20, return_fwd_bwd_separate=True)
    print('Top replacements for:')
    display_sent = sent3.strip().split()
    display_sent[word_idx] = '__SLOT__'
    print(' '.join(display_sent))
    print('Top-ranked words from forward-pass:')
    print(fw_words)
    print()
    print('Top-ranked words from backward-pass:')
    print(bw_words)
    print()

    word_idx = 9
    s = "fun crowd last night ! tonight's gonna be a great show . it's nearly the weekend ! ! ! ! :-)"
    slot_words = model_predictor.rank_slot_entries(sent=s, word_idx=word_idx, n_best=20)
    print(slot_words)
    fw_words, bw_words = model_predictor.rank_slot_entries(sent=s, word_idx=word_idx, n_best=20, return_fwd_bwd_separate=True)
    print(fw_words)
    print(bw_words)