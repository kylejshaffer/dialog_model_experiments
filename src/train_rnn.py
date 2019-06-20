from argparse import ArgumentParser

from recurrent import RNNSeq2Seq, HanRnnSeq2Seq
import os
import data_utils

def train(args):
    vocab = data_utils.get_vocab(vocab_file=args.vocab_file, min_freq=args.min_vocab_freq)

    print('Vocab loaded...')
    print('VOCAB SIZE = ', len(vocab))

    if args.model_type == 'rnn':
        print('Training RNN...')
        rnn = RNNSeq2Seq(args=args, vocab=vocab)
        rnn.train()
    elif args.model_type == 'han_rnn':
        print('Training HAN-RNN...')
        han_rnn = HanRnnSeq2Seq(args=args, vocab=vocab)
        han_rnn.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--model_name', type=str, required=False, default='lstm_att_movie_transfer_chatbot_epoch{:02d}_loss{:.3f}.h5')
    parser.add_argument('--train_file', type=str, required=False, default='/data/users/kyle.shaffer/ased_data/combined_multilabel_train.jl')
    parser.add_argument('--valid_file', type=str, required=False, default='/data/users/kyle.shaffer/ased_data/combined_multilabel_valid.jl')
    parser.add_argument('--vocab_file', type=str, required=False, default='')
    parser.add_argument('--min_vocab_freq', type=int, required=False, default=0)
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--model_type', type=str, required=False, default='han_rnn')
    parser.add_argument('--encoder_type', type=str, required=False, default='uni')
    parser.add_argument('--train_from', type=str, required=False, default='')
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--rec_cell', type=str, required=False, default='lstm')
    parser.add_argument('--embedding_dim', type=int, required=False, default=300)
    parser.add_argument('--encoder_dim', type=int, required=False, default=256)
    parser.add_argument('--decoder_dim', type=int, required=False, default=256)
    parser.add_argument('--num_encoder_layers', type=int, required=False, default=2)
    parser.add_argument('--num_decoder_layers', type=int, required=False, default=1)
    parser.add_argument('--n_train_examples', type=int, required=False, default=158037)
    parser.add_argument('--n_valid_examples', type=int, required=False, default=17561)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    train(args)
