from argparse import ArgumentParser

from models import Transformer, RNNSeq2Seq, ConvSeq2Seq
import os
import data_utils

def train(args):
    vocab = data_utils.get_vocab(vocab_file=args.vocab_file, min_freq=args.min_vocab_freq)
    # vocab = {}
    # with open(args.vocab_file, mode='r') as infile:
    #     for line in infile:
    #         w, w_id = line.split('\t')
    #         vocab[w] = int(w_id)

    print('Vocab loaded...')
    print('VOCAB SIZE = ', len(vocab))

    if args.model_type == 'transformer':
        transformer = Transformer(args=args, vocab=vocab)
        transformer.train_generator()
    elif args.model_type == 'rnn':
        rnn_params = {'rec_cell': 'lstm',
                     'encoder_dim': 800,
                     'decoder_dim': 800,
                     'num_encoder_layers': 2,
                     'num_decoder_layers': 2
                     }
        rnn = RNNSeq2Seq(args=args, rnn_params=rnn_params, vocab=vocab)
        # rnn.train()
        rnn.train_keras()
    elif args.model_type == 'han_rnn':
        han_rnn = HanRnnSeq2Seq(args=args, vocab=vocab)
        han_rnn.train()
    elif args.model_type == 'cnn':
        cnn = ConvSeq2Seq(args=args, vocab=vocab)
        cnn.train_keras()

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, required=False, default='transformer')
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--train_file', type=str, required=False, default='/data/users/kyle.shaffer/dialog_data/ubuntu_train_seq.tsv')
    parser.add_argument('--valid_file', type=str, required=False, default='/data/users/kyle.shaffer/dialog_data/ubuntu_valid_seq.tsv')
    parser.add_argument('--vocab_file', type=str, required=False, default='/data/users/kyle.shaffer/dialog_data/ubuntu_vocab.txt')
    parser.add_argument('--min_vocab_freq', type=int, required=False, default=3)
    parser.add_argument('--model_name', type=str, required=False, default='transformer')
    parser.add_argument('--n_train_examples', type=int, required=False, default=1190799)
    parser.add_argument('--n_valid_examples', type=int, required=False, default=43837)
    parser.add_argument('--train_from', type=str, required=False, default='')
    # Model training params
    parser.add_argument('--n_epochs', type=int, required=False, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    parser.add_argument('--n_layers', type=int, required=False, default=4)
    parser.add_argument('--n_heads', type=int, required=False, default=6)
    parser.add_argument('--embedding_dim', type=int, required=False, default=256)
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    train(args)
