from argparse import ArgumentParser

from models import *
import data_utils

def train(args):
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--n_train_instances', type=int, required=False, default=1000000)
    parser.add_argument('--n_valid_instances', type=int, required=False, default=60000)
    parser.add_argument('--max_seq_len', type=int, required=False, default=60)
    # Model training params
    parser.add_argument('--epochs', type=int, required=False, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    parser.add_argument('--n_layers', type=int, required=False, default=2)
    parser.add_argument('--n_heads', type=int, required=False, default=6)
    parser.add_argument('--embedding_dim', type=int, required=False, default=256)
    args = parser.parse_args()
    train(args)
