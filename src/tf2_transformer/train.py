import argparse
import os
import sys
sys.path.append('..')

from utils import *
from model import *
# from transformer import *

def log_params(args):
    print('\nTRAINING PARAMS')
    print('=' * 40)
    print('Embedding size:', args.embedding_dim)
    print('Model dim:', args.d_model)
    print('Num units:', args.units)
    print('Num layers:', args.num_layers)
    print('Num heads:', args.num_heads)
    print('Dropout:', args.dropout)
    print('Training epochs:', args.n_epochs)
    print('Batch size:', args.batch_size)
    print('=' * 40)
    print()
    return

def train(args):
    # Log training parameters for sanity-check
    log_params(args)

    with open(args.all_data_file, mode='r') as infile:
        movie_lines = []
        for line in infile:
            movie_lines.append(line.strip())

    bpe_tok = get_bpe_tokenizer(input_text=movie_lines, tgt_vocab_size=args.target_voc_size)
    del movie_lines

    bpe_tok.save_to_file('cornell_bpe_tokenizer.tok')

    data_processor = DataProcessor(max_len=100, tokenizer=bpe_tok, train_file=args.train_file,
                                   valid_file=args.valid_file, batch_size=args.batch_size)

    trainer = Trainer(d_model=args.d_model, units=args.units, vocab_size=data_processor.vocab_size,
                      num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                      epochs=args.n_epochs, batch_size=args.batch_size, data_generator=data_processor)

    # Train
    trainer.train()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--embedding_dim', type=int, required=False, default=512)
    parser.add_argument('--d_model', type=int, required=False, default=512)
    parser.add_argument('--units', type=int, required=False, default=2048)
    parser.add_argument('--num_layers', type=int, required=False, default=6)
    parser.add_argument('--num_heads', type=int, required=False, default=8)
    parser.add_argument('--dropout', type=float, required=False, default=0.3)

    # Data params
    parser.add_argument('--all_data_file', type=str, required=False, default='/data/users/kyle.shaffer/dialog_data/cornell_movie/dialogs_text.txt')
    parser.add_argument('--train_file', type=str, required=False, default="/data/users/kyle.shaffer/dialog_data/cornell_movie/cornell_movie_dialog_no_context_train.txt")
    parser.add_argument('--valid_file', type=str, required=False, default="/data/users/kyle.shaffer/dialog_data/cornell_movie/cornell_movie_dialog_no_context_valid.txt")
    parser.add_argument('--target_voc_size', type=int, required=False, default=15000)

    # Training params
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    parser.add_argument('--n_epochs', type=int, required=False, default=50)
    parser.add_argument('--gpu', type=int, required=False, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    train(args)

