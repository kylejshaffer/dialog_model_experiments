import argparse
import os
import csv
import random
import logging
import json
import sys
from tqdm import tqdm, trange

import numpy as np
import torch

from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     OpenAIAdam, cached_path, WEIGHTS_NAME, CONFIG_NAME)


def load_data_with_tok(data_path, tokenizer, max_len):
    tok_lines = []
    start_tok= '_start_'
    end_tok = '_delimiter_'
    with open(data_path, mode='r') as infile:
        for ix, line in enumerate(infile):
            sys.stdout.write('\r Loading line {}...'.format(ix))
            json_line = json.loads(line.strip())
            for utt in json_line:
                text = start_tok + ' ' + utt['text'].strip() + ' ' + end_tok
                toks = tokenizer.tokenize(text)
                tok_ids = tokenizer.convert_tokens_to_ids(toks)
                if len(tok_ids) > max_len:
                    tok_ids = tok_ids[:max_len]
                tok_lines.append(tok_ids)

    x_lines = [i[1:] for i in tok_lines]
    y_lines = [i[:-1] for i in tok_lines]

    x = pad_sequences(x_lines, padding='post', maxlen=max_len)
    y = pad_sequences(y_lines, padding='post', maxlen=max_len)

    return x, y

def data_to_torch(x_train, y_train, x_valid, y_valid, batch_size):
    train_inputs = torch.tensor(x_train)
    valid_inputs =torch.tensor(x_valid)
    train_tags = torch.tensor(y_train)
    valid_tags = torch.tensor(y_valid)
    # train_mask = torch.tensor(mask_train)
    # valid_mask = torch.tensor(mask_valid)
    print('Data converted to Torch Tensors...')

    # train_data = TensorDataset(train_inputs, train_mask, train_tags)
    train_data = TensorDataset(train_inputs, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # valid_data = TensorDataset(valid_inputs, valid_mask, valid_tags)
    valid_data = TensorDataset(valid_inputs, valid_tags)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
    print('Data converted to Torch Datasets...')

    return train_dataloader, valid_dataloader

def prep_optimizer(model, epochs, learning_rate, warmup_proportion, max_grad_norm, weight_decay):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_dataloader) * epochs
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            max_grad_norm=max_grad_norm,
                            weight_decay=weight_decay,
                            t_total=num_train_optimization_steps)

    return optimizer

def train_model(model, train_dataloader, opt, lm_coef, epochs, valid_dataloader=None):
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    for _ in trange(int(epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            loss = lm_coef * losses[0] + losses[1]
            loss.backward()
            opt.step()
            opt.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, opt.get_lr()[0])

        # Optionally evaluate after each epoch
        if valid_dataloader is not None:
            eval_model(model, valid_dataloader, '', save_model=False, save_results=False)

def eval_model(model, eval_dataloader, output_dir, save_model=False, save_results=False):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels = batch
        with torch.no_grad():
            _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            _, mc_logits = model(input_ids, mc_token_ids)

        mc_logits = mc_logits.detach().cpu().numpy()
        mc_labels = mc_labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

        eval_loss += mc_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    # train_loss = tr_loss/nb_tr_steps if args.do_train else None
    result = {'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy}

    if save_results:
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return


if __name__ == '__main__':
    global device
    global n_gpu

    epochs = 10
    max_len = 100
    batch_size = 128
    learning_rate = 6.25e-5
    warmup_proportion = 0.002
    max_grad_norm = 1
    weight_decay = 0.01

    train_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/corpora_processed/train_no_tok.txt'
    valid_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/corpora_processed/valid_no_tok.txt'

    model_name = 'openai-gpt'
    lm_coef = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(model_name, num_special_tokens=len(special_tokens))
    
    x_train_arr, y_train_arr = load_data_with_tok(train_path, tokenizer, max_len)
    x_valid_arr, y_valid_arr = load_data_with_tok(valid_path, tokenizer, max_len)

    train_dataloader, valid_dataloader = data_to_torch(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr, batch_size)

    opt = prep_optimizer(model=model, epochs=epochs, learning_rate=learning_rate, warmup_proportion=warmup_proportion, 
                         max_grad_norm=max_grad_norm, weight_decay=weight_decay)
    train_model(model=model, train_dataloader=train_dataloader, opt=opt, lm_coef=lm_coef, epochs=epochs, valid_dataloader=valid_dataloader)
