import os
import argparse
import random
import pprint

import numpy as np
import pandas as pd
import jsonlines

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertConfig, BertModel, BertTokenizer, BertTokenizerFast
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from ignite.utils import manual_seed

from simcse.utils import *
from simcse.trainer import Trainer

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument(
        '--train_fn', 
        required=True,
        help='Training datasset file name'
    )
    p.add_argument(
        '--valid_fn',
        required=True,
        help='Validation datasset file name',
    )
    p.add_argument(
        '--model_name_or_path', 
        required=True,
        help='The transformer model checkpoint for weights initialization.',
    )
    p.add_argument(
        '--output_dir', 
        required=True,
        help='Model file name to save. Additional information would be annotated to the file name.',
    )
    p.add_argument(
        '--gpu_id', 
        type=int, 
        default=0,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s',
    )
    p.add_argument(
        '--verbose', 
        type=int, 
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s',
    )
    p.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Mini batch size for gradient descent. Default=%(default)s',
    )
    p.add_argument(
        '--max_length', 
        type=int, 
        default=64,
        help='The maximum total input sequence length after tokenization. Sequences longer. Default=%(default)s',
    )
    p.add_argument(
        '--n_epochs', 
        type=int, 
        default=1,
        help='Number of epochs to train. Default=%(default)s',
    )
    p.add_argument(
        '--lr', 
        type=float, 
        default=3e-5,
        help='Initial learning rate. Default=%(default)s',
    )
    p.add_argument(
        '--adam_eps', 
        type=float, 
        default=1e-8,
    )
    p.add_argument(
        '--warmup_ratio', 
        type=float, 
        default=0,
    )
    p.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed',
    )
    
    # SimCSE's arguments
    p.add_argument(
        '--pooler_type', 
        type=str, 
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], 
        default='cls',
        help='What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).',
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--metric_for_best_model', 
        type=str, 
        choices=["stsb_spearman", "sickr_spearman", "sts_avg"], 
        default='stsb_spearman',
        help='Metric for saving best_model.',
    )
    p.add_argument(
        '--temp', 
        type=float, 
        default=0.05,
        help='Temperature for softmax. Default=%(default)s',
    )
    p.add_argument(
        '--do_eval_step', 
        action='store_true',
        help='Whether to use eval_step (if you want to save model checkpoint on every epoch, do not use.)',
    )
    p.add_argument(
        '--eval_step', 
        type=int,
        default=250,   
    )
    p.add_argument(
        '--do_mlm', 
        action='store_true',
        help='Whether to use MLM auxiliary objective.',
    )
    p.add_argument(
        '--mlm_weight', 
        type=float, 
        default=0.1,
        help='Weight for MLM auxiliary objective (only effective if --do_mlm).',
    )
    p.add_argument(
        '--mixed_precision', 
        action='store_true',
        help='Whether to use mixed precision.',
    )
    args = p.parse_args()

    return args

def main(args):
    manual_seed(args.seed)
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(args)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path).cuda(args.gpu_id)
    
    config = BertConfig.from_pretrained(args.model_name_or_path)
    train_loader, valid_loader = get_loaders(args, tokenizer)    
    
    print(
        '|train| =', len(train_loader) * args.batch_size,
        '|valid| =', len(valid_loader) * args.batch_size,
    )
    n_total_iterations = len(train_loader) * args.n_epochs
    n_warmup_steps = int(n_total_iterations * args.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    optimizer = get_optimizer(model, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )
    if args.gpu_id >= 0:
        model.cuda(args.gpu_id)

    trainer = Trainer(args, config)
    model = trainer.train(
                model,
                optimizer,
                scheduler,
                train_loader,
                valid_loader,
                )

if __name__ == '__main__':
    args = define_argparser()
    main(args)
    