import numpy as np

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from simcse.dataset import *

def get_data(file, mode='train'):
    if mode == 'train':
        sent = []
        with open(file) as f:
            for line in f:
                sent.append(line)
            return sent
                
    elif mode == 'valid':
        sent1, sent2, score = [], [], []
        with open(file) as f:
            for line in f:
                x, y, z = line.split('\t')
                sent1.append(x)
                sent2.append(y)
                score.append(z.replace('\n', ''))                
            return sent1, sent2, score
        
def get_loaders(args, tokenizer):
    sent = get_data(args.train_fn, mode='train')
    sent1, sent2, score = get_data(args.valid_fn, mode='valid')
    
    wiki = list(zip(sent))
    sts_dev = list(zip(sent1, sent2, score))
    
    sent = [e[0] for e in wiki]
    
    sent1 = [e[0] for e in sts_dev]
    sent2 = [e[1] for e in sts_dev]    
    score = [e[2] for e in sts_dev]
    
    train_loader = DataLoader(
        TrainDataset(sent),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer, args, mode="train"),
    )

    sts_dev_loader = DataLoader(
        ValidDataset(sent1, sent2, score),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer, args, mode="valid"),
    )
    return train_loader, sts_dev_loader 

def get_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_eps
    )

    return optimizer
        
def get_mask_tokens(self, tokenizer, inputs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    self.tokenizer = tokenizer
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class MLPLayer(nn.Module):

    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "max", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "max":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            last_hidden[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(last_hidden, 1)[0]
            return max_over_time
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
