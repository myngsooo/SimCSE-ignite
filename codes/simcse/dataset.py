import torch
from torch.utils.data import Dataset
from simcse.utils import *

class DataCollator():
    def __init__(self, tokenizer, args, mode='train'):
        self.tokenizer = tokenizer        
        self.args = args
        self.mode = mode
        
    def __call__(self, samples):
        if self.mode == 'train':
            sent = [s['sent'] for s in samples]
            
            if self.args.do_mlm:
                encoding = self.tokenizer(
                    sent, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=self.args.max_length
                )
                encoding_ = self.tokenizer(
                    sent, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=self.args.max_length
                )
                m_encoding, m_encoding_label = get_mask_tokens(
                    self, 
                    self.tokenizer, 
                    encoding['input_ids']
                )
                return_value = {
                    'input_ids': encoding['input_ids'],
                    'input_ids_': encoding_['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'attention_mask_': encoding_['attention_mask'],
                    'masked_input_ids' : m_encoding,
                    'masked_input_ids_label' : m_encoding_label,
                    }
            else:          
                encoding = self.tokenizer(
                    sent, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=self.args.max_length
                )
                encoding_ = self.tokenizer(
                    sent, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=self.args.max_length
                )
                return_value = {
                        'input_ids': encoding['input_ids'],
                        'input_ids_': encoding_['input_ids'],
                        'attention_mask': encoding['attention_mask'],
                        'attention_mask_': encoding_['attention_mask'],
                        }
                
        elif self.mode == 'valid':
            sent1 = [s['sent1'] for s in samples]
            sent2 = [s['sent2'] for s in samples]
            score = [s['score'] for s in samples]
            
            encoding1 = self.tokenizer(
                sent1,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.args.max_length
            )
            
            encoding2 = self.tokenizer(
                sent2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.args.max_length
            )     
            return_value = {
                'input_ids1': encoding1['input_ids'],
                'attention_mask1': encoding1['attention_mask'],
                'input_ids2': encoding2['input_ids'],
                'attention_mask2': encoding2['attention_mask'],
                'score' : torch.tensor(score, dtype=torch.long),
            }
        return return_value


class TrainDataset(Dataset):

    def __init__(self, sent):
        self.sent = sent
    
    def __len__(self):
        return len(self.sent)
    
    def __getitem__(self, item):
        sent = str(self.sent[item])
        
        return {
            'sent': sent,
        }
        
class ValidDataset(Dataset):
    def __init__(self, sent1, sent2, score):
        self.sent1 = sent1
        self.sent2 = sent2
        self.score = score
        
    def __len__(self):
        return len(self.sent1)
    
    def __getitem__(self, item):
        sent1 = str(self.sent1[item])
        sent2 = str(self.sent2[item])
        score = float(self.score[item])
        
        return {
            'sent1': sent1,
            'sent2': sent2,
            'score': score,
        }