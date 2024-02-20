import torch
from torch.utils.data import Dataset
from simcse.utils import *

class DataCollator():
    def __init__(self, tokenizer, args, mode='train'):
        self.tokenizer = tokenizer        
        self.args = args
        self.mode = mode
         
    def encode_sentences(self, sent):
        encoding = self.tokenizer(
            sent, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.args.max_length
        )
        return encoding['input_ids'], encoding['attention_mask']
        
    def __call__(self, samples):
        if self.mode == 'train':
            sent = [s['sent'] for s in samples]
            input_ids, attention_mask = self.encode_sentences(sent)
            if self.args.do_mlm:
                masked_input_ids, masked_input_ids_label = get_mask_tokens(self, self.tokenizer, input_ids)
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'input_ids_': input_ids,
                    'attention_mask_': attention_mask,
                    'masked_input_ids': masked_input_ids,
                    'masked_input_ids_label': masked_input_ids_label,
                }
            else:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'input_ids_': input_ids,
                    'attention_mask_': attention_mask,
                }
        elif self.mode == 'valid':
            sent1 = [s['sent1'] for s in samples]
            sent2 = [s['sent2'] for s in samples]
            score = [s['score'] for s in samples]
            
            input_ids1, attention_mask1 = self.encode_sentences(sent1)
            input_ids2, attention_mask2 = self.encode_sentences(sent2)
            
            return {
                'input_ids1': input_ids1,
                'attention_mask1': attention_mask1,
                'input_ids2': input_ids2,
                'attention_mask2': attention_mask2,
                'score': torch.tensor(score, dtype=torch.long),
            }


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