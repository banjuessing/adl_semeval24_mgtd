import json
import torch
from torch.utils.data import Dataset

from utils.augmentation import get_augmentation


def read_data(file_path):
    """Read data from jsonl files"""
    txts, lbls = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            txts.append(json_data['text'])
            lbls.append(json_data['label'])
    return txts, lbls


def collate_batch(batch, tokenizer, max_len, num_labels, loss_func):
    texts, labels = zip(*batch)

    labels = torch.tensor(labels, dtype=torch.long)

    if loss_func == "dualcl":
        tokenized_inputs = tokenizer(texts, 
                                truncation=True, padding=True, max_length=max_len, 
                                is_split_into_words=True, add_special_tokens=True, 
                                return_tensors='pt')
        
        positions = torch.zeros_like(tokenized_inputs['input_ids'])
        positions[:, num_labels:] = torch.arange(0, tokenized_inputs['input_ids'].size(1)-num_labels)
        tokenized_inputs['position_ids'] = positions

        return tokenized_inputs, labels
    
    else:
        tokenized_inputs = tokenizer(texts, 
                                truncation=True, padding=True, max_length=max_len, 
                                return_tensors='pt')
        
        return tokenized_inputs, labels


class MGTDDataset(Dataset):
    def __init__(self, txts, lbls, 
                 label_dict, model_name, loss_func,
                 use_aug=False, augmodel_path=None, aug_all=False,
                 synonym=False, antonym=False, swap=False, spelling=False, word2vec=False):
        self.txts = txts
        self.lbls = lbls
        self.model_name = model_name
        self.loss_func = loss_func
        self.label_list = list(label_dict.keys()) if loss_func == "dualcl" else []
        if 'gpt' in model_name:
            self.sep_token = ['<|endoftext|>']
        elif 'xlnet' in model_name:
            self.sep_token = ['<sep>']
        else:
            self.sep_token = ['</s>']
        self.use_aug = use_aug

        if use_aug:
            self.aug = get_augmentation(augmodel_path, aug_all, 
                                        synonym, antonym, swap, spelling, word2vec)
        else:
            self.aug = None

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        text = self.txts[idx]
        label = self.lbls[idx]
        
        if self.use_aug:
            text = self.aug.augment(text)[0]

        if self.loss_func == "dualcl":
            """
            For dualcl, transform input sentence into:
            [label0, label1, label2, ... , labeln, sep_token, token1, token2, ... , tokenn]
            """
            tokens = text.lower().split()
            text = self.label_list + self.sep_token + tokens

        return text, label