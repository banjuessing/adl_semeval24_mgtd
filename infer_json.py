import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from train_st import TransformerModel, SentenceTransformerModel


def get_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--infer_data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--saving_path", type=str)
    return parser


def read_infer_data(file_path):
    """Read inference data from jsonl files"""
    ids, txts = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            ids.append(json_data['id'])
            txts.append(json_data['text'])
    return ids, txts


def collate_infer_batch(batch, tokenizer, max_len, num_labels, model_name, loss_func):
    ids, texts = zip(*batch)

    if "sentence-transformers" in model_name:
        tokenized_inputs = tokenizer.batch_encode_plus(list(texts), 
                                                       truncation=True, padding=True, max_length=max_len, 
                                                       return_tensors="pt")

        batch = {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'ids': ids
        }

        return batch

    else:
        tokenized_inputs = tokenizer(texts, 
                                truncation=True, padding=True, max_length=max_len, 
                                is_split_into_words=True, add_special_tokens=True, 
                                return_tensors='pt')
        if loss_func == "dualcl":
            positions = torch.zeros_like(tokenized_inputs['input_ids'])
            positions[:, num_labels:] = torch.arange(0, tokenized_inputs['input_ids'].size(1)-num_labels)
            tokenized_inputs['position_ids'] = positions

        return tokenized_inputs, ids


class MGTDINFERDataset(Dataset):
    def __init__(self, ids, txts,
                 label_dict, model_name, loss_func):
        self.ids = ids
        self.txts = txts
        self.model_name = model_name
        self.label_list = list(label_dict.keys()) if loss_func == "dualcl" else []
        self.sep_token = ['</s>'] if 'roberta' in model_name else ['[SEP]']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        text = self.txts[idx]
        
        if "sentence-transformers" not in self.model_name:
            tokens = text.lower().split()
            text = self.label_list + self.sep_token + tokens

        return id, text


def infer_st(model, loader, device):
    ids, preds = [], []

    for batch in tqdm(loader, desc="Inference: ", file=sys.stdout, unit="batches"):

        batch_input = {k: v.to(device) for k, v in batch.items() if k != 'ids'}
        id = batch['ids']

        with torch.no_grad():
            outputs = model(**batch_input)

        pred = outputs['predicts'].argmax(dim=1)

        preds.extend(pred.detach().tolist())
        ids.extend(id)

    return ids, preds


def infer_t(model, loader, device):
    ids, preds = [], []

    for inputs, id in tqdm(loader, desc="Inference: ", file=sys.stdout, unit="batches"):

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(inputs)

        pred = torch.argmax(outputs['predicts'], -1)

        preds.extend(pred.detach().tolist())
        ids.extend(id)

    return ids, preds


def save_infer_results(model_path, saving_path, model_name, loss_func, loader, num_labels, device):
    # load model
    if "sentence-transformers" in model_name:
        model = SentenceTransformerModel(model_name, num_labels).to(device)
    else:
        model = TransformerModel(model_name, num_labels, loss_func).to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()
    if "sentence-transformers" in model_name:
        ids, predicted_labels = infer_st(model, loader, device)
    else:
        ids, predicted_labels = infer_t(model, loader, device)

    assert len(ids) == len(predicted_labels)
        
    with open(os.path.join(saving_path, f'infer_results_from_{model_path[-24:]}.jsonl'), 'w') as file:
        for id, pred in zip(ids, predicted_labels):
            json.dump({'id': id, 'label': pred}, file)
            file.write('\n')
    

if __name__ == "__main__":
    # set configs
    parser = get_infer_args()
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_length
    LOSS_FUNC = args.loss
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = args.device
    INFER_DATA_PATH = args.infer_data_path
    MODEL_PATH = args.model_path
    SAVING_PATH = args.saving_path

    lbl2idx = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    
    # load data
    infer_data = read_infer_data(INFER_DATA_PATH)
    infer_dataset = MGTDINFERDataset(infer_data[0], infer_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)

    # get tokenizer
    if ("sentence-transformers" not in MODEL_NAME) and ("roberta" in MODEL_NAME):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # create DataLoader
    infer_loader = DataLoader(infer_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                              collate_fn=lambda batch: collate_infer_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), MODEL_NAME, LOSS_FUNC))
    
    # save predicted results
    save_infer_results(MODEL_PATH, SAVING_PATH, MODEL_NAME, LOSS_FUNC, infer_loader, len(lbl2idx), DEVICE)

