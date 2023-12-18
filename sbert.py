import os
import sys
import argparse
import json
import random
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
)


def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--dev_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--saving_path", type=str)
    parser.add_argument("--use_amp", default=False, action="store_true")

    return parser


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_data(file_path):
  """Read data from jsonl files"""
  txts, lbls = [], []
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_data = json.loads(line)
        txts.append(json_data['text'])
        lbls.append(json_data['label'])
  return txts, lbls


def collate_batch(batch, max_len):
    texts, labels = zip(*batch)

    tokenized_inputs = tokenizer.batch_encode_plus(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    labels = torch.tensor(labels, dtype=torch.long)

    batch = {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }

    return batch


class MGTDDataset(Dataset):
    def __init__(self, txts, lbls):
        self.txts = txts
        self.lbls = lbls

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        return self.txts[idx], self.lbls[idx]


class SentenceTransformerForClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.sentence_transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.sentence_transformer.config.hidden_size, num_labels)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, **batch):
        # Forward pass through the sentence transformer
        model_output = self.sentence_transformer(**batch)

        # Perform pooling
        pooled_output = self.mean_pooling(model_output, batch['attention_mask'])

        # Normalize embeddings
        pooled_output = F.normalize(pooled_output, p=2, dim=1)

        # Classifier
        logits = self.classifier(pooled_output)

        return logits


def process(model, loader, device, criterion, optim=None):
    epoch_loss, epoch_acc, total = 0, 0, 0
    preds, lbls = [], []

    for batch in tqdm(
        loader,
        desc="Train: " if optim is not None else "Eval: ",
        file=sys.stdout,
        unit="batches"
    ):

        batch_input = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        lbl = batch['labels'].to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(**batch_input)
            loss = criterion(logits, lbl)

        if optim is not None:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        pred = logits.argmax(dim=1)
        epoch_loss += loss.item() * lbl.shape[0]
        epoch_acc += (pred == lbl).sum().item()
        total += lbl.shape[0]
        preds.extend(pred.detach().tolist())
        lbls.extend(lbl.detach().tolist())

    return epoch_loss / total, epoch_acc / total, f1_score(lbls, preds, average='micro')


def regularized_f1(train_f1, dev_f1, threshold=0.0025):
    """
    Returns development f1 if overfitting is below threshold, otherwise 0.
    """
    return dev_f1 if (train_f1 - dev_f1) < threshold else 0


def save_metrics(*args, path, fname):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path + fname):
        with open(path + fname, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "train_f1",
                        "val_loss",
                        "val_acc",
                        "val_f1",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path + fname, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")


if __name__ == "__main__":
    # set configs
    parser = get_runtime_args()
    args = parser.parse_args()

    SEED = args.random_seed
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_length
    NUM_EPOCHS = args.num_epochs
    EARLY_STOP = args.early_stop
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LEARNING_RATE = args.lr
    DEVICE = args.device

    TRAIN_PATH = args.train_data_path
    DEV_PATH = args.dev_data_path
    TEST_PATH = args.test_data_path

    SAVING_PATH = args.saving_path
    logging_file = 'metrics.csv'

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    seed_everything(SEED)

    # avoids parallelism errors when both tokenizers and torch dataloaders use multiprocessing 
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # create idx2lbl and lbl2idx dictionaries
    idx2lbl = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    lbl2idx = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}

    # load data
    train_data = read_data(TRAIN_PATH)
    dev_data = read_data(DEV_PATH)
    test_data = read_data(TEST_PATH)

    train_dataset = MGTDDataset(train_data[0], train_data[1])
    dev_dataset = MGTDDataset(dev_data[0], dev_data[1])
    test_dataset = MGTDDataset(test_data[0], test_data[1])

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda batch: collate_batch(batch, MAX_LEN))
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda batch: collate_batch(batch, MAX_LEN))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda batch: collate_batch(batch, MAX_LEN))

    # load model
    model = SentenceTransformerForClassification(MODEL_NAME, 6).to(DEVICE)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # main training loop
    highest_val_f1 = 0
    f1_decreasing_count = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_acc, train_f1 = process(model, train_loader, DEVICE, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_f1 = process(model, dev_loader, DEVICE, criterion)

        # save metrics
        save_metrics(
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
            path=SAVING_PATH,
            fname=logging_file
        )

        print(f"Training:   [Epoch {epoch:2d}, Loss: {train_loss:8.6f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}]")
        print(f"Evaluation: [Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}]")

        reg_f1 = regularized_f1(train_f1, val_f1)
        if reg_f1 > highest_val_f1:
            highest_val_f1 = reg_f1
            f1_decreasing_count = 0
            _path = SAVING_PATH + f"val_f1_{val_f1:.4f}_epoch{epoch}.pt"
            torch.save(model.state_dict(), _path)
            print('saved at:', _path)
        else:
            f1_decreasing_count += 1
            if f1_decreasing_count >= EARLY_STOP:
                print(f"Early stopping triggered at epoch {epoch}")
                break