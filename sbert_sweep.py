import os
import sys
import json
import yaml
import random
import wandb
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
)


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

    return epoch_loss / total, epoch_acc / total, f1_score(lbls, preds, average='macro'), preds, lbls


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


def run_epochs(num_epochs, early_stop, model, train_loader, dev_loader, device, criterion, saving_path, logging_file, optimizer=None):
    # main training loop
    highest_val_acc = 0
    lowest_val_loss = float('inf')
    num_neg_progress = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_acc, train_f1, _, _ = process(model, train_loader, device, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_f1, _, _ = process(model, dev_loader, device, criterion)

        # save metrics
        save_metrics(
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
            path=saving_path,
            fname=logging_file
        )

        wandb.log({"train_loss": train_loss, 
               "train_acc": train_acc, 
               "train_f1": train_f1,
               "val_loss": val_loss,
               "val_acc": val_acc,
               "val_f1": val_f1
               })

        print(f"Training:   [Epoch {epoch:2d}, Loss: {train_loss:8.6f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}]")
        print(f"Evaluation: [Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}]")

        # save the model if the validation acc is the highest
        if val_acc > highest_val_acc:
            highest_val_acc = val_acc
            _path = saving_path + f"val_acc_{val_acc:.4f}_epoch{epoch}.pt"
            torch.save(model.state_dict(), _path)
            print('Model saved at:', _path)

        # early stopping based on loss
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            num_neg_progress = 0
        else:
            num_neg_progress += 1
            if num_neg_progress >= early_stop:
                print(f"Early stopping triggered at epoch {epoch} due to no improvement in loss")
                break


def test_results(saving_path, model, test_loader, device, criterion):
    # select saved best model
    best_model_path = sorted([f for f in os.listdir(saving_path) if f.endswith('.pt')], reverse=True)[0]
    model.load_state_dict(torch.load(saving_path + best_model_path))
    print("Loaded best model for testing: ", best_model_path)

    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_f1, predicted_labels, true_labels = process(model, test_loader, device, criterion)

    print(f"Test: [Loss: {test_loss:8.6f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}]")
    
    # generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=idx2lbl.values(), yticklabels=idx2lbl.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalised Confusion Matrix with Acc {test_acc:.4f}')
    plt.savefig(os.path.join(saving_path, 'confusion_matrix.png')) 

    # save the test results
    test_results = {
    'Test Loss': test_loss,
    'Test Accuracy': test_acc,
    'Test F1 Score': test_f1,
    }

    with open(os.path.join(saving_path, f'test_results_acc_{test_acc:.4f}.json'), 'w') as file:
        json.dump(test_results, file)


if __name__ == "__main__":
    # set configs
    with open('./configs/stmpnetv1_sweep_config.yaml', 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    run = wandb.init(project='Machine-Generated-Text-Detection', config=sweep_config)
    
    # Use hyperparameters from wandb
    config = wandb.config

    MODEL_NAME = config.model_name
    MAX_LEN = config.max_len
    LEARNING_RATE = config.learning_rate
    BATCH_SIZE = config.batch_size
    SEED = config.random_seed
    NUM_EPOCHS = config.num_epochs
    EARLY_STOP = config.early_stop
    NUM_WORKERS = config.num_workers
    DEVICE = config.device

    TRAIN_PATH = config.train_data_path
    DEV_PATH = config.dev_data_path
    TEST_PATH = config.test_data_path

    if 'all-mpnet-base-v1' in MODEL_NAME:
        SAVING_PATH = f'./stmpnetv1_lr{LEARNING_RATE}_bs{BATCH_SIZE}_results/'
    elif 'all-mpnet-base-v2' in MODEL_NAME:
        SAVING_PATH = f'./stmpnetv2_lr{LEARNING_RATE}_bs{BATCH_SIZE}_results/'
    elif 'all-roberta-large-v1' in MODEL_NAME:
        SAVING_PATH = f'./strbta_lr{LEARNING_RATE}_bs{BATCH_SIZE}_results/'
    logging_file = 'metrics.csv'

    scaler = torch.cuda.amp.GradScaler(enabled=False)

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
    model = SentenceTransformerForClassification(MODEL_NAME, len(idx2lbl)).to(DEVICE)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    run_epochs(NUM_EPOCHS, EARLY_STOP, model, train_loader, dev_loader, DEVICE, criterion, SAVING_PATH, logging_file, optimizer)

    # load the best model and test on test set
    test_results(SAVING_PATH, model, test_loader, DEVICE, criterion)
