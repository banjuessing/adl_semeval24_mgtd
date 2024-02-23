import os
import sys
import argparse
import json
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

from augmentation import get_augmentation
from loss_func import CELoss, SupConLoss, DualLoss

from transformers import AutoTokenizer, AutoModel


def get_runtime_args():
    parser = argparse.ArgumentParser()
    # specify model
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    # specify loss function
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"])
    # specify training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    # specify augmentation methods
    parser.add_argument('--use_aug', default=False, action='store_true')
    parser.add_argument('--augmodel_path', type=str, default=None)
    parser.add_argument('--aug_all', default=True, action='store_true')
    parser.add_argument('--synonym', default=True, action='store_true')
    parser.add_argument('--antonym', default=False, action='store_true')
    parser.add_argument('--swap', default=False, action='store_true')
    parser.add_argument('--spelling', default=False, action='store_true')
    parser.add_argument('--word2vec', default=False, action='store_true')
    parser.add_argument('--contextual', default=False, action='store_true')
    # others
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    # paths
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


def collate_batch(batch, tokenizer, max_len, num_labels, model_name, loss_func):
    texts, labels = zip(*batch)

    labels = torch.tensor(labels, dtype=torch.long)

    if "sentence-transformers" in model_name:
        if loss_func == "dualcl":
            tokenized_inputs = tokenizer(texts, 
                                truncation=True, padding=True, max_length=max_len, 
                                is_split_into_words=True, add_special_tokens=True, 
                                return_tensors='pt')

            batch = {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels
            }   
        else:
            tokenized_inputs = tokenizer.batch_encode_plus(list(texts), 
                                                        truncation=True, padding=True, max_length=max_len, 
                                                        return_tensors="pt")

            batch = {
                'input_ids': tokenized_inputs['input_ids'],
                'attention_mask': tokenized_inputs['attention_mask'],
                'labels': labels
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

        return tokenized_inputs, labels


class MGTDDataset(Dataset):
    def __init__(self, txts, lbls, 
                 label_dict, model_name, loss_func,
                 use_aug=False, augmodel_path=None, aug_all=False,
                 synonym=False, antonym=False, swap=False, spelling=False, word2vec=False, contextual=False):
        self.txts = txts
        self.lbls = lbls
        self.model_name = model_name
        self.label_list = list(label_dict.keys()) if loss_func == "dualcl" else []
        self.sep_token = ['</s>'] if 'roberta' in model_name else ['[SEP]']
        self.use_aug = use_aug

        if use_aug:
            self.aug = get_augmentation(augmodel_path, aug_all, 
                                        synonym, antonym, swap, spelling, word2vec, contextual)
        else:
            self.aug = None

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        text = self.txts[idx]
        label = self.lbls[idx]
        
        if self.use_aug:
            text = self.aug.augment(text)[0]
        
        tokens = text.lower().split()
        text = self.label_list + self.sep_token + tokens

        return text, label


class TransformerModel(nn.Module):
    def __init__(self, model_name, num_labels, loss_func):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.loss_func = loss_func
        self.linear = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.5)
        for param in self.base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        if self.loss_func == "dualcl":
            label_feats = hiddens[:, 1:self.num_labels+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        else:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }

        return outputs


class SentenceTransformerModel(nn.Module):
    def __init__(self, model_name, num_labels, loss_func):
        super().__init__()
        self.sentence_transformer = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.loss_func = loss_func
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
        normed_pooled_output = F.normalize(pooled_output, p=2, dim=1)

        # Classifier
        logits = self.classifier(normed_pooled_output)

        if self.loss_func == "dualcl":
            label_feats = model_output[0][:, 1:self.num_labels+1, :]
            logits = torch.einsum('bd,bcd->bc', pooled_output, label_feats)

        outputs = {
            'predicts': logits,
            'cls_feats': pooled_output,
            'label_feats': label_feats
        }

        return outputs


def process_st(model, loader, device, scaler, criterion, optim=None):
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
            outputs = model(**batch_input)
            loss = criterion(outputs, lbl)

        if optim is not None:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        pred = outputs['predicts'].argmax(dim=1)
        epoch_loss += loss.item() * lbl.shape[0]
        epoch_acc += (pred == lbl).sum().item()
        total += lbl.shape[0]
        preds.extend(pred.detach().tolist())
        lbls.extend(lbl.detach().tolist())

    return epoch_loss / total, epoch_acc / total, f1_score(lbls, preds, average='macro'), preds, lbls


def process_t(model, loader, device, scaler, criterion, optim=None):
    epoch_loss, epoch_acc, total = 0, 0, 0
    preds, lbls = [], []

    for inputs, labels in tqdm(
        loader,
        desc="Train: " if optim is not None else "Eval: ",
        file=sys.stdout,
        unit="batches"
    ):

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if optim is not None:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        pred = torch.argmax(outputs['predicts'], -1)
        epoch_loss += loss.item() * labels.shape[0]
        epoch_acc += (pred == labels).sum().item()
        total += labels.shape[0]
        preds.extend(pred.detach().tolist())
        lbls.extend(labels.detach().tolist())

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


def run_epochs(num_epochs, early_stop, model, model_name, train_loader, dev_loader, device, scaler,
               criterion, saving_path, logging_file, optimizer):
    # main training loop
    highest_val_acc = 0
    lowest_val_loss = float('inf')
    num_neg_progress = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        if "sentence-transformers" in model_name:
            train_loss, train_acc, train_f1, _, _ = process_st(model, train_loader, device, scaler, criterion, optimizer)
        else:
            train_loss, train_acc, train_f1, _, _ = process_t(model, train_loader, device, scaler, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            if "sentence-transformers" in model_name:
                val_loss, val_acc, val_f1, _, _ = process_st(model, dev_loader, device, scaler, criterion)
            else:
                val_loss, val_acc, val_f1, _, _ = process_t(model, dev_loader, device, scaler, criterion)

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

        # save the model if the validation acc is the highest or the validation loss is the lowest
        if val_acc > highest_val_acc or val_loss < lowest_val_loss:
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


def test_results(saving_path, model, model_name, test_loader, device, scaler, criterion):
    # select saved best model
    best_model_path = sorted([f for f in os.listdir(saving_path) if f.endswith('.pt')], reverse=True)[0]
    model.load_state_dict(torch.load(saving_path + best_model_path))
    print("Loaded best model for testing: ", best_model_path)

    model.eval()
    with torch.no_grad():
        if "sentence-transformers" in model_name:
            test_loss, test_acc, test_f1, predicted_labels, true_labels = process_st(model, test_loader, device, scaler, criterion)
        else:
            test_loss, test_acc, test_f1, predicted_labels, true_labels = process_t(model, test_loader, device, scaler, criterion)

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
    parser = get_runtime_args()
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_length

    LOSS_FUNC = args.loss

    LEARNING_RATE = args.lr
    DECAY = args.decay
    ALPHA = args.alpha
    TEMP = args.temp
    BATCH_SIZE = args.batch_size

    USE_AUG = args.use_aug
    AUGMODEL_PATH = args.augmodel_path
    AUG_ALL = args.aug_all
    SYN = args.synonym
    ANT = args.antonym
    SWAP = args.swap
    SPELLING = args.spelling
    WORD2VEC = args.word2vec
    CONTEXTUAL = args.contextual

    SEED = args.random_seed
    NUM_EPOCHS = args.num_epochs
    EARLY_STOP = args.early_stop
    NUM_WORKERS = args.num_workers
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

    train_dataset = MGTDDataset(train_data[0], train_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC,
                                use_aug=USE_AUG, augmodel_path=AUGMODEL_PATH, aug_all=AUG_ALL,
                                synonym=SYN, antonym=ANT, swap=SWAP, spelling=SPELLING, word2vec=WORD2VEC, contextual=CONTEXTUAL)
    dev_dataset = MGTDDataset(dev_data[0], dev_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)
    test_dataset = MGTDDataset(test_data[0], test_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)

    # get tokenizer
    if "roberta" in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
                              collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), MODEL_NAME, LOSS_FUNC))
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                            collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), MODEL_NAME, LOSS_FUNC))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                             collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), MODEL_NAME, LOSS_FUNC))

    # load model
    if "sentence-transformers" in MODEL_NAME:
        model = SentenceTransformerModel(MODEL_NAME, len(idx2lbl), LOSS_FUNC).to(DEVICE)
    else:
        model = TransformerModel(MODEL_NAME, len(idx2lbl), LOSS_FUNC).to(DEVICE)
        
    # define loss function and optimizer
    if LOSS_FUNC == 'dualcl':
        criterion = DualLoss(ALPHA, TEMP)
    elif LOSS_FUNC == 'scl':
        criterion = SupConLoss(ALPHA, TEMP)
    elif LOSS_FUNC == 'ce':
        criterion = CELoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    # start a new run in wandb
    wandb.init(
        project="Machine-Generated-Text-Detection",
        name=SAVING_PATH[2:-9],
        config={
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "loss": LOSS_FUNC,
        "learning_rate": LEARNING_RATE,
        "decay": DECAY,
        "alpha": ALPHA,
        "temp": TEMP,
        "batch_size": BATCH_SIZE,
        "use_aug": USE_AUG,
        "aug_all": AUG_ALL,
        "synonym": SYN,
        "antonym": ANT,
        "swap": SWAP,
        "spelling": SPELLING,
        "word2vec": WORD2VEC,
        "contextual": CONTEXTUAL,
        "use_amp": args.use_amp,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "early_stop": EARLY_STOP,
        "num_workers": NUM_WORKERS,
        "device": DEVICE
    })

    run_epochs(NUM_EPOCHS, EARLY_STOP, model, MODEL_NAME, train_loader, dev_loader, DEVICE, scaler,
               criterion, SAVING_PATH, logging_file, optimizer)

    wandb.finish()

    # load the best model and test on test set
    test_results(SAVING_PATH, model, MODEL_NAME, test_loader, DEVICE, scaler, criterion)
