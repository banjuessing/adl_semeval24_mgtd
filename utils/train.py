import os
import sys
import json
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import f1_score, confusion_matrix

from utils.common import save_metrics
from model import ClassificationModel


def process(model, loader, device, scaler, criterion, optim=None):
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

        pred = outputs['predicts'].argmax(dim=1)
        epoch_loss += loss.item() * labels.shape[0]
        epoch_acc += (pred == labels).sum().item()
        total += labels.shape[0]
        preds.extend(pred.detach().tolist())
        lbls.extend(labels.detach().tolist())

    return epoch_loss / total, epoch_acc / total, f1_score(lbls, preds, average='macro'), preds, lbls


def run_epochs(num_epochs, early_stop, model, train_loader, dev_loader, device, scaler,
               criterion, saving_path, logging_file, optimizer):
    # main training loop
    highest_val_acc = 0
    lowest_val_loss = float('inf')
    num_neg_progress = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_acc, train_f1, _, _ = process(model, train_loader, device, scaler, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_f1, _, _ = process(model, dev_loader, device, scaler, criterion)

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

        # optinal: use wanb to log training process
        # wandb.log({"train_loss": train_loss, 
        #        "train_acc": train_acc, 
        #        "train_f1": train_f1,
        #        "val_loss": val_loss,
        #        "val_acc": val_acc,
        #        "val_f1": val_f1
        #        })

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


def test_results(saving_path, model, idx2lbl, test_loader, device, scaler, criterion, 
                 loss_func=None, model_name=None, specific_model_path=None):
    if specific_model_path:
        model = ClassificationModel(model_name, len(idx2lbl), loss_func).to(device)
        model.load_state_dict(torch.load(specific_model_path))
        print("Loaded model for testing: ", specific_model_path)
    else:
        # select saved best model
        best_model_path = sorted([f for f in os.listdir(saving_path) if f.endswith('.pt')], reverse=True)[0]
        model.load_state_dict(torch.load(saving_path + best_model_path))
        print("Loaded best model for testing: ", best_model_path)

    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_f1, predicted_labels, true_labels = process(model, test_loader, device, scaler, criterion)

    print(f"Test: [Loss: {test_loss:8.6f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}]")
    
    # generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=idx2lbl.values(), yticklabels=idx2lbl.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalised Confusion Matrix with Acc {test_acc:.4f}')
    if specific_model_path:
        plt.savefig(os.path.join(saving_path, f'confusion_matrix_from_{specific_model_path[-24:]}.png'))
    else:
        plt.savefig(os.path.join(saving_path, 'confusion_matrix.png')) 

    # save the test results
    test_results = {
    'Test Loss': test_loss,
    'Test Accuracy': test_acc,
    'Test F1 Score': test_f1,
    }

    if specific_model_path:
        with open(os.path.join(saving_path, f'test_results_acc_{test_acc:.4f}_from_{specific_model_path[-24:]}.json'), 'w') as file:
            json.dump(test_results, file)
    else:
        with open(os.path.join(saving_path, f'test_results_acc_{test_acc:.4f}.json'), 'w') as file:
            json.dump(test_results, file)
