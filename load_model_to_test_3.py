import os
import json

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix

from train_st_3 import *
from loss_func import *


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--saving_path", type=str)
    parser.add_argument('--mixed_domain', default=False, action='store_true')
    return parser


def save_test_results(model_path, saving_path, model_name, loss_func, test_loader, num_labels, device, scaler, criterion, mixed):
    # load model
    if "sentence-transformers" in model_name:
        model = SentenceTransformerModel(model_name, num_labels).to(device)
    else:
        model = TransformerModel(model_name, num_labels, loss_func).to(device)
    
    model.load_state_dict(torch.load(model_path))

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
    if mixed:
        plt.savefig(os.path.join(saving_path, f'mixed_confusion_matrix_from_{model_path[-24:]}.png'))
    else:
        plt.savefig(os.path.join(saving_path, f'confusion_matrix_from_{model_path[-24:]}.png'))

    # save the test results
    test_results = {
    'Test Loss': test_loss,
    'Test Accuracy': test_acc,
    'Test F1 Score': test_f1,
    }

    if mixed:
        with open(os.path.join(saving_path, f'mixed_test_results_acc_{test_acc:.4f}_from_{model_path[-24:]}.json'), 'w') as file:
            json.dump(test_results, file)
    else:
        with open(os.path.join(saving_path, f'test_results_acc_{test_acc:.4f}_from_{model_path[-24:]}.json'), 'w') as file:
            json.dump(test_results, file)


if __name__ == "__main__":
    # set configs
    parser = get_test_args()
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_length
    LOSS_FUNC = args.loss
    ALPHA = args.alpha
    TEMP = args.temp
    BATCH_SIZE = args.batch_size
    SEED = args.random_seed
    NUM_WORKERS = args.num_workers
    DEVICE = args.device
    TEST_DATA_PATH = args.test_data_path
    MODEL_PATH = args.model_path
    SAVING_PATH = args.saving_path
    MIXED = args.mixed_domain

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    seed_everything(SEED)

    idx2lbl = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    lbl2idx = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    
    # load data
    test_data = read_data(TEST_DATA_PATH)
    test_dataset = MGTDDataset(test_data[0], test_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)

     # get tokenizer
    if "gpt2" in MODEL_NAME:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    # create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                             collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), MODEL_NAME, LOSS_FUNC))
    
    # define loss function
    if LOSS_FUNC == 'dualcl':
        criterion = DualLoss(ALPHA, TEMP)
    elif LOSS_FUNC == 'scl':
        criterion = SupConLoss(ALPHA, TEMP)
    elif LOSS_FUNC == 'ce':
        criterion = CELoss()

    # save test results
    save_test_results(MODEL_PATH, SAVING_PATH, MODEL_NAME, LOSS_FUNC, test_loader, len(lbl2idx), DEVICE, scaler, criterion, MIXED)
