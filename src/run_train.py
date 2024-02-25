import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.args import get_runtime_args
from utils.common import seed_everything
from utils.dataloader import read_data, collate_batch, MGTDDataset
from model import ClassificationModel
from loss_func import CELoss, SupConLoss, DualLoss
from utils.train import run_epochs, test_results


############################# Setup Configs #############################
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

############################# Load Data #############################
# create Dataset
train_data = read_data(TRAIN_PATH)
dev_data = read_data(DEV_PATH)
test_data = read_data(TEST_PATH)

train_dataset = MGTDDataset(train_data[0], train_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC,
                            use_aug=USE_AUG, augmodel_path=AUGMODEL_PATH, aug_all=AUG_ALL,
                            synonym=SYN, antonym=ANT, swap=SWAP, spelling=SPELLING, word2vec=WORD2VEC)
dev_dataset = MGTDDataset(dev_data[0], dev_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)
test_dataset = MGTDDataset(test_data[0], test_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)

# get tokenizer
if ("roberta" in MODEL_NAME or "gpt" in MODEL_NAME) and LOSS_FUNC == "dualcl":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if "gpt" in MODEL_NAME:
    tokenizer.pad_token = tokenizer.eos_token

# create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
                            collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), LOSS_FUNC))
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                        collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), LOSS_FUNC))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                            collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), LOSS_FUNC))

############################# Load Model #############################
# load model
model = ClassificationModel(MODEL_NAME, len(idx2lbl), LOSS_FUNC).to(DEVICE)
    
# define loss function and optimizer
if LOSS_FUNC == 'dualcl':
    criterion = DualLoss(ALPHA, TEMP)
elif LOSS_FUNC == 'scl':
    criterion = SupConLoss(ALPHA, TEMP)
elif LOSS_FUNC == 'ce':
    criterion = CELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

############################# Run Training #############################
# optional: record a new run in wandb
# wandb.init(
#     project="Machine-Generated-Text-Detection",
#     name=SAVING_PATH[2:-9],
#     config={
#     "model_name": MODEL_NAME,
#     "max_len": MAX_LEN,
#     "loss": LOSS_FUNC,
#     "learning_rate": LEARNING_RATE,
#     "decay": DECAY,
#     "alpha": ALPHA,
#     "temp": TEMP,
#     "batch_size": BATCH_SIZE,
#     "use_aug": USE_AUG,
#     "aug_all": AUG_ALL,
#     "synonym": SYN,
#     "antonym": ANT,
#     "swap": SWAP,
#     "spelling": SPELLING,
#     "word2vec": WORD2VEC,
#     "seed": SEED,
#     "num_epochs": NUM_EPOCHS,
#     "early_stop": EARLY_STOP,
#     "num_workers": NUM_WORKERS,
#     "device": DEVICE,
#     "use_amp": args.use_amp,
# })

run_epochs(NUM_EPOCHS, EARLY_STOP, model, train_loader, dev_loader, DEVICE, scaler,
            criterion, SAVING_PATH, logging_file, optimizer)

# wandb.finish()

# load the best model and test on test set
test_results(SAVING_PATH, model, idx2lbl, test_loader, DEVICE, scaler, criterion)