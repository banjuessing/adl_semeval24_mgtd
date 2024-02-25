import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.args import get_test_args
from utils.common import seed_everything
from utils.dataloader import read_data, collate_batch, MGTDDataset
from loss_func import CELoss, SupConLoss, DualLoss
from utils.train import test_results


############################# Setup Configs #############################
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

scaler = torch.cuda.amp.GradScaler(enabled=True)

seed_everything(SEED)

idx2lbl = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
lbl2idx = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}

############################# Load Data #############################
# create Dataset
test_data = read_data(TEST_DATA_PATH)
test_dataset = MGTDDataset(test_data[0], test_data[1], lbl2idx, MODEL_NAME, LOSS_FUNC)

# get tokenizer
if "roberta" in MODEL_NAME and LOSS_FUNC == "dualcl":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if "gpt" in MODEL_NAME:
    tokenizer.pad_token = tokenizer.eos_token

# create DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                            collate_fn=lambda batch: collate_batch(batch, tokenizer, MAX_LEN, len(lbl2idx), LOSS_FUNC))

############################# Test Model #############################
# define loss function
if LOSS_FUNC == 'dualcl':
    criterion = DualLoss(ALPHA, TEMP)
elif LOSS_FUNC == 'scl':
    criterion = SupConLoss(ALPHA, TEMP)
elif LOSS_FUNC == 'ce':
    criterion = CELoss()

# save test results
test_results(SAVING_PATH, None, idx2lbl, test_loader, DEVICE, scaler, criterion, LOSS_FUNC, MODEL_NAME, MODEL_PATH)
