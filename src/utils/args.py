import argparse

def get_runtime_args():
    parser = argparse.ArgumentParser()

    # specify model
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=512)

    # specify loss function
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"],
                        help="Which loss function to use for training, ce for cross-entropy, \
                        scl for supervised contrastive learning, \
                        dualcl for dual contrastive learning.")
    
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
    return parser