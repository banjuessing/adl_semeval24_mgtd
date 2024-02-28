import argparse

def get_runtime_args():
    parser = argparse.ArgumentParser()

    # specify model
    parser.add_argument("--model_name", type=str, 
                        choices=["gpt2", "xlnet-base-cased", "roberta-base", "roberta-large", 
                                 "sentence-transformers/all-mpnet-base-v1", 
                                 "sentence-transformers/all-mpnet-base-v2", 
                                 "sentence-transformers/all-roberta-large-v1"], 
                        help="Which model to use for training.")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="The maximum input sequence length, which should be matched with the chosen model.")

    # specify loss function
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"],
                        help="Which loss function to use for training, ce for cross-entropy, \
                        scl for supervised contrastive learning, \
                        dualcl for dual contrastive learning.")
    
    # specify training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="The learning rate for AdamW optimizer.")
    parser.add_argument("--decay", type=float, default=0.01, 
                        help="The weight decay coefficient for AdamW optimizer.")
    parser.add_argument("--alpha", type=float, default=0.1, choices=[0.01, 0.1, 0.2], 
                        help="The weight value which controls the influence of the contrastive loss term.")
    parser.add_argument("--temp", type=float, default=0.1, 
                        help="The temperature factor.")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # specify augmentation methods
    parser.add_argument('--use_aug', default=False, action='store_true',
                        help="Whether to use data augmentation.")
    parser.add_argument('--augmodel_path', type=str, default=None,
                        help="The path to the augmodel, if word2vec augmentation is used,\
                            place first the GoogleNews-vectors-negative300.bin under this augmodel_path.")
    parser.add_argument('--aug_all', default=True, action='store_true', 
                        help="Whether to apply all augmentation methods or apply them randomly.")
    parser.add_argument('--synonym', default=True, action='store_true', 
                        help="Whether to use synonym augmentation.")
    parser.add_argument('--antonym', default=False, action='store_true',
                        help="Whether to use antonym augmentation.")
    parser.add_argument('--swap', default=False, action='store_true',
                        help="Whether to use random word swap augmentation.")
    parser.add_argument('--spelling', default=False, action='store_true',
                        help="Whether to use random word spelling augmentation.")
    parser.add_argument('--word2vec', default=False, action='store_true', 
                        help="WHether to use random word2vec augmentation.")
    
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

    parser.add_argument("--use_amp", default=False, action="store_true", 
                        help="Whether to use mixed precision training.")

    return parser

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, 
                        choices=["gpt2", "xlnet-base-cased", "roberta-base", "roberta-large", 
                                 "sentence-transformers/all-mpnet-base-v1", 
                                 "sentence-transformers/all-mpnet-base-v2", 
                                 "sentence-transformers/all-roberta-large-v1"], 
                        help="Which model to use for testing."),
    parser.add_argument("--max_length", type=int, default=512,
                        help="The maximum input sequence length, which should be matched with the chosen model.")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "scl", "dualcl"],
                        help="Which loss function to use for testing, ce for cross-entropy, \
                        scl for supervised contrastive learning, \
                        dualcl for dual contrastive learning.")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="The weight value which controls the influence of the contrastive loss term.")
    parser.add_argument("--temp", type=float, default=0.1, 
                        help="The temperature factor.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--model_path", type=str, 
                        help="The path to the model to be tested.")
    parser.add_argument("--saving_path", type=str)
    return parser