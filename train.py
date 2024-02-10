import argparse
import json
import pandas as pd
import os
import sys
import multiprocessing
from class_util import TextClassificationTrainer

def parse_args_and_config():
    """
    Parse command-line arguments and configuration file.
    """
    parser = argparse.ArgumentParser(description="Training script for text classification.")
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    parser.add_argument('--config_save_path', type=str, help='Path to save the configuration file.')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data file.')
    parser.add_argument('--val_data_path', type=str, help='Path to the validation data file.')
    parser.add_argument('--model_name', type=str, help='Model name or path.')
    parser.add_argument('--text_col', type=str, help='Name of the text column in the dataset.')
    parser.add_argument('--label_col', type=str, help='Name of the label column in the dataset.')
    parser.add_argument('--regex_query', type=str, help='Regex pattern for data cleaning.')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length.')
    parser.add_argument('--lowercase', type=bool, help='Convert text to lowercase before processing.')
    parser.add_argument('--augmodel_path', type=str, help='Path to augmentation model.')
    parser.add_argument('--all', type=bool, help='Apply augementation sequentially otherwise its sometimes skipped.')
    parser.add_argument('--synonym', type=bool, help='Apply synonym augmentation.')
    parser.add_argument('--antonym', type=bool, help='Apply antonym augmentation.')
    parser.add_argument('--swap', type=bool, help='Apply swap augmentation.')
    parser.add_argument('--spelling', type=bool, help='Apply spelling augmentation.')
    parser.add_argument('--word2vec', type=bool, help='Apply word2vec augmentation.')
    parser.add_argument('--contextual', type=bool, help='Apply contextual augmentation.')
    parser.add_argument('--augmentation', type=bool, help='Enable data augmentation.')
    parser.add_argument('--labels_to_augment', nargs='+', type=int, help='List of labels to augment.')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving the model.')
    parser.add_argument('--logging_dir', type=str, help='Logging directory for TensorBoard logs.')
    parser.add_argument('--per_device_train_batch_size', type=int, help='Training batch size per device.')
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Evaluation batch size per device.')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of gradient accumulation steps.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, help='Number of warmup steps.')
    parser.add_argument('--weight_decay', type=float, help='Weight decay.')
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--evaluation_strategy', type=str, help='Evaluation strategy.')
    parser.add_argument('--logging_strategy', type=str, help='Logging strategy.')
    parser.add_argument('--logging_steps', type=int, help='Logging steps.')
    parser.add_argument('--save_strategy', type=str, help='Save strategy.')
    parser.add_argument('--seed', type=int, help='Random seed for initialization.')
    parser.add_argument('--data_seed', type=int, help='Seed for data shuffling.')
    parser.add_argument('--fp16', type=bool, help='Whether to use 16-bit (mixed) precision instead of 32-bit')
    parser.add_argument('--dataloader_num_workers', type=int, help='Number of workers for data loading.')
    parser.add_argument('--load_best_model_at_end', type=bool, help='Whether to load the best model found during training at the end of training.')
    parser.add_argument('--metric_for_best_model', type=str, help='The metric to use to compare model performance.')
    parser.add_argument('--greater_is_better', type=bool, help='Whether the `metric_for_best_model` should be maximized or not.')

    
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f"Configuration file not found: {args.config}")
            sys.exit(1)
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON configuration file: {args.config}")
            sys.exit(1)
    else:
        config = {}

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config


def save_config(config, path):
    """
    Save configuration to a file.
    """
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving configuration: {e}")
        sys.exit(1)


def load_data(config):
    """
    Load training and validation data.
    """
    try:
        train_data = pd.read_json(config['train_data_path'], lines=True)
        val_data = pd.read_json(config['val_data_path'], lines=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    return train_data, val_data

def run_training(config, train_data, val_data):
    """
    Run training.
    """
    try:
        trainer = TextClassificationTrainer(config, train_data, val_data)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def main():
    config = parse_args_and_config()
    save_config(config, config['config_save_path'])
    train_data, val_data = load_data(config)
    run_training(config, train_data, val_data)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
