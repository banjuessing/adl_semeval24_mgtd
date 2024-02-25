import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from augmentation import get_augmentation


class MetricsCallback(TrainerCallback):
    "A callback that calls a function with the evaluation metrics after each evaluation."

    def __init__(self, metrics_processor):
        self.metrics_processor = metrics_processor
        self.metrics = []  # Add this line

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics_processor(metrics)
        self.metrics.append(metrics)  # Add this line

class CustomTrainer(Trainer):
    def __init__(self, *args, metrics_capture_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        if metrics_capture_callback is not None:
            self.add_callback(metrics_capture_callback)

class TextClassificationTrainer:
    def __init__(self, config, train_data, val_data, metrics_processor):
        required_fields = ['model_name', 'text_col', 'label_col']

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        self.config = config
        # universal
        self.model_name = config.get('model_name')
        self.text_col = config.get('text_col')
        self.label_col = config.get('label_col')
        self.regex_query = config.get('regex_query')
        self.max_length = config.get('max_length', 512)
        self.lowercase = config.get('lowercase')
        # augmentation
        self.augmodel_path = config.get('augmodel_path', './augmodel')
        self.all = config.get('all')
        self.synonym = config.get('synonym')
        self.antonym = config.get('antonym')
        self.swap = config.get('swap')
        self.spelling = config.get('spelling')
        self.word2vec = config.get('word2vec')
        self.contextual = config.get('contextual')
        self.augmentation = config.get('augmentation')
        self.augmenter = None
        self.labels_to_augment = config.get('labels_to_augment', [0,5])
        # training args
        self.output_dir = config.get('output_dir', f"./{config.get('model_name')}_output")
        self.logging_dir = config.get('logging_dir', f"./{config.get('model_name')}_logs")
        self.per_device_train_batch_size = config.get('per_device_train_batch_size', 8)
        self.per_device_eval_batch_size = config.get('per_device_eval_batch_size', 40)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.weight_decay = config.get('weight_decay', 1e-3)
        self.num_train_epochs = config.get('num_train_epochs', 8)
        self.evaluation_strategy = config.get('evaluation_strategy', 'epoch')
        self.logging_strategy = config.get('logging_strategy', 'steps')
        self.logging_steps = config.get('logging_steps', len(train_data) // self.per_device_train_batch_size)
        self.save_strategy = config.get('save_strategy', 'epoch')
        self.seed = config.get('seed', 42)
        self.data_seed = config.get('data_seed', 42)
        self.fp16 = config.get('fp16', True)
        self.dataloader_num_workers = config.get('dataloader_num_workers', 0)
        self.load_best_model_at_end = config.get('load_best_model_at_end')
        self.metric_for_best_model = config.get('metric_for_best_model', 'accuracy')
        self.greater_is_better = config.get('greater_is_better', True)

        self.metrics_capture_callback = MetricsCallback(metrics_processor)

        self.train_data = self._process_dataset(train_data)
        self.val_data = self._process_dataset(val_data)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model_name == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = len(np.unique(self.train_data[self.label_col]))
        )

        if self.model_name == 'gpt2':
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if self.augmentation is True:
            self.augmenter = get_augmentation(self.augmodel_path, self.all, self.synonym, self.antonym, self.swap, self.spelling, self.word2vec, self.contextual)
            self.dataloader_num_workers = 0

    def _clean_text(self, text):
        try:
            if self.regex_query:
                text = re.sub(self.regex_query, ' ', text)
            if self.lowercase:
                text = text.lower()
            return text.strip()
        except Exception as e:
            print(f"Error during text cleaning: {e}")
            return ""

    def _process_dataset(self, dataset):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Dataset must be a pandas DataFrame.")
        try:
            dataset[self.text_col] = dataset[self.text_col].apply(self._clean_text)
            return dataset
        except Exception as e:
            raise RuntimeError(f"Error processing dataset: {e}")
        
    def get_current_epoch(self):
        return self.trainer.state.epoch

    class TextClassificationDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length, augmenter=None, labels_to_augment=None):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.augmenter = augmenter
            self.labels_to_augment = labels_to_augment
            self.max_length = max_length

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            if self.augmenter is None:
                pass
            elif self.augmenter is not None:
                if self.labels_to_augment is None:
                    self.labels_to_augment = np.unique(self.labels)
                if self.augmenter and label in self.labels_to_augment:
                    text = self.augmenter.augment(text)[0]

            encoding = self.tokenizer(text, padding = 'max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label)
            }
            return item

        def __len__(self):
            return len(self.labels)

    def _build_dataset(self):
        if self.train_data.empty or self.val_data.empty:
            raise ValueError("Training or validation data is empty.")
        
        self.train_dataset = self.TextClassificationDataset(self.train_data[self.text_col].to_list(), self.train_data[self.label_col].to_list(), self.tokenizer, self.max_length, augmenter=self.augmenter, labels_to_augment=self.labels_to_augment)
        self.eval_dataset = self.TextClassificationDataset(self.val_data[self.text_col].to_list(), self.val_data[self.label_col].to_list(), self.tokenizer, self.max_length, augmenter=None, labels_to_augment=None)

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
        }


    def train(self):
        try:
            self._build_dataset()

            training_args_dict = {
                "output_dir": self.output_dir,
                "logging_dir": self.logging_dir,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "per_device_eval_batch_size": self.per_device_eval_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.weight_decay,
                "num_train_epochs": self.num_train_epochs,
                "evaluation_strategy": self.evaluation_strategy,
                "logging_strategy": self.logging_strategy,
                "logging_steps": self.logging_steps,
                "save_strategy": self.save_strategy,
                "seed": self.seed,
                "data_seed": self.data_seed,
                "fp16": self.fp16,
                "dataloader_num_workers": self.dataloader_num_workers,
                "load_best_model_at_end": self.load_best_model_at_end,
                "metric_for_best_model": self.metric_for_best_model,
                "greater_is_better": self.greater_is_better,
            }

            training_args_dict = {k: v for k, v in training_args_dict.items() if v is not None}

            training_args = TrainingArguments(**training_args_dict)

            self.trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self._compute_metrics,
                tokenizer=self.tokenizer,
                callbacks=[self.metrics_capture_callback]
            )


            self.trainer.train()

            last_evaluation_metrics = self.metrics_capture_callback.metrics[-1] if self.metrics_capture_callback.metrics else {}
            return last_evaluation_metrics

        except Exception as e:
            print(f"Error during training: {e}")
            raise e
