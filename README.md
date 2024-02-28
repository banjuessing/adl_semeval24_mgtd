# Leveraging (Sentence) Transformer Models with Contrastive Learning for Identifying Machine-Generated Text
## Overview
This is the code base of the project for Applied Deep Learning WS23/24 & SemEval-2024 Task 8: Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection.

Our detection system is built upon Transformer-based techniques, leveraging various pre-trained language models (PLMs), including sentence transformer models. Additionally, we incorporate Contrastive Learning (CL) into the classifier to improve the detecting capabilities and employ Data Augmentation methods. Ultimately, our system achieves a peak accuracy of 76.96% on the test set of the competition, configured using a sentence transformer model integrated with CL methodology.

## Quick Links
| Section                                 | Description                                                     |
| :-------------------------------------: |:--------------------------------------------------------------: |
| [Requirements](#Requirements)           | How to set up the python environemnt of our experiments         |
| [Data Preparation](#Data-Preparation)   | How to download and prepare the data for our experiments        |
| [Experiments](#Experiments)             | How to run our experiments and use our best model for inference |
| [Archive](#Archive)                     | Additional experimental code and configurations                 |

## Requirements
Our experiments are based on:
* python 3.10.12
* gensim 4.3.2
* matplotlib 3.7.1
* nlpaug 1.1.11
* nltk 3.8.1
* numpy 1.25.2
* pandas 1.5.3
* scikit-learn 1.2.2
* seaborn 0.13.1
* torch 2.1.0
* tqdm 4.66.2
* transformers 4.37.2

Use [requirements.txt](requirements.txt) for quick installation

## Data Preparation
The datasets that we used to train and evaluate the models contain three subsets: `train_set`, `val_set` and `test_set`, where `train_set` and `val_set` are 90% and 10% of the [original training dataset](https://github.com/mbzuai-nlp/SemEval2024-task8) provided by the SemEval-2024 Task 8 organizers, and `test_set` is all available samples from the [original M4 dataset](https://github.com/mbzuai-nlp/M4) specific to the PeerRead domain.

To prepare the datasets for training and evaluation, download the folder [`SubtaskB`](https://drive.google.com/drive/folders/1Hh8kD9NlbKfJLpJ_BRvcckN20Xpxxjh_?usp=share_link) and put it under the folder `data`.

(The dataset splits that we used in our experiments as described above are under the subfolder `enrich_comp_data`. The other two subfolders contain different splits of the wohle dataset, where subfolder `comp` contains the orginal training and development dataset from SemEval-2024 Task 8, and `mixed_domain_data` contains splits with a 80:10:10 ratio of all data from SemEval-2024 Task 8 and all PeerRead domain data from M4 dataset.)

## Experiments
### Training, Evaluation and Testing
It is possible to start reproducing all results by running `zsh run.sh` from inside the `src` directory. This will start executing training, evaluation, and testing of all possible model configurations in `train_configs` in a loop. Random seeds are set in all scripts.

Or to train, evaluate and test a single model configuration:
```
cat train_configs/[a_cfg_file_under_the_train_configs].cfg | xargs python run_train.py
```

Or to train, evaluate and test a single model configuration with customized hyperparameters, use the output of following code under `src` for reference to create the corresponding configuration file.
```
python run_train.py --help
```

### Testing
To load a single model checkpoint and test it on the test set:
```
cat test_configs/[a_cfg_file_under_the_test_configs].cfg | xargs python load_model_to_test.py
```

Or to load a single model checkpoint and test it with customized hyperparameters, use the output of following code under `src` for reference to create the corresponding configuration file.
```
python load_model_to_test.py --help
```

### Inference
To load our [best performed model](https://drive.google.com/file/d/10VTeF4KGdZMtmkXCbNIFK49bsjOQKbuz/view?usp=share_link) for inference:
```
python infer.py -[path_to_the_saved_model_checkpoint] -[text_to_be_detected]
```
(Our best performed model is a sentence transformer model `all-roberta-large-v1` fintuned with Supervised Contrastive Loss.)

## Archive
The Archive branch contains additional code developed during our project, offering insights into our experimentation with Transformer models and Bayesian Optimization techniques like Tree-Parzen Estimators using Optuna. This code diverges from the main codebase and may require additional packages (optuna 3.1.1).

To replicate these experiments, please refer to the specific setup and execution instructions provided within this section. Use the following command as a starting point: ```python train.py --config 'path/to/config.json'```. Additionally, example JSON config files are included to illustrate possible configurations. This section is particularly valuable for those interested in deeper insights into the experimental aspects of our machine-generated text detection system.
