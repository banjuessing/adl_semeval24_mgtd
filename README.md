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
