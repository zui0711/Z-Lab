# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './output_xlarge/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

                break

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
                break
        self.backup = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

# ====================================================
# CFG
# ====================================================
class CFG:
    wandb = True
    competition = 'NBME'
    _wandb_kernel = 'nakama'
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    model = "microsoft/deberta-xlarge"
    model_pre = "nbme_deberta_xlarge_pretrain"
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 4
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 4
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# ====================================================
# wandb
# ====================================================
'''
if CFG.wandb:

    import wandb

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print(
            'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


    run = wandb.init(project='NBME-Public',
                     name=CFG.model,
                     config=class2dict(CFG),
                     group=CFG.model,
                     job_type="train",
                     anonymous=anony)

'''
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

# os.system('pip uninstall -y transformers')
# os.system('python -m pip install transformers==4.16.2')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

#%env TOKENIZERS_PARALLELISM=true

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#Root_path = "../input/nbme-score-clinical-patient-notes/"
Root_path = "./nbme-score-clinical-patient-notes/"

# From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline
def display(data):
    print(data)

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)


def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        # if len(result) > 0:
        #     if result[0][0] == 1:
        #         result[0][0] = 0

            # if result[-1][-1] >= len(char_prob):
            #     result[-1][-1] = len(char_prob) -1

        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=12345)

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv(Root_path + 'train.csv')
train['annotation'] = train['annotation'].apply(ast.literal_eval)
train['location'] = train['location'].apply(ast.literal_eval)
features = pd.read_csv(Root_path + 'features.csv')
def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features
features = preprocess_features(features)
patient_notes = pd.read_csv(Root_path + 'patient_notes.csv')

print(f"train.shape: {train.shape}")
display(train.head())
print(f"features.shape: {features.shape}")
display(features.head())
print(f"patient_notes.shape: {patient_notes.shape}")
display(patient_notes.head())

train = train.merge(features, on=['feature_num', 'case_num'], how='left')
train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
display(train.head())

# incorrect annotation
train.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
train.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

train.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
train.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

train.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
train.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

train.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
train.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

train.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
train.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

train.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
train.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

train.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
train.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

train.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
train.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

train.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
train.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

train.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
train.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

train.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
train.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

train.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
train.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

train.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
train.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

train.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
train.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

train.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
train.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

train.loc[3858, 'annotation'] = ast.literal_eval('[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
train.loc[3858, 'location'] = ast.literal_eval('[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

train.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
train.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

train.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
train.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

train.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
train.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

train.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
train.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

train.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
train.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

train.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
train.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

train.loc[6380, 'annotation'] = ast.literal_eval('[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
train.loc[6380, 'location'] = ast.literal_eval('[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

train.loc[6562, 'annotation'] = ast.literal_eval('[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
train.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

train.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
train.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

train.loc[7022, 'annotation'] = ast.literal_eval('[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
train.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

train.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
train.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

train.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
train.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

train.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
train.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

train.loc[9938, 'annotation'] = ast.literal_eval('[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
train.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

train.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
train.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

train.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
train.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

train.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
train.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

train.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
train.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

train.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
train.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

train.loc[12279, 'annotation'] = ast.literal_eval('[["heard what she described as a party later than evening these things did not actually happen"]]')
train.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

train.loc[12289, 'annotation'] = ast.literal_eval('[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
train.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

train.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
train.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

train.loc[13297, 'annotation'] = ast.literal_eval('[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
train.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

train.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
train.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

train.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
train.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

train.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
train.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')

train['annotation_length'] = train['annotation'].apply(len)
display(train['annotation_length'].value_counts())

Add_fake = True
# ADD Fake data -----------------------
if Add_fake:
    #train_fake = pd.read_csv(Root_path + 'train_pseudo_0.889_1000pn_new.csv')
    train_fake = pd.read_csv(Root_path + 'train_pseudo_v1_fixed.csv')

    train_fake['annotation'] = train_fake['annotation'].apply(ast.literal_eval)
    train_fake['location'] = train_fake['location'].apply(ast.literal_eval)

    display(train_fake.head())

    train_fake = train_fake.merge(features, on=['feature_num', 'case_num'], how='left')
    train_fake = train_fake.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

    display(train_fake.head())
    display(train_fake.tail())

    Fold_f = GroupKFold(n_splits=CFG.n_fold)
    groups_f = train_fake['pn_num'].values
    for n, (train_index, val_index) in enumerate(Fold_f.split(train_fake, train_fake['location'], groups_f)):
        train_fake.loc[val_index, 'fold'] = int(n)
    train_fake['fold'] = train_fake['fold'].astype(int)
    display(train_fake.groupby('fold').size())

    train_fake['annotation_length'] = train_fake['annotation'].apply(len)

# ADD Fake data -----------------------


# def process_feature_text(text):
#     text = re.sub('I-year', '1-year', text)
#     text = re.sub('-OR-', " or ", text)
#     text = re.sub('-', ' ', text)
#     return text
#
#
# def clean_spaces(txt):
#     txt = re.sub('\n', ' ', txt)
#     txt = re.sub('\t', ' ', txt)
#     txt = re.sub('\r', ' ', txt)
# #     txt = re.sub(r'\s+', ' ', txt)
#     return txt
#
#
# train['pn_history'] = train['pn_history'].apply(lambda x: x.strip())
# train['feature_text'] = train['feature_text'].apply(process_feature_text)
#
# train['feature_text'] = train['feature_text'].apply(clean_spaces)
# train['pn_history'] = train['pn_history'].apply(clean_spaces)




# ====================================================
# CV split
# ====================================================
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = train['pn_num'].values
for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
display(train.groupby('fold').size())

if CFG.debug:
    display(train.groupby('fold').size())
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)
    display(train.groupby('fold').size())

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================
for text_col in ['pn_history']:
    pn_history_lengths = []
    tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        pn_history_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(pn_history_lengths)}')

for text_col in ['feature_text']:
    features_lengths = []
    tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        features_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(features_lengths)}')

CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3 # cls & sep & sep
LOGGER.info(f"max_len: {CFG.max_len}")

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(text,
                            add_special_tokens=True,
                            max_length=CFG.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])
        label = create_label(self.cfg,
                             self.pn_historys[item],
                             self.annotation_lengths[item],
                             self.locations[item])
        return inputs, label


# ====================================================
# Model
# ====================================================
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model_pre, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_pre, config=self.config)
        else:
            self.model = AutoModel(self.config)

        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        #self.dropouts = nn.ModuleList([nn.Dropout(d_p) for d_p in np.linspace(0.05, 0.5, 10) ])

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

"""# Loss Function"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def binary_cross_entropy(inputs, target, weight=None, reduction='none', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.)
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='none', smooth_eps=None, from_logits=True):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)


class BCEWithLogitsLoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average,
                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)



def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, fgm, pgd,
             valid_loader,valid_folds,valid_texts,valid_labels,best_score):

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    save_step = int(len(train_loader) / 3)

    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # ---------------------fgm-------------
        fgm.attack()  # embedding被修改了

        #optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss_avd = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss_avd = torch.masked_select(loss_avd, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss_avd = loss_avd / CFG.gradient_accumulation_steps
        losses.update(loss_avd.item(), batch_size)
        scaler.scale(loss_avd).backward()

        fgm.restore()  # 恢复Embedding的参数
        # ---------------------fgm-------------


        # ---------------------pgd-------------
        # K_pgd = 3
        #
        # pgd.backup_grad()  # 保存正常的grad
        # # 对抗训练
        # for t in range(K_pgd):
        #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        #     if t != K_pgd - 1:
        #         optimizer.zero_grad()
        #     else:
        #         pgd.restore_grad()  # 恢复正常的grad
        #
        #     with torch.cuda.amp.autocast(enabled=CFG.apex):
        #         y_preds = model(inputs)
        #     loss_pgd = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        #     loss_pgd = torch.masked_select(loss_pgd, labels.view(-1, 1) != -1).mean()
        #     if CFG.gradient_accumulation_steps > 1:
        #         loss_pgd = loss_pgd / CFG.gradient_accumulation_steps
        #     losses.update(loss_pgd.item(), batch_size)
        #     scaler.scale(loss_pgd).backward()
        #
        # pgd.restore()  # 恢复embedding参数
        # 梯度下降，更新参数
        # ---------------------pgd-------------




        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if CFG.wandb:
            if step % 20 == 0:
                print({f"[fold{fold}] loss": losses.val, f"[fold{fold}] lr": scheduler.get_lr()[0]})


        if (step+1) % save_step == 0 and epoch > 0:
            # eval
            avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
            predictions = predictions.reshape((len(valid_folds), CFG.max_len))

            # scoring
            char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
            results = get_results(char_probs, th=0.5)
            preds = get_predictions(results)
            score = get_score(valid_labels, preds)


            LOGGER.info(f'Epoch {epoch + 1} - step: {step:.4f}  avg_val_loss: {avg_val_loss:.4f}')
            LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
            if CFG.wandb:
                print({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score,
                       f"[fold{fold}] best_score": best_score})

            if best_score < score:
                best_score = score
                LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                           OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
    return losses.avg, best_score


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)

    # ADD Fake ===================
    if Add_fake:
        train_folds_fake = train_fake[train_fake['fold'] != fold].reset_index(drop=True)
        train_folds = train_folds.append(train_folds_fake)

        #train_folds = train_folds.append(train_fake)

    # ADD Fake ===================

    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_texts = valid_folds['pn_history'].values
    valid_labels = create_labels_for_scoring(valid_folds)

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)

    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="none")


    # criterion = BCEWithLogitsLoss(reduction="none")#nn.BCEWithLogitsLoss(reduction="none")
    # criterion = DiceBCELoss()

    best_score = 0.
    fgm = FGM(model)

    pgd = PGD(model)
    K = 3

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss, new_score = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, fgm, pgd,
                            valid_loader,valid_folds,valid_texts,valid_labels, best_score)

        best_score = new_score



    predictions = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']

    # valid_folds_pre = []
    # for i in range(CFG.max_len):
    #     valid_folds_pre.append([])
    #
    # for i in range(CFG.max_len):
    #     valid_folds_pre[[i]] = predictions
    #
    # #valid_folds_pre[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds,predictions



if __name__ == '__main__':

    def get_result(oof_df,oof_df_pre):
        labels = create_labels_for_scoring(oof_df)
        #predictions = oof_df_pre[[i for i in range(CFG.max_len)]].values
        predictions = oof_df_pre
        char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')


    if CFG.train:
        oof_df = pd.DataFrame()

        for fold in range(CFG.n_fold):

            if fold in CFG.trn_fold:
                _oof_df,_oof_df_pre = train_loop(train, fold)
                #oof_df = pd.concat([oof_df, _oof_df])

                np.save(OUTPUT_DIR + str(fold) + "_fold_prop.npy", _oof_df_pre)
                _oof_df.to_pickle(OUTPUT_DIR + str(fold) + '_fold_oof_df.pkl')

                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df,_oof_df_pre)
        # oof_df = oof_df.reset_index(drop=True)
        # LOGGER.info(f"========== CV ==========")
        # get_result(oof_df,pred_all)
        # oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')

    # if CFG.wandb:
    #     wandb.finish()

