import pandas as pd
import shutil
from pathlib import Path

transformers_path = Path("/home/syzong/anaconda3/lib/python3.7/site-packages/transformers")

input_dir = Path("./deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py', "deberta__init__.py"]:
    if str(filename).startswith("deberta"):
        filepath = deberta_v2_path/str(filename).replace("deberta", "")
    else:
        filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)



import warnings
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


patient_notes = pd.read_csv('./nbme-score-clinical-patient-notes/patient_notes.csv')

def display(data):
    return print(data)


print(f"patient_notes.shape: {patient_notes.shape}")
display(patient_notes.head())



# def process_feature_text(text):
#     text = re.sub('I-year', '1-year', text)
#     text = re.sub('-OR-', " or ", text)
#     text = re.sub('-', ' ', text)
#     return text
#
#
def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
#     txt = re.sub(r'\s+', ' ', txt)
    return txt

# train['pn_history'] = train['pn_history'].apply(lambda x: x.strip())
# train['feature_text'] = train['feature_text'].apply(process_feature_text)
#
# train['feature_text'] = train['feature_text'].apply(clean_spaces)
# train['pn_history'] = train['pn_history'].apply(clean_spaces)

patient_notes['pn_history'] = patient_notes['pn_history'].apply(clean_spaces)

comment_text = patient_notes['pn_history']
print("len comment_text : ", len(comment_text))

text  = '\n'.join(comment_text.tolist())

with open('text.txt','w') as f:
    f.write(text)



model_name = './deberta-v3-large'
model = AutoModelForMaskedLM.from_pretrained(model_name)


from transformers.models.deberta_v2 import DebertaV2TokenizerFast

tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")

# abbreviations = ['hpi','pmhx','rx','fhx','shx','phi','meds','sh','ros','psh','hx','pshx','fmhx']
#
# tokenizer.add_tokens(abbreviations, special_tokens=False)


# ====================================================
# tokenizer
# ====================================================
# from transformers.models.deberta_v2 import DebertaV2TokenizerFast
# tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)


tokenizer.save_pretrained('./nbme_deberta_v3_large_pretrain');


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",
    block_size=512)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",
    block_size=512)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./checkpoints_largev3",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    gradient_accumulation_steps=3,
    eval_steps=10000,
    save_steps=10000,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
    report_to = "none",
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)



trainer.train()
trainer.save_model(f'./nbme_deberta_v3_large_pretrain')
