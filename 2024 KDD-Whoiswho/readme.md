# Whoiswho

## Team **LGB YYDS** RANK9

## Prerequisites
- Windows
- python==3.9
- pandas==1.4.4
- numpy==1.21.6
- scikit-learn==1.4.2
- gensim==4.1.2
- cogdl==0.6
- tqdm==4.64.1
- lightgbm==4.1.0
- xgboost==2.0.2
- pytorch==1.13.1+cu117
- pyarrow==16.0

## Hardware device
- CPU AMD 5600X
- GPU 3080Ti 12G
- RAM 64G

## Parameter count
total ~110,500,000
- oagbert-v2 ~110M
- mlp ~430k
- hand-crafted features ~2k

## File structure
- data [dataset given by organizer]
  - train_author.json
  - pid_to_info_all.json
  - ind_test_author_submit.json
  - ind_test_author_filter_public.json]
- usr_data [dataset generated by codes]
- models [models trained]
- ans_test [single model answers and the final answer(named ensemble.json)]
- code files

files of usr_data/models/ans_test can be downloaded from

Link: https://pan.baidu.com/s/1iCtXYY-1jIp51lVAAcduMg Password: fq70

## Run code
- Train+infer: sh train.sh
- Only infer: sh infer.sh

## Method
1. extract embedding of article information with w2v and tfidf
2. extract embedding of article information with OAG Bert
3. do feature engineering: statistics features and distance features
4. train mlp model with features to get an oof prediction
5. train xgboost and lightgbm models with different groups of features and mlp oof prediction to get 4 single model predictions
6. ensemble, just weighted average of 4 single-model answers
 