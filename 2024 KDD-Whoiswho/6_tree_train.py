import pandas as pd
import numpy as np
import json

from sklearn.model_selection import StratifiedKFold, GroupKFold

import lightgbm as lgb
import xgboost as xgb
import pickle

df = pd.read_parquet('usr_data/df.parquet')

df_train = df[df['label'].notna()].reset_index(drop=True)
df_test = df[df['label'].isna()].reset_index(drop=True)

nn_oof = pd.read_csv('usr_data/nn_valid.csv', usecols=['name_id', 'article_id', 'preds'])

nn_test = pd.read_csv('usr_data/nn_test.csv')
nn_test.columns = ['preds']

df_train = df_train.merge(nn_oof, on=['name_id', 'article_id'], how='left')
df_test = pd.concat([df_test, nn_test], axis=1)

f = 'preds'
df_train[f] = df_train[f] / df_train.groupby('name_id')[f].transform('mean')
df_test[f] = df_test[f] / df_test.groupby('name_id')[f].transform('mean')


fold_num = 5
seeds = [2222, 222, 22]

"""
xgb model1
"""

feats = [f for f in df_train.columns if
         f not in
         ['label', 'name_id', 'article_id', 'authors_name', 'name', 'oof', 'venue_le_mode_count', 'st_emb_dis',
          'authors', 'author_dic', 'author_org', 'author_org_mode', 'fold',
          # 'preds'
          ]
         +['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
         ]

print(df_train[feats].shape, df_test[feats].shape)
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
LABEL = 'label'
for seed in seeds:
    kf = GroupKFold(n_splits=fold_num)
    params_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 8,
        'subsample':0.8,
        'colsample_bytree': 0.9,
        'seed': seed
    }
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL], groups=df_train['name_id'])):
        print('-----------', fold)
        train = xgb.DMatrix(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = xgb.DMatrix(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        watchlist = [(train, 'train'), (val, 'eval')]
        model = xgb.train(params_xgb, dtrain=train, evals=watchlist, num_boost_round=1500,
                          maximize=True,
                          verbose_eval=500)

        oof[val_idx] += model.predict(xgb.DMatrix(df_train.loc[val_idx, feats])) / len(seeds)
        # pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(xgb.DMatrix(df_test[feats]))
        d = model.get_score(importance_type='total_gain')
        d = np.array(list(d.values()))
        importance += np.pad(d, (0, len(feats) - len(d)))
        pickle.dump(model, open(f'models/xgb_oag_nn_seed{seed}_fold{fold}.pkl', 'wb'))


"""
xgb model2
"""

feats = [f for f in df_train.columns if
         f not in
         ['label', 'name_id', 'article_id', 'authors_name', 'name', 'oof', 'venue_le_mode_count', 'st_emb_dis',
          'authors', 'author_dic', 'author_org', 'author_org_mode', 'fold',
          # 'preds'
          ]
         # +['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
         ]

print(df_train[feats].shape, df_test[feats].shape)
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
LABEL = 'label'
for seed in seeds:
    kf = GroupKFold(n_splits=fold_num)
    params_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 8,
        'subsample':0.8,
        'colsample_bytree': 0.9,
        'seed': seed
    }
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL], groups=df_train['name_id'])):
        print('-----------', fold)
        train = xgb.DMatrix(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = xgb.DMatrix(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        watchlist = [(train, 'train'), (val, 'eval')]
        model = xgb.train(params_xgb, dtrain=train, evals=watchlist, num_boost_round=1500,
                          maximize=True,
                          verbose_eval=500)

        oof[val_idx] += model.predict(xgb.DMatrix(df_train.loc[val_idx, feats])) / len(seeds)
        # pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(xgb.DMatrix(df_test[feats]))
        d = model.get_score(importance_type='total_gain')
        d = np.array(list(d.values()))
        importance += np.pad(d, (0, len(feats) - len(d)))
        pickle.dump(model, open(f'models/xgb_oag_nn_emb_seed{seed}_fold{fold}.pkl', 'wb'))



"""
lgb model1
"""

feats = [f for f in df_train.columns if
         f not in
         ['label', 'name_id', 'article_id', 'authors_name', 'name', 'oof', 'venue_le_mode_count', 'st_emb_dis',
          'authors', 'author_dic', 'author_org', 'author_org_mode', 'fold',
          'preds'
          ]
         +['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
         ]

print(df_train[feats].shape, df_test[feats].shape)
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
LABEL = 'label'
for seed in seeds:
    kf = GroupKFold(n_splits=fold_num)
    params_lgb = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 64,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,

        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 4,
    }
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL], groups=df_train['name_id'])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        model = lgb.train(params_lgb, train, valid_sets=[val], num_boost_round=1500,
                          callbacks=[  # lgb.early_stopping(100),
                              lgb.log_evaluation(500)]
                          )

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        # pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain') / fold_num
        pickle.dump(model, open(f'models/lgb_oag_seed{seed}_fold{fold}.pkl', 'wb'))


"""
lgb model2
"""

feats = [f for f in df_train.columns if
         f not in
         ['label', 'name_id', 'article_id', 'authors_name', 'name', 'oof', 'venue_le_mode_count', 'st_emb_dis',
          'authors', 'author_dic', 'author_org', 'author_org_mode', 'fold',
          'preds'
          ]
         # +['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
         ]

print(df_train[feats].shape, df_test[feats].shape)
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
LABEL = 'label'
for seed in seeds:
    kf = GroupKFold(n_splits=fold_num)
    params_lgb = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 64,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,

        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 4,
    }
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL], groups=df_train['name_id'])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        model = lgb.train(params_lgb, train, valid_sets=[val], num_boost_round=1500,
                          callbacks=[  # lgb.early_stopping(100),
                              lgb.log_evaluation(500)]
                          )

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        # pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain') / fold_num
        pickle.dump(model, open(f'models/lgb_oag_emb_seed{seed}_fold{fold}.pkl', 'wb'))


