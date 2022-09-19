import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import StratifiedKFold, KFold
from matplotlib.pyplot import plot, show


df_train = pd.read_csv('data/train_dataset.csv')
df_test = pd.read_csv('data/evaluation_public.csv')

df = pd.concat([df_train, df_test]).reset_index(drop=True)

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['dayofweek'] = df['time'].dt.dayofweek
df['ts'] = df['hour']*60 + df['minute']

df_train = df[:len(df_train)].reset_index(drop=True)
df_test = df[len(df_train):].reset_index(drop=True)
feats = [f for f in df_test if f not in ['time', 'Label1', 'Label2']]

df_train['Label1_log'] = np.log1p(df_train['Label1'])
df_train['Label2_log'] = np.log1p(df_train['Label2'])

df_train = df_train.dropna(subset=['Label1', 'Label2']).reset_index(drop=True)


def calc_score(df):
    loss = (mse(df['Label1'], df['Label1_oof'])**0.5 +
            mse(df['Label2'], df['Label2_oof'])**0.5) / 2
    score = (1 / (1 + loss)) * 1000
    return score


for LABEL in ['Label1', 'Label2']:
    print(LABEL)
    print(df_train[feats].shape, df_test[feats].shape)
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'metric': 'mse',
        'num_leaves': 64,
        'verbose': -1,
        'seed': 2222,
        'n_jobs': -1,

        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 4,
        # 'min_child_weight': 10,
    }

    fold_num = 5
    seeds = [2222]
    oof = np.zeros(len(df_train))
    importance = 0
    pred_y = pd.DataFrame()
    for seed in seeds:
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
            print('-----------', fold)
            train = lgb.Dataset(df_train.loc[train_idx, feats],
                                df_train.loc[train_idx, LABEL+'_log'])
            val = lgb.Dataset(df_train.loc[val_idx, feats],
                              df_train.loc[val_idx, LABEL+'_log'])
            model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

            oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
            pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
            importance += model.feature_importance(importance_type='gain') / fold_num
    feats_importance = pd.DataFrame()
    feats_importance['name'] = feats
    feats_importance['importance'] = importance
    print(feats_importance.sort_values('importance', ascending=False)[:30])

    df_train[LABEL+'_oof'] = np.expm1(oof)
    print(np.sqrt(mse(df_train[LABEL], df_train[LABEL+'_oof'])))

    df_test[LABEL] = np.expm1(pred_y.mean(axis=1).values)

score = calc_score(df_train)
print(score)

sub = pd.read_csv('data/sample_submission.csv')
sub['Label1'] = df_test['Label1']
sub['Label2'] = df_test['Label2']
# sub.to_csv(time.strftime('ans/lgb_%Y%m%d%H%M_') + '%.5f.csv' % score, index=False)
sub.to_csv(time.strftime('ans/base_') + '%.5f.csv' % score, index=False)
