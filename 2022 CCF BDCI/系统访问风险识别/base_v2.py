# 公榜分数0.92860690761
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

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/evaluation_public.csv')
df = pd.concat([df_train, df_test])
df['op_datetime'] = pd.to_datetime(df['op_datetime'])
df['hour'] = df['op_datetime'].dt.hour
df['dayofweek'] = df['op_datetime'].dt.dayofweek

df = df.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)
df['ts'] = df['op_datetime'].values.astype(np.int64) // 10 ** 9
df['ts1'] = df.groupby('user_name')['ts'].shift(1)
df['ts2'] = df.groupby('user_name')['ts'].shift(2)
df['ts_diff1'] = df['ts1'] - df['ts']
df['ts_diff2'] = df['ts2'] - df['ts']

df['hour_sin'] = np.sin(df['hour']/24*2*np.pi)
df['hour_cos'] = np.cos(df['hour']/24*2*np.pi)

LABEL = 'is_risk'

cat_f = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
          'os_type', 'os_version', 'ip_type', 'op_city', 'log_system_transform', 'url',]

for f in cat_f:
    le = LabelEncoder()
    df[f] = le.fit_transform(df[f])
    df[f+'_ts_diff_mean'] = df.groupby([f])['ts_diff1'].transform('mean')
    df[f+'_ts_diff_std'] = df.groupby([f])['ts_diff1'].transform('std')

df_train = df[df[LABEL].notna()].reset_index(drop=True)
df_test = df[df[LABEL].isna()].reset_index(drop=True)


feats = [f for f in df_test if f not in [LABEL, 'id',
                                         'op_datetime', 'op_month', 'ts', 'ts1', 'ts2']]
print(feats)
print(df_train[feats].shape, df_test[feats].shape)
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
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
score = []
for seed in seeds:
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    # kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain') / fold_num
        score.append(auc(df_train.loc[val_idx, LABEL], model.predict(df_train.loc[val_idx, feats])))
feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

df_train['oof'] = oof
print(np.mean(score), np.std(score))

score = np.mean(score)
df_test[LABEL] = pred_y.mean(axis=1).values
df_test = df_test.sort_values('id').reset_index(drop=True)

sub = pd.read_csv('data/submit_sample.csv')
sub['ret'] = df_test[LABEL].values
sub.columns = ['id', LABEL]
sub.to_csv(time.strftime('ans/lgb_%Y%m%d%H%M_')+'%.5f.csv'%score, index=False)
