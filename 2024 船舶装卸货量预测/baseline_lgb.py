import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

LABEL = '装载量'

df_train = pd.read_csv('data/船舶装卸货量预测-训练集-20240611.csv', encoding='gbk')
df_test = pd.read_csv('data/船舶装卸货量预测-测试集X-20240611.csv', encoding='gbk')

df = pd.concat([df_train, df_test])
df['离泊时间'] = df['离泊时间'].replace({' None': np.nan}).astype(float)
df['time_diff'] = df['离泊时间'] - df['进泊时间']

df['进泊时间'] = pd.to_datetime(df['进泊时间'], unit='s')
df['离泊时间'] = pd.to_datetime(df['离泊时间'], unit='s')

for f in ['进泊时间', '离泊时间']:
    df[f+'_month'] = df[f].dt.month
    df[f+'_hour'] = df[f].dt.hour
    df[f+'_dayofweek'] = df[f].dt.dayofweek

df['A_le'] = df['船舶类型代码A'].factorize()[0]
df['B_le'] = df['船舶类型代码B'].factorize()[0]

df['AB_le'] = (df['船舶类型代码A'] + '_' + df['船舶类型代码B']).factorize()[0]
df['面积'] = df['船长'] * df['船宽']
for num_f in ['载重吨', 'time_diff',]:
    for cat_f in ['船舶类型代码B', '船舶类型代码A']:
        df[cat_f + '_' + num_f + '_mean'] = df.groupby(cat_f)[num_f].transform('mean')
        df[cat_f + '_' + num_f + '_std'] = df.groupby(cat_f)[num_f].transform('std')
        df[cat_f + '_' + num_f + '_max'] = df.groupby(cat_f)[num_f].transform('max')
        df[cat_f + '_' + num_f + '_min'] = df.groupby(cat_f)[num_f].transform('min')

df['loc0'] = df['泊位位置'].map(lambda x: float(x.split(' ')[0]))
df['loc1'] = df['泊位位置'].map(lambda x: float(x.split(' ')[1]))

df_train = df[df[LABEL].notna()]
df_test = df[df[LABEL].isna()]

feats = [f for f in df_test if f not in [LABEL, '航次ID', '船舶类型代码A', '船舶类型代码B', '泊位位置',
                                         '进泊时间', '离泊时间'
                                         ]]
print(df_train[feats].shape, df_test[feats].shape)
params_lgb = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'mse',
    'num_leaves': 32,
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
    # kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            (df_train.loc[train_idx, LABEL]))
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          (df_train.loc[val_idx, LABEL]))
        model = lgb.train(params_lgb, train, valid_sets=[val], num_boost_round=20000,
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)])

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain') / fold_num

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

df_train['oof'] = oof
score = mse(df_train[LABEL], df_train['oof'], squared=False)
print(score)
df_test[LABEL] = pred_y.mean(axis=1).values
df_test[LABEL].to_csv('ans/lgb_'+time.strftime('%Y%m%d-%H%M%S')+'_%d.txt'%score, index=False, header=None)

