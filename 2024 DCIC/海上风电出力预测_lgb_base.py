import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import StratifiedKFold, KFold
from matplotlib.pyplot import plot, show, title

df_train = pd.read_csv('data/A榜-训练集_海上风电预测_气象变量及实际功率数据.csv', encoding='gbk')
df_test = pd.read_csv('data/A榜-测试集_海上风电预测_气象变量数据.csv', encoding='gbk')

add_df = pd.read_csv('data/A榜-训练集_海上风电预测_基本信息.csv', encoding='gbk')
# add_df = pd.read_csv('data/A榜-训练集_海上风电预测_基本信息.csv')
print(df_test.columns)
df = pd.concat([df_train, df_test])
df = df.merge(add_df[['站点编号', '装机容量(MW)']], on='站点编号', how='left')
df['站点编号_le'] = df['站点编号'].map(lambda x: int(x[1]))

df['time'] = pd.to_datetime(df['时间'])
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['min'] = df['time'].dt.minute

LABEL = '出力(MW)'

df_train = df[df[LABEL].notna()]
df_test = df[df[LABEL].isna()].reset_index(drop=True)

df_train = df_train[df_train[LABEL]!='<NULL>'].reset_index(drop=True)
df_train[LABEL] = df_train[LABEL].astype('float32')

params_lgb = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'mse',
    'num_leaves': 64,
    'verbose': -1,
    'seed': 2,
    'n_jobs': -1,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4,
}

importance = 0
MODEL_TYPE = 'lgb'

sub_train_df = df_train[df_train['time'] < '2023-02-01 0:0:0']
sub_val_df = df_train[df_train['time'] >= '2023-02-01 0:0:0']

feats = [f for f in sub_train_df.columns if f not in [LABEL, '时间', 'time',  '站点编号', 'min'
                                                      ]]

train = lgb.Dataset(sub_train_df[feats],
                    sub_train_df[LABEL])
val = lgb.Dataset(sub_val_df[feats],
                  sub_val_df[LABEL])

model = lgb.train(params_lgb, train, valid_sets=[train, val], num_boost_round=5000,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

val_pred = model.predict(sub_val_df[feats])

s_mse = mse(sub_val_df[LABEL],
            val_pred,
            squared=False)
score = 1/(1+s_mse)
print('score... %.5f'%score, 'rmse...%.5f'%s_mse)

importance += model.feature_importance(importance_type='gain')
feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

plot(sub_val_df[LABEL].values)
plot(val_pred)
show()


model = lgb.train(params_lgb, lgb.Dataset(df_train[feats], df_train[LABEL]),
                  num_boost_round=model.best_iteration)

pred_y = model.predict(df_test[feats])

df_test[LABEL] = pred_y

df_test[['站点编号','时间','出力(MW)']].to_csv('ans/lgb_base_%.5f.csv'%score, index=False)
