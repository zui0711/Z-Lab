# 公榜分数 0.90967858758

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

# 读取文件并拼接
df_train = pd.read_csv('data/dataTrain.csv')
df_test = pd.read_csv('data/dataA.csv')
df = pd.concat([df_train, df_test])


LABEL = 'label'

# 对非数值类别特征进行编码
for f in ['f3']:
    le = LabelEncoder()
    df[f] = le.fit_transform(df[f])

df_train = df[df[LABEL].notna()]
df_test = df[df[LABEL].isna()]


feats = [f for f in df_test if f not in [LABEL, 'id']]
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

# 5折交叉验证
fold_num = 5
seeds = [2222]
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
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

# 输出特征重要度
feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

df_train['oof'] = oof
score = auc(df_train[LABEL], df_train['oof'])
print(score)

# 保存提交文件
sub = pd.read_csv('data/submit_example_A.csv')
sub[LABEL] = pred_y.mean(axis=1).values
sub.to_csv('ans/base2.csv', index=False)
