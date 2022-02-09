import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, log_loss
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train_user = pd.read_csv('data/training_dataset/训练集_用户基本信息表.txt')
train_day = pd.read_csv('data/training_dataset/训练集_用户日电量明细表.txt')
train_month = pd.read_csv('data/training_dataset/训练集_行业户均月电量.txt')

test_user = pd.read_csv('data/test_dataset/测试集_用户基本信息表.txt')
test_day = pd.read_csv('data/test_dataset/测试集_用户日电量明细表.txt')
test_month = pd.read_csv('data/test_dataset/测试集_行业户均月电量.txt')

df_user = pd.concat([train_user, test_user])
df_day = pd.concat([train_day, test_day])
df_month = pd.concat([train_month, test_month])

feats = ['contract_cap', 'run_cap']

tmp = df_month.groupby(['trade_code', 'trade_name', 'county_code'])['avg_settle_pq'].agg(
    ['mean', 'median', 'skew', 'sum']).reset_index()
tmp.columns = list(tmp.columns[:3]) + ['avg_settle_pq_mean', 'avg_settle_pq_median', 'avg_settle_pq_skew', 'avg_settle_pq_sum']
df_user = pd.merge(df_user, tmp, on=['trade_code', 'trade_name', 'county_code'], how='left')
feats += ['avg_settle_pq_mean', 'avg_settle_pq_median', 'avg_settle_pq_skew']

cat_feats = ['county_code', 'volt_name', 'elec_type_name', 'status_name', 'trade_name']
for name in cat_feats:
    le = LabelEncoder()
    df_user[name] = le.fit_transform(df_user[name])
feats += cat_feats

df_train = df_user[~df_user['flag'].isna()].reset_index()
df_test = df_user[df_user['flag'].isna()].reset_index()

print(feats)
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'verbose': -1,
    'seed': 2222,
    'n_jobs': -1,
}

fold_num = 5
seed = 2222
kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)

oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
LABEL = 'flag'
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
    print('-----------', fold)
    train = lgb.Dataset(df_train.loc[train_idx, feats],
                        df_train.loc[train_idx, LABEL])
    val = lgb.Dataset(df_train.loc[val_idx, feats],
                      df_train.loc[val_idx, LABEL])
    model = lgb.train(params, train, valid_sets=val, num_boost_round=10000,
                      early_stopping_rounds=100, verbose_eval=200)
    oof[val_idx] += model.predict(df_train.loc[val_idx, feats])
    pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
    importance += model.feature_importance(importance_type='gain') / fold_num

thre = 0.1
score = f1_score(df_train[LABEL],
                 list(map(lambda x: 1 if x > thre else 0, oof)), average='macro')
print('\nF1... ', score)
print('AUC  %0.5f, LOGLOSS  %0.5f' % (
    roc_auc_score(df_train['flag'], oof),
    log_loss(df_train['flag'], oof)))

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:10])

pred_y = pred_y.mean(axis=1).map(lambda x: 1 if x > thre else 0)
print(pred_y.sum())
df_test['flag'] = pred_y
df_test[['user_id', 'flag']].to_csv('ans/baseline.csv', index=False, header=['id', 'flag'])
