import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import plot, show
from sklearn.metrics import roc_auc_score
import json
from gensim.models.word2vec import Word2Vec

df_train = pd.read_csv('data/train_dataset.csv', sep='\t')

df_test = pd.read_csv('data/test_dataset.csv', sep='\t')
sub = pd.read_csv('data/submit_example.csv')
df_test['id'] = sub['id']
df = pd.concat([df_train, df_test])

df['location_first_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
df['location_sec_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
df['location_third_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

feats = ['user_name', 'action', 'auth_type', 'ip_location_type_keyword', 'ip_risk_level', 'ip', 'location',
         'device_model', 'os_type', 'os_version', 'browser_type', 'browser_version', 'bus_system_code', 'op_target',
         'location_first_lvl', 'location_sec_lvl', 'location_third_lvl',
         ]

cat = []

LABEL = 'risk_label'

df['sec'] = df['session_id'].apply(lambda x: int(x[-7:-5]))
df['sec_sin'] = np.sin(df['sec']/60*2*np.pi)
df['sec_cos'] = np.cos(df['sec']/60*2*np.pi)
df['op_date'] = pd.to_datetime(df['op_date'])
df['hour'] = df['op_date'].dt.hour
df['weekday'] = df['op_date'].dt.weekday
df['year'] = df['op_date'].dt.year
df['month'] = df['op_date'].dt.month
df['day'] = df['op_date'].dt.day
df['op_ts'] = df["op_date"].values.astype(np.int64) // 10 ** 9
df = df.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
df['last_ts'] = df.groupby(['user_name'])['op_ts'].shift(1)
df['last_ts2'] = df.groupby(['user_name'])['op_ts'].shift(2)
df['ts_diff'] = df['op_ts'] - df['last_ts']
df['ts_diff2'] = df['op_ts'] - df['last_ts2']
feats += ['sec',
          'sec_sin', 'sec_cos',
          'op_ts', 'last_ts', 'ts_diff',
          # 'last_ts2',
          'ts_diff2',
          ]

for name in ['auth_type']:
    df[name+'_fillna'] = df[name].astype('str')
    sent = df.groupby(['user_name', 'year', 'month', 'day'])[name+'_fillna'].agg(list).values

    vec_size = 6
    w2v_model = Word2Vec(sentences=sent, vector_size=vec_size, window=12, min_count=1, workers=1)
    tmp = df[name+'_fillna'].map(lambda x: w2v_model.wv[x])
    tmp = pd.DataFrame(list(tmp))
    tmp.columns = ['_'.join([name, 'emb', str(i)]) for i in range(vec_size)]
    df = pd.concat([df, tmp], axis=1)
    feats += list(tmp.columns)

# for name in df['auth_type']:
for w in w2v_model.wv.key_to_index:
    # print(w)
    print(w, w2v_model.wv[w])

for name in ['mean', 'std', 'max', 'min', 'median', 'skew']:
    for name1 in ['user_name', 'bus_system_code', 'auth_type', 'action',
                  ]:  # 'op_target'

        df[name1+'_ts_diff_'+name] = df.groupby([name1])['ts_diff'].transform(name)
        feats.append(name1+'_ts_diff_'+name)


df['if_out'] = (df['location'] == '{"first_lvl":"成都分公司","sec_lvl":"9楼","third_lvl":"销售部"}')
feats.append('if_out')

for name in ['user_name', 'action', 'auth_type', 'ip', 'ip_location_type_keyword', 'ip_risk_level', 'location',
             'device_model', 'os_type', 'os_version', 'browser_type', 'browser_version', 'bus_system_code', 'op_target',
             'location_first_lvl', 'location_sec_lvl', 'location_third_lvl',
             ]+cat:
    le = LabelEncoder()
    df[name] = le.fit_transform(df[name])


df_train = df[~df[LABEL].isna()].reset_index(drop=True)
df_test = df[df[LABEL].isna()].reset_index(drop=True)

params = {
    'learning_rate': 0.08,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'seed': 2222,
    'n_jobs': -1,
}

print(feats)
print(df_train[feats].shape, df_test[feats].shape)

seeds = [2021]
oof = np.zeros(len(df_train))
importance = 0
fold_num = 10
pred_y = pd.DataFrame()
for seed in seeds:
    print('############################', seed)
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        model = lgb.train(params, train, valid_sets=val, num_boost_round=10000,
                          early_stopping_rounds=100, verbose_eval=100)

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        pred_y['fold_%d_seed_%d'%(fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain')/fold_num


df_train['oof'] = oof
score = roc_auc_score(df_train[LABEL], df_train['oof'])
print(score)

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:10])

sub = pd.read_csv('data/submit_example.csv')

pred_y = pred_y.mean(axis=1)
sub['ret'] = pred_y

# ans = pd.read_csv('ans/ans_lgb_202111192052_0.526827.csv')
# print(np.corrcoef(ans['ret'].rank().values.astype('float'), sub['ret'].rank().values.astype('float'))[0, 1])

# laofei = pd.read_csv('ans/sub_5225.csv')
# print(np.corrcoef(laofei['ret'].rank().values.astype('float'), sub['ret'].rank().values.astype('float'))[0, 1])

# plot(ans['ret'].rank(), '-x')
# plot(sub['ret'].rank(), '-x')
# show()


sub[['id', 'ret']].to_csv('ans/lgb_5285.csv', index=False)
# df_train.to_csv('train.csv')