import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import StratifiedKFold, KFold
import time
from matplotlib.pyplot import plot, show
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec

LABEL = 'paper_tension_vertically_average'

df_train = pd.read_csv('data/train.csv')
# plot(df_train[LABEL])
# show()
df_test = pd.read_csv('data/test_A.csv')
df_add = pd.read_csv('data/paper_machine_data.csv', index_col=0)
df_add['end_time'] = pd.to_datetime(df_add['time'].map(lambda x: x[:-9]))
df_add = df_add.drop_duplicates(subset=['end_time'], keep='last')

# 尝试用一分钟内均值代替，无效
# f = list(df_add.columns)[1:-1]
# df_add = df_add.groupby('end_time')[f].agg('mean').reset_index()

df = pd.concat([df_train, df_test])
df['end_time'] = pd.to_datetime(df['end_time'])


# tfidf+w2v 即插即用，无效
# sent_name = 'formula'
# print('JIEBA...')
# df[sent_name+'_1'] = df[sent_name].map(lambda x: [xx for xx in jieba.lcut(x) if xx not in '：（）+=()件'])
# df[sent_name+'_2'] = df[sent_name+'_1'].map(lambda x: ' '.join(x))
#
# print('W2V...')
# vec_size = 16
# emb_model_w2v = Word2Vec(df[sent_name+'_1'], vector_size=vec_size, window=6, min_count=1, seed=2222, workers=1, epochs=20)
# emb_data = df[sent_name+'_1'].map(lambda x: np.mean([emb_model_w2v.wv[xx] for xx in x], axis=0))
# df[[sent_name+'_w2v_emb_%d'%i for i in range(vec_size)]] = list(np.array(emb_data))
# emb_model_w2v.save('tmp/emb_model_w2v')

# print('TFIDF...')
# tfidf_size = 4
# tdidf_model = TfidfVectorizer(ngram_range=(1, 5)).fit(df[sent_name+'_2'])
# svd = TruncatedSVD(n_components=tfidf_size, n_iter=50, random_state=2222)
# svd_tfidf_data = svd.fit_transform(tdidf_model.transform(df[sent_name+'_2']))
# df[[sent_name+'_tfidf_emb_%d'%i for i in range(tfidf_size)]] = svd_tfidf_data
# f = [sent_name+'_w2v_emb_%d'%i for i in range(vec_size)]+[sent_name+'_tfidf_emb_%d'%i for i in range(tfidf_size)]
# df[f] = df[f].astype('float32')
# df.to_pickle('tmp/df_emb.pkl')

df = df.merge(df_add, on='end_time', how='left')
le = LabelEncoder()
df['formula'] = le.fit_transform(df['formula'])

# 尝试交叉，过拟合严重
# df['sp'] = df['flow'] * df['concentration']
# df['sp'] = df['check_weight'] / df['paper_thickness']
# df['sp1'] = df['paper_thickness'] / df['check_weight']

df_train = df[df[LABEL].notna()].reset_index(drop=True)
df_test = df[df[LABEL].isna()].reset_index(drop=True)

feats=[i for i in df_train.columns if i not in ['id', 'end_time', LABEL, 'time', 'formula_1', 'formula_2',
                                                ]]

# 尝试了两种label变换，无效
# df_train['new_label'] = df_train[LABEL] / df_train['check_weight']
# df_train['new_label'] = np.log1p(df_train[LABEL])

# df_train[feats+[LABEL]].corr().to_csv('corr.csv')

# def get_adv_feats(df_train, df_test, feats):
#     df_train['adv'] = 1
#     df_test['adv'] = 0
#     df = pd.concat([df_train, df_test]).reset_index(drop=True)
#     params = {
#         'learning_rate': 0.1,
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'metric': 'auc',
#         'seed': 2222,
#         'n_jobs': 4,
#         'verbose': -1,
#     }
#
#     fold_num = 5
#     seeds = [2222]
#     new_feats = []
#     for f in feats:
#         # print('*****************', f)
#         oof = np.zeros(len(df))
#         for seed in seeds:
#             kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
#             for fold, (train_idx, val_idx) in enumerate(kf.split(df[[f]], df['adv'])):
#                 train = lgb.Dataset(df.loc[train_idx, [f]],
#                                     df.loc[train_idx, 'adv'])
#                 val = lgb.Dataset(df.loc[val_idx, [f]],
#                                   df.loc[val_idx, 'adv'])
#                 model = lgb.train(params, train, valid_sets=[val], num_boost_round=10000,  # feval=recall_score,
#                                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
#                 oof[val_idx] += model.predict(df.loc[val_idx, [f]]) / len(seeds)
#                 score = auc(df.loc[val_idx, 'adv'], oof[val_idx])
#                 if score > 0.95:
#                     print('--------------------------------------', f, score)
#                 else:
#                     new_feats.append(f)
#                 break
#     return new_feats


# 暴力对抗验证，掉分严重，可能是删了重要特征
# feats = get_adv_feats(df_train, df_test, feats)

params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'mape',
    'metric': 'mape',
    'num_leaves': 16,
    'verbose': -1,
    'seed': 2222,
    'n_jobs': -1,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4,
    'min_child_weight': 10,
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
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
        model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

        oof[val_idx] += (model.predict(df_train.loc[val_idx, feats])) / len(seeds)
        pred_y['fold_%d_seed_%d' % (fold, seed)] = (model.predict(df_test[feats]))
        importance += model.feature_importance(importance_type='gain') / fold_num
        # score.append(auc(df_train.loc[val_idx, LABEL], model.predict(df_train.loc[val_idx, feats])))

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

df_train['oof'] = oof # * df_train['check_weight']
# print(np.mean(score), np.std(score))
m = mape(df_train[LABEL], df_train['oof'])
score = 1 / (1 + m)
print(score)
plot(df_train[LABEL], '-x')
plot(df_train['oof'], '-x')
show()
pred_y = pred_y.mean(axis=1)

df_test['value'] = pred_y
# 从训练集观察到的较大数据预测有问题，尝试粗暴后处理，无效
# df_test['value'] = df_test['value'].map(lambda x: 1.5*x if x > 1500 else x)
plot(df_test['value'])
show()
# df_test[['id', 'value']].to_csv('ans/base.csv', index=False)
df_test[['id', 'value']].to_csv(time.strftime('ans/lgb_%Y%m%d%H%M_')+'%.5f.csv'%score, index=False)
