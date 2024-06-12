import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics.pairwise import paired_cosine_distances

article_info = pd.read_pickle('usr_data/article_info.pkl')

article_info['year'] = article_info['year'].replace({'': np.nan})

article_info['title_len'] = article_info['title'].map(lambda x: len(x.split(' ')))
article_info['authors_len'] = article_info['authors'].map(len)
article_info['abstract_len'] = article_info['abstract'].map(lambda x: len(x.split(' ')))
article_info['keywords_len'] = article_info['keywords'].map(len)

article_info['venue_le'] = article_info['venue'].factorize()[0]

article_info['article_le'] = article_info['article_id'].factorize()[0]

article_info['n_author_org'] = article_info['authors'].map(lambda x: len(np.unique([xx['org'] for xx in x])))
article_info['authors_name'] = article_info['authors'].map(lambda x: np.unique([xx['name'].lower().replace("-", "") for xx in x]))
article_info['authors_org'] = article_info['authors'].map(lambda x: np.unique([xx['org'].lower() for xx in x]))

sent_name = 'abstract'
sent_name1 = 'authors_org'
sent_name2 = 'title'
vec_size = 32
tfidf_size = 32

article_info = pd.concat([article_info,
                          pd.read_pickle('usr_data/article_info_abstract_w2v_32_re0522.pkl'),
                          pd.read_pickle('usr_data/article_info_authors_org_tfidf_32.pkl'),
                          pd.read_pickle('usr_data/article_info_title_w2v_32.pkl')#.astype('float32')
                         ], axis=1)

article_info[['oag_emb_%d'%i for i in range(768)]] = np.load('usr_data/oag_emb.npy')

article_feats = (['article_id', 'year', 'title_len', 'authors_len',
                 'abstract_len', 'keywords_len', #'authors_org_len',
                 'venue_le', 'article_le',
                  ]
                 +[sent_name+'_w2v_emb_%d'%i for i in range(vec_size)]
                 +[sent_name1+'_tfidf_emb_%d'%i for i in range(tfidf_size)]
                 +[sent_name2+'_w2v_emb_%d'%i for i in range(vec_size)]
                 +['oag_emb_%d'%i for i in range(768)]
                )

path = 'data/'

_df_train = pd.read_json('data/train_author.json').T.reset_index()
_df_train.columns = ['name_id', 'name', 'normal_data', 'outliers']

_df_test = pd.read_json('data/ind_test_author_filter_public.json').T.reset_index()
_df_test.columns = ['name_id', 'name', 'papers']

normal_data_l = []
outliers_l = []
for r in _df_train.iterrows():
    normal_data_l.append(len(r[1]['normal_data']))
    outliers_l.append(len(r[1]['outliers']))
print(np.sum(outliers_l), np.sum(normal_data_l))

df_train = []
for r in tqdm(_df_train.iterrows()):
    v = r[1]
    _tmp = pd.concat([
        pd.DataFrame({'name_id': v['name_id'], 'name': v['name'], 'article_id': v['normal_data'], 'label': 1}),
        pd.DataFrame({'name_id': v['name_id'], 'name': v['name'], 'article_id': v['outliers'], 'label': 0}),
    ])
    df_train.append(_tmp)
df_train = pd.concat(df_train).reset_index(drop=True)

df_test = []
for r in tqdm(_df_test.iterrows()):
    v = r[1]
    _tmp = pd.DataFrame({'name_id': v['name_id'], 'name': v['name'], 'article_id': v['papers']})
    df_test.append(_tmp)
df_test = pd.concat(df_test).reset_index(drop=True)

df = pd.concat([df_train, df_test])
df = df.merge(article_info[article_feats], how='left', on='article_id')

for f in (['year', 'title_len', 'authors_len', 'abstract_len', 'keywords_len']
          + [sent_name + '_w2v_emb_%d' % i for i in range(vec_size)]
          + [sent_name1 + '_tfidf_emb_%d' % i for i in range(tfidf_size)]
          + [sent_name2 + '_w2v_emb_%d' % i for i in range(vec_size)]
          + ['oag_emb_%d' % i for i in range(768)]
):
    df[f + '_mean'] = df.groupby('name_id')[f].transform('mean')

for f in (['year', 'title_len', 'authors_len', 'abstract_len', 'keywords_len',  # 'n_author_org'
           #          'title_char_l', 'abstract_char_l'
           ]):
    df[f + '_ratio_mean'] = df[f] / df.groupby('name_id')[f].transform('mean')
    df[f + '_ratio_ptp'] = df[f] / (df.groupby('name_id')[f].transform('max') - df.groupby('name_id')[f].transform(
        'min') + 1)  # (df[f+'_ptp']+1)

for f in (['venue_le',  ]):
    df[f + '_count_ratio'] = df.groupby(['name_id', f])['name_id'].transform('count') / df.groupby('name_id')[
        'name_id'].transform('count')

df[sent_name + '_w2v_emb_dis'] = paired_cosine_distances(
    df[[sent_name + '_w2v_emb_%d' % i for i in range(vec_size)]].values,
    df[[sent_name + '_w2v_emb_%d_mean' % i for i in range(vec_size)]].values,
)
df[sent_name1 + '_tfidf_emb_dis'] = paired_cosine_distances(
    df[[sent_name1 + '_tfidf_emb_%d' % i for i in range(tfidf_size)]].values,
    df[[sent_name1 + '_tfidf_emb_%d_mean' % i for i in range(tfidf_size)]].values,
)
df[sent_name2 + '_w2v_emb_dis'] = paired_cosine_distances(
    df[[sent_name2 + '_w2v_emb_%d' % i for i in range(vec_size)]].values,
    df[[sent_name2 + '_w2v_emb_%d_mean' % i for i in range(vec_size)]].values,
)
df['oag_emb_dis'] = paired_cosine_distances(
    df[['oag_emb_%d' % i for i in range(768)]].values,
    df[['oag_emb_%d_mean' % i for i in range(768)]].values,
)


feats = [f for f in df.columns if
         f not in ['label', 'name_id', 'article_id', 'authors_name', 'name', 'oof', 'venue_le_mode_count',
                   'st_emb_dis',
                   'authors', 'author_dic', 'author_org', 'author_org_mode', 'fold',
                   ] #+ ['oag_emb_%d' % i for i in range(768)] + ['oag_emb_%d_mean' % i for i in range(768)]

         ]
df[['name_id', 'article_id', 'label']+feats].to_parquet('usr_data/df.parquet')
