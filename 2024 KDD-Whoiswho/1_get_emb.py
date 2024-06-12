import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import re

article_info = pd.read_json('data/pid_to_info_all.json').T.reset_index(drop=True)
article_info = article_info.rename({'id': 'article_id'}, axis=1)
article_info.to_pickle('usr_data/article_info.pkl')

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

"""abstract"""
remove_char = r'[,\.:\-\(\)1234567890/%\<\>]'

sent_name = 'abstract'
article_info[sent_name+'_1'] = article_info[sent_name].map(lambda x: re.split(r'\s+', re.sub(remove_char, ' ', x.lower())))
article_info[sent_name+'_2'] = article_info[sent_name+'_1'].map(lambda x: ' '.join(x))

print('W2V...')
vec_size = 32
emb_model_w2v = Word2Vec(article_info[sent_name+'_1'], vector_size=vec_size, window=10, min_count=1, seed=2222, workers=1, epochs=10)

emb_data = article_info[sent_name+'_1'].map(lambda x: np.mean([emb_model_w2v.wv[xx] for xx in x], axis=0))
article_info[[sent_name+'_w2v_emb_%d'%i for i in range(vec_size)]] = list(np.array(emb_data))

article_info[[sent_name+'_w2v_emb_%d'%i for i in range(vec_size)]].to_pickle('usr_data/article_info_abstract_w2v_32_re0522.pkl')


"""authors_org"""
sent_name1 = 'authors_org'
article_info[sent_name1+'_1'] = article_info[sent_name1].map(lambda x: ['_'.join(xx.split(' ')) for xx in x] if len(x) > 0 else ['NAN'])
article_info[sent_name1+'_2'] = article_info[sent_name1].map(lambda x: ' '.join(x))

print('TFIDF...')
tfidf_size = 32
tdidf_model = TfidfVectorizer(ngram_range=(1,2)).fit(article_info[sent_name1+'_2'])
svd = TruncatedSVD(n_components=tfidf_size, n_iter=20, random_state=2222)
svd_tfidf_data = svd.fit_transform(tdidf_model.transform(article_info[sent_name1+'_2']))
article_info[[sent_name1+'_tfidf_emb_%d'%i for i in range(tfidf_size)]] = svd_tfidf_data
article_info[[sent_name1+'_tfidf_emb_%d'%i for i in range(tfidf_size)]].to_pickle('usr_data/article_info_authors_org_tfidf_32.pkl')

"""title"""
sent_name2 = 'title'
article_info[sent_name2+'_1'] = article_info[sent_name2].map(lambda x: x.lower().replace('.', '').replace(',', '').split(' '))
article_info[sent_name2+'_2'] = article_info[sent_name2+'_1'].map(lambda x: ' '.join(x))

print('W2V...')
vec_size = 32
emb_model_w2v = Word2Vec(article_info[sent_name2+'_1'], vector_size=vec_size, window=10, min_count=1, seed=2222, workers=1, epochs=10)
emb_data = article_info[sent_name2+'_1'].map(lambda x: np.mean([emb_model_w2v.wv[xx] for xx in x], axis=0))
article_info[[sent_name2+'_w2v_emb_%d'%i for i in range(vec_size)]] = list(np.array(emb_data))
article_info[[sent_name2+'_w2v_emb_%d'%i for i in range(vec_size)]].to_pickle('usr_data/article_info_title_w2v_32.pkl')

