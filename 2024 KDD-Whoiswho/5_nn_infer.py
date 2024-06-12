import os

import pandas as pd
import numpy as np
import torch

import time
import warnings
import random
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import FullFC

warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sent_name = 'abstract'
sent_name1 = 'authors_org'
sent_name2 = 'title'
EMB_FEATS = (
        ['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
        +[sent_name+'_w2v_emb_%d'%i for i in range(32)]+[sent_name+'_w2v_emb_%d_mean'%i for i in range(32)]
        +[sent_name1+'_tfidf_emb_%d'%i for i in range(32)]+[sent_name1+'_tfidf_emb_%d_mean'%i for i in range(32)]
        +[sent_name2+'_w2v_emb_%d'%i for i in range(32)]+[sent_name2+'_w2v_emb_%d_mean'%i for i in range(32)]
)

OTHER_FEATS = ['year', 'title_len', 'authors_len', 'abstract_len', 'keywords_len', 'venue_le', 'article_le',
               'year_mean', 'title_len_mean', 'authors_len_mean', 'abstract_len_mean', 'keywords_len_mean', 'year_ratio_mean',
               'year_ratio_ptp', 'title_len_ratio_mean', 'title_len_ratio_ptp', 'authors_len_ratio_mean', 'authors_len_ratio_ptp',
               'abstract_len_ratio_mean', 'abstract_len_ratio_ptp', 'keywords_len_ratio_mean', 'keywords_len_ratio_ptp', 'venue_le_count_ratio',
               'abstract_w2v_emb_dis', 'authors_org_tfidf_emb_dis', 'title_w2v_emb_dis', 'oag_emb_dis']


def load_model(weight_path):
    # print(weight_path)
    model = FullFC(emb_dim=len(EMB_FEATS), other_f_dim=len(OTHER_FEATS))

    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(test, model_list):
    ret = 0
    for i, model in enumerate(model_list):
        # print('----models_base ', i)
        inputs = torch.from_numpy(test).to(device)
        outputs = model({'input': inputs}, mode='test')
        ret += outputs.cpu().numpy()
    ret = ret / len(model_list)

    return ret

class myDataset(Dataset):
    def __init__(self, df):
        self.data_emb = df[EMB_FEATS].values.astype('float32')
        self.data_other = df[OTHER_FEATS].values.astype('float32')
        # self.name_id = df['name_id']
        # self.label = df['label'].astype('float32')
        # self.label = torch.FloatTensor(df[LABEL + '_log1p'].values)

    def __getitem__(self, index):
        ret_dic = {
            'emb': self.data_emb[index],
            'other': self.data_other[index],
            # 'name_id': self.name_id[index],
            # 'label': self.label[index]
        }
        return ret_dic

    def __len__(self):
        return len(self.data_emb)


if __name__ == '__main__':
    df = pd.read_parquet('usr_data/df.parquet')

    df[OTHER_FEATS] = (df[OTHER_FEATS] - df[OTHER_FEATS].mean()) / df[OTHER_FEATS].std()
    df[OTHER_FEATS] = df[OTHER_FEATS].astype('float32').fillna(0)

    df_train = df[df['label'].notna()].reset_index(drop=True)
    df_test = df[df['label'].isna()].reset_index(drop=True)

    test_dataset = myDataset(df_test)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=4096,
                                 pin_memory=True,
                                 shuffle=False,
                                 num_workers=6)
    model_list = [load_model('models/fold_%d.pth' % i) for i in range(1, 6)]

    for i in range(1, 6):
        pred_y = 0
        with torch.no_grad():
            model = FullFC(emb_dim=len(EMB_FEATS), other_f_dim=len(OTHER_FEATS))
            model.load_state_dict(torch.load(f'models/fold_{i}.pth'))
            model.to(device)
            model.eval()

            preds = []
            for data in tqdm(test_dataloader):
                for name in data.keys():
                    data[name] = data[name].to(device)
                pred = model(data, mode='test')
                preds.extend(pred.detach().cpu().numpy().flatten())
        pred_y += np.array(preds) / 5

    # pred_y = predict(np.array(df_test, dtype='float32'), model_list)
    # this_time = time.strftime('%Y%m%d-%H%M%S')
    df_test['label'] = pred_y
    df_test[['label']].to_csv('usr_data/nn_test.csv', index=False)
