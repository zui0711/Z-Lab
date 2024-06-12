import pandas as pd
import numpy as np
from sklearn import metrics
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold
import torch
from torch.utils.data import Dataset, DataLoader

import time
import warnings
import random

from model import FullFC

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 2222
# 设置随机数种子
setup_seed(SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_save_dir = 'models'
# if ~os.path.exists(model_save_dir):
# os.makedirs(model_save_dir)

MAX_EPOCH = 1
OPT_TYPE = 'MAX'
METRIC_TYPE = 'CLASSIFICATION'
# METRIC_TYPE = 'REGRESSION'
BATCH_SIZE = 128
USE_EMA = False

kf = GroupKFold(n_splits=5)
# kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2222)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print_interval = -1

kfold_best_metric = []
# print(len(df))
sent_name = 'abstract'
sent_name1 = 'authors_org'
sent_name2 = 'title'

EMB_FEATS = (
        ['oag_emb_%d'%i for i in range(768)]+['oag_emb_%d_mean'%i for i in range(768)]
        +
        [sent_name+'_w2v_emb_%d'%i for i in range(32)]+[sent_name+'_w2v_emb_%d_mean'%i for i in range(32)]
        +[sent_name1+'_tfidf_emb_%d'%i for i in range(32)]+[sent_name1+'_tfidf_emb_%d_mean'%i for i in range(32)]
        +[sent_name2+'_w2v_emb_%d'%i for i in range(32)]+[sent_name2+'_w2v_emb_%d_mean'%i for i in range(32)]
)

OTHER_FEATS = ['year', 'title_len', 'authors_len', 'abstract_len', 'keywords_len', 'venue_le', 'article_le',
               'year_mean', 'title_len_mean', 'authors_len_mean', 'abstract_len_mean', 'keywords_len_mean', 'year_ratio_mean',
               'year_ratio_ptp', 'title_len_ratio_mean', 'title_len_ratio_ptp', 'authors_len_ratio_mean', 'authors_len_ratio_ptp',
               'abstract_len_ratio_mean', 'abstract_len_ratio_ptp', 'keywords_len_ratio_mean', 'keywords_len_ratio_ptp', 'venue_le_count_ratio',
               'abstract_w2v_emb_dis', 'authors_org_tfidf_emb_dis', 'title_w2v_emb_dis', 'oag_emb_dis']


class myDataset(Dataset):
    def __init__(self, df):
        self.data_emb = df[EMB_FEATS].values.astype('float32')
        self.data_other = df[OTHER_FEATS].values.astype('float32')
        self.name_id = df['name_id']
        self.label = df['label'].astype('float32')
        # self.label = torch.FloatTensor(df[LABEL + '_log1p'].values)

    def __getitem__(self, index):

        ret_dic = {
            'emb': self.data_emb[index],
            'other': self.data_other[index],
            'name_id': self.name_id[index],
            'label': self.label[index]
        }
        return ret_dic

    def __len__(self):
        return len(self.data_emb)



@torch.no_grad()
def val_model(model, val_dataloader):
    # dset_sizes=len(val_dataset)
    model.eval()
    preds = []
    labels = []
    losses = []

    ids = []
    for data in val_dataloader:
        for name in data.keys():
            if name != 'name_id':
                data[name] = data[name].to(device)

        loss, pred, label = model(data, mode='val')
        preds.extend(pred.cpu().numpy().flatten())
        labels.extend(label.cpu().numpy())
        losses.append(loss.cpu().numpy())
        ids.extend(data['name_id'])
    loss = sum(losses) / len(losses)
    dfs = pd.DataFrame({'name_id': ids, 'label': labels, 'pred': preds})
    score = []
    for name_id in tqdm(dfs['name_id'].unique()):
        _tmp = dfs[dfs['name_id'] == name_id]
        score.append(
            metrics.roc_auc_score(_tmp['label'], _tmp['pred']) * (1 - _tmp['label']).sum() / (1-dfs['label']).sum()
        )
    score = np.sum(score)

    return loss, score, preds


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':
    df = pd.read_parquet('usr_data/df.parquet')

    df[OTHER_FEATS] = (df[OTHER_FEATS] - df[OTHER_FEATS].mean()) / df[OTHER_FEATS].std()
    df[OTHER_FEATS] = df[OTHER_FEATS].astype('float32').fillna(0)
    # df[EMB_FEATS] = df[EMB_FEATS].astype('float32')

    df_train = df[df['label'].notna()].reset_index(drop=True)
    df_test = df[df['label'].isna()].reset_index(drop=True)

    print(df_train.shape, df_test.shape)
    # print(df_test.shape, df_test_y.shape)

    # print(df)

    kfold_best_metric = []
    all_valid = []
    dataset = myDataset(df_train)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, df_train['label'], groups=df_train['name_id'])):
        dataset_train = myDataset(df_train.loc[train_idx].reset_index(drop=True))
        dataset_val = myDataset(df_train.loc[val_idx].reset_index(drop=True))

        train_dataloader = DataLoader(dataset_train,
                                      batch_size=BATCH_SIZE,
                                      # sampler=train_sampler,
                                      pin_memory=True,
                                      shuffle=True,
                                      num_workers=6)
        val_dataloader = DataLoader(dataset_val,
                                    batch_size=BATCH_SIZE*4,
                                    # sampler=val_sampler,
                                    pin_memory=True,
                                    shuffle=False,
                                    num_workers=6)
        model = FullFC(emb_dim=len(EMB_FEATS), other_f_dim=len(OTHER_FEATS))
        model.to(device)
        if USE_EMA:
            ema = EMA(model, 0.99)
            ema.register()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)
        total_iters = len(train_dataloader)
        print('--------------total_iters:{}'.format(total_iters))
        since = time.time()
        if OPT_TYPE == 'MIN':
            best_metric = 1e7
        else:
            best_metric = -1e7
        best_loss = 1e7
        best_epoch = 0

        iters = len(train_dataloader)
        for epoch in range(1, MAX_EPOCH + 1):
            model.train(True)
            begin_time = time.time()
            # print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
            print('Fold{} Epoch {}/{}'.format(fold + 1, epoch, MAX_EPOCH))

            count = 0
            train_loss = []
            for train_data in tqdm(train_dataloader):
                # print(inputs)
                count += 1
                for name in train_data.keys():
                    if name != 'name_id':
                        train_data[name] = train_data[name].to(device)
                loss, _, _ = model(train_data)
                # loss = model(train_data)

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                optimizer.step()
                if USE_EMA:
                    if epoch >= 0:
                        ema.update()
                scheduler.step()
                # 更新cosine学习率

                # if scheduler != None:
                #     scheduler.step(epoch + count / iters)
                if print_interval > 0 and (i % print_interval == 0 or loss.size()[0] < BATCH_SIZE):
                    spend_time = time.time() - begin_time
                    print(
                        ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                            fold + 1, epoch, count, total_iters,
                            loss.item(), optimizer.param_groups[-1]['lr'],
                            spend_time / count * total_iters // 60 - spend_time // 60))
                # #
                train_loss.append(loss.item())

            if USE_EMA:
                if epoch >= 0:
                    ema.apply_shadow()
            val_loss, val_metric, preds = val_model(model, val_dataloader)
            print('val loss: {:.4f}, val metric: {:.4f}  '.format(val_loss, val_metric))
            # best_model_out_path = model_save_dir + '/fold_%d_epoch_%d_%.5f.pth'%(fold + 1, epoch, val_metric)
            best_model_out_path = model_save_dir + '/fold_%d.pth'%(fold + 1)
            # torch.save(model.state_dict(), best_model_out_path)
            # save the best models_base
            # if val_loss < best_loss:
            # if (OPT_TYPE == 'MIN' and val_metric < best_metric) or (OPT_TYPE == 'MAX' and val_metric > best_metric):
            best_metric = val_metric
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_out_path)
                # print("save best epoch: {} best val_metric: {:.5f}".format(best_epoch, val_metric))

            if USE_EMA:
                if epoch >= 0:
                    ema.restore()
        print('Fold{} Best metric: {:.5f} Best epoch:{}'.format(fold + 1, best_metric, best_epoch))
        time_elapsed = time.time() - since

        kfold_best_metric.append(best_metric)

        valid = df_train.loc[val_idx, ['name_id', 'article_id']].reset_index(drop=True)
        valid['preds'] = preds
        all_valid.append(valid)

        # break
    print(kfold_best_metric)
    print(np.mean(kfold_best_metric))
    pd.concat(all_valid).to_csv('usr_data/nn_valid.csv', index=False)
