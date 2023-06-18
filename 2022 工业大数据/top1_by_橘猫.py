import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
import gc
import glob
import lightgbm as lgb

import warnings
warnings.simplefilter('ignore')


'''
读取训练集、决赛测试集，并且将各个csv文件整合合并在一起
'''
header = ['Elapsed', 'step_n', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9',
          'param10',
          'param11', 'param12', 'param13', 'param14', 'param15', 'param16', 'param17', 'param18', 'param19', 'param20',
          'param21',
          'param22', 'param23', 'param24', 'param25', 'param26', 'param27']
data_path = '../data/初赛训练数据_X输入值/初赛训练数据_X输入值'
train_files = glob.glob(data_path + '/*.csv')

data_path = '../data/决赛验证数据_X输入值/决赛验证数据_X输入值'
test_files = glob.glob(data_path + '/*.csv')

## 整合训练集
train = pd.DataFrame()
for i in tqdm(train_files):
    name_id = i[-18:-4]
    news_list = []
    for idx, line in enumerate(open(i, encoding='utf-8')):
        if idx == 0:
            cols = header
        else:
            line_list = ' '.join(line.split()).split(' ')
            news_list.append(line_list)
    train_temp = pd.DataFrame(news_list, columns=cols)
    train_temp = pd.concat([train_temp.iloc[:, :2], train_temp.iloc[:, 2:].astype(float)], axis=1)
    train_temp['glass_id'] = name_id
    train = pd.concat([train, train_temp]).reset_index(drop=True)

## 整合测试集
test = pd.DataFrame()
for i in tqdm(test_files):
    name_id = i[-18:-4]
    news_list = []
    for idx, line in enumerate(open(i, encoding='utf-8')):
        if idx == 0:
            cols = header
        else:
            line_list = ' '.join(line.split()).split(' ')
            news_list.append(line_list)
    train_temp = pd.DataFrame(news_list, columns=cols)
    train_temp = pd.concat([train_temp.iloc[:, :2], train_temp.iloc[:, 2:].astype(float)], axis=1)
    train_temp['glass_id'] = name_id
    test = pd.concat([test, train_temp]).reset_index(drop=True)
print(test)

## 整合标签
test_df = pd.DataFrame()
for i in test['glass_id'].unique():
    temp = pd.DataFrame([i for i in range(1, 29)], columns=['site'])
    temp['glass_id'] = i
    test_df = pd.concat([test_df, temp]).reset_index(drop=True)
train_df = pd.read_csv('../data/train_y.csv')

## 合并标签和x值
data = pd.concat([train, test]).reset_index(drop=True)
data_df = pd.concat([train_df, test_df]).reset_index(drop=True)







'''
数据增强
'''
data_df_group = data.groupby('glass_id')
group_data = []
df_odd = []
df_even = []
for idx, data_ in tqdm(data_df_group):
    group_data.append(data_)

df1 = []
df2 = []
df3 = []
for idx, m in tqdm(enumerate(group_data)):
    for i in range(3):
        df = m.iloc[i::3]
        name = np.array(df)
        if i==0:
            df1.append(df)
        elif i==1:
            df2.append(df)
        elif i==2:
            df3.append(df)
df1 = pd.concat(df1).reset_index(drop = True)
df2 = pd.concat(df2).reset_index(drop = True)
df3 = pd.concat(df3).reset_index(drop = True)




'''
特征工程、特征衍生

'''
feasts = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10',
          'param11',  'param13', 'param15', 'param17', 'param19', 'param20', 'param21',
          'param22', 'param23', 'param24', 'param25', 'param26', 'param27']
pw_feas = ['param3','param4','param6','param7','param8','param9','param13','param15',
                'param17','param19','param15']
## groupby glass_id
zong_data = pd.DataFrame()
for data in tqdm([df1, df2, df3]):

    get_feas = list(set(feasts) - set([]))
    data = data[['glass_id', 'step_n'] + get_feas]
    data_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    data_diff = data[['glass_id', 'step_n'] + get_feas].copy()

    ## 原始特征describe
    temp = data.groupby(['glass_id']).mean()
    temp.columns = [i + '_mean' for i in temp.columns]
    data_df = data_df.merge(temp.reset_index(), how='left', on='glass_id')

    temp = data.groupby(['glass_id']).std()
    temp.columns = [i + '_std' for i in temp.columns]
    data_df = data_df.merge(temp.reset_index(), how='left', on='glass_id')

    temp = data.groupby(['glass_id']).skew()
    temp.columns = [i + '_skew' for i in temp.columns]
    data_df = data_df.merge(temp.reset_index(), how='left', on='glass_id')


    ## 特征diff describe
    for dif in tqdm([1]):
        for i in get_feas:
            data_diff[i] = data_diff.groupby(['glass_id', ])[i].diff(dif).values

        temp = data_diff.groupby(['glass_id']).mean()
        temp.columns = [i + '_mean_diff' + str(dif) for i in temp.columns]
        data_df = data_df.merge(temp.reset_index(), how='left')

        temp = data_diff.groupby(['glass_id']).std()
        temp.columns = [i + '_std_diff' + str(dif) for i in temp.columns]
        data_df = data_df.merge(temp.reset_index(), how='left')

        temp = data_diff.groupby(['glass_id']).min()
        temp.columns = [i + '_min_diff' + str(dif) for i in temp.columns]
        data_df = data_df.merge(temp.reset_index(), how='left')

        temp = data_diff.groupby(['glass_id']).max()
        temp.columns = [i + '_max_diff' + str(dif) for i in temp.columns]
        data_df = data_df.merge(temp.reset_index(), how='left')

    zong_data = pd.concat([zong_data, data_df]).reset_index(drop=True)



'''
Target目标编码
'''
zong_data['glass_id_cate_number'] = zong_data['glass_id'].apply(lambda x:x[-8:-2]).astype('category')
zong_data['site_label_mean'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].mean())
zong_data['site_label_std'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].std())
zong_data['site_label_skew'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].skew())
zong_data['site_label_median'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].median())
zong_data['site_label_max'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].max())
zong_data['site_label_min'] = zong_data['site'].map(train_df.groupby(['site'])['param_value'].min())
zong_data['site_label_max-min'] = zong_data['site_label_max'] - zong_data['site_label_min']
zong_data['site_label_mean-median'] = zong_data['site_label_mean'] - zong_data['site_label_median']
zong_data['site_label_mean/std'] = zong_data['site_label_mean'] / zong_data['site_label_std']





'''
标签转换：最大最小归一化
'''
zong_data['param_value'] = (zong_data['param_value'] - zong_data['site_label_min'])/zong_data['site_label_max-min']




'''
开始训练，划分训练集、测试集，
通过对抗验证、特征重要性等删除一部分噪音特征
'''
data_df = zong_data.copy()
cate_list = ['glass_id_cate_number','site']
train_xl = data_df[data_df['param_value'].notnull()].reset_index(drop = True)
test_xl = data_df[~data_df['param_value'].notnull()].reset_index(drop = True)
feas = [i  for i in train_xl.columns.tolist() if i not in ['param_name', 'glass_id','param_value', 'Elapsed_min','step_n_min',
                                                           'Elapsed_max','step_n_max','step_n_max_diff1','step_n_min_diff1',
                                                           'glass_id_number',
                                                       ]+['param19_mean_diff1', 'param20_mean_diff1', 'param22_mean_diff1',
                                                           'param27_mean_diff1', 'param26_mean_diff1', 'param24_mean_diff1',
                                                           'param21_mean_diff1', 'param25_mean_diff1', 'param23_mean_diff1',
                                                           'param22_std_diff1', 'param27_std_diff1', 'param21_std_diff1',
                                                           'param3_min_diff1', 'param4_max_diff1','param6_max_diff1','param13_mean',
                                                           'param19_skew','param13_skew']]

x_train = train_xl[feas]
y_train = train_xl['param_value']
x_test = test_xl[feas]
print('训练集测试集的维度：',x_train.shape,x_test.shape)




'''
使用模型LightGBM，分层交叉验证
'''
def lgb_model(clf, train_x, train_y, test_x, is_label_bool=False, kfod=10):
    folds = kfod
    seed = 2020
    if is_label_bool:
        train_y = np.log1p(train_y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    test_pre = []
    Feass = pd.DataFrame()

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                     train_y[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        fea = pd.DataFrame()

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'min_child_weight': 6,
            'num_leaves': 2 ** 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.5,
            'lambda_l2': 8,
            'bagging_freq': 4,
            'learning_rate': 0.03,
            'njobs': 8,
            'seed': seed,
            'silent': True,
            'verbose': -1,
        }

        model = clf.train(params, train_matrix, 
                          num_boost_round=20000, 
                          valid_sets=[train_matrix, valid_matrix],
                          categorical_feature=cate_list, 
                          verbose_eval=500, 
                          early_stopping_rounds=150)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        test_pre.append(test_pred)
        fea['feas'] = train_x.columns.tolist()
        fea['sorce'] = model.feature_importance()
        Feass = pd.concat([Feass, fea], axis=0)
        print(list(sorted(zip(train_x.columns.tolist(), model.feature_importance()), key=lambda x: x[1], reverse=True))[
              :20])

        if is_label_bool:
            val_pred = [i if i >= 0 else 0 for i in val_pred]
            train[valid_index] = np.expm1(val_pred)
            test = sum(test_pre) / folds
            test = np.expm1(test)
            cv_scores.append(mean_squared_error(np.expm1(val_y), np.expm1(val_pred)) ** 0.5)
        else:
            train[valid_index] = val_pred
            test = sum(test_pre) / folds
            cv_scores.append(mean_squared_error(val_y, val_pred) ** 0.5)

        print(cv_scores)
    print("s_scotrainre_list:", cv_scores)
    print("s_score_mean:", np.mean(cv_scores))
    print("s_score_std:", np.std(cv_scores))
    print('feature_importance:\n',
          Feass.groupby(['feas'])['sorce'].mean().reset_index().sort_values('sorce', ascending=False).iloc[:20])

    return train, test, Feass.groupby(['feas'])['sorce'].mean().reset_index().sort_values('sorce', ascending=False)

## 开始训练预测
lgb_train, lgb_test, Feas = lgb_model(lgb, x_train, y_train, x_test, kfod=10)


sub = test_xl[['glass_id','site']]
sub['param_value'] = test_xl['site_label_max-min'].values * lgb_test + test_xl['site_label_min'].values
sub = sub.groupby(['glass_id','site'])['param_value'].min().reset_index()
sub = pd.pivot(sub, index = 'glass_id',columns = 'site', values = 'param_value').reset_index()
sub.columns = ['glass_id'] + ['site_'+str(i) for i in sub.columns[1:]]
sub.to_csv('../结果文件/决赛最终提交结果文件.csv', index=False)
print(sub)



























