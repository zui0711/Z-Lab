import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import lightgbm as lgb
import warnings
warnings.simplefilter('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)

train = pd.read_csv('data/train_dataset.csv', sep='\t')
print(train.shape)
test = pd.read_csv('data/test_dataset.csv', sep='\t')
print(test.shape)

data = pd.concat([train, test])
print(data.shape)

# location
data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
data['auth_type'].fillna('__NaN__', inplace=True)

for col in tqdm(['user_name', 'action', 'auth_type', 'ip',
                 'ip_location_type_keyword', 'ip_risk_level', 'location', 'device_model',
                 'os_type', 'os_version', 'browser_type', 'browser_version',
                 'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
                 'location_third_lvl']):
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])


data['op_date'] = pd.to_datetime(data['op_date'])
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
data['ts_diff1'] = data['op_ts'] - data['last_ts']

# 

for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')

# 

for method in ['mean', 'max', 'min', 'std', 'sum', 'median','prod']:
    for col in ['user_name', 'ip', 'location', 'device_model', 'os_version', 'browser_version']:
        data[f'ts_diff1_{method}_' + str(col)] = data.groupby(col)['ts_diff1'].transform(method)

# 

train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]


# 
print(train.shape, test.shape)
###, 'last_ts'   ts_diff1_std_os_version
ycol = 'risk_label'

#
feature_names = ['last_ts', 'op_ts', 'ts_diff1', 'ts_diff1_mean_user_name', 'ts_diff1_sum_user_name',
       'browser_version', 'ts_diff1_max_user_name','ts_diff1_max_browser_version', 
       'ts_diff1_mean_browser_version', 'ts_diff1_sum_browser_version','user_name', 
       'ts_diff1_std_browser_version','ts_diff1_std_user_name','bus_system_code', 'ts_diff1_mean_ip', 
       'auth_type', 'location','ip', 'action','op_target', 'device_model', 'browser_type']

x_train = train[feature_names]
y_train = train['risk_label']
x_test  = test[feature_names]

def lgb_model(data_, test_, y_):
    df_importance_list = []
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    folds_ = StratifiedKFold(n_splits=20, shuffle=True, random_state=1983)
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_,y_)):
        trn_x, trn_y = data_.iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_.iloc[val_idx], y_.iloc[val_idx]
        #cat_feats =  ['auth_type','bus_system_code','browser_type','action']
        cat_feats =  ['auth_type','bus_system_code','op_target','browser_type','action']
        clf = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=2 ** 8,
                           max_depth=16,
                           learning_rate=0.2,
                           n_estimators=10000,
                           subsample=0.75,
                           feature_fraction=0.55,
                           # max_bin = 63,
                           reg_alpha=0.2,
                           reg_lambda=0.2,
                           random_state=1983,
                           is_unbalance=True,
                           metric='auc',
                           device='gpu',
                           gpu_platform_id=0, 
                           gpu_device_id=0,
                           )

        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)],categorical_feature=cat_feats,
                eval_metric='auc', verbose=100, early_stopping_rounds=40  #30
               )
        
        vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        minmin= min(oof_preds[val_idx])
        maxmax= max(oof_preds[val_idx])
        oof_preds[val_idx] = vfunc(oof_preds[val_idx])
        

        sub_preds += clf.predict_proba(test_, num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        minmin= min(sub_preds)
        maxmax= max(sub_preds)
        sub_preds = vfunc(sub_preds)
        
        df_importance = pd.DataFrame({
            'column': feature_names,
            'importance': clf.feature_importances_,
        })
        df_importance_list.append(df_importance)
        joblib.dump(clf, './model/lgb_'+ str(n_fold) +'.pkl')   
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))   
    score  = roc_auc_score(y_, oof_preds)
    print('Full AUC score %.6f' % score) 

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg(
        'mean').sort_values(ascending=False).reset_index()
    print(df_importance)
    
    return oof_preds, sub_preds

lgb_train, lgb_test = lgb_model(x_train, x_test, y_train)

####
submit = pd.DataFrame([])
submit['id'] = range(len(lgb_test))
submit['id'] = submit['id'] + 1
submit['ret'] = lgb_test 
submit.to_csv('ans/submit28.csv', index=False)