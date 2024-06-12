import json
import pandas as pd

# xgb 加入nn oof 不加emb
xgb_ans = json.load(open('ans_test/xgb_B oag_nn 3seed.json', 'r'))
# xgb 加入nn oof 加入emb
xgb_ans1 = json.load(open('ans_test/xgb_B oag_nn emb 3seed.json', 'r'))
# lgb 不加nn oof 不加emb
xgb_ans2 = json.load(open('ans_test/lgb_B oag 3seed.json', 'r'))
# lgb 不加nn oof 加入emb
xgb_ans3 = json.load(open('ans_test/lgb_B oag emb 3seed.json', 'r'))

with open("data/ind_test_author_submit.json") as f:
    submission = json.load(f)


for id, names in submission.items():
    xgb_rank = pd.Series(list(xgb_ans[id].values())).rank()
    xgb_rank1 = pd.Series(list(xgb_ans1[id].values())).rank()
    xgb_rank2 = pd.Series(list(xgb_ans2[id].values())).rank()
    xgb_rank3 = pd.Series(list(xgb_ans3[id].values())).rank()
    y = (xgb_rank*0.4 + xgb_rank1*0.4 + xgb_rank2*0.1 + xgb_rank3*0.1)
    y = y / y.max()
    cnt=0
    for i, name in enumerate(names):
        submission[id][name] = y[i]

with open('ans_test/ensemble.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)
