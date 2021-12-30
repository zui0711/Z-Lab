import pandas as pd
from matplotlib.pyplot import plot, show
from seaborn import heatmap

ans1 = pd.read_csv('ans/lgb_5285.csv')  # 嘴爷5285
ans2 = pd.read_csv('ans/lgb_5272.csv')  # 嘴爷5272
ans3 = pd.read_csv('ans/submit28.csv')  # 小渡
ans4 = pd.read_csv('ans/sub_5225.csv')  # 老肥


ans = pd.DataFrame()
# ans['ans1'] = ans1['ret'].rank()
# ans['ans2'] = ans2['ret'].rank()
# ans['ans3'] = ans3['ret'].rank()
# ans['ans4'] = ans4['ret'].rank()

ans['A_0.5285'] = ans1['ret'].rank()
ans['A_0.5272'] = ans2['ret'].rank()
ans['A_0.5265'] = ans3['ret'].rank()
ans['A_0.5224'] = ans4['ret'].rank()
heatmap(ans.corr())
show()
print(ans.corr())

ans = ans1.copy()

ans['ret'] = ans1['ret'].rank()*0.3 + ans2['ret'].rank()*0.3 + ans3['ret'].rank()*0.4 + ans4['ret'].rank()*0.3
ans['ret'] = (ans['ret'] - ans['ret'].min())/(ans['ret'].max() - ans['ret'].min())

# best = pd.read_csv('ans/ans_lgb_202111201304_merge_rank_334.csv')
# plot(ans['ret'].rank(), '-x')
# plot(best['ret'].rank(), '-x')
# show()

# ans[['id', 'ret']].to_csv('ans/final.csv', index=False)