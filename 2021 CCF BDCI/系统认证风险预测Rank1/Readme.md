# CCF-BDCI 系统风险认证-保住绿码团队-AB榜rank1

## **1. 环境依赖**

- Python==3.6
- pandas==0.24.0
- numpy==1.19.0
- scikit-learn==0.24.1
- gensim==4.0.1
- lightgbm==3.2.1
- imbalanced-learn==0.8.1
- tqdm==4.62.3

## **2. 模型方案**
- 模型：Lightgbm【1】
- 使用4个lgb基模型的rank加权融合，每个模型采取了不同的特征处理方式
