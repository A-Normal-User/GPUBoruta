# https://archive.ics.uci.edu/dataset/332/online+news+popularity
# End time: 2025-09-15 14:43:46.539213
# Time taken: 0 days 00:21:44.007992
from sklearn.datasets import load_breast_cancer, load_diabetes
import pandas as pd
import numpy as np
import boruta
from sklearn.ensemble import RandomForestRegressor as curfr

np.random.seed(42)
X = pd.read_csv('OnlineNewsPopularity.csv')
y = X.pop(' shares')

index = np.random.permutation(len(X))[:10000]

feature_names = X.columns.tolist()
X = X.values[index]
y = y.values[index]

ranking_df = pd.DataFrame(index=range(1, 21), columns=feature_names)
# 计时
timestart = pd.Timestamp.now()
print(f"Start time: {timestart}")
# 运行 Boruta 20 次
for i in range(20):
    print(f"Iteration {i+1}")
    # 初始化随机森林模型
    rfcv = curfr(n_jobs = 128, random_state=i, max_depth=8)
    
    # 初始化Boruta特征选择器
    boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=i, max_iter=100, perc = 50)
    
    # 对训练数据进行特征选择
    boruta_selector.fit(X, y)
    
    # 获取特征排名
    feature_ranks = boruta_selector.ranking_
    
    # 将特征排名保存到 DataFrame 中
    ranking_df.loc[i+1] = feature_ranks

# 打印结果
print("\nFeature Ranking across 20 iterations:")
timeend = pd.Timestamp.now()
print(f"End time: {timeend}")
print(f"Time taken: {timeend - timestart}")
ranking_df.to_csv('origin.csv', index=False)