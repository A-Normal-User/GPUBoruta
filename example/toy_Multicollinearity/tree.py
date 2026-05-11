# End time: 2025-09-19 10:05:05.546925
# Time taken: 0 days 00:04:53.898219
import numpy as np
import pandas as pd
import boruta_tree as boruta
from cuml.ensemble import RandomForestRegressor as curfr

np.random.seed(42)
index = np.random.permutation(50)
index = [13, 39, 30, 45, 17, 48, 26, 25, 32, 19, 12, 4, 37, 8, 3, 6, 41, 46, 47, 15, 9, 16, 24, 34, 31, 0, 44, 27, 33, 5, 29, 11, 36, 1, 21, 2, 43, 35, 23, 40, 10, 22, 18, 49, 20, 7, 42, 14, 28, 38]
# print(f"True important feature indices: {list(index + 1)}")
X = np.random.uniform(0,2,size=(10000, 50))
feature_names = [f'feature_{i}' for i in range(1, 51)]

# X[np.random.permutation(10000)[:9900],index[9]] = -1
y = 2 * X[:,index[0]] - 2.5 * X[:,index[1]] + 5 * X[:,index[2]] * X[:,index[3]] + 3.5 * np.cbrt(X[:,index[4]]) + 4 * (X[:,index[5]]**2) + 5 * (X[:,index[6]] ** 3) + 2 * np.sin(3.14 * X[:,index[7]]) + 5 * np.cos(3.14 * X[:,index[8]]) + np.exp(X[:,index[9]]) + np.random.normal(size=10000)
X[:,index[10]] = 0.1 * X[:,index[0]] + 0.2 * X[:,index[1]] + 0.3 * X[:,index[15]] + 0.4 * X[:,index[16]] + np.random.normal(0, 0.01, size=10000)
X[:,index[11]] = 0.4 * X[:,index[2]] + 0.2 * X[:,index[3]] + 0.3 * X[:,index[17]] + 0.1 * X[:,index[18]] + np.random.normal(0, 0.01, size=10000)
X[:,index[12]] = 0.1 * X[:,index[4]] + 0.3 * X[:,index[5]] + 0.2 * X[:,index[19]] + 0.4 * X[:,index[20]] + np.random.normal(0, 0.01, size=10000)
X[:,index[13]] = 0.4 * X[:,index[6]] + 0.3 * X[:,index[7]] + 0.2 * X[:,index[21]] + 0.1 * X[:,index[22]] + np.random.normal(0, 0.01, size=10000)
X[:,index[14]] = 0.2 * X[:,index[8]] + 0.4 * X[:,index[9]] + 0.1 * X[:,index[23]] + 0.3 * X[:,index[24]] + np.random.normal(0, 0.01, size=10000)

ranking_df = pd.DataFrame(index=range(1, 21), columns=feature_names)
# 计时
timestart = pd.Timestamp.now()
print(f"Start time: {timestart}")
# 运行 Boruta 20 次
for i in range(20):
    print(f"Iteration {i+1}")
    # 初始化随机森林模型
    rfcv = curfr(random_state=i, max_depth=10)
    
    # 初始化Boruta特征选择器
    boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=i, max_iter=100, perc = 100)
    
    # 对训练数据进行特征选择
    boruta_selector.fit(X, y)
    
    # 获取特征排名
    feature_ranks = boruta_selector.ranking_
    
    # 将特征排名保存到 DataFrame 中
    ranking_df.loc[i+1] = feature_ranks
    ranking_df.to_csv('tree.csv', index=False)

# 打印结果
print("\nFeature Ranking across 20 iterations:")
# print(ranking_df)
# 输出用时
timeend = pd.Timestamp.now()
print(f"End time: {timeend}")
print(f"Time taken: {timeend - timestart}")