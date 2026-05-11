# End time: 2025-09-19 09:13:15.960747
# Time taken: 0 days 00:04:42.412772
import pandas as pd
import numpy as np
import boruta_tree as boruta
from cuml.ensemble import RandomForestClassifier as curfr
X = pd.read_csv('Madelon.csv')
y = X.pop('Class')

feature_names = X.columns.tolist()
# 只要前2000行数据
X = X.values[:2000]
y = y.values[:2000]

ranking_df = pd.DataFrame(index=range(1, 21), columns=feature_names)
# 计时
timestart = pd.Timestamp.now()
print(f"Start time: {timestart}")
# 运行 Boruta 20 次
for i in range(20):
    print(f"Iteration {i+1}")
    # 初始化随机森林模型
    rfcv = curfr(random_state=i, max_depth = 10)
    
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
import matplotlib.pyplot as plt
# 绘制imp_history的箱形图
plt.figure(figsize=(12, 6))
imp = boruta_selector.imp_history[1:, :]
# 替换nan为0
imp = np.nan_to_num(imp)
# 对imp的行进行Z-score标准化
imp = (imp - np.mean(imp, axis=1, keepdims=True)) / np.std(imp, axis=1, keepdims=True)
# 计算中位值并获取排序索引
medians = np.median(imp, axis=0)
sorted_idx = np.argsort(medians)
# print(sorted_idx)
# 按中位值升序重新排列
sorted_data = imp[:, sorted_idx]
sorted_labels = [feature_names[i] for i in sorted_idx]
# 绘制箱线图
plt.figure(figsize=(12, 6))
plt.boxplot(sorted_data, tick_labels=sorted_labels, vert=True)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Value')
plt.title('Boxplot Sorted by Median (Ascending)')
plt.tight_layout()
plt.savefig('tree_imp.png', dpi=300)