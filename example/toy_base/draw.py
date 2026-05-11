# End time: 2025-09-18 09:45:26.864808
# Time taken: 0 days 00:19:33.655133
import numpy as np
import pandas as pd
import boruta
import boruta_score
import boruta_tree
from sklearn.ensemble import RandomForestRegressor as skfr
from cuml.ensemble import RandomForestRegressor as curfr

which = "tree"  # "sklearn" "score" "tree"

np.random.seed(42)
index = np.random.permutation(50)
index = [13, 39, 30, 45, 17, 48, 26, 25, 32, 19, 12, 4, 37, 8, 3, 6, 41, 46, 47, 15, 9, 16, 24, 34, 31, 0, 44, 27, 33, 5, 29, 11, 36, 1, 21, 2, 43, 35, 23, 40, 10, 22, 18, 49, 20, 7, 42, 14, 28, 38]
# print(f"True important feature indices: {list(index + 1)}")
X = np.random.uniform(0,2,size=(10000, 50))
feature_names = [f'feature_{i}' for i in range(1, 51)]

# X[np.random.permutation(10000)[:9900],index[9]] = -1
y = 2 * X[:,index[0]] - 2.5 * X[:,index[1]] + 5 * X[:,index[2]] * X[:,index[3]] + 3.5 * np.cbrt(X[:,index[4]]) + 4 * (X[:,index[5]]**2) + 5 * (X[:,index[6]] ** 3) + 2 * np.sin(3.14 * X[:,index[7]]) + 5 * np.cos(3.14 * X[:,index[8]]) + np.exp(X[:,index[9]]) + np.random.normal(size=10000)

if which == "sklearn":
    rfcv = skfr(n_jobs = 128, random_state=0, max_depth=10)
else:
    rfcv = curfr(random_state=0, max_depth=10)

# 初始化Boruta特征选择器
if which == "sklearn":
    boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=0, max_iter=100, perc = 100)
elif which == "score":
    boruta_selector = boruta_score.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=0, max_iter=100, perc = 100)
elif which == "tree":
    boruta_selector = boruta_tree.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=0, max_iter=100, perc = 100)

# 对训练数据进行特征选择
boruta_selector.fit(X, y)

# 获取特征排名
feature_ranks = boruta_selector.ranking_
support = boruta_selector.support_
tentative = boruta_selector.support_weak_

import matplotlib.pyplot as plt
# 绘制imp_history的箱形图
imp = boruta_selector.imp_history[1:,:]
# 替换nan为0
imp = np.nan_to_num(imp)
# 对imp的行进行Z-score标准化
imp = (imp - np.mean(imp, axis=1, keepdims=True)) / np.std(imp, axis=1, keepdims=True)
# 计算中位值并获取排序索引
medians = np.median(imp, axis=0)
sorted_idx = np.argsort(-medians)
print(sorted_idx)
# 按中位值升序重新排列
sorted_data = imp[:, sorted_idx]
sorted_labels = [feature_names[i] for i in sorted_idx]
# 绘制箱线图
plt.figure(figsize = (18, 6))
plt.boxplot(sorted_data, tick_labels=sorted_labels, vert=True)
# support的特征填充绿色
for i in range(len(sorted_labels)):
    original_idx = feature_names.index(sorted_labels[i])
    if support[original_idx]:
        plt.gca().get_xticklabels()[i].set_color('green')
    elif tentative[original_idx]:
        plt.gca().get_xticklabels()[i].set_color('blue')
    else:
        plt.gca().get_xticklabels()[i].set_color('red')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Importance (Z-score)', fontsize=16)
# 放大字体到16号
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f'{which}_boxplot.png', dpi=300)