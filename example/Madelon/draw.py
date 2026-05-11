# End time: 2025-09-18 09:45:26.864808
# Time taken: 0 days 00:19:33.655133
import numpy as np
import pandas as pd
import boruta
import boruta_clf
import boruta_tree
from sklearn.ensemble import RandomForestClassifier as skfr
from cuml.ensemble import RandomForestClassifier as curfr

which = "sklearn"  # "sklearn" "score" "tree"

np.random.seed(42)
X = pd.read_csv('Madelon.csv')
y = X.pop('Class')

feature_names = X.columns.tolist()
# 只要前2000行数据
X = X.values[:2000]
y = y.values[:2000]

if which == "sklearn":
    rfcv = skfr(n_jobs = 128, random_state=0, max_depth=8)
else:
    rfcv = curfr(random_state=0, max_depth=8)

# 初始化Boruta特征选择器
if which == "sklearn":
    boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=0, max_iter=100, perc = 100)
elif which == "score":
    boruta_selector = boruta_clf.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=0, max_iter=100, perc = 100)
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
sorted_idx = np.argsort(-medians)[:50]  # 取前50个
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
# plt.title('Boxplot Sorted by Median (Ascending)')
plt.tight_layout()
plt.savefig(f'{which}_boxplot.png', dpi=300)