import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as curfr
from sklearn.model_selection import train_test_split
np.random.seed(42)

model = 'tree' 
ranking_df = pd.read_csv(f'{model}.csv')
# 确保数据集中只有数值列
numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

# 计算每个特征的中位数
median_values = numeric_ranking_df.median()
features = list(median_values[median_values < 2].index)

anwser = [14, 40, 31, 46, 18, 49, 27, 26, 33, 20]
# 检查特征是否在列表中
correct_features = [f'feature_{i}' for i in anwser]


correct = 0
error = 0
who_error = []
for feature in features:
    if feature in correct_features:
        correct += 1
        correct_features.remove(feature)
    else:
        error += 1
        who_error.append(feature)

print(f"Selected features: {features}")
print(f"Number of correct features selected: {correct}")
print(f"Number of incorrect features selected: {error}")
print(f"Incorrect features: {who_error}")
print(f"Missed correct features: {correct_features}")