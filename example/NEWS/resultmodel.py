import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as curfr
from sklearn.model_selection import train_test_split
np.random.seed(42)

X = pd.read_csv('OnlineNewsPopularity.csv')
y = X.pop(' shares')

model = 'tree' 
ranking_df = pd.read_csv(f'{model}.csv')
# 确保数据集中只有数值列
numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

# 计算每个特征的中位数
median_values = numeric_ranking_df.median()
feature = list(median_values[median_values < 2].index)
print(f'model: {model}, count of selected feature: {len(feature)}')
X = X[feature]
X = X.values
y = y.values

# 进行五折交叉验证cross_val_score
from sklearn.model_selection import cross_validate
rfcv = curfr(n_jobs = -1, random_state=42, max_depth=None)
scoring = {
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error', 
    'r2': 'r2'
}
cv_results = cross_validate(rfcv, X, y, cv=5, scoring=scoring)
# 5. 提取和计算结果
mse_scores = -cv_results['test_neg_mse']  # 转换为正值
mae_scores = -cv_results['test_neg_mae']  # 转换为正值
r2_scores = cv_results['test_r2']
# 6. 输出结果
print("五折交叉验证结果:")
print(f"MSE: {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
print(f"MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")
print(f"R²:  {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")

# model: ../BorutaShap/NEWS, count of selected feature: 5
# 五折交叉验证结果:
# MSE: 145761740.6660 (+/- 149715246.5174)
# MAE: 3504.0866 (+/- 878.4114)
# R²:  -0.1241 (+/- 0.1767)

# model: all, count of selected feature: 58
# 五折交叉验证结果:
# MSE: 144059901.8515 (+/- 152889299.5718)
# MAE: 3553.3707 (+/- 662.9454)
# R²:  -0.0918 (+/- 0.0946)

# model: origin, count of selected feature: 16
# 五折交叉验证结果:
# MSE: 147082032.6941 (+/- 148398262.0615)
# MAE: 3512.7931 (+/- 765.0627)
# R²:  -0.1667 (+/- 0.3918)

# model: score, count of selected feature: 13
# 五折交叉验证结果:
# MSE: 142585751.4345 (+/- 154766537.4025)
# MAE: 3509.1945 (+/- 793.7199)
# R²:  -0.0735 (+/- 0.0944)

# model: tree, count of selected feature: 5
# 五折交叉验证结果:
# MSE: 146042249.2378 (+/- 153854330.8910)
# MAE: 3461.0728 (+/- 870.0110)
# R²:  -0.1126 (+/- 0.1297)