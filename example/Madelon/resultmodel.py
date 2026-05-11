import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as curfr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
np.random.seed(42)

cancer = load_breast_cancer()
X = pd.read_csv('Madelon.csv')
y = X.pop('Class')

model = 'tree' 
ranking_df = pd.read_csv(f'{model}.csv')
# 确保数据集中只有数值列
numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

# 计算每个特征的中位数
median_values = numeric_ranking_df.median()
feature = list(median_values[median_values < 2].index)
print(f'model: {model}, count of selected feature: {len(feature)}')
X = X[feature]
X_train = X.values[:2000]
y_train = y.values[:2000]
X_test = X.values[2000:]
y_test = y.values[2000:]

rfcv = curfr(n_jobs = 128, random_state=42, max_depth=None)

rfcv.fit(X_train, y_train)

from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score
y_pred = rfcv.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))
# root@a95d8f153d56:/ml/Boruta/Madelon# python3 resultmodel.py 
# model: origin, count of selected feature: 20
# Accuracy: 0.8883333333333333
# F1 Score: 0.8896210873146623


#               precision    recall  f1-score   support

#            1       0.88      0.90      0.89       300
#            2       0.90      0.88      0.89       300

#     accuracy                           0.89       600
#    macro avg       0.89      0.89      0.89       600
# weighted avg       0.89      0.89      0.89       600

# root@a95d8f153d56:/ml/Boruta/Madelon# python3 resultmodel.py 
# model: clf, count of selected feature: 19
# Accuracy: 0.8983333333333333
# F1 Score: 0.8991735537190083


#               precision    recall  f1-score   support

#            1       0.89      0.91      0.90       300
#            2       0.91      0.89      0.90       300

#     accuracy                           0.90       600
#    macro avg       0.90      0.90      0.90       600
# weighted avg       0.90      0.90      0.90       600

# model: all, count of selected feature: 500
# Accuracy: 0.685
# F1 Score: 0.6834170854271356


#               precision    recall  f1-score   support

#            1       0.69      0.68      0.68       300
#            2       0.68      0.69      0.69       300

#     accuracy                           0.69       600
#    macro avg       0.69      0.69      0.68       600
# weighted avg       0.69      0.69      0.68       600

# model: tree, count of selected feature: 16
# Accuracy: 0.8916666666666667
# F1 Score: 0.8925619834710744


#               precision    recall  f1-score   support

#            1       0.89      0.90      0.89       300
#            2       0.90      0.88      0.89       300

#     accuracy                           0.89       600
#    macro avg       0.89      0.89      0.89       600
# weighted avg       0.89      0.89      0.89       600