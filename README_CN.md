## GPUBoruta
本项目是原始Boruta算法的GPU版本实现。
原始[python版本的Boruta算法](https://github.com/scikit-learn-contrib/boruta_py)在处理大规模数据集时可能会非常慢，而GPU版本通过利用并行计算能力从而降低计算成本。

## 安装依赖
要使用GPUBoruta，推荐您使用Python 3.10或更高版本，并且至少需要安装以下依赖项：
```
cupy-cuda12x
cuml==25.8.0
numpy
orjson
pandas
scikit-learn
scipy
```
我们默认您已经安装了CUDA工具包，并且正确配置了环境变量。

## 源码介绍
GPUBoruta的核心代码位于`src`中，其中：
*   `boruta_clf.py`是分类任务的排列重要性（使用KL散度）的实现版本。
*   `boruta_score.py`是回归任务的排列重要性（使用MSE）的实现版本。
*   `boruta_tree.py`是回归任务的树重要性分析（类似sklearn原始RandomForest的feature_importances_）的实现版本。

## 相关示例
`example`文件夹中包含了GPUBoruta论文中使用的示例代码，您可以参考这些代码来了解如何使用GPUBoruta进行特征选择。
*   `Madelon`：一个分类任务的示例，使用了[Madelon数据集](https://doi.org/10.24432/C5602H).
*   `NEWS`：一个回归任务的示例，使用了[Online News Popularity数据集](https://doi.org/10.24432/C5NS3V).
*   `toy_base`和`toy_Multicollinearity`：toy数据集的示例，具体需要参考论文。

示例中包含以下共同的文件：
*   `origin.py`：使用CPU-based的Boruta算法进行特征选择。
*   `score.py`：使用GPU-based的Boruta算法（`boruta_score.py`）进行特征选择。
*   `tree.py`：使用GPU-based的Boruta算法（`boruta_tree.py`）进行特征选择。
*   `clf.py`：使用GPU-based的Boruta算法（`boruta_clf.py`）进行特征选择。
*   `resultmodel.py`：评估特征选择效果的代码，将计算对应的性能指标，使用了sklearn的RandomForest进行评估。
*   `draw.py`：结果可视化代码。

## 使用方法
首先需要引入GPUBoruta的库，对于分类任务：
```python
import boruta_clf as boruta
from cuml.ensemble import RandomForestClassifier as curfr
```

对于回归任务：
```python
import boruta_score as boruta
from cuml.ensemble import RandomForestRegressor as curfr
```

请自行处理数据预处理和特征工程等步骤，确保输入数据适合Boruta算法的要求。然后，您可以使用以下代码来进行特征选择：
```python
# 初始化随机森林模型，其余超参数可以根据需要进行调整
rfcv = curfr(random_state=seed_)
# 初始化Boruta算法，n_estimators='auto'表示使用随机森林的默认树数量，verbose=2表示输出详细的日志信息，max_iter=100表示内部最多迭代100次，perc=100表示使用Shadow特征的100分位处的值作为比较基准（即Shadow特征的最大值）
boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=seed_, max_iter=100, perc = 100)
# 进行特征选择
boruta_selector.fit(X, y)
# 获取特征排名
feature_ranks = boruta_selector.ranking_
```