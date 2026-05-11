## GPUBoruta
This project is a GPU-based implementation of the original Boruta algorithm.
The original [Python version of the Boruta algorithm](https://github.com/scikit-learn-contrib/boruta_py) can be quite slow when handling large-scale datasets, while the GPU version reduces computational cost by leveraging parallel computing capabilities.

## Installing Dependencies
To use GPUBoruta, we recommend Python 3.10 or higher, and at least the following dependencies must be installed:
```
cupy-cuda12x
cuml==25.8.0
numpy
orjson
pandas
scikit-learn
scipy
```
We assume that you have already installed the CUDA toolkit and configured the environment variables correctly.

## Source Code Overview
The core code of GPUBoruta is located in `src`, where:
*   `boruta_clf.py` is the implementation of permutation importance (using KL divergence) for classification tasks.
*   `boruta_score.py` is the implementation of permutation importance (using MSE) for regression tasks.
*   `boruta_tree.py` is the implementation of tree-based importance analysis (similar to the feature_importances_ of the original sklearn RandomForest) for regression tasks.

## Examples
The `example` folder contains the example code used in the GPUBoruta paper. You can refer to these examples to learn how to perform feature selection with GPUBoruta.
*   `Madelon`: an example of a classification task using the [Madelon dataset](https://doi.org/10.24432/C5602H).
*   `NEWS`: an example of a regression task using the [Online News Popularity dataset](https://doi.org/10.24432/C5NS3V).
*   `toy_base` and `toy_Multicollinearity`: examples on toy datasets; please refer to the paper for details.

The examples share the following common files:
*   `origin.py`: performs feature selection using the CPU-based Boruta algorithm.
*   `score.py`: performs feature selection using the GPU-based Boruta algorithm (`boruta_score.py`).
*   `tree.py`: performs feature selection using the GPU-based Boruta algorithm (`boruta_tree.py`).
*   `clf.py`: performs feature selection using the GPU-based Boruta algorithm (`boruta_clf.py`).
*   `resultmodel.py`: evaluates the feature selection performance and computes the corresponding metrics using the sklearn RandomForest.
*   `draw.py`: visualizes the results.

## Usage
First, import the GPUBoruta library. For classification tasks:
```python
import boruta_clf as boruta
from cuml.ensemble import RandomForestClassifier as curfr
```

For regression tasks:
```python
import boruta_score as boruta
from cuml.ensemble import RandomForestRegressor as curfr
```

Please handle data preprocessing and feature engineering on your own to ensure that the input data meets the requirements of the Boruta algorithm. Then, you can use the following code to perform feature selection:
```python
# Initialize the random forest model; other hyperparameters can be adjusted as needed
rfcv = curfr(random_state=seed_)
# Initialize the Boruta algorithm. n_estimators='auto' means using the default number of trees in the random forest, verbose=2 means outputting detailed log information, max_iter=100 means at most 100 internal iterations, and perc=100 means using the value at the 100th percentile of the shadow features as the comparison baseline (i.e., the maximum value of the shadow features)
boruta_selector = boruta.BorutaPy(rfcv, n_estimators='auto', verbose=2, random_state=seed_, max_iter=100, perc = 100)
# Perform feature selection
boruta_selector.fit(X, y)
# Obtain the feature ranking
feature_ranks = boruta_selector.ranking_
```