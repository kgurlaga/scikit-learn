## 1.13. Feature selection
## 1.13.1. Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

## 1.13.2. Univariate feature selection
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
X, y = load_iris(return_X_y=True)
X.shape
X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
X_new.shape

## 1.13.3. Recursive feature elimination
    