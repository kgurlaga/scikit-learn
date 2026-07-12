## 1.11.1. Gradient-boosted trees
## 1.11.1.1. Histogram-based gradient boosting
## 1.11.1.1.1 Usage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
clf.score(X_test, y_test)

## 1.11.1.1.2 Missing values support
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
gbdt.predict(X)

X = np.array([0, np.nan, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 1, 0, 0, 1]
gbdt = HistGradientBoostingClassifier(min_samples_leaf=1,
                                      max_depth=2,
                                      learning_rate=1,
                                      max_iter=1).fit(X, y)
gbdt.predict(X)

## 1.11.1.1.3. Sample weight support
X = [[1, 0],
     [1, 0],
     [1, 0],
     [0, 1]]
y = [0, 0, 1, 0]
# ignore the first 2 training samples by setting their weight to 0
sample_weight = [0, 0, 1, 1]
gb = HistGradientBoostingClassifier(min_samples_leaf=1)
gb.fit(X, y, sample_weight=sample_weight)
gb.predict([[1, 0]])
gb.predict_proba([[1, 0]])[0, 1]

## 1.11.1.1.4. Castegorical Features Support
gbdt = HistGradientBoostingClassifier(categorical_features=[True, False])
gbdt = HistGradientBoostingClassifier(categorical_features=[0])
gbdt = HistGradientBoostingClassifier(categorical_features=["site", "manufacturer"])

## 1.11.1.1.5. Monotonic Constraints
from sklearn.ensemble import HistGradientBoostingRegressor
gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, -1, 0])

## 1.11.1.1.6. Intercation constraints

## 1.11.1.1.7. Low-level parallelism

## 1.11.1.1.8. Why it's faster

## 1.11.1.2. GradientBoostingClassifier and GradientBoostingRegressor
