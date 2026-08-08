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

## 1.11.1.2.1. Fitting additional weak-learners
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
    loss="squared_error"
)
est = est.fit(X_train, y_train) # fit with 100 trees
mean_squared_error(y_test, est.predict(X_test))
_ = est.set_params(n_estimators=200, warm_start=True) # set warm_start and increase num of trees
_ = est.fit(X_train, y_train) # fit additional 100 trees to est
mean_squared_error(y_test, est.predict(X_test))


## 1.11.1.2.2. Controlling the tree size

## 1.11.1.2.3. Mathematical formulation

## 1.11.1.2.4. Loss Functions

## 1.11.1.2.5. Shrinkage via learning rate

## 1.11.1.2.6. Subsampling

## 1.11.1.2.7. Interpretation with feature importance
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0).fit(X, y)
clf.feature_importances_

## 1.11.2. Random forests and other randomized tree ensembles
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

## 1.11.2.1. Random Forests

## 1.11.2.2. Extremely Randomized Trees
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                             random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() > 0.999

## 1.11.2.3. Parameters

## 1.11.2.4. Parallelization

## 1.11.2.5. Feature importance evaluation

## 1.11.2.6. Totally Random Trees Embedding

## 1.11.2.7. Fitting additional trees
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, random_state=1)
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y) # fit with 10 trees
len(clf.estimators_)

# set warm start and increase num of estimators
_ = clf.set_params(n_estimators=20, warm_start=True)
_ = clf.fit(X, y) # fit additional 10 trees
len(clf.estimators_)

clf = RandomForestClassifier(n_estimators=20)
_ = clf.fit(X, y)

## 1.11.3. Bagging meta-estimator
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
print(bagging)