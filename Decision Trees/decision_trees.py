## 1.10.1. Classification
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])
clf.predict_proba([[2., 2.]])

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)
plt.show()

## 1.10.2. Regression
from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])

## 1.10.3. Multi-output problems

## 1.10.4. Complexity

## 1.10.5. Tips on practical use

## 1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART

## 1.10.7. Mathematical formulation

## 1.10.8. Missing Values Support
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([0, 1, 6, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]
tree = DecisionTreeClassifier(random_state=0).fit(X, y)
tree.predict(X)

X = np.array([np.nan, -1, np.nan, 1]).reshape(-1, 1)
y = [0, 0, 1, 1]
tree = DecisionTreeClassifier(random_state=0, max_depth=1).fit(X, y)
X_test = np.array([np.nan]).reshape(-1, 1)
tree.predict(X_test)

X = np.array([0, 1, 2, 3]).reshape(-1, 1)
y = [0, 1, 1, 1]
tree = DecisionTreeClassifier(random_state=0).fit(X, y)
X_test = np.array([np.nan]).reshape(-1, 1)
tree.predict(X_test)

## 1.10.9. Minimal Cost-Complexity Pruning
