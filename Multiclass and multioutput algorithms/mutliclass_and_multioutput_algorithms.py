## 1.12.1. Multiclass classification

## 1.12.1.1. Target format
import numpy as np
y = np.array(['apple', 'pear', 'apple', 'orange'])
print(y)

from sklearn.preprocessing import LabelBinarizer
y = np.array(['apple', 'pear', 'apple', 'orange'])
y_dense = LabelBinarizer().fit_transform(y)
print(y_dense)

from scipy import sparse
y_sparse = sparse.csr_array(y_dense)
print(y_sparse)

## 1.12.1.2. OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

## 1.12.1.3. OneVsOneClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

## 1.12.1.4. OutputCodeClassifier
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
clf.fit(X, y).predict(X)

## 1.12.2. Multilabel classification
## 1.12.2.1. Target format
import numpy as np
y = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
print(y)

from scipy import sparse
y_sparse = sparse.csr_array(y)
print(y_sparse)

## 1.12.2.2. MultiOutputClassifier

## 1.12.2.3. ClassifierChain

## 1.12.3. Multiclass-multioutput classification
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape
n_outputs = Y.shape[1]
n_classes = 3
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
multi_target_forest.fit(X, Y).predict(X)

## 1.12.3.1. Target format
y = np.array([['apple', 'green'], ['orange', 'orange'], ['pear', 'green']])
print(y)
