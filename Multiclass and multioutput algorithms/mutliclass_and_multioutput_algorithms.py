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