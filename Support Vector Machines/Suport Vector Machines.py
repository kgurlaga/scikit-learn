### CLASSIFICATION
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)

clf.predict([[2., 2.]])

clf.support_vectors_
clf.support_
clf.n_support_

####################################################
## MULTI-CLASS CLASSIFICATION

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
dec = clf.decision_function([[1]])
dec.shape[1]
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1]

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
dec = lin_clf.decision_function([[1]])
dec.shape[1]

####################################################
## SCORES AND PROBABILITIES

####################################################
## UNBALANCED

####################################################
## REGRESSION
from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
regr = svm.SVR()
regr.fit(X, y)
regr.predict([[1, 1]])


####################################################
## KERNEL FUNCTIONS
linear_svc = svm.SVC(kernel='linear')
linear_svc.kernel
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.kernel

####################################################
## CUSTOM KERNELS
import numpy as np
from sklearn import svm
def my_kernel(X, Y):
    return np.dot(X, Y.T)
clf = svm.SVC(kernel=my_kernel)

# Using the Gram matrix
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
X, y = make_classification(n_samples=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = svm.SVC(kernel='precomputed')
gram_train = np.dot(X_train, X_train.T)
clf.fit(gram_train, y_train)
gram_test = np.dot(X_test, X_train.T)
clf.predict(gram_test)