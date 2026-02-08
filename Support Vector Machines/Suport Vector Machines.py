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