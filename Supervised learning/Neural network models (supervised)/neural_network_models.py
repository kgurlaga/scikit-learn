## 1.17.1. Multi-layer perceptron
## 1.17.2. Classification

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

clf.predict([[2., 2.], [-1, -2]])
[coef.shape for coef in clf.coefs_]

clf.predict_proba([[2., 2.], [1., 2.]])

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
clf.predict([[1., 2.]])
clf.predict([[0., 0.]])

## 1.17.3. Regression

## 1.17.4. Regularization

## 1.17.5. Algorithms

## 1.17.6. Complexity

## 1.17.7. Tips on Practical Use
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## 1.17.8. More control with warm_start
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
for i in range(10):
    clf.fit(X, y)