# 1.5.1 Classification #
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty = "l2", max_iter=5)
clf.fit(X, y)

clf.predict([[2., 2.]])
clf.coef_
clf.intercept_
clf.decision_function([[2., 2.]])

clf = SGDClassifier(loss="log_loss", max_iter=5).fit(X, y)
clf.predict_proba([[1., 1.]])

# 1.5.2 Regression

# 1.5.3 Online One-Class SVM

# 1.5.4 Stochastic Gradient Descent for sparse data

# 1.5.5 Complexity

# 1.5.6 Stopping criterion

# 1.5.7 Tips and practical use
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) # apply same transformation to test data

# or better yet: use a pipeline!
from sklearn.pipeline import make_pipeline
est = make_pipeline(StandardScaler(), SGDClassifier())
est.fit(X_train)
est.predict(X_test)

# 1.5.8 Mathematical formulation
