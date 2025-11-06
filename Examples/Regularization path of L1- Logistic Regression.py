# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Load data
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

X = X[y != 2]
y = y[y != 2]

## Compute regularization path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

cs = l1_min_c(X, y, loss= "log") * np.logspace(0, 1, 16)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty = "l1",
        solver = "liblinear",
        tol = 1e-6,
        max_iter = int(1e6),
        warm_start = True,
        fit_intercept = False,
    ),
)
coefs_ = []
for c in cs:
    clf.set_params(logisticregression__C = c)
    clf.fit(X, y)
    coefs_.append(clf["logisticregression"].coef_.ravel().copy())

coefs_ = np.array(coefs_)

## Plot
import matplotlib.pyplot as plt

# Colorblind-friendly palette (IBM Color)
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]

plt.figure(figsize = (10, 6))
for i in range(coefs_.shape[1]):
    plt.semilogx(cs, coefs_[:, i], marker = "o", color = colors[i], label = feature_names[i])

ymin, ymax = plt.ylim()
plt.xlabel("C")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Path")
plt.legend()
plt.axis("tight")
plt.show()
