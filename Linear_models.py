## 1.1.1. Ordinary Least Squere
from sklearn import linear_model
import numpy as np
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)


intercept = reg.intercept_
intercept = 0 if np.isclose(intercept, 0) else intercept
print(intercept)

## Ordinary Least Squares and Ridge Regression
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Data Loading and Preparation
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # Use only one feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

# Linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression().fit(X_train, y_train)

# Model evaluation
from sklearn.metrics import mean_squared_error, r2_score

y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")

# Plotting the results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].scatter(X_train, y_train, label="Train data points")
ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0].set(xlabel="Feature", ylabel="Target", title="Train set")
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[1].legend()

fig.suptitle("Linear Regression")

plt.show()

## Ordinary Least Squares and Ridge Regression Variance
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

X_train = np.c_[0.5, 1].T
y_train = [0.5, 1]
X_test = np.c_[0, 2].T

np.random.seed(0)

classifiers = dict(ols = linear_model.LinearRegression(), ridge = linear_model.Ridge(alpha = 0.1))

for name, clf in classifiers.items():
    fig, ax = plt.subplots(figsize = (4, 3))

    for _ in range(6):
        this_X = 0.1 * np.random.normal(size = (2, 1)) + X_train
        clf.fit(this_X, y_train)

        ax.plot(X_test, clf.predict(X_test), color = "gray")
        ax.scatter(this_X, y_train, s = 3, c = "gray", marker = "o", zorder = 10)
    
    clf.fit(X_train, y_train)
    ax.plot(X_test, clf.predict(X_test), linewidth = 2, color = "blue")
    ax.scatter(X_train, y_train, s = 30, c = "red", marker = "+", zorder = 10)

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel("X")
    ax.set_ylabel("y")

    fig.tight_layout()
plt.show()

## 1.1.1.1. Non-Negative Least Squares
## Non-negative least squares
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

## Generate some random data
np.random.seed(42)
n_samples, n_features = 200, 50
X = np.random.randn(n_samples, n_features)
true_coef = 3 * np.random.randn(n_features)

# Threshold coefficients to render them non-negative
true_coef[true_coef < 0] = 0
y = np.dot(X, true_coef)

# Add some noise
y += 5 * np.random.normal(size = (n_samples,))

## Split the data in train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

## Fit the Non-Negative least squares
from sklearn.linear_model import LinearRegression

reg_nnls = LinearRegression(positive = True)
y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
r2_score_nnls = r2_score(y_test, y_pred_nnls)
print("NNLS R2 score", r2_score_nnls)

## Fit an OLS
reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
r2_score_ols = r2_score(y_test, y_pred_ols)
print("OLS R2 score", r2_score_ols)

fig, ax = plt.subplots()
ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth = 0, marker = ".")

low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)
ax.plot([low, high], [low, high], ls = "--", c = ".3", alpha = 0.5)
ax.set_xlabel("OLS regression coefficients", fontweight = "bold")
ax.set_ylabel("NNLS regression coefficients", fontweight = "bold")
plt.show()

## 1.1.2. Ridge regression and classification

## Plot ridge coefficients as a function of the regularization
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

# Compute paths
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha = a, fit_intercept = False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# Display results
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge Coefficients vs Regularization Strength (alpha)")
plt.axis("tight")
plt.legend([f"Feature {i + 1}" for i in range(X.shape[1])], loc = "best", fontsize = "small")
plt.show()

## 1.1.2.1 Regression
from sklearn import linear_model
reg = linear_model.Ridge(alpha = .5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.coef_
reg.intercept_

## Porównanie solverów (cholesky, sparse_cg)
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Funkcja pomocniczna do pomiaru czasu
def benchmark_solver(solver, n_samples = 1000, n_features = 100):
    X, y = make_regression(n_samples = n_samples, n_features = n_features, noise = 0.1)
    model = Ridge(alpha = 1.0, solver = solver)
    start = time.time()
    model.fit(X, y)
    end = time.time()
    return end - start

if __name__ == "__main__":
    # mały zbiór danych
    print("--- Mały zbiór danych (1000 próbek, 100 cech) ---")
    for solver in ["cholesky", "sparse_cg"]:
        t = benchmark_solver(solver, n_samples = 1000, n_features = 100)
        print(f"Solver {solver:10s} czas: {t:.5f} s")

    # Duży zbiór danych
    print("\n--- Duży zbiór danych (100000 próbek, 1000 cech) ---")
    for solver in ["cholesky", "sparse_cg"]:
        t = benchmark_solver(solver, n_samples = 100000, n_features = 1000)
        print(f"Solver {solver:10s} czas: {t:.5f} s")
