# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time
from scipy import linalg, sparse
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

## Comparing the two Lasso implementations on Dense data
X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
# Create a copy of X in sparse format
X_sp = sparse.coo_array(X)

alpha = 1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

t0 = time()
sparse_lasso.fit(X_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(X, y)
print(f"Dense Lasso done in {(time() - t0):.3f}s")

# compare the regression coefficients
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients : {coeff_diff:2e}")

## Comparing the two Lasso implementations on Sparse data
# make a copy of the previous data
Xs = X.copy()
# make Xs sparse by replacing the values lower then 2.5 with 0s
Xs[Xs < 2.5] = 0.0
# create a copy of Xs in sparse format
Xs_sp = sparse.coo_array(Xs)
Xs_sp = Xs_sp.tocsc()

# compute the proportion of non-zero coefficient in the data matrix
print(f"Matrix density: {(Xs_sp.nnz / float(X.size) * 100):.3f}%")

alpha = 0.1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

t0 = time()
sparse_lasso.fit(Xs_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(Xs, y)
print(f"Dense Lasso done in {(time() - t0):.3f}s")

# compare the regression coefficients
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients: {coeff_diff:.2e}")

