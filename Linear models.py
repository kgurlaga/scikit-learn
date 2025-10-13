import matplotlib.pyplot as plt
low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)
ax.plot([low, high], [low, high], ls = "--", c = ".3", alpha = 0.5)
ax.set_xlabel("OLS regression coefficients", fontweight = "bold")
ax.set_ylabel("NNLS regression coefficients", fontweight = "bold")
plt.show()

## 1.1.1. Ordinary Least Squere
from sklearn import linear_model
import numpy as np
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)


intercept = reg.intercept_
intercept = 0 if np.isclose(intercept, 0) else intercept
print(intercept)




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


# Setting the regularization parameter: leave-one-out Cross-Validation
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.alpha_

## LASSO
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])

# AIC and BIC criteria

## Multi-task Lasso
