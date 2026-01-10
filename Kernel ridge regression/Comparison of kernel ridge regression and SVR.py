# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

####################################################
## GENERATE SAMPLE DATA
import numpy as np

# Generator liczb losowych z ustalonym ziarnem (powtarzalność wyników)
rng = np.random.RandomState(42)

# Dane wejściowe: 10 000 punktów losowych z zakresu [0, 5)
# Jedna cecha (kształt: [n_samples, 1])
X = 5 * rng.rand(10000, 1)

# Prawdziwa zależność: funkcja sinus
# ravel() zamienia (10000, 1) -> (10000,)
y = np.sin(X).ravel()

# Dodanie silnego szumu tylko do co 5-tego punktu
# Symulacja błędnych pomiarów / outlierów
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

# Gęsta siatka punktów do rysowania gładkiej krzywej predykcji
# Nie służy do trenowania modelu
X_plot = np.linspace(0, 5, 100000)[:, None]

####################################################
## CONSTRUCT THE KERNEL-BASED REGRESSION MODELS
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Liczba próbek treningowych (celowo mała)
train_size = 100

# ======================
# Grid Search dla SVR
# ======================
svr = GridSearchCV(
    # Model SVR z jądrem RBF (nieliniowa regresja)
    SVR(kernel="rbf", gamma=0.1),
    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
)

# ======================
# Grid Search dla Kernel Ridge Regression
# ======================
kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
)


####################################################
## COMPARE TIMES OF SVR AND KERNEL RIDGE REGRESSION
import time

t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score: {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))