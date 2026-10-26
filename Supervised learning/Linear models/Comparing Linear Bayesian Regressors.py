# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Models robustness to recover the ground truth weights
# Generate synthetic dataset

from sklearn.datasets import make_regression

X, y, true_weights = make_regression(
    n_samples = 100,
    n_features = 100,
    n_informative = 10,
    noise = 8.0,
    coef = True,
    random_state = 42,
)

# Fit the regressors
import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression, LinearRegression

olr = LinearRegression().fit(X, y)
brr = BayesianRidge(compute_score = True, max_iter = 30).fit(X, y)
ard = ARDRegression(compute_score = True, max_iter = 30).fit(X, y)
df = pd.DataFrame(
    {
        "Weights of true generative process": true_weights,
        "ARDRegression": ard.coef_,
        "BayesianRidge": brr.coef_,
        "LinearRegression": olr.coef_,
    }
)

# Plot the true and estimated coefficients
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

plt.figure(figsize = (10, 6))
ax = sns.heatmap(
    df.T,
    norm = SymLogNorm(linthresh = 10e-4, vmin = -80, vmax = 80),
    cbar_kws = {"label": "coefficients' values"},
    cmap = "seismic_r",
)
plt.ylabel("linear model")
plt.xlabel("coefficients")
plt.tight_layout(rect = (0, 0, 1, 0.95))
_ = plt.title("Models' coefficients")
plt.show()

# Plot the marginal log-likelihood
import numpy as np

ard_scores = -np.array(ard.scores_)
brr_scores = -np.array(brr.scores_)
plt.plot(ard_scores, color= "navy", label = "ARD")
plt.plot(brr_scores, color = "red", label = "BayesianRidge")
plt.ylabel("Log-likehood")
plt.xlabel("Iterations")
plt.xlim(1, 30)
plt.legend()
_ = plt.title("Models log-likelihood")
plt.show()

## Bayesian regressions with polynomial feature expansion
# Generate synthetic dataset

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.RandomState(0)
n_samples = 110

# sort the data to make plotting easier later
X = np.sort(-10 * rng.rand(n_samples) + 10)
noise = rng.normal(0, 1, n_samples) * 1.35
y = np.sqrt(X) * np.sin(X) + noise
full_data = pd.DataFrame({"input_feature": X, "target": y})
X = X.reshape((-1, 1))

# extrapolation
X_plot = np.linspace(10, 10.4, 10)
y_plot = np.sqrt(X_plot) *np.sin(X_plot)
X_plot = np.concatenate((X, X_plot.reshape((-1, 1))))
y_plot = np.concatenate((y - noise, y_plot))

# Fit the regressors
ard_poly = make_pipeline(
    PolynomialFeatures(degree = 10, include_bias = False),
    StandardScaler(),
    ARDRegression(),
).fit(X, y)
brr_poly = make_pipeline(
    PolynomialFeatures(degree = 10, include_bias = False),
    StandardScaler(),
    BayesianRidge(),
).fit(X, y)

y_ard, y_ard_std = ard_poly.predict(X_plot, return_std = True)
y_brr,y_brr_std = brr_poly.predict(X_plot, return_std  = True)

# Plotting polynomial regressions with std errors of the scores
ax = sns.scatterplot(
    data = full_data, x = "input_feature", y = "target", color = "black", alpha = 0.75
)
ax.plot(X_plot, y_plot, color = "black", label = "Ground Truth")
ax.plot(X_plot, y_brr, color = "red", label = "BayesianRidge with polynomial features")
ax.plot(X_plot, y_ard, color = "navy", label = "ARD with polynomial features")
ax.fill_between(
    X_plot.ravel(),
    y_ard - y_ard_std,
    y_ard + y_ard_std,
    color = "navy",
    alpha = 0.3,
)
ax.fill_between(
    X_plot.ravel(),
    y_brr - y_brr_std,
    y_brr + y_brr_std,
    color = "red",
    alpha = 0.3,
)
ax.legend()
_ = ax.set_title("Polynomial fit of a non-linear feature")
plt.show()