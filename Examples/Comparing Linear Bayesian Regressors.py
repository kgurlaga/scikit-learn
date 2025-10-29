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