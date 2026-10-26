# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Generate data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)
X = rng.uniform(-3, 3, size=500)
trend = 2.4 * X
seasonal = 3.1 * np.sin(3.2 * X)
drop = 10.0 * (X > 2).astype(float)
sigma = 0.75 + 0.75 * X**2
y = trend + seasonal - drop + rng.normal(loc=0.0, scale=np.sqrt(sigma))
df = pd.DataFrame({"X": X, "y": y})
_ = df.plot.scatter(x="X", y="y")
plt.show()

## Stack of predictors on a single data set
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

linear_ridge = make_pipeline(StandardScaler(), RidgeCV())
spline_ridge = make_pipeline(
    SplineTransformer(n_knots=6, degree=3),
    PolynomialFeatures(interaction_only=True),
    RidgeCV(),
)
hgbt = HistGradientBoostingRegressor(random_state=0)

estimators = [
    ("Linear Ridge", linear_ridge),
    ("Spline Ridge", spline_ridge),
    ("HGBT", hgbt),
]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
print(stacking_regressor)

## Measure and plot results
import matplotlib.pyplot as plt
X = X.reshape(-1, 1)
linear_ridge.fit(X, y)
spline_ridge.fit(X, y)
hgbt.fit(X, y)
stacking_regressor.fit(X, y)

x_plot = np.linspace(X.min() - 0.1, X.max() + 0.1, 500).reshape(-1, 1)
preds = {
    "Linear Ridge": linear_ridge.predict(x_plot),
    "Spline Ridge": spline_ridge.predict(x_plot),
    "HGBT": hgbt.predict(x_plot),
    "Stacking (Ridge final estimator)": stacking_regressor.predict(x_plot),
}
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (name, y_pred) in zip(axes, preds.items()):
    ax.scatter(
        X[:, 0],
        y,
        s=6,
        alpha=0.35,
        linewidths=0,
        label="observed (sample)",
    )
    ax.plot(x_plot.ravel(), y_pred, linewidth=2, alpha=0.9, label=name)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")

plt.suptitle("Base Models Predictions versus Stacked Predictions", y=1)
plt.tight_layout()
plt.show()


import time
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, cross_validate

fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)

for ax, (name, est) in zip(
    axs, estimators + [("Stacking Regressor", stacking_regressor)]
):
    scorers = {r"$R^2$": "r2", "MAE": "neg_mean_absolute_error"}
    start_time = time.time()
    scores = cross_validate(est, X, y, scoring=list(scorers.values()), n_jobs=-1)
    elapsed_time = time.time() - start_time

    y_pred = cross_val_predict(est, X, y, n_jobs=-1)
    scores = {
        key: (
            f"{np.abs(np.mean(scores[f'test_{value}'])):.2f}"
            r" $\pm$ "
            f"{np.std(scores[f'test_{value}']):.2f}"
        )
        for key, value in scorers.items()
    }

    display = PredictionErrorDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=ax,
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    ax.set_title(f"{name}\nEvaluation in {elapsed_time:.2f} seconds")

    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")

plt.suptitle("Prediction Errors of Base versus Stacked Predictors", y=1)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

stacking_regressor.fit(X, y)
stacking_regressor.final_estimator_.coef_

from sklearn.linear_model import LassoCV
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=LassoCV())
stacking_regressor.fit(X, y)
stacking_regressor.final_estimator_.coef_

## How to mimic SuperLearner with scikit-learn
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression(fit_intercept=False, positive=True)
super_learner_like = StackingRegressor(
    estimators=estimators, final_estimator=linear_reg
)
super_learner_like.fit(X, y)
super_learner_like.final_estimator_.coef_
super_learner_like.final_estimator_.coef_.sum()