# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Preparing the data
from sklearn.datasets import fetch_openml
electricity = fetch_openml(
    name="electricity", version=1, as_frame=True, parser="pandas"
)
df = electricity.frame
df["transfer"][:17_760].unique()

import matplotlib.pyplot as plt
import seaborn as sns

df = electricity.frame.iloc[17_760:]
X = df.drop(columns=["transfer", "class"])
y = df["transfer"]

fig, ax = plt.subplots(figsize=(15, 10))
pointplot = sns.lineplot(x=df["period"], y=df["transfer"], hue=df["day"], ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.set(
    title="Hourly energy transfer for different days of the week",
    xlabel="Normalized time of the day",
    ylabel="Normalized energy transfer",
)
_ = ax.legend(handles, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
plt.show()

## Effect of number of trees and early stopping
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
print(f"Training sample size: {X_train.shape[0]}")
print(f"Test sample size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

max_iter_list = [5, 50]
average_week_demand = (
    df.loc[X_test.index].groupby(["day", "period"], observed=False)["transfer"].mean()
)
colors = sns.color_palette("colorblind")
fig, ax = plt.subplots(figsize=(10, 5))
average_week_demand.plot(color=colors[0], label="recorded average", linewidth=2, ax=ax)

for idx, max_iter in enumerate(max_iter_list):
    hgbt = HistGradientBoostingRegressor(
        max_iter=max_iter, categorical_features=None, random_state=42
    )
    hgbt.fit(X_train, y_train)

    y_pred = hgbt.predict(X_test)
    prediction_df = df.loc[X_test.index].copy()
    prediction_df["y_pred"] = y_pred
    average_pred = prediction_df.groupby(["day", "period"], observed=False)["y_pred"].mean()
    average_pred.plot(color=colors[idx + 1], label=f"max_iter={max_iter}", linewidth=2, ax=ax)

ax.set(
    title="Predicted average energy transfer during this week",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
_ = ax.legend()
plt.show()


common_params = {
    "max_iter": 1_000,
    "learning_rate": 0.3,
    "validation_fraction": 0.2,
    "random_state": 42,
    "categorical_features": None,
    "scoring": "neg_root_mean_squared_error",
}
hgbt = HistGradientBoostingRegressor(early_stopping=True, **common_params)
hgbt.fit(X_train, y_train)

_, ax = plt.subplots()
plt.plot(-hgbt.validation_score_)
_ = ax.set(
    xlabel="number of iterations",
    ylabel="root mean squared error",
    title=f"Loss of hgbt with early stopping (n_iter={hgbt.n_iter_})",
)
plt.show()

import math
common_params["max_iter"] = math.ceil(hgbt.n_iter_ / 100) * 100
common_params["early_stopping"] = False
hgbt = HistGradientBoostingRegressor(**common_params)

## Support for missing values
import numpy as np
from sklearn.metrics import root_mean_squared_error

rng = np.random.RandomState(42)
first_week = slice(0, 336)  # first week in the test set as 7 * 48 = 336
missing_fraction_list = [0, 0.01, 0.03]


def generate_missing_values(X, missing_fraction):
    total_cells = X.shape[0] * X.shape[1]
    num_missing_cells = int(total_cells * missing_fraction)
    row_indices = rng.choice(X.shape[0], num_missing_cells, replace=True)
    col_indices = rng.choice(X.shape[1], num_missing_cells, replace=True)
    X_missing = X.copy()
    X_missing.iloc[row_indices, col_indices] = np.nan
    return X_missing


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.values[first_week], label="Actual transfer")

for missing_fraction in missing_fraction_list:
    X_train_missing = generate_missing_values(X_train, missing_fraction)
    X_test_missing = generate_missing_values(X_test, missing_fraction)
    hgbt.fit(X_train_missing, y_train)
    y_pred = hgbt.predict(X_test_missing[first_week])
    rmse = root_mean_squared_error(y_test[first_week], y_pred)
    ax.plot(
        y_pred[first_week],
        label=f"missing_fraction={missing_fraction}, RMSE={rmse:.3f}",
        alpha=0.5,
    )
ax.set(
    title="Daily energy transfer predictions on data with MCAR values",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
_ = ax.legend(loc="lower right")
plt.show()