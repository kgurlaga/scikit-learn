# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Load Ames Housing datasetfrom sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_openml
X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

# Select only a subset of features of X to make the example faster to run
categorical_columns_subset = [
    "BldgType",
    "GarageFinish",
    "LotConfig",
    "Functional",
    "MasVnrType",
    "HouseStyle",
    "FireplaceQu",
    "ExterCond",
    "ExterQual",
    "PoolQC",
]

numerical_columns_subset = [
    "3SsnPorch",
    "Fireplaces",
    "BsmtHalfBath",
    "HalfBath",
    "GarageCars",
    "TotRmsAbvGrd",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "GrLivArea",
    "ScreenPorch",
]

X = X[categorical_columns_subset + numerical_columns_subset]
X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")

categorical_columns = X.select_dtypes(include="category").columns
n_categorical_features = len(categorical_columns)
n_numerical_features = X.select_dtypes(include="number").shape[1]

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of categorical features: {n_categorical_features}")
print(f"Number of numerical features: {n_numerical_features}")

## Gradient boosting estimator with dropped categorical features
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

dropper = make_column_transformer(
    ("drop", make_column_selector(dtype_include="category")), remainder="passthrough"
)
hist_dropped = make_pipeline(dropper, HistGradientBoostingRegressor(random_state=42))
print(hist_dropped)

## Gradient boosting estimator with one-hot encoding
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

hist_one_hot = make_pipeline(
    one_hot_encoder, HistGradientBoostingRegressor(random_state=42)
)
print(hist_one_hot)

## Gradient boosting estimator with ordinal encoding
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

hist_ordinal = make_pipeline(
    ordinal_encoder, HistGradientBoostingRegressor(random_state=42)
)
hist_ordinal

## Gradient boosting estimator with target encoding
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

target_encoder = make_column_transformer(
    (
        TargetEncoder(
            target_type="continuous", cv=5
        ),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

hist_target = make_pipeline(
    target_encoder, HistGradientBoostingRegressor(random_state=42)
)
hist_target

## Gradient boosting estimator with native categorical support
hist_native = HistGradientBoostingRegressor(
    random_state=42, categorical_features="from_dtype"
)
hist_native

## Model comparison
from sklearn.model_selection import cross_validate

common_params = {"cv": 5, "scoring": "neg_mean_absolute_percentage_error", "n_jobs": -1}

dropped_result = cross_validate(hist_dropped, X, y, **common_params)
one_hot_result = cross_validate(hist_one_hot, X, y, **common_params)
ordinal_result = cross_validate(hist_ordinal, X, y, **common_params)
target_result = cross_validate(hist_target, X, y, **common_params)
native_result = cross_validate(hist_native, X, y, **common_params)
results = [
    ("Dropped", dropped_result),
    ("One Hot", one_hot_result),
    ("Ordinal", ordinal_result),
    ("Target", target_result),
    ("Native", native_result),
]

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_performance_tradeoff(results, title):
    fig, ax = plt.subplots()
    markers = ["s", "o", "^", "x", "D"]

    for idx, (name, result) in enumerate(results):
        test_error = -result["test_score"]
        mean_fit_time = np.mean(result["fit_time"])
        mean_score = np.mean(test_error)
        std_fit_time = np.std(result["fit_time"])
        std_score = np.std(test_error)

        ax.scatter(
            result["fit_time"],
            test_error,
            label=name,
            marker=markers[idx],
        )
        ax.scatter(
            mean_fit_time,
            mean_score,
            color="k",
            marker=markers[idx],
        )
        ax.errorbar(
            x=mean_fit_time,
            y=mean_score,
            yerr=std_score,
            c="k",
            capsize=2,
        )
        ax.errorbar(
            x=mean_fit_time,
            y=mean_score,
            xerr=std_fit_time,
            c="k",
            capsize=2,
        )
    ax.set_xscale("log")

    nticks = 7
    x0, x1 = np.log10(ax.get_xlim())
    ticks = np.logspace(x0, x1, nticks)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1e"))
    ax.minorticks_off()

    ax.annotate(
        " best\nmodels",
        xy=(0.04, 0.04),
        xycoords="axes fraction",
        xytext=(0.09, 0.14),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax.set_xlabel("Time to fit (seconds)")
    ax.set_ylabel("Mean Absolute Percentage Error")
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_performance_tradeoff(results, "Gradient Boosting on Ames Housing")

## Limiting the number of splits
for pipe in (hist_dropped, hist_one_hot, hist_ordinal, hist_target, hist_native):
    if pipe is hist_native:
        # The native model does not use a pipeline so, we can set the parameters
        # directly.
        pipe.set_params(max_depth=3, max_iter=15)
    else:
        pipe.set_params(
            histgradientboostingregressor__max_depth=3,
            histgradientboostingregressor__max_iter=15,
        )

dropped_result = cross_validate(hist_dropped, X, y, **common_params)
one_hot_result = cross_validate(hist_one_hot, X, y, **common_params)
ordinal_result = cross_validate(hist_ordinal, X, y, **common_params)
target_result = cross_validate(hist_target, X, y, **common_params)
native_result = cross_validate(hist_native, X, y, **common_params)
results_underfit = [
    ("Dropped", dropped_result),
    ("One Hot", one_hot_result),
    ("Ordinal", ordinal_result),
    ("Target", target_result),
    ("Native", native_result),
]
plot_performance_tradeoff(
    results_underfit, "Gradient Boosting on Ames Housing (few and shallow trees)"
)