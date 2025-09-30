
### Common pitfalls in the interpretation of coefficients of linear models
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

# The dataset: wages
from sklearn.datasets import fetch_openml
iris = fetch_openml(name="iris", as_frame=True)
print(iris.frame.head())

from sklearn.datasets import fetch_openml
survey = fetch_openml(data_id = 534, as_frame = True)
print(survey.DESCR[:500])
print(survey.frame.head())

X = survey.data[survey.feature_names]
X.describe(include = "all")
X.head()

y = survey.target.values.ravel()
survey.target.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

train_dataset = X_train.copy()
train_dataset.insert(0, "WAGE", y_train)
_ = sns.pairplot(train_dataset, kind = "reg", diag_kind = "kde")
plt.show()

# The machine-learning pipeline
survey.data.info()

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

preprocessor = make_column_transformer(
    (OneHotEncoder(drop = "if_binary"), categorical_columns),
    remainder = "passthrough",
    verbose_feature_names_out = False,
)

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    preprocessor,
    TransformedTargetRegressor(
        regressor = Ridge(alpha = 1e-10), func = np.log10, inverse_func = sp.special.exp10,
    )
)

# Processing the dataset
model.fit(X_train, y_train)

from sklearn.metrics import PredictionErrorDisplay, median_absolute_error

mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MadAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize = (5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind = "actual_vs_predicted", ax = ax, scatter_kwargs = {"alpha": 0.5}
)
ax.set_title("Ridge model, small regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label = f"{name}: {score}")
ax.legend(loc = "upper left")
plt.tight_layout()
plt.show()

# Interpreting coefficients: scale matters
feature_names = model[:-1].get_feature_names_out()

coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns = ["Coeffcients"],
    index = feature_names,
)

coefs.plot.barh(figsize = (9, 7))
plt.title("Ridge model, small regularization")
plt.axvline(x = 0, color = ".5")
plt.xlabel("Raw coefficient values")
plt.subplots_adjust(left = 0.3)
plt.show()

X_train_preprocessed = pd.DataFrame(
    model[:-1].transform(X_train), columns = feature_names
)

X_train_preprocessed.std(axis = 0).plot.barh(figsize = (9, 7))
plt.title("Feature ranges")
plt.xlabel("Std. dev. of feature values")
plt.subplots_adjust(left = 0.3)
plt.show()

coefs = pd.DataFrame(
    model[-1].regressor_.coef_ * X_train_preprocessed.std(axis = 0),
    columns = ["Coefficient importance"],
    index = feature_names,
)
coefs.plot(kind = "barh", figsize = (9, 7))
plt.xlabel("Coefficient values corrected by the feature's std. dev.")
plt.title("Ridge model, small regularization")
plt.axvline(x = 0, color = ".5")
plt.subplots_adjust(left = 0.3)
plt.show()

# Checking the variability of the coefficients
from sklearn.model_selection import RepeatedKFold, cross_validate

cv = RepeatedKFold(n_splits = 5, n_repeats = 5, random_state = 0)
cv_model = cross_validate(
    model, X, y, cv = cv, return_estimator = True, n_jobs= 2,
)

coefs = pd.DataFrame(
    [
        est[-1].regressor_.coef_ * est[:-1].transform(X.iloc[train_idx]).std(axis = 0)
        for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
    ],
    columns = feature_names,
)

plt.figure(figsize = (9, 7))
sns.stripplot(data = coefs, orient = "h", palette = "dark:k", alpha = 0.5)
sns.boxplot(data =coefs, orient = "h", color = "cyan", saturation = 0.5, whis = 10)
plt.axvline(x = 0, color = ".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.suptitle("Ridge model, small regularization")
plt.subplots_adjust(left = 0.3)
plt.show()

# The problem of correlated variables
plt.ylabel("Age coefficient")
plt.xlabel("Experience coefficient")
plt.grid(True)
plt.xlim(-0.4, 0.5)
plt.ylim(-0.4, 0.5)
plt.scatter(coefs["AGE"], coefs["EXPERIENCE"])
_ = plt.title("Co-variations of coefficients for AGE and EXPERIENCE across folds")
plt.show()


column_to_drop = ["AGE"]
cv_model = cross_validate(
    model, X.drop(columns = column_to_drop),
    y,
    cv = cv,
    return_estimator = True,
    n_jobs = 2,
)

coefs = pd.DataFrame(
    [
    est[-1].regressor_.coef_ * est[:-1].transform(X.drop(columns = column_to_drop).iloc[train_idx]).std(axis = 0)
    for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
    ],
    columns = feature_names[:-1],
)

plt.figure(figsize = (9, 7))
sns.stripplot(data = coefs, orient = "h", palette = "dark:k", alpha = 0.5)
sns.boxplot(data = coefs, orient = "h", color = "cyan", saturation = 0.5)
plt.axvline(x = 0, color = ".5")
plt.title("Coefficient importance and its variability")
plt.xlabel("Coefficient importance")
plt.suptitle("Ridge model, small regularization, AGE dropped")
plt.subplots_adjust(left = 0.3)
plt.show()

# Preprocessing numerical variables
from sklearn.preprocessing import StandardScaler
preprocessor = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), categorical_columns),
    (StandardScaler(), numerical_columns),
)

model = make_pipeline(
    preprocessor, TransformedTargetRegressor(regressor = Ridge(alpha = 1e-10), func = np.log10, inverse_func = sp.special.exp10),
)
model.fit(X_train, y_train)

mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize = (5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind = "actual_vs_predicted", ax = ax, scatter_kwargs = {"alpha": 0.5}
)
ax.set_title("Ridge model, small regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label = f"{name}: {score}")
ax.legend(loc = "upper left")
plt.tight_layout()
plt.show()

coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns = ["Coefficients importance"],
    index = feature_names,
)
coefs.plot.barh(figsize = (9, 7))
plt.title("Ridge model, small regularization, normalized variables")
plt.xlabel("Raw coefficien values")
plt.axvline(x = 0, color = ".5")
plt.subplots_adjust(left = 0.3)
plt.show()

cv_model = cross_validate(
    model, X, y, cv = cv, return_estimator = True, n_jobs = 2,
)
coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in cv_model["estimator"]], columns = feature_names
)

plt.figure(figsize = (9, 7))
sns.stripplot(data = coefs, orient = "h", palette = "dark:k", alpha = 0.5)
sns.boxplot(data = coefs, orient = "h", color = "cyan", saturation = 0.5, whis = 10)
plt.axvline(x = 0, color = ".5")
plt.title("Coefficient variability")
plt.subplots_adjust(left = 0.3)
plt.show()

# Linear models with regularization
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-10, 10, 21)
model = make_pipeline(
    preprocessor, TransformedTargetRegressor(
        regressor = RidgeCV(alphas = alphas), func = np.log10, inverse_func = sp.special.exp10,
    ),
)
model.fit(X_train, y_train)

model[-1].regressor_.alpha_

mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize = (5, 5))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind = "actual_vs_predicted", ax = ax, scatter_kwargs = {"alpha": 0.5}
)
ax.set_title("Ridge model, optimum regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label = f"{name}: {score}")
ax.legend(loc = "upper left")
plt.tight_layout()
plt.show()

coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns = ["Coefficients importance"],
    index = feature_names,
)
coefs.plot.barh(figsize = (9, 7))
plt.title("Ridge model, with regularization, normalized variables")
plt.xlabel("Raw coefficients values")
plt.axvline(x = 0, color = ".5")
plt.subplots_adjust(left = 0.3)
plt.show()


cv_model = cross_validate(
    model, X, y, cv = cv, return_estimator = True, n_jobs = 2,
)
coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in cv_model["estimator"]], columns = feature_names
)

plt.ylabel("Age coefficient")
plt.xlabel("Experience coefficient")
plt.grid(True)
plt.xlim(-0.4, 0.5)
plt.ylim(-0.4, 0.5)
plt.scatter(coefs["AGE"], coefs["EXPERIENCE"])
_ = plt.title("Co-variations of coefficients for AGE and EXPERIENCE across folds")
plt.show()

# Linear models with sparse coefficients
from sklearn.linear_model import LassoCV

alphas = np.logspace(-10, 10, 21)
model = make_pipeline(
    preprocessor, TransformedTargetRegressor(
        regressor = LassoCV(alphas = alphas, max_iter = 100_000),
        func = np.log10,
        inverse_func = sp.special.exp10,
    ),
)

_ = model.fit(X_train, y_train)

model[-1].regressor_.alpha_


mae_train = median_absolute_error(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
mae_test = median_absolute_error(y_test, y_pred)
scores = {
    "MedAE on training set": f"{mae_train:.2f} $/hour",
    "MedAE on testing set": f"{mae_test:.2f} $/hour",
}

_, ax = plt.subplots(figsize = (6, 6))
display = PredictionErrorDisplay.from_predictions(
    y_test, y_pred, kind = "actual_vs_predicted", ax = ax, scatter_kwargs = {"alpha": 0.5}
)
ax.set_title("Lasso model, optimum regularization")
for name, score in scores.items():
    ax.plot([], [], " ", label = f"{name}: {score}")
ax.legend(loc = "upper left")
plt.tight_layout()
plt.show()


coefs = pd.DataFrame(
    model[-1].regressor_.coef_,
    columns = ["Coefficients importance"],
    index = feature_names,
)
coefs.plot(kind = "barh", figsize = (9, 7))
plt.title("Lasso model, optimum regularization, normalized variables")
plt.axvline(x = 0, color = ".5")
plt.subplots_adjust(left = 0.3)
plt.show()

cv_model = cross_validate(
    model, X, y, cv = cv, return_estimator = True, n_jobs = 2,
)
coefs = pd.DataFrame(
    [est[-1].regressor_.coef_ for est in  cv_model["estimator"]], columns = feature_names
)

plt.figure(figsize = (9, 7))
sns.stripplot(data = coefs, orient = "h", palette = "dark:k", alpha = 0.5)
sns.boxplot(data = coefs, orient = "h", color = "cyan", saturation = 0.5, whis = 100)
plt.axvline(x = 0, color = ".5")
plt.title("Coefficient variability")
plt.subplots_adjust(left = 0.3)
plt.show()