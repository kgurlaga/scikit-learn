# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

n_samples = 500
rng = np.random.default_rng(0)
feature_names = ["Feature #0", "Feature #1"]
common_scatter_plot_params = dict(
    cmap=ListedColormap(["tab:red", "tab:blue"]),
    edgecolor="white",
    linewidth=1,
)

xor = pd.DataFrame(
    np.random.RandomState(0).uniform(low=-1, high=1, size=(n_samples, 2)),
    columns=feature_names,
)
noise = rng.normal(loc=0, scale=0.1, size=(n_samples, 2))
target_xor = np.logical_xor(
    xor["Feature #0"] + noise[:, 0] > 0, xor["Feature #1"] + noise[:, 1] > 0
)

X = xor[feature_names]
y = target_xor.astype(np.int32)

fig, ax = plt.subplots()
ax.scatter(X["Feature #0"], X["Feature #1"], c=y, **common_scatter_plot_params)
ax.set_title("The XOR dataset")
plt.show()


from sklearn.ensemble import VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

clf1 = make_pipeline(
    SplineTransformer(degree=2, n_knots=2),
    PolynomialFeatures(interaction_only=True),
    LogisticRegression(C=10),
)
clf2 = make_pipeline(
    SplineTransformer(
        degree=2,
        n_knots=4,
        extrapolation="periodic",
        include_bias=True,
    ),
    PolynomialFeatures(interaction_only=True),
    LogisticRegression(C=10),
)
clf3 = make_pipeline(
    StandardScaler(),
    Nystroem(gamma=2, random_state=0),
    LogisticRegression(C=10),
)
weights = [2, 1, 3]
eclf = VotingClassifier(
    estimators=[
        ("constant splines model", clf1),
        ("periodic splines model", clf2),
        ("nystroem model", clf3),
    ],
    voting="soft",
    weights=weights,
)

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)


from itertools import product
from sklearn.inspection import DecisionBoundaryDisplay

fig, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
for idx, clf, title in zip(
    product([0, 1], [0, 1]),
    [clf1, clf2, clf3, eclf],
    [
        "Splines with\nconstant extrapolation",
        "Splines with\nperiodic extrapolation",
        "RBF Nystroem",
        "Soft Voting",
    ],
):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict_proba",
        plot_method="pcolormesh",
        cmap="RdBu",
        alpha=0.8,
        ax=axarr[idx[0], idx[1]],
    )
    axarr[idx[0], idx[1]].scatter(
        X["Feature #0"],
        X["Feature #1"],
        c=y,
        **common_scatter_plot_params,
    )
    axarr[idx[0], idx[1]].set_title(title)
    fig.colorbar(disp.surface_, ax=axarr[idx[0], idx[1]], label="Probability estimate")
plt.show()


test_sample = pd.DataFrame({"Feature #0": [-0.5], "Feature #1": [1.5]})
predict_probas = [est.predict_proba(test_sample).ravel() for est in eclf.estimators_]
for (est_name, _), est_probas in zip(eclf.estimators, predict_probas):
    print(f"{est_name}'s predicted probabilities: {est_probas}")

print(
    "Weighted average of soft-predictions: "
    f"{np.dot(weights, predict_probas) / np.sum(weights)}"
)

print(
    "Predicted probability of VotingClassifier: "
    f"{eclf.predict_proba(test_sample).ravel()}"
)

print(
    "Class with the highest weighted average of soft-predictions: "
    f"{np.argmax(np.dot(weights, predict_probas) / np.sum(weights))}"
)

print(f"Predicted class of VotingClassifier: {eclf.predict(test_sample).ravel()}")

from sklearn.model_selection import FixedThresholdClassifier

eclf_other_threshold = FixedThresholdClassifier(
    eclf, threshold=0.7, response_method="predict_proba"
).fit(X, y)
print(
    "Predicted class of thresholded VotingClassifier: "
    f"{eclf_other_threshold.predict(test_sample)}"
)