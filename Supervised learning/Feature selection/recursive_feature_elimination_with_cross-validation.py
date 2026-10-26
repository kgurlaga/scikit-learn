# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Data generation
from sklearn.datasets import make_classification
n_features = 15
feat_names = [f"feature_{i}" for i in range(15)]

X, y = make_classification(
    n_samples=500,
    n_features=n_features,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=0,
)

## Model training and selection
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1 # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)
print(f"Optimal number of features: {rfecv.n_features_}")

## Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
import pandas as pd

data = {
    key: value
    for key, value in rfecv.cv_results_.items()
    if key in ["n_features", "mean_test_score", "std_test_score"]
}
cv_results = pd.DataFrame(data)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x = cv_results["n_features"],
    y = cv_results["mean_test_score"],
    yerr = cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()

import numpy as np

for i in range(cv.n_splits):
    mask = rfecv.cv_results_[f"split{i}_support"][
        rfecv.n_features_ - 1
    ] # mask of features selected by the RFE
    features_selected = np.ma.compressed(np.ma.masked_array(feat_names, mask=1 - mask))
    print(f"Features selected in fold {i}: {features_selected}")