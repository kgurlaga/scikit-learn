# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import OAS
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# liczba próbek treningowych i testowych
n_train = 20
n_test = 200

# ile razy uśredniamy wynik (redukcja losowości)
n_averages = 50

# maksymalna liczba cech
n_features_max = 75
step = 4

def generate_data(n_samples, n_features):
    """
    Generate random blob-ish data with noisy features.
    This returns a array of input data with shape `(n_samples, n_features)`
    and an array of 'n_samples' target labels.
    Only one feature contains discriminative information, the other features contain only noise.
    """

    # dane z 2 klasami oddzielonymi w 1D
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # dokładamy cechy szumu
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

# listy na accuracy
acc_clf1, acc_clf2, acc_clf3 = [], [], []

# zakres liczby cech
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2, score_clf3 = 0, 0, 0

    # uśrednianie wyników
    for _ in range(n_averages):

        # dane treningowe
        X, y = generate_data(n_train, n_features)
        
        # zwykłe LDA
        clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None).fit(X, y)

        # LDA + Ledoit-Wolf
        clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)

        # LDA + OAS
        oa = OAS(store_precision=False, assume_centered=False)
        clf3 = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa).fit(X, y)

        # dane testowe
        X, y = generate_data(n_test, n_features)

        # zbieramy accuracy
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)
        score_clf3 += clf3.score(X, y)
    
    # średnia accuracy
    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)
    acc_clf3.append(score_clf3 / n_averages)

# stosunek cech do próbek
features_samples_ratio = np.array(n_features_range) / n_train

# rysowanie wykresu
plt.plot(
    features_samples_ratio,
    acc_clf1,
    linewidth=2,
    label="LDA",
    color="gold",
    linestyle="solid",
)
plt.plot(
    features_samples_ratio,
    acc_clf2,
    linewidth=2,
    label="LDA with Ledoit Wolf",
    color="navy",
    linestyle="dashed",
)
plt.plot(
    features_samples_ratio,
    acc_clf3,
    linewidth=2,
    label="LDA with OAS",
    color="red",
    linestyle="dotted",
)

plt.xlabel("n_features / n_samples")
plt.ylabel("Classification accuracy")

plt.legend(loc="lower left")
plt.ylim((0.65, 1.0))
plt.suptitle(
    "LDA (Linear Discriminant Analysis) vs."
    "\n"
    "LDA with Ledoit Wolf vs."
    "\n"
    "LDA with OAS (1 discriminative feature)"
)
plt.show()