# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

# liczba próbek w każdej klasie (celowo niezbalansowane)
n_samples_1 = 1000
n_samples_2 = 100

# środki klastrów (gdzie są chmury punktów)
centers = [[0.0, 0.0], [2.0, 2.0]]

# rozrzut punktów w każdej klasie
clusters_std = [1.5, 0.5]

# generowanie sztucznych danych
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)

# -------- SVM BEZ WAG KLAS --------
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# -------- SVM Z WAGAMI KLAS --------
# klasa 1 jest 10 razy ważniejsza niż klasa 0
wclf = svm.SVC(kernel="linear", class_weight={1: 10})
wclf.fit(X, y)

# rysowanie punktów danych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# plot the decision functions for both classifiers
ax = plt.gca()

# granica decyzyjna dla SVM bez wag (czarna linia)
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

# granica decyzyjna dla SVM z wagami (czerwona linia)
wdisp = DecisionBoundaryDisplay.from_estimator(
    wclf,
    X,
    plot_method="contour",
    colors="r",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

# legenda tłumacząca linie
plt.legend(
    [
        mlines.Line2D([], [], color="k", label="non weighted"),
        mlines.Line2D([], [], color="r", label="weighted"),
    ],
    ["non weighted", "weighted"],
    loc="upper right",
)
plt.show()