# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

### LOAD DATA
import numpy as np
from sklearn.datasets import load_iris

# Wczytujemy klasyczny zbiór Iris (150 próbek, 4 sensowne cechy)
X, y = load_iris(return_X_y=True)

# Generator liczb losowych (powtarzalność)
rng = np.random.RandomState(0)

# Doklejamy 36 losowych (nieinformatywnych) cech
# → razem 40 cech, z czego tylko 4 niosą informację
X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))

### CREATE THE PIPELINE
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create a feature-selection transform, a scaler and an instance of SVM that we combine together to have a full-blown estimator

# Pipeline = sekwencja operacji wykonywanych w poprawnej kolejności:
# 1. wybór najlepszych cech (ANOVA F-test)
# 2. standaryzacja cech
# 3. klasyfikacja SVM
clf = Pipeline(
    [
        ("anova", SelectPercentile(f_classif)),
        ("scaler", StandardScaler()),
        ("svc", SVC(gamma="auto")),
    ]
)

### PLOT THE CROSS-VALIDATION SCORE AS A FUNCTION OF PERCENTILE OF FEATURES
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

score_means = list()
score_stds = list()

# Różne procenty cech, które testujemy
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:

    # Ustawiamy, ile % najlepszych cech ma zostać wybranych
    clf.set_params(anova__percentile=percentile)

    # Walidacja krzyżowa (domyślnie 5-fold)
    # Każdy fold:
    # - wybiera cechy
    # - skaluje
    # - trenuje SVM
    this_scores = cross_val_score(clf, X, y)

    # Średnia skuteczność i odchylenie
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title("Performance of the SVM-Anova varying the percentile of features selected")
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel("Percentile")
plt.ylabel("Accuracy Score")
plt.axis("tight")
plt.show()