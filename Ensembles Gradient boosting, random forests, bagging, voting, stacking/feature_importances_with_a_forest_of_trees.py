# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

## Data generation and model fitting
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Generowanie przykładowego zbioru danych
# ==========================================
X, y = make_classification(
    n_samples=1000, # liczba obserwacji (wierszy)
    n_features=10, # całkowita liczba cech (kolumn)
    n_informative=3, # 3 cechy rzeczywiście zawierają informacje do rozróżnienia klas
    n_redundant=0, # brak cech będących kombinacją innych cech
    n_repeated=0, # brak zduplikowanych cech
    n_classes=2, # problem klasyfikacji binarnej (0 lub 1)
    random_state=0, # ziarno losowości - zawsze wygeneruje te same dane
    shuffle=False, # nie miesza kolejności cech dzięki temu pierwsze 3 kolumny są informacyjne
)

# X -> macierz cech (1000 x 10)
# y -> etykiety klas (1000 wartości: 0 lub 1)

# ==========================================
# 2. Podział danych na zbiór treningowy i testowy
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.ensemble import RandomForestClassifier

feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

## Feature importance based on mean decrease in impurity
import time
import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

## Feature importance based on feature permutation
from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(F"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(result.importances_mean, index=feature_names)


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()