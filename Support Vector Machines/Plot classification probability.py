# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, PolynomialFeatures, SplineTransformer,)

####################################################
## DATA 2D PROJECTION OF THE IRIS DATASET

# ===============================
# 1. Wczytanie danych Iris
# ===============================
iris = datasets.load_iris()

# Wybieramy tylko 2 pierwsze cechy (sepal length, sepal width),
# aby możliwa była wizualizacja w 2D
X = iris.data[:, 0:2]
y = iris.target

# ===============================
# 2. Podział na zbiór treningowy i testowy
# ===============================
# 50% danych do testu
# random_state zapewnia powtarzalność wyników
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

####################################################
## PROBABLISTIC CLASSIFIERS
# Wszystkie klasyfikatory muszą obsługiwać predict_proba(),
# ponieważ będziemy wizualizować prawdopodobieństwa klas
classifiers = {

    # Regresja logistyczna z silną regularyzacją (małe C)
    "Logistic regression\n(C=0.01)": LogisticRegression(C=0.1),

    # Regresja logistyczna z bardzo słabą regularyzacją (duże C)
    "Logistic regression\n(C=1)": LogisticRegression(C=100),

    # Klasyfikator procesów Gaussowskich z kernelem RBF
    # Model silnie nieliniowy
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),

    # Regresja logistyczna + cechy RBF (aproksymacja kernel trick)
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),

    # Gradient Boosting – model drzewiasty, nieliniowy
    "Gradient Boosting": HistGradientBoostingClassifier(),

    # Regresja logistyczna na cechach zdyskretyzowanych (binning)
    # + interakcje między cechami
    "Logistic regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),

    # Regresja logistyczna na cechach spline’owych
    # + interakcje (gładsza nieliniowość)
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}

####################################################
## PLOTTING THE DECISION BOUNDARIES
n_classifiers = len(classifiers)

# Parametry rysowania punktów testowych
scatter_kwargs = {
    "s": 25,
    "marker": "o",
    "linewidth": 0.8,
    "edgecolor": "k",
    "alpha": 0.7,
}

# Unikalne klasy (0, 1, 2)
y_unique = np.unique(y)

# Zapobiega obcinaniu legend i colorbarów
mpl.rcParams["savefig.bbox"] = "tight"

# Układ wykresów:
# - wiersze: klasyfikatory
# - kolumny: klasy + kolumna "Max class"
fig, axes = plt.subplots(
    nrows=n_classifiers,
    ncols=len(iris.target_names) + 1,
    figsize=(4 * 2.2, n_classifiers * 2.2),
)

# Lista do zapisu wyników ewaluacji
evaluation_results = []

# Liczba poziomów konturu dla map prawdopodobieństw
levels = 100

####################################################
## 5. TRENOWANIE, EWALUACJA I RYSOWANIE
####################################################
for classifier_idx, (name, classifier) in enumerate(classifiers.items()):

    # Predykcja klas (twarda)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Predykcja prawdopodobieństw klas
    y_pred_proba = classifier.predict_proba(X_test)

    # Metryki jakości
    accuracy_test = accuracy_score(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    log_loss_test = log_loss(y_test, y_pred_proba)

    # Zapis wyników
    evaluation_results.append(
        {
            "name": name.replace("\n", " "),
            "accuracy": accuracy_test,
            "roc_auc": log_loss_test,
        }
    )

    # ===============================
    # Rysowanie map prawdopodobieństw
    # ===============================
    for label in y_unique:

        # Rysowanie powierzchni prawdopodobieństwa dla danej klasy
        disp = DecisionBoundaryDisplay.from_estimator(
            classifier,
            X_train,
            response_method="predict_proba",
            class_of_interest=label,
            ax=axes[classifier_idx, label],
            vmin=0,
            vmax=1,
            cmap="Blues",
            levels=levels,
        )
        axes[classifier_idx, label].set_title(f"Class {label}")

        # Punkty testowe, które model przypisał do tej klasy
        mask_y_pred = y_pred == label
        axes[classifier_idx, label].scatter(
            X_test[mask_y_pred, 0], X_test[mask_y_pred, 1], c="w", **scatter_kwargs
        )
        axes[classifier_idx, label].set(xticks=(), yticks=())
        
        # ===============================
        # Kolumna "Max class"
        # ===============================
        # Pokazuje klasę o największym prawdopodobieństwie
        max_class_disp = DecisionBoundaryDisplay.from_estimator(
            classifier,
            X_train,
            response_method="predict_proba",
            class_of_interest=None,
            ax=axes[classifier_idx, len(y_unique)],
            vmin=0,
            vmax=1,
            levels=levels,
        )

        # Rysowanie punktów testowych w kolorze przewidzianej klasy
        for label in y_unique:
            mask_label = y_test == label
            axes[classifier_idx, 3].scatter(
                X_test[mask_label, 0],
                X_test[mask_label, 1],
                c=max_class_disp.multiclass_colors_[[label], :],
                **scatter_kwargs,
            )
        axes[classifier_idx, 3].set(xticks=(), yticks=())
        axes[classifier_idx, 3].set_title("Max class")
        axes[classifier_idx, 0].set_ylabel(name)
    
    # ===============================
    # Colorbar dla pojedynczych klas
    # ===============================
    ax_single = fig.add_axes([0.15, 0.01, 0.5, 0.02])
    plt.title("Probability")
    _ = plt.colorbar(
        cm.ScalarMappable(norm=None, cmap=disp.surface_.cmap),
        cax=ax_single,
        orientation="horizontal",
    )

    # ===============================
    # Colorbary dla kolumny "Max class"
    # ===============================
    max_class_cmaps = [s.cmap for s in max_class_disp.surface_]

    for label in y_unique:
        ax_max = fig.add_axes([0.73, (0.06 - (label * 0.04)), 0.16, 0.015])
        plt.title(f"Probability class {label}", fontsize=10)
        _ = plt.colorbar(
            cm.ScalarMappable(norm=None, cmap=max_class_cmaps[label]),
            cax=ax_max,
            orientation="horizontal",
        )
        if label in (0, 1):
            ax_max.set(xticks=(), yticks=())
    
# Wyświetlenie wszystkich wykresów
plt.show()

# Tabela z wynikami ewaluacji modeli
pd.DataFrame(evaluation_results).round(2)