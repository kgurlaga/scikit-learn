# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


####################################################
## DATA GENERATION
import numpy as np
def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):
    # Ustawiamy generator liczb losowych, żeby wyniki były powtarzalne
    rng = np.random.RandomState(seed)

    # Tworzymy dane dla dwóch klas
    X = np.concatenate(
        [
            # ===== KLASA 0 =====
            # rng.randn(...) -> losowe punkty z rozkładu normalnego
            # @ cov_class_1  -> "rozciąga i obraca" chmurę punktów
            #                  (nadaje jej określoną kowariancję)
            rng.randn(n_samples, n_features) @ cov_class_1,

            # ===== KLASA 1 =====
            # Analogicznie, ale z inną kowariancją
            # + np.array([1, 1]) -> przesunięcie całej chmury w prawo i w górę,
            #                        żeby klasy nie leżały w tym samym miejscu
            rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1])
        ]
    )
    # Tworzymy etykiety klas:
    # 0 dla pierwszej chmury, 1 dla drugiej
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    # Zwracamy dane wejściowe X oraz etykiety y
    return X, y

####################################################

# ==========================
# 1. Dane z isotropową (sferyczną) kowariancją
# ==========================
# Obie klasy mają tę samą macierz kowariancji, która jest jednostkowa (okrągłe chmury)
covariance = np.array([[1, 0], [0, 1]])
X_isotropic_covariance, y_isotropic_covariance = make_data(
    n_samples=1_000,
    n_features=2,
    cov_class_1=covariance,
    cov_class_2=covariance,
    seed=0,
)

# ==========================
# 2. Dane z tą samą kowariancją, ale nie sferyczną
# ==========================
# Obie klasy mają tę samą macierz kowariancji, ale chmura jest „pochylona”/rozciągnięta
covariance = np.array([[0.0, -0.23], [0.83, 0.23]])
X_shared_covariance, y_shared_covariance = make_data(
    n_samples=300,
    n_features=2,
    cov_class_1=covariance,
    cov_class_2=covariance,
    seed=0,
)

# ==========================
# 3. Dane z różnymi kowariancjami dla każdej klasy
# ==========================
# Pierwsza klasa: nieregularna, „pochylona” elipsa
cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0

# Druga klasa: transpozycja pierwszej, czyli elipsa w innym kierunku
cov_class_2 = cov_class_1.T
X_different_covariance, y_different_covariance = make_data(
    n_samples=300,
    n_features=2,
    cov_class_1=cov_class_1,
    cov_class_2=cov_class_2,
    seed=0,
)


####################################################
## PLOTTING FUNCTIONS
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import colors

def plot_ellipse(mean, cov, color, ax):
    # Wyznaczenie osi elipsy z macierzy kowariancji
    # v = wartości własne (rozrzut wzdłuż osi głównych), w = wektory własne (kierunki osi)
    v, w = np.linalg.eigh(cov)

    # Wektor jednostkowy w kierunku głównej osi elipsy
    u = w[0] / np.linalg.norm(w[0])

    # Kąt obrotu elipsy w radianach, potem konwersja na stopnie
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # konwersja do stopni

    # Tworzymy elipsę obejmującą 2 odchylenia standardowe (~95% punktów dla Gaussa)
    ell = mpl.patches.Ellipse(
        mean,                  # środek elipsy = średnia klasy
        2 * v[0] ** 0.5,       # szerokość elipsy
        2 * v[1] ** 0.5,       # wysokość elipsy
        angle=180 + angle,     # obrót elipsy
        facecolor=color,       # kolor wypełnienia
        edgecolor="black",     # kolor krawędzi
        linewidth=2,
    )
    ell.set_clip_box(ax.bbox)  # przycinamy elipsę do obszaru wykresu
    ell.set_alpha(0.4)         # przezroczystość elipsy
    ax.add_artist(ell)         # dodanie elipsy do wykresu

def plot_result(estimator, X, y, ax):
    cmap = colors.ListedColormap(["tab:red", "tab:blue"])

    # Granica decyzyjna: kolorowe tło z prawdopodobieństwem klasy
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="pcolormesh",
        ax=ax,
        cmap="RdBu",
        alpha=0.3,
    )
    # Granica decyzyjna jako kontur
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="contour",
        ax=ax,
        alpha=1.0,
        levels=[0.5],
    )

    # Rozdzielenie poprawnie i błędnie sklasyfikowanych punktów
    y_pred = estimator.predict(X)
    X_right, y_right = X[y == y_pred], y[y == y_pred]
    X_wrong, y_wrong = X[y != y_pred], y[y != y_pred]

    ax.scatter(X_right[:, 0], X_right[:, 1], c=y_right, s=20, cmap=cmap, alpha=0.5)
    ax.scatter(X_wrong[:, 0], X_wrong[:, 1], c=y_wrong, s=30, cmap=cmap, alpha=0.9, marker="x")

    # Środki klas
    ax.scatter(estimator.means_[:, 0], estimator.means_[:, 1], c="yellow", s=200,
               marker="*", edgecolor="black")

    # Dobranie kowariancji
    covariance = None
    if isinstance(estimator, LinearDiscriminantAnalysis):
        covariance = [estimator.covariance_] * 2
    else:
        # QDA: jeśli istnieje atrybut covariances_, użyj go, inaczej pomiń elipsy
        if hasattr(estimator, "covariances_"):
            covariance = estimator.covariances_

    # Rysowanie elips (tylko jeśli kowariancje istnieją)
    if covariance is not None:
        plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
        plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)

    # Estetyka wykresu
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xticks=[], yticks=[])


####################################################
## COMPARISON OF LDA AND QDA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

fig, axs = plt.subplots(nrows=3, ncols=2, sharex="row", sharey="row", figsize=(8, 12))

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
qda = QuadraticDiscriminantAnalysis()

for ax_row, X, y in zip(
    axs,
    (X_isotropic_covariance, X_shared_covariance, X_different_covariance),
    (y_isotropic_covariance, y_shared_covariance, y_different_covariance),
):
    lda.fit(X, y)
    plot_result(lda, X, y, ax_row[0])
    qda.fit(X, y)
    plot_result(qda, X, y, ax_row[1])

axs[0, 0].set_title("Linear Discriminant Analysis")
axs[0, 0].set_ylabel("Data with fixed and spherical covariance")
axs[1, 0].set_ylabel("Data with fixed covariance")
axs[0, 1].set_title("Quadratic Discriminant Analysis")
axs[2, 0].set_ylabel("Data with varying covariances")
fig.suptitle(
    "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
    y=0.94,
    fontsize=15,
)
plt.show()