# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# Wczytanie zbioru danych Iris
iris = datasets.load_iris()

# Wybieramy tylko dwie pierwsze cechy (żeby móc rysować w 2D)
X = iris.data[:, :2]
y = iris.target

# Parametr regularyzacji
C = 1.0

# Definicja czterech różnych modeli SVM
models = (
    svm.SVC(kernel="linear", C=C),                          # SVM liniowy (kernel)
    svm.LinearSVC(C=C, max_iter=10000),                     # Szybki liniowy SVM
    svm.SVC(kernel="rbf", gamma=0.7, C=C),                  # SVM z kernelem RBF
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),    # Kernel wielomianowy
)

# Trenowanie wszystkich modeli
models = (clf.fit(X, y) for clf in models)

# Tytuły wykresów
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Przygotowanie siatki 2x2 wykresów
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Rozdzielenie cech do rysowania punktów
X0, X1 = X[:, 0], X[:, 1]

# Pętla: rysujemy granicę decyzyjną dla każdego modelu
for clf, title, ax in zip(models, titles, sub.flatten()):
    # Rysowanie obszarów decyzyjnych modelu
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )

    # Rysowanie punktów danych
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

    # Uproszczenie wykresu
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()