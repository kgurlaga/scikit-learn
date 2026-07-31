# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =========================
# 1. Wczytanie danych IRIS
# =========================
iris = datasets.load_iris()

# X – macierz cech:
# 4 kolumny = [sepal length, sepal width, petal length, petal width]
# każdy wiersz = jeden kwiat
X = iris.data

# y – etykiety klas:
# 0 = Setosa, 1 = Versicolour, 2 = Virginica
y = iris.target

# Nazwy klas (do legendy na wykresach)
target_names = iris.target_names

# =========================
# 2. PCA – redukcja wymiaru
# =========================
# PCA bez użycia etykiet klas (metoda niesuperwizyjna)
# Redukujemy z 4 cech do 2 nowych osi (PC1, PC2)
pca = PCA(n_components=2)

# fit -> uczy się, w jakim kierunku dane mają największą wariancję
# transform -> rzutuje dane na nowe osie
X_r = pca.fit(X).transform(X)

# =========================
# 3. LDA – redukcja wymiaru
# =========================
# LDA korzysta z etykiet klas (metoda superwizyjna)
# Szuka osi, które najlepiej ROZDZIELAJĄ klasy
lda = LinearDiscriminantAnalysis(n_components=2)

# fit -> uczy się, jak najlepiej rozdzielić klasy
# transform -> rzutuje dane na osie maksymalnej separacji
X_r2 = lda.fit(X, y).transform(X)

# =========================
# 4. Informacja diagnostyczna PCA
# =========================
# Ile wariancji danych tłumaczą pierwsze dwie składowe PCA
# (np. 0.92 oznacza 92% informacji z oryginalnych danych)
print(
    "explained variance ratio (first two components):  %s"
    % str(pca.explained_variance_ratio_)
)

# =========================
# 5. Wykres PCA
# =========================
plt.figure()

# Kolory dla trzech klas irysa
colors = ["navy", "turquoise", "darkorange"]
lw = 2

# Rysujemy punkty każdej klasy osobno
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0],    # współrzędna PC1
        X_r[y == i, 1],    # współrzędna PC2 
        color=color, alpha=0.8, lw=lw, label=target_name,
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")
# Ten wykres pokazuje:
# „Jak dane wyglądają po rzutowaniu na kierunki największej wariancji”


# =========================
# 6. Wykres LDA
# =========================
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0],  # pierwsza oś rozdzielająca klasy (LD1)
        X_r2[y == i, 1],  # druga oś rozdzielająca klasy (LD2)
        alpha=0.8, color=color, label=target_name,
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

# Ten wykres pokazuje:
# „Jak dane wyglądają po rzutowaniu na osie maksymalnej separacji klas”
plt.show()