# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

####################################################
## GENERATE SAMPLE DATA
import numpy as np

# Generator liczb losowych z ustalonym ziarnem (powtarzalność wyników)
rng = np.random.RandomState(42)

# Dane wejściowe: 10 000 punktów losowych z zakresu [0, 5)
# Jedna cecha (kształt: [n_samples, 1])
X = 5 * rng.rand(10000, 1)

# Prawdziwa zależność: funkcja sinus
# ravel() zamienia (10000, 1) -> (10000,)
y = np.sin(X).ravel()

# Dodanie silnego szumu tylko do co 5-tego punktu
# Symulacja błędnych pomiarów / outlierów
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

# Gęsta siatka punktów do rysowania gładkiej krzywej predykcji
# Nie służy do trenowania modelu
X_plot = np.linspace(0, 5, 100000)[:, None]

####################################################
## CONSTRUCT THE KERNEL-BASED REGRESSION MODELS
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Liczba próbek treningowych (celowo mała)
train_size = 100

# ======================
# Grid Search dla SVR
# ======================
svr = GridSearchCV(
    # Model SVR z jądrem RBF (nieliniowa regresja)
    SVR(kernel="rbf", gamma=0.1),
    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
)

# ======================
# Grid Search dla Kernel Ridge Regression
# ======================
kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
)


####################################################
## COMPARE TIMES OF SVR AND KERNEL RIDGE REGRESSION
import time

t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score: {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))

####################################################
## LOOK AT THE RESULTS
import matplotlib.pyplot as plt

# Indeksy punktów, które SVR uznał za support vectors
# (czyli tych, które faktycznie wpływają na kształt modelu)
sv_ind = svr.best_estimator_.support_

# Rysujemy support vectors jako czerwone, większe punkty
plt.scatter(
    X[sv_ind],                      # współrzędne X support vectors
    y[sv_ind],                      # odpowiadające im wartości y
    c="r",                          # kolor czerwony
    s=50,                           # większy rozmiar punktów
    label="SVR support vectors",
    zorder=2,                       # na wierzchu wykresu
    edgecolors=(0, 0, 0),           # czarna obwódka punktów
)

# Rysujemy dane treningowe (pierwsze 100 punktów)
# jako małe czarne kropki
plt.scatter(X[:100], y[:100], c="k", label="data", zorder=1, edgecolors=(0, 0, 0))

# Rysujemy krzywą nauczoną przez SVR
# y_svr to predykcje modelu SVR dla gęstej siatki X_plot
plt.plot(
    X_plot,
    y_svr,
    c="r",
    label="SVR (fit: %.3fs, predict: %.3fs)" % (svr_fit, svr_predict),
)

# Rysujemy krzywą nauczoną przez Kernel Ridge Regression
# y_kr to predykcje KRR dla tych samych punktów X_plot
plt.plot(
    X_plot, y_kr, c="g", label="KRR (fit: %.3fs, predict: %.3fs)" % (kr_fit, kr_predict)
)

# Opisy osi
plt.xlabel("data")
plt.ylabel("target")

# Tytuł wykresu
plt.title("SVR versus Kernel Ridge")

# Dodanie legendy
_ = plt.legend()
plt.show()

####################################################
## VISUALIZE TRAINING AND PREDICTION TIMES
plt.figure()

# Różne rozmiary zbioru treningowego (od ~10 do ~6000 próbek)
# Skala logarytmiczna, żeby zobaczyć trend dla małych i dużych danych
sizes = np.logspace(1, 3.8, 7).astype(int)

# Porównujemy dwa algorytmy:
# - Kernel Ridge Regression
# - Support Vector Regression
for name, estimator in {
    "KRR": KernelRidge(kernel="rbf", alpha=0.01, gamma=10),
    "SVR": SVR(kernel="rbf", C=1e2, gamma=10),
}.items():
    train_time = []     # czasy uczenia
    test_time = []      # czasy predykcji

    # Dla każdego rozmiaru zbioru treningowego
    for train_test_size in sizes:

        # Pomiar czasu uczenia (fit)
        t0 = time.time()
        estimator.fit(X[:train_test_size], y[:train_test_size])
        train_time.append(time.time() - t0)

        # Pomiar czasu predykcji (predict)
        # Predykcja na stałej liczbie punktów (1000),
        # żeby porównanie było uczciwe
        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)

    # Rysujemy czas uczenia (linia ciągła)
    plt.plot(
        sizes,
        train_time,
        "o-",
        color="r" if name == "SVR" else "g",
        label="%s (train)" % name,
    )

    # Rysujemy czas predykcji (linia przerywana)
    plt.plot(
        sizes,
        test_time,
        "o--",
        color="r" if name == "SVR" else "g",
        label="%s (test)" % name,
    )

# Skale logarytmiczne – czasy i rozmiary rosną wykładniczo
plt.xscale("log")
plt.yscale("log")

# Opisy osi
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")

# Tytuł wykresu
plt.title("Execution Time")

# Legenda
_ = plt.legend(loc="best")
plt.show()


####################################################
## VISUALIZE THE LEARNING CURVES
from sklearn.model_selection import LearningCurveDisplay

# Tworzymy figurę i oś do rysowania wykresu
_, ax = plt.subplots()

# Definicja modeli, które chcemy porównać
svr = SVR(kernel="rbf", C=1e1, gamma=0.1)
kr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)

# Wspólne parametry dla obu modeli – żeby porównanie było uczciwe
common_params = {
    "X": X[:100],                           # używamy tylko 100 próbek
    "y": y[:100],
    "train_sizes": np.linspace(0.1, 1, 10), # 10 punktów: od 10% do 100% danych treningowych
    "scoring": "neg_mean_squared_error",    # sklearn maksymalizuje score → MSE jest ujemne
    "negate_score": True,                   # odwracamy znak, żeby dostać klasyczne MSE (> 0)
    "score_name": "Mean Squared Error",
    "score_type": "test",                   # pokazujemy błąd na zbiorze testowym
    "std_display_style": None,              # nie rysujemy odchylenia standardowego
    "ax": ax,                               # oba wykresy rysujemy na tej samej osi
}

# Rysujemy krzywą uczenia dla SVR
LearningCurveDisplay.from_estimator(svr, **common_params)

# Rysujemy krzywą uczenia dla KRR
LearningCurveDisplay.from_estimator(kr, **common_params)

# Tytuł wykresu
ax.set_title("Learning curves")

# Ręcznie ustawiamy legendę (kolejność modeli)
ax.legend(handles=ax.get_legend_handles_labels()[0], labels=["SVR", "KRR"])
plt.show()