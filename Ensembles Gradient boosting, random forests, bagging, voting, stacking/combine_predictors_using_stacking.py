# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

## Generate data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)
X = rng.uniform(-3, 3, size=500)
trend = 2.4 * X
seasonal = 3.1 * np.sin(3.2 * X)
drop = 10.0 * (X > 2).astype(float)
sigma = 0.75 + 0.75 * X**2
y = trend + seasonal - drop + rng.normal(loc=0.0, scale=np.sqrt(sigma))
df = pd.DataFrame({"X": X, "y": y})
_ = df.plot.scatter(x="X", y="y")
plt.show()

## Stack of predictors on a single data set
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

linear_ridge = make_pipeline(StandardScaler(), RidgeCV())
spline_ridge = make_pipeline(
    SplineTransformer(n_knots=6, degree=3),
    PolynomialFeatures(interaction_only=True),
    RidgeCV(),
)
hgbt = HistGradientBoostingRegressor(random_state=0)

estimators = [
    ("Linear Ridge", linear_ridge),
    ("Spline Ridge", spline_ridge),
    ("HGBT", hgbt),
]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
print(stacking_regressor)
