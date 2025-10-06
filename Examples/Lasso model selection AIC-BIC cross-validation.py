# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

##DATASET
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y = True, as_frame = True)
X.head()
y.head()

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

rng = np.random.RandomState(42)
n_random_features = 14
X_random = pd.DataFrame(
    rng.randn(X.shape[0], n_random_features),
    columns = [f"random_{i:02d}" for i in range(n_random_features)],
)
X = pd.concat([X, X_random], axis = 1)

X[X.columns[::3]].head()

## Selecting Lasso via an information criterion