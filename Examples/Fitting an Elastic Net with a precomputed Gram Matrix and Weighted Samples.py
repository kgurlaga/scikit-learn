# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.datasets import make_regression

rng = np.random.RandomState(0)
n_samples = int(1e5)
X, y = make_regression(n_samples = n_samples, noise = 0.5, random_state = rng)

sample_weight = rng.lognormal(size = n_samples)
normalized_weights = sample_weight * (n_samples / (sample_weight.sum()))

X_offset = np.average(X, axis = 0, weights = normalized_weights)
X_centered = X - np.average(X, axis = 0, weights = normalized_weights)
X_scaled = X_centered * np.sqrt(normalized_weights)[:, np.newaxis]
gram = np.dot(X_scaled.T, X_scaled)

from sklearn.linear_model import ElasticNet

lm = ElasticNet(alpha = 0.01, precompute = gram)
lm.fit(X_centered, y, sample_weight = normalized_weights)