# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import sys

try:
    import nmslib
except ImportError:
    print("The package 'nmslib' is required to run this example.")
    sys.exit()

try:
    from pynndescent import PyNNDescentTransformer
except ImportError:
    print("The package 'pynndescent' is required to run this example.")
    sys.exit()

import nmslib
from pynndescent import PyNNDescentTransformer
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

