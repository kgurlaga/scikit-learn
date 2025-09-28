# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Loading and vectorizing the 20 newsgroups text dataset
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer

categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

def load_dataset(verbose = False, remove = ()):
    """Load and vectorize the 20 newsgroups dataset."""

    data_train = fetch_20newsgroups(
        subset = "train",
        categories = categories,
        shuffle = True,
        random_state = 42,
        remove = remove,
    )

    data_test = fetch_20newsgroups(
        subset = "test",
        categories = categories,
        shuffle = True,
        random_state = 42,
        remove = remove,
    )

    # order of labels in 'target_name' can be different from 'categories'
    target_names = data_train.target_names

    # split target