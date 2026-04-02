## 1.6.1. Unsupervised Nearest Neighbors
# 1.6.1.1. Finding the Nearest Neighbors
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
indices
distances
nbrs.kneighbors_graph(X).toarray()

# 1.6.1.2. KDTree and BallTree Classes
from sklearn.neighbors import KDTree
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
kdt.query(X, k=2, return_distance=False)

from sklearn.neighbors import KDTree, BallTree
KDTree.valid_metrics
BallTree.valid_metrics

## 1.6.2. Nearest Neighbors Classification
