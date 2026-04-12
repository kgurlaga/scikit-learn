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
## 1.6.3. Nearest Neighbors Regression
## 1.6.4. Nearest Neighbor Algorithms
# 1.6.4.1. Brute Force
# 1.6.4.2. K-D Tree
# 1.6.4.3. Ball Tree

## 1.6.5. Nearest Centroid Classifier
from sklearn.neighbors import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))

# 1.6.5.1. Nearest Shrunken Centroid

## 1.6.6. Nearest Neighbors Transformer
import tempfile
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

cache_path = tempfile.gettempdir()
X, _ = make_regression(n_samples=50, n_features=25, random_state=0)
estimator = make_pipeline(
    KNeighborsTransformer(mode='distance'),
    Isomap(n_components=3, metric='precomputed'),
    memory=cache_path)
X_embedded = estimator.fit_transform(X)
X_embedded.shape