# 2.3. Clustering
# 2.3.1. Overview of clustering methods
# 2.3.2. K-means
# 2.3.2.1. Low-level parallelism
# 2.3.2.2. MiniBatch K-Means
# 2.3.3. Affinity Propagation
# 2.3.4. Mean Shift
# 2.3.5. Spectral clustering
# 2.3.5.1. Different label assignment strategies
# 2.3.5.2. Spectral Clustering Graphs
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(3, affinity='precomputed', n_init=100,
                        assign_labels='discretize')
sc.fit_predict(adjacency_matrix)

# 2.3.6. Hierarchical clustering
# 2.3.6.1. Different linkage type: Ward, complete, average, and single linkage
