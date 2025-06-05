from copy import deepcopy
from typing import Callable

import numpy as np


class AgglomerativeHierarchicalClustering():
    def __init__(self, metric: Callable[[np.ndarray, np.ndarray], float], linkage: str = "complete",
                 minimum_n_clusters: int = 1):
        """
        Agglomerative hierarchical clustering algorithm. Only "complete" linkage is supported for now.

        Args:
            metric (Callable[[np.ndarray, np.ndarray], float]): metric function for measuring element distance.
            linkage (str, optional, default="complete"): Currently, only "complete" is supported.
            minimum_n_clusters  (int, optional, default=1): The minimum number of clusters for clustering.
        """
        supported_linkages = ["complete"]
        if linkage not in supported_linkages:
            raise ValueError(f"linkage must be one of {supported_linkages}")
        if minimum_n_clusters < 1:
            raise ValueError(f"Minimum number of clusters must be >= 1")
        self.metric = metric
        self.linkage = linkage
        self.minimum_n_clusters = minimum_n_clusters

    def fit(self, X):
        n_samples = X.shape[0]

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples for clustering.")

        cluster_distance = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.metric(X[i], X[j])
                cluster_distance[i][j] = distance
                cluster_distance[j][i] = distance

        # This mask is used to cover unused/invalid cluster distances to avoid verbose searching for merging clusters.
        # Initially, only the upper triangle of the distance matrix are used to search clusters with minimum distance.
        mask = np.ones((X.shape[0], X.shape[0]), dtype=bool)
        mask = ~np.tril(mask, k=0)

        k_steps = n_samples - self.minimum_n_clusters
        linkage_matrix = np.zeros((k_steps, 2), dtype=int)

        # active_cluster_indices is the unmerged cluster indices
        active_cluster_indices = list(range(n_samples))

        for k_step in range(k_steps):
            cluster_distance_filtered = cluster_distance[mask]
            argmin = np.argmin(cluster_distance_filtered)

            i_indices, j_indices = np.where(mask)
            i_index, j_index = i_indices[argmin], j_indices[argmin]

            merged_cluster_idx_i = min(i_index, j_index)
            merged_cluster_idx_j = max(i_index, j_index)

            active_cluster_indices.remove(merged_cluster_idx_j)

            mask[:, merged_cluster_idx_j] = False
            mask[merged_cluster_idx_j, :] = False

            linkage_matrix[k_step][0] = merged_cluster_idx_i
            linkage_matrix[k_step][1] = merged_cluster_idx_j

            for i in active_cluster_indices:
                if i != merged_cluster_idx_i:
                    distance = self._complete_linkage(cluster_distance, linkage_matrix[k_step], i)
                    cluster_distance[i][merged_cluster_idx_i] = distance
                    cluster_distance[merged_cluster_idx_i][i] = distance

        return linkage_matrix, self.get_clusters(linkage_matrix)

    @staticmethod
    def _complete_linkage(cluster_distance, current_linkage, cluster_idx):
        merged_cluster_idx_i, merged_cluster_idx_j = current_linkage
        max_distance = max(cluster_distance[cluster_idx][merged_cluster_idx_i],
                           cluster_distance[cluster_idx][merged_cluster_idx_j])
        return max_distance

    @staticmethod
    def get_clusters(linkage_matrix: np.ndarray):
        unfiltered_linkage_clusters = [[i] for i in range(linkage_matrix.max() + 1)]
        merged_cluster_indices = []
        clusters = []
        for linkage in linkage_matrix:
            unfiltered_linkage_clusters[linkage[0]].extend(unfiltered_linkage_clusters[linkage[1]])
            merged_cluster_indices.append(linkage[1])
            filtered_current_clusters = []
            for i, x in enumerate(unfiltered_linkage_clusters):
                if i not in merged_cluster_indices:
                    x = deepcopy(x)
                    x.sort()
                    filtered_current_clusters.append(x)
            clusters.append(filtered_current_clusters)
        return clusters
