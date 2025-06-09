from typing import Callable

import numpy as np


class AgglomerativeHierarchicalClustering:
    def __init__(self, metric: Callable[[np.ndarray, np.ndarray], float], linkage: str = "complete",
                 n_clusters: int = 1):
        """
        Agglomerative hierarchical clustering algorithm. Only "complete" linkage is supported for now.
        Note that `sklearn.cluster.AgglomerativeClustering` is much faster, this implementation is used when
        AgglomerativeClustering in sklearn fails to meet the data dimension restriction when the sample is
        represented by a matrix/tensor with feature dimensions greater than 1. While AgglomerativeClustering in sklearn
        only supports data shape of (n_samples, n_features).

        Args:
            metric (Callable[[np.ndarray, np.ndarray], float]): metric function for measuring element distance.
            linkage (str, optional, default="complete"): Currently, only "complete" is supported.
            n_clusters  (int, optional, default=1): The number of clusters for clustering.
        """
        supported_linkages = ["complete"]
        if linkage not in supported_linkages:
            raise ValueError(f"linkage must be one of {supported_linkages}")
        if n_clusters < 1:
            raise ValueError(f"Minimum number of clusters must be >= 1")
        self.metric = metric
        self.linkage = linkage
        self.n_clusters = n_clusters

    def fit(self, X):
        """
        Perform agglomerative hierarchical clustering on `X`.
        Args:
            X (np.ndarray): The input data `X` is expected to have be (n_samples, ...).

        Returns:
        The clustering with desired number of clusters and the full grouping tree.
        """
        n_samples = X.shape[0]

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples for clustering.")

        cluster_distance = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.metric(X[i], X[j])
                cluster_distance[i][j] = distance
                cluster_distance[j][i] = distance

        np.fill_diagonal(cluster_distance, np.inf)

        k_steps = n_samples - self.n_clusters
        linkage_matrix = np.zeros((k_steps, 2), dtype=int)

        # active_cluster_indices is the unmerged cluster indices
        active_cluster_indices = list(range(n_samples))

        actual_cluster_ids = list(range(n_samples))
        new_cluster_id = n_samples

        for k_step in range(k_steps):

            argmin = np.argmin(cluster_distance)

            i_idx, j_idx = np.unravel_index(argmin, cluster_distance.shape)

            i_id, j_id = actual_cluster_ids[i_idx], actual_cluster_ids[j_idx]
            linkage_matrix[k_step][0] = i_id
            linkage_matrix[k_step][1] = j_id

            actual_cluster_ids[i_idx] = new_cluster_id
            new_cluster_id += 1

            active_cluster_indices.remove(j_idx)

            for i in active_cluster_indices:
                if i != i_idx:
                    distance = self._complete_linkage(cluster_distance, i_idx, j_idx, i)
                    cluster_distance[i][i_idx] = distance
                    cluster_distance[i_idx][i] = distance

            cluster_distance[j_idx, :] = np.inf
            cluster_distance[:, j_idx] = np.inf

        clusterings = self.get_clusters(linkage_matrix, n_samples)
        return clusterings[0], clusterings

    @staticmethod
    def get_clusters(linkage_matrix, n_samples: int):
        """
        Build the clusterings from linkage matrix. This function can also be applied to
        sklearn.cluster.AgglomerativeClustering for building the clusterings:

        Args:
            linkage_matrix:
            n_samples:

        Returns: Clusterings with different numbers of clusters.

        Examples:
        >>> from sklearn.cluster import AgglomerativeClustering
        >>> import numpy as np
        >>> X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
        >>> clustering = AgglomerativeClustering(compute_full_tree=True).fit(X)
        >>> linkage_matrix = clustering.children_
        >>> clusterings = AgglomerativeHierarchicalClustering.get_clusters(linkage_matrix, X.shape[0])
        >>> clusterings
        [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 5], [4]], [[0, 1], [2], [3, 5], [4]], [[0, 1], [2], [3], [4], [5]], [[0], [1], [2], [3], [4], [5]]]
        """
        cluster_contents = [[i] for i in range(n_samples)]
        active_cluster_ids = list(range(n_samples))

        history_of_cluster_sets = []

        current_set_of_clusters = [[i] for i in range(n_samples)]
        history_of_cluster_sets.append(current_set_of_clusters)

        for i in range(linkage_matrix.shape[0]):
            c1_id, c2_id = linkage_matrix[i]

            # Retrieve the cluster members of merged clusters
            samples_c1 = cluster_contents[c1_id]
            samples_c2 = cluster_contents[c2_id]

            new_cluster_samples = sorted(samples_c1 + samples_c2)

            cluster_contents.append(new_cluster_samples)
            new_cluster_id = n_samples + i

            active_cluster_ids.remove(c1_id)
            active_cluster_ids.remove(c2_id)
            active_cluster_ids.append(new_cluster_id)

            current_set_snapshot = [list(cluster_contents[k]) for k in active_cluster_ids]
            current_set_snapshot.sort()
            history_of_cluster_sets.append(current_set_snapshot)

        return history_of_cluster_sets[::-1]

    @staticmethod
    def _complete_linkage(cluster_distance, linkage_cluster_i_idx, linkage_cluster_j_idx, cluster_idx):

        max_distance = max(cluster_distance[cluster_idx][linkage_cluster_i_idx],
                           cluster_distance[cluster_idx][linkage_cluster_j_idx])
        return max_distance
