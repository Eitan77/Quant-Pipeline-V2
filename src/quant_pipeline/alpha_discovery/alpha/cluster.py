from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_clusters(returns: np.ndarray, maximum_distance: float = 0.30) -> np.ndarray:
    correlation = np.nan_to_num(np.corrcoef(returns, rowvar=False), nan=0.0)
    distance = np.clip(1 - correlation, 0, 2); np.fill_diagonal(distance, 0)
    if distance.shape[0] <= 1: return np.ones(distance.shape[0], dtype=int)
    return fcluster(linkage(squareform(distance, checks=False), method="average"), maximum_distance, criterion="distance")
