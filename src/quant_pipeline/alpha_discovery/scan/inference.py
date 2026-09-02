from __future__ import annotations

import numpy as np
from scipy.stats import norm


def clustered_mean_test(values: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(values) & ~np.asarray([item is None for item in clusters])
    y, g = np.asarray(values)[valid], np.asarray(clusters)[valid]
    if len(y) < 3:
        return {"mean": np.nan, "cluster_se": np.nan, "t": np.nan, "p": np.nan, "clusters": 0}
    unique, codes = np.unique(g, return_inverse=True)
    mean = y.mean(); centered = y - mean
    sums = np.bincount(codes, weights=centered)
    correction = len(unique) / max(len(unique) - 1, 1)
    variance = correction * np.sum(sums ** 2) / (len(y) ** 2)
    se = np.sqrt(max(variance, 0)); t = mean / se if se > 0 else np.nan
    return {"mean": float(mean), "cluster_se": float(se), "t": float(t),
            "p": float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan, "clusters": len(unique)}
