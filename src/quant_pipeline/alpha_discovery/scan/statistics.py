from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def pairwise_rank_ic(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, int(valid.sum())
    a, b = rankdata(x[valid]), rankdata(y[valid])
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, int(valid.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(valid.sum())


def quantile_spread(x: np.ndarray, y: np.ndarray, tail: float = 0.10) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return np.nan
    low, high = np.quantile(x[valid], [tail, 1 - tail])
    return float(np.mean(y[valid & (x >= high)]) - np.mean(y[valid & (x <= low)]))
