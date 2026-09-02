from __future__ import annotations

import numpy as np


def combine_alphas(alpha_returns: np.ndarray, turnover: np.ndarray | None = None, maximum_weight: float = .35, turnover_penalty: float = .10) -> np.ndarray:
    values = np.asarray(alpha_returns, dtype=float)
    covariance = np.cov(np.nan_to_num(values), rowvar=False) + np.eye(values.shape[1]) * 1e-8
    expected = np.nanmean(values, axis=0)
    if turnover is not None: expected = expected - turnover_penalty * np.asarray(turnover)
    raw = np.linalg.solve(covariance, expected); raw = np.maximum(raw, 0)
    weights = raw / raw.sum() if raw.sum() else np.repeat(1 / len(raw), len(raw))
    for _ in range(10):
        weights = np.minimum(weights, maximum_weight); weights /= weights.sum()
    return weights
