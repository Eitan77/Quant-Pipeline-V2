from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    p = values[valid]; order = np.argsort(p, kind="mergesort"); ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
    output[valid] = np.clip(adjusted[inverse], 0, 1)
    return output
