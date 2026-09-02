from __future__ import annotations

import numpy as np
import pandas as pd


def build_rank_bins(values: np.ndarray, decision_codes: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(values)
    groups = pd.Series(decision_codes)
    ranked = frame.groupby(groups, sort=False).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    valid = np.isfinite(ranked)
    labels = np.full(ranked.shape, 255, dtype=np.uint8)
    labels[valid] = np.minimum((ranked[valid] * bins).astype(np.uint8), bins - 1)
    return labels, valid.astype(np.uint8)
