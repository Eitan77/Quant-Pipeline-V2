from __future__ import annotations

import numpy as np


def positions_from_scores(scores: np.ndarray, mode: str = "rank_weighted_long_only", top_fraction: float = .10) -> np.ndarray:
    scores = np.asarray(scores, dtype=float); valid = np.isfinite(scores); output = np.zeros_like(scores)
    if not valid.any(): return output
    threshold = np.quantile(scores[valid], 1 - top_fraction)
    selected = valid & (scores >= threshold)
    if mode == "top_decile_long_only": output[selected] = 1 / max(selected.sum(), 1)
    elif mode == "rank_weighted_long_only":
        weights = np.maximum(scores[selected] - threshold, 0); output[selected] = weights / weights.sum() if weights.sum() else 1 / max(selected.sum(), 1)
    elif mode == "long_short":
        low = np.quantile(scores[valid], top_fraction); short = valid & (scores <= low)
        output[selected] = .5 / max(selected.sum(), 1); output[short] = -.5 / max(short.sum(), 1)
    else: raise ValueError(f"Unknown strategy mode: {mode}")
    return output
