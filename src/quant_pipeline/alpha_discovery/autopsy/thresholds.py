from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_curve(score: np.ndarray, target: np.ndarray, tails: list[float], costs_per_side_bp: float = 0.0) -> pd.DataFrame:
    valid = np.isfinite(score) & np.isfinite(target); x, y = score[valid], target[valid]
    rows = []
    for tail in tails:
        low, high = np.quantile(x, [tail, 1 - tail])
        for side, mask, direction in (("long", x >= high, 1), ("short", x <= low, -1)):
            values = direction * y[mask] - 2 * costs_per_side_bp / 10_000
            rows.append({"tail": tail, "side": side, "mean_future_return": float(np.mean(values)),
                         "median_future_return": float(np.median(values)), "hit_rate": float(np.mean(values > 0)),
                         "sample_count": int(mask.sum())})
    return pd.DataFrame(rows)
