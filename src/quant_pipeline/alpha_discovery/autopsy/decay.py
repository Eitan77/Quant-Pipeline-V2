from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def effect_decay(score: np.ndarray, targets: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for horizon, target in targets.items():
        valid = np.isfinite(score) & np.isfinite(target)
        if valid.sum() < 3: continue
        rank_ic = float(spearmanr(score[valid], target[valid]).statistic)
        high, low = np.quantile(score[valid], [.9, .1])
        spread = float(np.mean(target[valid & (score >= high)]) - np.mean(target[valid & (score <= low)]))
        rows.append({"target_horizon": horizon, "rank_ic": rank_ic, "top_bottom_spread": spread, "n_obs": int(valid.sum())})
    return pd.DataFrame(rows)
