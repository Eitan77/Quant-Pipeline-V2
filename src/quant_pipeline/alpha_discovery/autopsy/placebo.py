from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def placebo_test(score: np.ndarray, target: np.ndarray, decision_codes: np.ndarray, runs: int = 1000, seed: int = 1729) -> pd.DataFrame:
    valid = np.isfinite(score) & np.isfinite(target); x, y, groups = score[valid], target[valid], decision_codes[valid]
    observed = float(spearmanr(x, y).statistic); rng = np.random.default_rng(seed); statistics = np.empty(runs)
    unique = np.unique(groups)
    for run in range(runs):
        shuffled = x.copy()
        for group in unique:
            mask = groups == group; shuffled[mask] = rng.permutation(shuffled[mask])
        statistics[run] = spearmanr(shuffled, y).statistic
    p = (1 + np.sum(np.abs(statistics) >= abs(observed))) / (runs + 1)
    return pd.DataFrame({"observed_statistic": [observed], "empirical_p_value": [p], "number_of_placebo_runs": [runs], "placebo_mean": [statistics.mean()], "placebo_std": [statistics.std(ddof=1)]})
