from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import pairwise_rank_ic, quantile_spread


def scan_singles(features: np.ndarray, targets: np.ndarray, feature_ids: list[str], target_ids: list[str]) -> pd.DataFrame:
    if features.shape[0] != targets.shape[0] or features.shape[1] != len(feature_ids) or targets.shape[1] != len(target_ids):
        raise ValueError("Single-scan dimensions disagree")
    rows = []
    for feature_index, feature_id in enumerate(feature_ids):
        for target_index, target_id in enumerate(target_ids):
            x, y = features[:, feature_index], targets[:, target_index]
            rank_ic, count = pairwise_rank_ic(x, y)
            rows.append({"feature_id": feature_id, "target_id": target_id, "n_obs": count,
                         "rank_ic": rank_ic, "top_bottom_spread": quantile_spread(x, y)})
    return pd.DataFrame(rows)
