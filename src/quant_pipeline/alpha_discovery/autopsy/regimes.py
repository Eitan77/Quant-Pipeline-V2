from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_effects(returns: np.ndarray, metadata: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in metadata: continue
        frame = pd.DataFrame({"bucket": metadata[column], "return": returns}).dropna()
        for bucket, group in frame.groupby("bucket", observed=True):
            rows.append({"dimension": column, "bucket": str(bucket), "mean_return": group["return"].mean(),
                         "median_return": group["return"].median(), "hit_rate": (group["return"] > 0).mean(), "sample_count": len(group)})
    return pd.DataFrame(rows)
