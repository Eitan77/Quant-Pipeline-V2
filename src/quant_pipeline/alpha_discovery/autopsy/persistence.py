from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_turnover(rank: np.ndarray, security_codes: np.ndarray, lags: tuple[int, ...] = (1, 2, 5)) -> pd.DataFrame:
    frame = pd.DataFrame({"rank": rank, "security": security_codes})
    rows = []
    for lag in lags:
        previous = frame.groupby("security", sort=False).rank.shift(lag)
        valid = frame.rank.notna() & previous.notna()
        rows.append({"lag": lag, "rank_autocorrelation": float(frame.loc[valid, "rank"].corr(previous[valid], method="spearman")),
                     "remain_top_decile_probability": float(((frame.rank >= .9) & (previous >= .9)).sum() / max((previous >= .9).sum(), 1))})
    return pd.DataFrame(rows)
