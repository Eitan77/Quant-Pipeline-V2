from __future__ import annotations

import numpy as np
import pandas as pd


def select_cross_sectional_tails(frame: pd.DataFrame, tail: float, signal_direction: int = 1) -> pd.DataFrame:
    """Deterministically select tails within each decision timestamp only."""
    required = {"symbol", "decision_ts", "signal"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Selection input missing: {sorted(missing)}")
    if not 0 < tail < 0.5:
        raise ValueError("tail must be in (0, 0.5)")
    work = frame.dropna(subset=["signal"]).sort_values(["decision_ts", "signal", "symbol"], kind="mergesort").copy()
    work["rank_ascending"] = work.groupby("decision_ts", sort=False).cumcount() + 1
    work["eligible_count"] = work.groupby("decision_ts", sort=False)["symbol"].transform("size")
    work["rank_descending"] = work["eligible_count"] - work["rank_ascending"] + 1
    target_count = np.maximum(1, np.floor(work["eligible_count"] * tail).astype(int))
    high = work["rank_descending"].le(target_count)
    low = work["rank_ascending"].le(target_count)
    work["side"] = 0
    if signal_direction >= 0:
        work.loc[high, "side"] = 1
        work.loc[low, "side"] = -1
    else:
        work.loc[high, "side"] = -1
        work.loc[low, "side"] = 1
    return work
