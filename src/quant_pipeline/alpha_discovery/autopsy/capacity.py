from __future__ import annotations

import numpy as np
import pandas as pd


def capacity_proxy(trade_notional: np.ndarray, bar_dollar_volume: np.ndarray, returns: np.ndarray, caps: tuple[float, ...] = (.001, .0025, .005, .01, .02, .05)) -> pd.DataFrame:
    participation = np.asarray(trade_notional) / np.asarray(bar_dollar_volume)
    rows = []
    for cap in caps:
        eligible = np.isfinite(participation) & (participation <= cap) & np.isfinite(returns)
        rows.append({"participation_cap": cap, "fraction_liquidity_constrained": float(np.mean(participation > cap)),
                     "performance_after_dropping_constrained": float(np.mean(np.asarray(returns)[eligible])) if eligible.any() else np.nan,
                     "eligible_trades": int(eligible.sum())})
    return pd.DataFrame(rows)
