from __future__ import annotations

import numpy as np
import pandas as pd


def cost_delay_curve(gross_returns: dict[int, np.ndarray], costs: list[float]) -> pd.DataFrame:
    rows = []
    for delay, values in gross_returns.items():
        values = np.asarray(values, dtype=float)
        for cost in costs:
            net = values - 2 * cost / 10_000
            rows.append({"entry_delay_minutes": delay, "cost_bp_per_side": cost, "gross_edge_per_trade": float(np.nanmean(values)),
                         "net_edge_per_trade": float(np.nanmean(net)), "sample_count": int(np.isfinite(values).sum())})
    return pd.DataFrame(rows)
