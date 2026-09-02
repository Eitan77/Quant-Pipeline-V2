from __future__ import annotations

import pandas as pd


def promote(summary: pd.DataFrame) -> pd.DataFrame:
    """Transparent, deliberately conservative discovery-only status labels."""
    out = summary.copy(); out["status"] = "invalid_strategy"
    valid = out.get("trade_count", 0).ge(100) & out.get("sessions", 0).ge(100)
    out.loc[valid, "status"] = "fails_execution_costs"
    robust = valid & out.get("net_cagr_3bps", out.get("net_cagr", 0)).ge(0)
    out.loc[robust, "status"] = "historically_profitable_but_recently_weak"
    stable = robust & out.get("recent_cagr", 0).gt(0) & out.get("positive_year_fraction", 0).ge(0.5)
    out.loc[stable, "status"] = "high_drawdown"
    candidate = stable & out.get("maximum_drawdown", -1).ge(-0.10)
    out.loc[candidate, "status"] = "standalone_candidate"
    return out
