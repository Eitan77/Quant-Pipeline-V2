from __future__ import annotations

import pandas as pd


def return_correlations(daily_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily and monthly correlations from long-form strategy returns."""
    wide = daily_returns.pivot(index="session_date", columns="strategy_id", values="net_return").fillna(0.0)
    daily = wide.corr()
    monthly = (1 + wide).groupby(pd.to_datetime(wide.index).to_period("M")).prod().sub(1).corr()
    return daily, monthly
