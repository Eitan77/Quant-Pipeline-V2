from __future__ import annotations

import numpy as np
import pandas as pd

from .vol_target import causal_leverage


def inverse_volatility_blend(daily_returns: pd.DataFrame, lookback: int = 60, cap: float = 0.40) -> tuple[pd.Series, pd.DataFrame]:
    wide = daily_returns.pivot(index="session_date", columns="strategy_id", values="net_return").fillna(0.0).sort_index()
    prior_vol = wide.rolling(lookback, min_periods=20).std().shift(1).clip(lower=1e-4)
    weights = (1 / prior_vol).clip(upper=(1 / prior_vol).quantile(0.90, axis=1), axis=0)
    weights = weights.div(weights.sum(axis=1), axis=0).clip(upper=cap)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(1 / len(wide.columns))
    return (wide * weights).sum(axis=1), weights


def vol_target_portfolio(returns: pd.Series, target: float, cap: float) -> tuple[pd.Series, pd.Series]:
    leverage = causal_leverage(returns, target, cap)
    return returns * leverage, leverage
