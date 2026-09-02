from __future__ import annotations

import numpy as np
import pandas as pd


def causal_leverage(daily_returns: pd.Series, target_annual_volatility: float, max_leverage: float,
                    lookback_sessions: int = 60, volatility_floor: float = 0.03) -> pd.Series:
    """Prior-return-only volatility targeting with a leverage cap."""
    history = daily_returns.astype(float).rolling(lookback_sessions, min_periods=lookback_sessions).std().shift(1) * np.sqrt(252)
    denominator = history.clip(lower=volatility_floor)
    leverage = (target_annual_volatility / denominator).clip(upper=max_leverage)
    return leverage.fillna(1.0)
