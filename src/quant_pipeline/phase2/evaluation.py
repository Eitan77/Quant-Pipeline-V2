from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    r = returns.fillna(0.0).astype(float).sort_index()
    if r.empty:
        return {"net_total_return": 0.0, "net_cagr": -1.0, "annualized_volatility": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "maximum_drawdown": 0.0, "calmar": np.nan, "worst_day": np.nan,
                "profitable_day_fraction": np.nan, "average_daily_return": np.nan, "daily_return_std": np.nan,
                "skew": np.nan, "excess_kurtosis": np.nan, "positive_month_fraction": np.nan}
    equity = (1 + r).cumprod(); drawdown = equity / equity.cummax() - 1
    annual_vol = r.std(ddof=1) * np.sqrt(252)
    downside = r[r < 0].std(ddof=1) * np.sqrt(252)
    years = max(len(r) / 252, 1 / 252)
    total = equity.iloc[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    return {"net_total_return": float(total), "net_cagr": float(cagr), "annualized_volatility": float(annual_vol),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) else np.nan,
            "sortino": float(r.mean() * 252 / downside) if downside else np.nan,
            "maximum_drawdown": float(drawdown.min()), "calmar": float(cagr / abs(drawdown.min())) if drawdown.min() else np.nan,
            "worst_day": float(r.min()), "profitable_day_fraction": float((r > 0).mean()),
            "average_daily_return": float(r.mean()), "daily_return_std": float(r.std(ddof=1)),
            "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()), "positive_month_fraction": float((1 + r).groupby(pd.to_datetime(r.index).to_period("M")).prod().sub(1).gt(0).mean())}


def drawdown_summary(returns: pd.Series) -> pd.DataFrame:
    r = returns.fillna(0.0).sort_index(); equity = (1 + r).cumprod(); dd = equity / equity.cummax() - 1
    return pd.DataFrame({"session_date": pd.to_datetime(r.index), "equity": equity.values, "drawdown": dd.values})
