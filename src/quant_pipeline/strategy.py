from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .holdout import assert_pre_holdout_frame


Direction = Literal["long_only", "short_only", "long_short", "adaptive"]
Selection = Literal["quantile", "top_n"]


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    family: str
    economic_hypothesis: str
    decision_time: str
    signal: str
    direction: Direction
    selection: Selection
    positions: int | None
    quantile: float | None
    entry_delay_minutes: int
    exit_rule: str
    holding_period_minutes: int | None
    signal_direction: int = 1
    cost_grid_bps: tuple[float, ...] = (0, 1, 2, 4, 6, 10)
    required_regime: str | None = None
    required_confirmation: str | None = None
    concentration_mode: str = "diversified"
    adaptive_min_history_sessions: int = 60
    adaptive_minimum_edge_bps: float = 0.0

    def as_dict(self) -> dict:
        result = asdict(self)
        result["cost_grid_bps"] = list(self.cost_grid_bps)
        return result


def evaluate_strategy(
    frame: pd.DataFrame,
    spec: StrategySpec,
    sealed_holdout_start: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate one deterministic strategy on point-in-time signal/return rows."""
    required = {"symbol", "session_date", "decision_ts", "entry_ts", "exit_ts", "signal", "raw_return"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Strategy input missing columns: {sorted(missing)}")
    assert_pre_holdout_frame(frame, sealed_holdout_start, f"Phase 2 strategy {spec.strategy_id}")
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["signal", "raw_return"]).copy()
    work["session_date"] = pd.to_datetime(work.session_date).dt.normalize()
    if work.empty:
        return _empty_metrics(spec), pd.DataFrame(), pd.DataFrame()
    # Signal extremes can contain many exact ties.  Make the documented
    # secondary key deterministic instead of inheriting Parquet/DuckDB order.
    work = work.sort_values(["session_date", "symbol"], kind="mergesort").reset_index(drop=True)

    group = work.groupby("session_date", sort=True)["signal"]
    if spec.selection == "quantile":
        if spec.quantile is None or not 0 < spec.quantile < 0.5:
            raise ValueError("Quantile selection requires 0 < quantile < 0.5")
        pct = group.rank(method="first", pct=True)
        high = pct.gt(1 - spec.quantile)
        low = pct.le(spec.quantile)
    elif spec.selection == "top_n":
        if not spec.positions or spec.positions < 1:
            raise ValueError("top_n selection requires positive positions")
        ascending_rank = group.rank(method="first", ascending=True)
        descending_rank = group.rank(method="first", ascending=False)
        high = descending_rank.le(spec.positions)
        low = ascending_rank.le(spec.positions)
    else:
        raise ValueError(f"Unsupported selection: {spec.selection}")

    favorable_long = high if spec.signal_direction >= 0 else low
    favorable_short = low if spec.signal_direction >= 0 else high
    long_candidates = work.loc[favorable_long].copy()
    short_candidates = work.loc[favorable_short].copy()
    long_candidates["side"] = 1
    short_candidates["side"] = -1

    if spec.direction == "long_only":
        trades = long_candidates
    elif spec.direction == "short_only":
        trades = short_candidates
    elif spec.direction == "long_short":
        trades = pd.concat([long_candidates, short_candidates], ignore_index=True)
    elif spec.direction == "adaptive":
        trades = _adaptive_trades(long_candidates, short_candidates, spec)
    else:
        raise ValueError(f"Unsupported direction: {spec.direction}")

    if trades.empty:
        return _empty_metrics(spec), pd.DataFrame(), pd.DataFrame()
    trades["gross_trade_return"] = trades.side * trades.raw_return
    daily = _daily_portfolio(trades, spec.direction)
    opportunity_sessions = pd.Index(sorted(work.session_date.unique()), name="session_date")
    daily = daily.reindex(opportunity_sessions)
    daily["traded"] = daily.gross_return.notna()
    daily["gross_return"] = daily.gross_return.fillna(0.0)
    daily["positions"] = daily.positions.fillna(0).astype(int)
    for cost in spec.cost_grid_bps:
        daily[f"net_return_{_cost_label(cost)}"] = daily.gross_return - daily.traded.astype(float) * 2 * cost / 10_000

    metrics = _summarize(spec, trades, daily)
    trades = trades.assign(strategy_id=spec.strategy_id)
    daily = daily.reset_index().assign(strategy_id=spec.strategy_id)
    assert_pre_holdout_frame(trades, sealed_holdout_start, f"Phase 2 trades {spec.strategy_id}")
    assert_pre_holdout_frame(daily, sealed_holdout_start, f"Phase 2 daily {spec.strategy_id}")
    return metrics, trades, daily


def evaluate_preselected_strategy(
    frame: pd.DataFrame,
    spec: StrategySpec,
    sealed_holdout_start: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate causal selections already encoded as side=-1/0/1."""
    required = {"symbol", "session_date", "decision_ts", "entry_ts", "exit_ts", "signal", "raw_return", "side"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Preselected strategy input missing columns: {sorted(missing)}")
    assert_pre_holdout_frame(frame, sealed_holdout_start, f"Phase 2 preselected strategy {spec.strategy_id}")
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["signal", "raw_return"]).copy()
    work["session_date"] = pd.to_datetime(work.session_date).dt.normalize()
    work = work.sort_values(["session_date", "symbol"], kind="mergesort").reset_index(drop=True)
    opportunity_sessions = pd.Index(sorted(work.session_date.unique()), name="session_date")
    trades = work.loc[work.side.isin((-1, 1))].copy()
    if trades.empty:
        return _empty_metrics(spec), pd.DataFrame(), pd.DataFrame()
    trades["side"] = trades.side.astype(int)
    trades["gross_trade_return"] = trades.side * trades.raw_return
    daily = _daily_portfolio(trades, spec.direction).reindex(opportunity_sessions)
    daily["traded"] = daily.gross_return.notna()
    daily["gross_return"] = daily.gross_return.fillna(0.0)
    daily["positions"] = daily.positions.fillna(0).astype(int)
    for cost in spec.cost_grid_bps:
        daily[f"net_return_{_cost_label(cost)}"] = daily.gross_return - daily.traded.astype(float) * 2 * cost / 10_000
    metrics = _summarize(spec, trades, daily)
    trades = trades.assign(strategy_id=spec.strategy_id)
    daily = daily.reset_index().assign(strategy_id=spec.strategy_id)
    assert_pre_holdout_frame(trades, sealed_holdout_start, f"Phase 2 preselected trades {spec.strategy_id}")
    assert_pre_holdout_frame(daily, sealed_holdout_start, f"Phase 2 preselected daily {spec.strategy_id}")
    return metrics, trades, daily


def _adaptive_trades(long_candidates: pd.DataFrame, short_candidates: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    long_daily = long_candidates.groupby("session_date").raw_return.mean().sort_index()
    short_daily = (-short_candidates.groupby("session_date").raw_return.mean()).sort_index()
    dates = long_daily.index.union(short_daily.index).sort_values()
    history = pd.DataFrame({"long": long_daily.reindex(dates), "short": short_daily.reindex(dates)})
    expected = history.expanding(spec.adaptive_min_history_sessions).mean().shift(1)
    threshold = spec.adaptive_minimum_edge_bps / 10_000
    choice = np.where(expected.long.ge(expected.short) & expected.long.gt(threshold), 1,
                      np.where(expected.short.gt(expected.long) & expected.short.gt(threshold), -1, 0))
    choice = pd.Series(choice, index=dates)
    longs = long_candidates.loc[long_candidates.session_date.map(choice).eq(1)]
    shorts = short_candidates.loc[short_candidates.session_date.map(choice).eq(-1)]
    return pd.concat([longs, shorts], ignore_index=True)


def _daily_portfolio(trades: pd.DataFrame, direction: Direction) -> pd.DataFrame:
    rows = []
    for session, group in trades.groupby("session_date", sort=True):
        longs = group.loc[group.side.eq(1), "raw_return"]
        shorts = group.loc[group.side.eq(-1), "raw_return"]
        if direction == "long_short":
            if longs.empty or shorts.empty:
                continue
            gross = 0.5 * (longs.mean() - shorts.mean())
            long_contribution = 0.5 * longs.mean()
            short_contribution = -0.5 * shorts.mean()
        elif not longs.empty:
            gross = longs.mean(); long_contribution = gross; short_contribution = 0.0
        elif not shorts.empty:
            gross = -shorts.mean(); long_contribution = 0.0; short_contribution = gross
        else:
            continue
        rows.append({"session_date": session, "gross_return": gross, "positions": len(group),
                     "long_contribution": long_contribution, "short_contribution": short_contribution})
    return pd.DataFrame(rows).set_index("session_date") if rows else pd.DataFrame()


def _summarize(spec: StrategySpec, trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    gross = daily.gross_return
    traded = daily.traded
    trade_returns = trades.gross_trade_return
    years = max(len(gross) / 252, 1 / 252)
    result = {**spec.as_dict(), "opportunity_sessions": len(gross), "traded_sessions": int(traded.sum()),
              "trade_count": len(trades), "trades_per_year": len(trades) / years,
              "percent_days_traded": float(traded.mean()), "gross_total_return": _total_return(gross),
              "gross_cagr": _cagr(gross), "gross_average_portfolio_return": float(gross[traded].mean()),
              "gross_average_trade": float(trade_returns.mean()), "gross_median_trade": float(trade_returns.median()),
              "win_rate": float((gross[traded] > 0).mean()), "trade_win_rate": float((trade_returns > 0).mean()),
              "maximum_drawdown": _max_drawdown(gross), "worst_trade": float(trade_returns.min()),
              "worst_day": float(gross.min()), "long_contribution": float(daily.long_contribution.fillna(0).sum()),
              "short_contribution": float(daily.short_contribution.fillna(0).sum()),
              "cost_break_even_bps_per_side": float(gross[traded].mean() * 10_000 / 2),
              "return_by_year": _returns_by_year(gross), "recent_period_return": _recent_return(gross),
              "selected_symbol_distribution": trades.symbol.value_counts().head(20).to_json(),
              "top_10_trade_profit_share": _top_profit_share(trade_returns, 10),
              "average_after_removing_top_10_trades": _average_without_top(trade_returns, 10)}
    for cost in spec.cost_grid_bps:
        column = f"net_return_{_cost_label(cost)}"; values = daily[column]
        result[f"net_total_return_{_cost_label(cost)}"] = _total_return(values)
        result[f"net_cagr_{_cost_label(cost)}"] = _cagr(values)
        result[f"net_average_portfolio_return_{_cost_label(cost)}"] = float(values[traded].mean())
    return result


def _total_return(series: pd.Series) -> float:
    return float((1 + series.fillna(0)).prod() - 1)


def _cagr(series: pd.Series) -> float:
    total = 1 + _total_return(series); years = len(series) / 252
    return float(total ** (1 / years) - 1) if total > 0 and years > 0 else -1.0


def _max_drawdown(series: pd.Series) -> float:
    equity = (1 + series.fillna(0)).cumprod(); return float((equity / equity.cummax() - 1).min())


def _returns_by_year(series: pd.Series) -> str:
    years = pd.to_datetime(series.index).year
    return pd.Series({int(year): _total_return(series[years == year]) for year in sorted(set(years))}).to_json()


def _recent_return(series: pd.Series) -> float:
    cutoff = pd.Timestamp("2025-05-01")
    return _total_return(series[pd.to_datetime(series.index) >= cutoff])


def _top_profit_share(series: pd.Series, n: int) -> float:
    positive = series[series > 0]; total = positive.sum()
    return float(positive.nlargest(n).sum() / total) if total > 0 else np.nan


def _average_without_top(series: pd.Series, n: int) -> float:
    return float(series.drop(series.nlargest(min(n, len(series))).index).mean()) if len(series) > n else np.nan


def _cost_label(cost: float) -> str:
    return f"{cost:g}bps".replace(".", "p")


def _empty_metrics(spec: StrategySpec) -> dict:
    return {**spec.as_dict(), "opportunity_sessions": 0, "traded_sessions": 0, "trade_count": 0}
