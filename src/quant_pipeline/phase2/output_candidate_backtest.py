from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from .data import load_phase1_signal_rows
from .selection import select_cross_sectional_tails
from .portfolio import assign_weights
from .execution import apply_next_bar_open_fills
from .evaluation import summarize_returns


SOURCE = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1_final_discovery_through_20260430"
OUT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_output_candidate_backtest")
COSTS = (-1.0, -0.5, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Sleeve:
    name: str
    feature: str
    horizon: int
    time: str
    direction: int = 1


SLEEVES = (
    Sleeve("range_0940_h30", "session_range_position", 30, "09:40"),
    Sleeve("range_0955_h120", "session_range_position", 120, "09:55"),
    Sleeve("vwap_0955_h120", "vwap_slope", 120, "09:55"),
    Sleeve("consistency_0955_h120", "return_consistency_5", 120, "09:55"),
    Sleeve("normalized_range_1025_h120", "range_position_10_z_4680", 120, "10:25"),
    Sleeve("residual_reversal_1445_h30", "market_residual_return_60", 30, "14:45", -1),
)


def duration_metrics(r: pd.Series) -> dict[str, float]:
    r = r.sort_index().fillna(0.0)
    eq = (1 + r).cumprod(); peaks = eq.cummax(); dd = eq / peaks - 1
    trough = int(np.argmin(dd.to_numpy()))
    peak = int(np.argmax(eq.iloc[:trough + 1].to_numpy())) if trough >= 0 else 0
    underwater = dd.lt(0).to_numpy(); longest = run = 0
    for value in underwater:
        run = run + 1 if value else 0; longest = max(longest, run)
    return {"peak_to_trough_sessions": trough - peak, "max_underwater_sessions": longest}


def one_sleeve(spec: Sleeve) -> tuple[list[dict], dict[float, pd.Series]]:
    frame = load_phase1_signal_rows(SOURCE, (spec.feature,), spec.horizon, spec.time, "2026-05-01")
    frame["signal"] = pd.to_numeric(frame[spec.feature], errors="coerce")
    selected = select_cross_sectional_tails(frame, 0.10, spec.direction)
    weighted = assign_weights(selected, "equal", "dollar_neutral", symbol_cap=0.10)
    rows, returns = [], {}
    for cost in COSTS:
        filled = apply_next_bar_open_fills(weighted, cost)
        filled["position_return"] = filled.final_weight.abs() * filled.net_return
        daily = filled.groupby("session_date").position_return.sum().sort_index()
        metrics = summarize_returns(daily); metrics.update(duration_metrics(daily))
        metrics.update({"strategy": spec.name, "feature": spec.feature, "decision_time": spec.time,
                        "horizon_minutes": spec.horizon, "cost_bps_per_side": cost,
                        "trades": len(filled), "sessions": len(daily),
                        "positive_year_fraction": float((1 + daily).groupby(pd.to_datetime(daily.index).year).prod().sub(1).gt(0).mean())})
        rows.append(metrics); returns[cost] = daily
    return rows, returns


def combine(name: str, members: tuple[str, ...], all_returns: dict[str, dict[float, pd.Series]]) -> tuple[list[dict], dict[float, pd.Series]]:
    rows, out = [], {}
    for cost in COSTS:
        wide = pd.concat([all_returns[m][cost].rename(m) for m in members], axis=1).fillna(0.0)
        daily = wide.mean(axis=1); metrics = summarize_returns(daily); metrics.update(duration_metrics(daily))
        metrics.update({"strategy": name, "feature": "+".join(members), "decision_time": "multi",
                        "horizon_minutes": -1, "cost_bps_per_side": cost,
                        "trades": np.nan, "sessions": len(daily),
                        "positive_year_fraction": float((1 + daily).groupby(pd.to_datetime(daily.index).year).prod().sub(1).gt(0).mean())})
        rows.append(metrics); out[cost] = daily
    return rows, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, all_returns = [], {}
    for sleeve in SLEEVES:
        sleeve_rows, ret = one_sleeve(sleeve); rows.extend(sleeve_rows); all_returns[sleeve.name] = ret
        pd.DataFrame(sleeve_rows).to_csv(OUT / f"checkpoint_{sleeve.name}.csv", index=False)
        print(f"completed {sleeve.name}", flush=True)
    combos = {
        "multi_all_equal": tuple(s.name for s in SLEEVES),
        "multi_nonredundant_equal": ("range_0940_h30", "vwap_0955_h120", "consistency_0955_h120", "residual_reversal_1445_h30"),
    }
    for name, members in combos.items():
        combo_rows, ret = combine(name, members, all_returns); rows.extend(combo_rows); all_returns[name] = ret
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    daily_rows = []
    for name, costs in all_returns.items():
        for cost, series in costs.items():
            daily_rows.append(pd.DataFrame({"session_date": series.index, "strategy": name, "cost_bps_per_side": cost, "net_return": series.values}))
    pd.concat(daily_rows, ignore_index=True).to_parquet(OUT / "daily_returns.parquet", index=False)
    base = pd.concat([all_returns[s.name][0.5].rename(s.name) for s in SLEEVES], axis=1).fillna(0.0)
    base.corr().to_csv(OUT / "sleeve_correlations_0p5bps.csv")
    gate = summary[(summary.cost_bps_per_side.eq(0.5)) & summary.maximum_drawdown.ge(-0.05) & summary.peak_to_trough_sessions.le(21)]
    (OUT / "README.md").write_text(
        "# Phase 2 output-list candidate backtest\n\n"
        "All sleeves were selected from the completed 1A/1B detailed candidate lists before backtest inspection. "
        "Results are bar-reference fills with the stated per-side cost, not quote fills.\n\n"
        f"Strategies passing the 0.5 bps, 5% drawdown, and 21-session peak-to-trough gate: {len(gate)}.\n",
        encoding="utf-8",
    )
    print(summary[["strategy","cost_bps_per_side","net_cagr","sharpe","maximum_drawdown","peak_to_trough_sessions","max_underwater_sessions"]].to_string(index=False))


if __name__ == "__main__":
    main()
