from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_phase1_signal_rows
from .evaluation import summarize_returns
from .execution import apply_next_bar_open_fills


SOURCE = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1_final_discovery_through_20260430"
OUT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_opening_top1_pairs")
COSTS = (-1.0, -0.5, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Candidate:
    name: str
    feature: str
    horizon: int
    time: str
    direction: int


CANDIDATES = (
    Candidate("opening_close_location_h30", "opening_close_location_20m", 30, "09:55", 1),
    Candidate("opening_close_location_h15", "opening_close_location_20m", 15, "09:55", 1),
    Candidate("opening_breakdown_reversal_h30", "opening_breakdown_5m", 30, "09:40", -1),
    Candidate("opening_breakdown_reversal_h120", "opening_breakdown_5m", 120, "09:40", -1),
)


def select_pair(frame: pd.DataFrame, feature: str, direction: int) -> pd.DataFrame:
    work = frame.dropna(subset=[feature]).sort_values(
        ["decision_ts", feature, "symbol"], kind="mergesort"
    ).copy()
    low = work.groupby("decision_ts", sort=False).head(1).copy()
    high = work.groupby("decision_ts", sort=False).tail(1).copy()
    low["side"] = -direction
    high["side"] = direction
    selected = pd.concat([low, high], ignore_index=True)
    selected["final_weight"] = selected["side"] * 0.5
    return selected


def duration_metrics(r: pd.Series) -> dict[str, int]:
    eq = (1 + r.sort_index().fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    trough = int(np.argmin(dd.to_numpy()))
    peak = int(np.argmax(eq.iloc[: trough + 1].to_numpy()))
    run = longest = 0
    for value in dd.lt(0).to_numpy():
        run = run + 1 if value else 0
        longest = max(longest, run)
    return {"peak_to_trough_sessions": trough - peak, "max_underwater_sessions": longest}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        frame = load_phase1_signal_rows(
            SOURCE, (candidate.feature,), candidate.horizon, candidate.time, "2026-05-01"
        )
        pair = select_pair(frame, candidate.feature, candidate.direction)
        for cost in COSTS:
            filled = apply_next_bar_open_fills(pair, cost)
            filled["position_return"] = filled.final_weight.abs() * filled.net_return
            daily = filled.groupby("session_date").position_return.sum().sort_index()
            metrics = summarize_returns(daily)
            metrics.update(duration_metrics(daily))
            metrics.update({
                "strategy": candidate.name,
                "feature": candidate.feature,
                "decision_time": candidate.time,
                "horizon_minutes": candidate.horizon,
                "cost_bps_per_side": cost,
                "positions": len(filled),
                "sessions": len(daily),
                "positive_year_fraction": float(
                    (1 + daily).groupby(pd.to_datetime(daily.index).year).prod().sub(1).gt(0).mean()
                ),
            })
            rows.append(metrics)
            daily_frames.append(pd.DataFrame({
                "session_date": daily.index,
                "strategy": candidate.name,
                "cost_bps_per_side": cost,
                "net_return": daily.values,
            }))
        print(f"completed {candidate.name}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_parquet(OUT / "daily_returns.parquet", index=False)


if __name__ == "__main__":
    main()
