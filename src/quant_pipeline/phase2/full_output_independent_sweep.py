from __future__ import annotations

from pathlib import Path
import math
import re

import duckdb
import numpy as np
import pandas as pd

from .data import _column_paths, _target_paths
from .evaluation import summarize_returns


SOURCE = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1_final_discovery_through_20260430"
OUT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_all_output_independent")
COSTS = (-1.0, -0.5, 0.5, 1.0, 2.0)
HOLDOUT = "2026-05-01"
OPENING_TIMES = {
    "opening_breakout_5m": "09:40",
    "opening_breakdown_5m": "09:40",
    "opening_breakout_10m": "09:45",
    "opening_close_location_20m": "09:55",
}


def raw_target(target: str) -> str:
    match = re.match(r"^(fwd_return_(?:\d+m|eod))", target)
    if not match:
        raise ValueError(f"Unsupported target {target}")
    return match.group(1)


def duration_metrics(r: pd.Series) -> dict[str, int]:
    eq = (1 + r.sort_index().fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    trough = int(np.argmin(dd.to_numpy()))
    peak = int(np.argmax(eq.iloc[: trough + 1].to_numpy()))
    longest = run = 0
    for value in dd.lt(0).to_numpy():
        run = run + 1 if value else 0
        longest = max(longest, run)
    return {"peak_to_trough_sessions": trough - peak, "max_underwater_sessions": longest}


def opening_time(feature: str) -> str:
    if feature not in OPENING_TIMES:
        raise ValueError(f"No predeclared safe opening time for {feature}")
    return OPENING_TIMES[feature]


def run_implementation(
    con: duckdb.DuckDBPyConnection,
    feature: str,
    target: str,
    direction: int,
    schedule: str,
    feature_path: Path,
    target_path: Path,
) -> tuple[pd.DataFrame, int]:
    if target == "fwd_return_eod":
        cohort_divisor = 78
    else:
        horizon = int(re.search(r"(\d+)m", target).group(1))
        cohort_divisor = max(1, math.ceil(horizon / 5))
    time_filter = ""
    if schedule == "opening":
        time_filter = (
            "AND strftime(f.decision_ts AT TIME ZONE 'America/New_York', '%H:%M') "
            f"= '{opening_time(feature)}'"
        )
        cohort_divisor = 1
    costs_sql = ",".join(f"({cost})" for cost in COSTS)
    exit_ts = f'exit_ts__{target}'
    exit_px = f'exit_close_raw__{target}'
    sql = f"""
    WITH base AS (
      SELECT f.symbol, f.session_date, f.decision_ts, f."{feature}" AS signal,
             t.entry_open_raw AS entry_px, t."{exit_px}" AS exit_px
      FROM read_parquet('{feature_path.as_posix()}') f
      JOIN read_parquet('{target_path.as_posix()}') t
      USING(symbol,session_date,bar_start_ts,decision_ts)
      WHERE f.analysis_eligible
        AND CAST(f.session_date AS DATE) < DATE '{HOLDOUT}'
        AND f."{feature}" IS NOT NULL
        AND t.entry_open_raw > 0 AND t."{exit_px}" > 0
        AND t."{exit_ts}" > t.entry_ts
        {time_filter}
    ), ranked AS (
      SELECT *, count(*) OVER(PARTITION BY decision_ts) AS n,
             row_number() OVER(PARTITION BY decision_ts ORDER BY signal, symbol) AS lo_rank,
             row_number() OVER(PARTITION BY decision_ts ORDER BY signal DESC, symbol) AS hi_rank
      FROM base
    ), sided AS (
      SELECT *, greatest(1, floor(n * 0.10))::INTEGER AS tail_n,
        CASE
          WHEN hi_rank <= greatest(1, floor(n * 0.10)) THEN {direction}
          WHEN lo_rank <= greatest(1, floor(n * 0.10)) THEN {-direction}
          ELSE 0
        END AS side
      FROM ranked
    ), trades AS (
      SELECT *, least(0.10, 0.5 / tail_n) / {cohort_divisor} AS abs_weight
      FROM sided WHERE side <> 0
    ), cost_grid(cost_bps) AS (VALUES {costs_sql})
    SELECT session_date, cost_bps,
      sum(abs_weight * CASE WHEN side=1
        THEN exit_px*(1-cost_bps/10000.0)/(entry_px*(1+cost_bps/10000.0))-1
        ELSE 1-exit_px*(1+cost_bps/10000.0)/(entry_px*(1-cost_bps/10000.0)) END) AS net_return,
      count(*) AS positions
    FROM trades CROSS JOIN cost_grid
    GROUP BY session_date, cost_bps
    ORDER BY session_date, cost_bps
    """
    frame = con.execute(sql).df()
    return frame, cohort_divisor


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(Path(SOURCE) / "detailed_candidates.csv")
    candidates["raw_target"] = candidates.target.map(raw_target)
    candidates["direction"] = np.where(candidates.top_bottom_spread.ge(0), 1, -1)
    candidates["schedule"] = np.where(
        candidates.feature.eq("universe_breadth_positive"), "not_cross_sectional",
        np.where(candidates.feature.str.startswith("opening_"), "opening", "continuous"),
    )
    candidates["implementation_id"] = (
        candidates.feature + "__" + candidates.raw_target + "__" +
        candidates.direction.astype(str) + "__" + candidates.schedule
    )
    feature_paths, target_paths = _column_paths(SOURCE), _target_paths(SOURCE)
    tradeable = candidates.loc[candidates.schedule.ne("not_cross_sectional")].copy()
    implementations = tradeable.drop_duplicates("implementation_id")
    summary_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    con = duckdb.connect()
    con.execute("SET threads=8")
    try:
        for number, row in enumerate(implementations.itertuples(index=False), 1):
            daily, divisor = run_implementation(
                con, row.feature, row.raw_target, int(row.direction), row.schedule,
                feature_paths[row.feature], target_paths[row.raw_target],
            )
            for cost, group in daily.groupby("cost_bps", sort=True):
                returns = group.set_index("session_date").net_return.sort_index()
                metrics = summarize_returns(returns)
                metrics.update(duration_metrics(returns))
                metrics.update({
                    "implementation_id": row.implementation_id,
                    "feature": row.feature,
                    "raw_target": row.raw_target,
                    "direction": int(row.direction),
                    "schedule": row.schedule,
                    "cohort_divisor": divisor,
                    "cost_bps_per_side": float(cost),
                    "positions": int(group.positions.sum()),
                    "sessions": len(returns),
                    "positive_year_fraction": float(
                        (1 + returns).groupby(pd.to_datetime(returns.index).year).prod().sub(1).gt(0).mean()
                    ),
                })
                summary_rows.append(metrics)
            daily["implementation_id"] = row.implementation_id
            daily_frames.append(daily)
            pd.DataFrame(summary_rows).to_csv(OUT / "summary_checkpoint.csv", index=False)
            print(f"{number}/{len(implementations)} {row.implementation_id}", flush=True)
    finally:
        con.close()
    summary = pd.DataFrame(summary_rows)
    mapping = candidates[[
        "feature", "target", "raw_target", "status", "top_bottom_spread", "valid_observations",
        "valid_sessions", "phase2_recommendation", "schedule", "implementation_id",
    ]].copy()
    mapping.to_csv(OUT / "candidate_coverage.csv", index=False)
    summary.to_csv(OUT / "implementation_summary.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_parquet(OUT / "daily_returns.parquet", index=False)
    evaluated = mapping.merge(summary, on=["implementation_id", "feature", "raw_target"], how="left")
    evaluated.to_csv(OUT / "all_candidate_cost_grid.csv", index=False)
    gate = evaluated[
        evaluated.cost_bps_per_side.eq(0.5)
        & evaluated.maximum_drawdown.ge(-0.05)
        & evaluated.peak_to_trough_sessions.le(21)
    ]
    print(f"tradeable_outputs={len(tradeable)} unique_implementations={len(implementations)} gate_passes={len(gate)}")


if __name__ == "__main__":
    main()
