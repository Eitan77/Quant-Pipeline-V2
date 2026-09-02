from __future__ import annotations

import json
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PHASE1A = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1_final_discovery_through_20260430")
PHASE1B = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1b_systematic_v1_through_20260430")
PARENT_RESULTS = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_all_output_independent")
OUT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_followup_73_ranking_through_20260430")
HOLDOUT = pd.Timestamp("2026-05-01")
RECENT_START = pd.Timestamp("2025-05-01")
COSTS = (-1.0, -0.5, 0.5, 1.0)


def raw_target(target: str) -> str:
    match = re.match(r"^(fwd_return_(?:\d+m|eod))", target)
    if not match:
        raise ValueError(f"Unsupported target: {target}")
    return match.group(1)


def column_paths(folder: Path, pattern: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in folder.glob(pattern):
        for column in pq.ParquetFile(path).schema.names:
            paths.setdefault(column, path)
    return paths


def interaction_daily(
    con: duckdb.DuckDBPyConnection,
    row: pd.Series,
    feature_path: Path,
    target_path: Path,
) -> tuple[pd.DataFrame, int, str]:
    target = raw_target(row.target)
    if target == "fwd_return_eod":
        divisor = 78
    else:
        divisor = max(1, math.ceil(int(re.search(r"(\d+)m", target).group(1)) / 5))
    direction = 1 if row.top_bottom_spread >= 0 else -1
    exit_ts = f"exit_ts__{target}"
    exit_px = f"exit_close_raw__{target}"
    costs = ",".join(f"({x})" for x in COSTS)
    is_binary = str(row.get("scan_kind", "")) == "binary"

    if is_binary:
        selection = f"""
        grouped AS (
          SELECT *, CASE WHEN signal > 0.5 THEN {direction} ELSE {-direction} END AS side,
                 count(*) FILTER (WHERE signal > 0.5) OVER (PARTITION BY decision_ts) AS on_n,
                 count(*) FILTER (WHERE signal <= 0.5) OVER (PARTITION BY decision_ts) AS off_n
          FROM base
        ), trades AS (
          SELECT *, least(0.10, 0.5 / CASE WHEN signal > 0.5 THEN on_n ELSE off_n END) / {divisor} AS abs_weight
          FROM grouped WHERE on_n > 0 AND off_n > 0
        )
        """
        construction = "binary_on_vs_off_equal_side_gross"
    else:
        selection = f"""
        ranked AS (
          SELECT *, count(*) OVER(PARTITION BY decision_ts) AS universe_n,
                 row_number() OVER(PARTITION BY decision_ts ORDER BY signal, symbol) AS lo_rank,
                 row_number() OVER(PARTITION BY decision_ts ORDER BY signal DESC, symbol) AS hi_rank
          FROM base
        ), sided AS (
          SELECT *, greatest(1, floor(universe_n * 0.10))::INTEGER AS tail_n,
                 CASE WHEN hi_rank <= greatest(1, floor(universe_n * 0.10)) THEN {direction}
                      WHEN lo_rank <= greatest(1, floor(universe_n * 0.10)) THEN {-direction}
                      ELSE 0 END AS side
          FROM ranked
        ), trades AS (
          SELECT *, least(0.10, 0.5 / tail_n) / {divisor} AS abs_weight
          FROM sided WHERE side <> 0
        )
        """
        construction = "top_bottom_10pct_equal_weight"

    sql = f"""
    WITH base AS (
      SELECT f.symbol, f.session_date, f.decision_ts, f."{row.feature}" AS signal,
             t.entry_open_raw AS entry_px, t."{exit_px}" AS exit_px
      FROM read_parquet('{feature_path.as_posix()}') f
      JOIN read_parquet('{target_path.as_posix()}') t
      USING(symbol,session_date,bar_start_ts,decision_ts)
      WHERE f.analysis_eligible
        AND CAST(f.session_date AS DATE) < DATE '2026-05-01'
        AND f."{row.feature}" IS NOT NULL
        AND t.entry_open_raw > 0 AND t."{exit_px}" > 0
        AND t."{exit_ts}" > t.entry_ts
    ), {selection}, cost_grid(cost_bps) AS (VALUES {costs})
    SELECT session_date, cost_bps,
      sum(abs_weight * CASE WHEN side=1
        THEN exit_px*(1-cost_bps/10000.0)/(entry_px*(1+cost_bps/10000.0))-1
        ELSE 1-exit_px*(1+cost_bps/10000.0)/(entry_px*(1-cost_bps/10000.0)) END) AS net_return,
      count(*) AS positions
    FROM trades CROSS JOIN cost_grid
    GROUP BY session_date, cost_bps ORDER BY session_date, cost_bps
    """
    return con.execute(sql).df(), divisor, construction


def metrics(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, float | int]:
    values = frame.set_index("session_date").net_return.reindex(calendar, fill_value=0.0).astype(float)
    equity = np.r_[1.0, np.cumprod(1.0 + values.to_numpy())]
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(equity[: trough + 1]))
    peak_index = 0
    max_underwater = 0
    for i in range(1, len(equity)):
        if equity[i] >= equity[peak_index]:
            peak_index = i
        else:
            max_underwater = max(max_underwater, i - peak_index)
    n = len(values)
    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (252.0 / n) - 1.0)
    vol = float(values.std(ddof=1) * np.sqrt(252.0))
    sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(252.0)) if values.std(ddof=1) else np.nan
    return {
        "total_return": total,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "peak_to_trough_sessions": trough - peak,
        "max_underwater_sessions": max_underwater,
        "positive_day_fraction": float((values > 0).mean()),
        "sessions": n,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(PHASE1B / "detailed_candidates.csv")
    follow = final.loc[final.phase2_recommendation.ne("reject_as_concentrated_or_unstable")].copy()
    if len(follow) != 73:
        raise RuntimeError(f"Expected 73 follow-up findings, found {len(follow)}")
    follow["raw_target"] = follow.target.map(raw_target)
    follow["direction"] = np.where(follow.top_bottom_spread.ge(0), 1, -1)
    follow["is_interaction"] = follow.status.eq("systematic_interaction_not_robust")
    follow["schedule"] = np.where(
        follow.feature.eq("universe_breadth_positive"), "not_cross_sectional",
        np.where(follow.feature.str.startswith("opening_"), "opening", "continuous"),
    )
    follow["implementation_id"] = (
        follow.feature + "__" + follow.raw_target + "__" + follow.direction.astype(str) + "__" + follow.schedule
    )

    parent_daily = pd.read_parquet(PARENT_RESULTS / "daily_returns.parquet")
    parent_daily["session_date"] = pd.to_datetime(parent_daily.session_date)
    if parent_daily.session_date.max() >= HOLDOUT:
        raise RuntimeError("Holdout breach in parent daily returns")
    calendar = pd.DatetimeIndex(sorted(parent_daily.session_date.unique()))

    feature_paths = column_paths(PHASE1B / "phase1b_systematic_features", "dual_*.parquet")
    target_paths = column_paths(PHASE1A / "blocks" / "targets", "*.parquet")
    interactions = follow.loc[follow.is_interaction].drop_duplicates("implementation_id").copy()
    interaction_frames: list[pd.DataFrame] = []
    interaction_meta: list[dict] = []
    con = duckdb.connect()
    con.execute("SET threads=8")
    try:
        for number, (_, row) in enumerate(interactions.iterrows(), 1):
            if row.feature not in feature_paths or row.raw_target not in target_paths:
                raise RuntimeError(f"Missing feature/target path: {row.feature} / {row.raw_target}")
            daily, divisor, construction = interaction_daily(
                con, row, feature_paths[row.feature], target_paths[row.raw_target]
            )
            daily["implementation_id"] = row.implementation_id
            interaction_frames.append(daily)
            interaction_meta.append({
                "implementation_id": row.implementation_id,
                "cohort_divisor": divisor,
                "construction": construction,
                "source_feature_file": feature_paths[row.feature].name,
            })
            print(f"interaction {number}/{len(interactions)} {row.implementation_id}", flush=True)
    finally:
        con.close()

    interaction_daily_all = pd.concat(interaction_frames, ignore_index=True)
    interaction_daily_all["session_date"] = pd.to_datetime(interaction_daily_all.session_date)
    if interaction_daily_all.session_date.max() >= HOLDOUT:
        raise RuntimeError("Holdout breach in interaction backtest")
    interaction_daily_all.to_parquet(OUT / "interaction_daily_returns.parquet", index=False)
    pd.DataFrame(interaction_meta).to_csv(OUT / "interaction_implementation_audit.csv", index=False)

    executable = follow.loc[follow.schedule.ne("not_cross_sectional")].drop_duplicates("implementation_id").copy()
    combined_daily = pd.concat([
        parent_daily.loc[parent_daily.implementation_id.isin(executable.implementation_id)],
        interaction_daily_all,
    ], ignore_index=True)
    combined_daily = combined_daily.drop_duplicates(["implementation_id", "session_date", "cost_bps"], keep="last")
    combined_daily.to_parquet(OUT / "daily_returns.parquet", index=False)

    summary_rows: list[dict] = []
    for strategy_id, strategy_daily in combined_daily.groupby("implementation_id"):
        for period, dates in {
            "full_in_sample": calendar,
            "recent_12m": calendar[(calendar >= RECENT_START) & (calendar < HOLDOUT)],
        }.items():
            for cost in COSTS:
                piece = strategy_daily.loc[strategy_daily.cost_bps.eq(cost), ["session_date", "net_return"]]
                result = metrics(piece, dates)
                result.update({
                    "implementation_id": strategy_id,
                    "period": period,
                    "cost_bps_per_side": cost,
                    "positions": int(strategy_daily.loc[strategy_daily.cost_bps.eq(cost), "positions"].sum()),
                })
                summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "strategy_period_cost_results.csv", index=False)

    identity = executable[[
        "implementation_id", "feature", "target", "raw_target", "status", "phase2_recommendation",
        "top_bottom_spread", "valid_observations", "valid_sessions", "recent_classification", "is_interaction",
    ]].drop_duplicates("implementation_id")
    wide = summary.pivot(index="implementation_id", columns=["period", "cost_bps_per_side"], values=[
        "cagr", "maximum_drawdown", "peak_to_trough_sessions", "max_underwater_sessions", "sharpe"
    ])
    wide.columns = [f"{metric}__{period}__{str(cost).replace('-', 'm').replace('.', 'p')}bps" for metric, period, cost in wide.columns]
    ranking = identity.merge(wide.reset_index(), on="implementation_id", how="left")
    conservative = [
        "cagr__full_in_sample__0p5bps", "cagr__full_in_sample__1p0bps",
        "cagr__recent_12m__0p5bps", "cagr__recent_12m__1p0bps",
    ]
    ranking["conservative_cagr_floor"] = ranking[conservative].min(axis=1)
    ranking["worst_1bps_drawdown"] = ranking[[
        "maximum_drawdown__full_in_sample__1p0bps", "maximum_drawdown__recent_12m__1p0bps"
    ]].min(axis=1)
    ranking["worst_1bps_peak_to_trough"] = ranking[[
        "peak_to_trough_sessions__full_in_sample__1p0bps", "peak_to_trough_sessions__recent_12m__1p0bps"
    ]].max(axis=1)
    ranking = ranking.sort_values(
        ["conservative_cagr_floor", "worst_1bps_drawdown", "valid_observations"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "robust_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(OUT / "robust_ranking.csv", index=False)

    top = summary.merge(identity, on="implementation_id", how="left")
    top = top.sort_values(["period", "cost_bps_per_side", "cagr"], ascending=[True, True, False])
    top["rank_within_period_cost"] = top.groupby(["period", "cost_bps_per_side"]).cumcount() + 1
    top.loc[top.rank_within_period_cost.le(15)].to_csv(OUT / "top15_by_period_and_cost.csv", index=False)

    coverage = follow[[
        "feature", "target", "status", "phase2_recommendation", "schedule", "implementation_id", "is_interaction"
    ]].copy()
    coverage["coverage_status"] = np.where(
        coverage.schedule.eq("not_cross_sectional"), "descriptive_not_standalone_tradeable",
        np.where(coverage.duplicated("implementation_id", keep=False), "mapped_to_duplicate_implementation", "implemented"),
    )
    coverage.to_csv(OUT / "followup_73_coverage.csv", index=False)
    audit = {
        "findings_flagged": int(len(follow)),
        "descriptive_not_tradeable": int(follow.schedule.eq("not_cross_sectional").sum()),
        "executable_unique_strategies": int(len(executable)),
        "parent_unique_strategies": int((~executable.is_interaction).sum()),
        "interaction_unique_strategies": int(executable.is_interaction.sum()),
        "first_session": str(calendar.min().date()),
        "last_session": str(calendar.max().date()),
        "recent_start": str(RECENT_START.date()),
        "sealed_holdout_start": str(HOLDOUT.date()),
        "holdout_access": False,
        "ranking_rule": "descending minimum CAGR across full/recent at +0.5/+1.0 bps per side",
    }
    (OUT / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
