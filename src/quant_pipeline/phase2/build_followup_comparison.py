from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


OUT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_followup_73_ranking_through_20260430")
PARENT = Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_all_output_independent")
COST_LABELS = {-1.0: "m1p0", -0.5: "m0p5", 0.5: "p0p5", 1.0: "p1p0"}


def short_name(row: pd.Series) -> str:
    feature = str(row.feature)
    if feature.startswith("dual_auto__"):
        feature = feature.removeprefix("dual_auto__")
    horizon = str(row.raw_target).removeprefix("fwd_return_")
    return f"{feature} / H{horizon.removesuffix('m').upper()}"


def main() -> None:
    results = pd.read_csv(OUT / "strategy_period_cost_results.csv")
    base = pd.read_csv(OUT / "robust_ranking.csv")[[
        "implementation_id", "feature", "target", "raw_target", "status", "phase2_recommendation",
        "top_bottom_spread", "valid_observations", "valid_sessions", "recent_classification", "is_interaction",
    ]].copy()
    base["strategy"] = base.apply(short_name, axis=1)
    base["schedule"] = base.implementation_id.str.rsplit("__", n=1).str[-1]

    parent_meta = pd.read_csv(PARENT / "implementation_summary.csv")
    parent_meta = parent_meta.drop_duplicates("implementation_id")[["implementation_id", "cohort_divisor"]]
    interaction_meta = pd.read_csv(OUT / "interaction_implementation_audit.csv")[[
        "implementation_id", "cohort_divisor", "construction"
    ]]
    meta = pd.concat([
        parent_meta.assign(construction="top_bottom_10pct_equal_weight"), interaction_meta
    ], ignore_index=True).drop_duplicates("implementation_id", keep="last")
    base = base.merge(meta, on="implementation_id", how="left")
    base["basket_entries_per_day"] = np.where(base.schedule.eq("opening"), 1.0, 78.0)
    base["full_portfolio_turns_per_day"] = base.basket_entries_per_day / base.cohort_divisor

    daily = pd.read_parquet(OUT / "daily_returns.parquet")
    daily["session_date"] = pd.to_datetime(daily.session_date)
    one_cost = daily.loc[daily.cost_bps.eq(0.5)].copy()
    for period, mask in {
        "full": one_cost.session_date.lt("2026-05-01"),
        "recent": one_cost.session_date.ge("2025-05-01") & one_cost.session_date.lt("2026-05-01"),
    }.items():
        freq = one_cost.loc[mask].groupby("implementation_id").agg(
            **{f"{period}_positions": ("positions", "sum"), f"{period}_active_sessions": ("session_date", "nunique")}
        ).reset_index()
        freq[f"{period}_positions_per_day"] = freq[f"{period}_positions"] / freq[f"{period}_active_sessions"]
        base = base.merge(freq, on="implementation_id", how="left")

    value_columns = ["cagr", "maximum_drawdown", "peak_to_trough_sessions", "max_underwater_sessions", "sharpe"]
    wide = results.pivot(index="implementation_id", columns=["period", "cost_bps_per_side"], values=value_columns)
    names = []
    for metric, period, cost in wide.columns:
        period_label = "full" if period == "full_in_sample" else "recent"
        names.append(f"{period_label}_{COST_LABELS[float(cost)]}_{metric}")
    wide.columns = names
    table = base.merge(wide.reset_index(), on="implementation_id", how="left")

    for period in ("full", "recent"):
        for cost_label in COST_LABELS.values():
            table[f"{period}_{cost_label}_cagr_rank"] = table[f"{period}_{cost_label}_cagr"].rank(
                method="min", ascending=False
            ).astype(int)

    full_p1 = table.full_p1p0_cagr
    recent_p1 = table.recent_p1p0_cagr
    full_p05 = table.full_p0p5_cagr
    recent_p05 = table.recent_p0p5_cagr
    full_m05 = table.full_m0p5_cagr
    recent_m05 = table.recent_m0p5_cagr
    table["execution_profile"] = np.select(
        [
            (full_p1 > 0) & (recent_p1 > 0),
            (full_p05 > 0) & (recent_p05 > 0),
            (recent_p05 > 0) & (full_p05 <= 0),
            (full_m05 > 0) & (recent_m05 > 0),
        ],
        [
            "positive_full_and_recent_at_1bps",
            "positive_full_and_recent_at_0p5bps_only",
            "recently_positive_at_0p5bps_only",
            "maker_dependent",
        ],
        default="not_positive_even_with_maker_assumption",
    )
    table["recent_minus_full_cagr_at_0p5bps"] = recent_p05 - full_p05
    table["recently_printing_at_0p5bps"] = recent_p05 > 0

    lead = [
        "strategy", "implementation_id", "execution_profile", "construction", "schedule",
        "cohort_divisor", "basket_entries_per_day", "full_portfolio_turns_per_day",
        "full_positions_per_day", "recent_positions_per_day", "valid_observations", "valid_sessions",
        "top_bottom_spread", "recent_classification", "is_interaction",
    ]
    performance = []
    for period in ("full", "recent"):
        for cost_label in COST_LABELS.values():
            performance.extend([
                f"{period}_{cost_label}_cagr_rank", f"{period}_{cost_label}_cagr",
                f"{period}_{cost_label}_maximum_drawdown", f"{period}_{cost_label}_peak_to_trough_sessions",
                f"{period}_{cost_label}_max_underwater_sessions", f"{period}_{cost_label}_sharpe",
            ])
    tail = ["recent_minus_full_cagr_at_0p5bps", "recently_printing_at_0p5bps"]
    table = table[lead + performance + tail].sort_values(
        ["full_p1p0_cagr_rank", "recent_p1p0_cagr_rank"]
    )
    table.to_csv(OUT / "ALL_STRATEGIES_OBJECTIVE_COMPARISON.csv", index=False)

    long_ranked = results.merge(base[[
        "implementation_id", "strategy", "construction", "schedule", "full_portfolio_turns_per_day",
        "full_positions_per_day", "recent_positions_per_day", "valid_observations", "recent_classification",
    ]], on="implementation_id", how="left")
    long_ranked["cagr_rank"] = long_ranked.groupby(["period", "cost_bps_per_side"])["cagr"].rank(
        method="min", ascending=False
    ).astype(int)
    long_ranked = long_ranked.sort_values(["period", "cost_bps_per_side", "cagr_rank"])
    long_ranked.to_csv(OUT / "ALL_RANKINGS_BY_PERIOD_AND_COST.csv", index=False)

    counts = table.execution_profile.value_counts()
    lines = [
        "# Objective follow-up comparison",
        "",
        "All executable follow-up findings use approximately 100% maximum concurrent gross exposure, "
        "50% long and 50% short gross, equal weighting with a 10% symbol cap, next-bar-open entry, "
        "fixed-time exit, no leverage, and costs stated per side. Continuous strategies enter every "
        "five minutes with capital divided across overlapping holding-period cohorts. Opening strategies "
        "enter one basket per day. Binary Phase 1B interactions use equal-side-gross signal-on versus "
        "signal-off portfolios because percentile tails are not meaningful for a binary feature.",
        "",
        "No composite winner score is used. Every strategy is ranked independently within each period "
        "and cost column by CAGR.",
        "",
        "## Coverage",
        "",
        f"- Executable unique strategies: {len(table)}",
        f"- Positive in full and recent periods at +1 bp/side: {counts.get('positive_full_and_recent_at_1bps', 0)}",
        f"- Positive in full and recent periods at +0.5 bp/side only: {counts.get('positive_full_and_recent_at_0p5bps_only', 0)}",
        f"- Positive at +0.5 bp/side only in the recent year: {counts.get('recently_positive_at_0p5bps_only', 0)}",
        f"- Maker-dependent: {counts.get('maker_dependent', 0)}",
        f"- Not positive even with maker assumption: {counts.get('not_positive_even_with_maker_assumption', 0)}",
        "",
        "Use `ALL_STRATEGIES_OBJECTIVE_COMPARISON.csv` for the wide comparison and "
        "`ALL_RANKINGS_BY_PERIOD_AND_COST.csv` for sortable period/cost leaderboards.",
    ]
    (OUT / "README_OBJECTIVE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
