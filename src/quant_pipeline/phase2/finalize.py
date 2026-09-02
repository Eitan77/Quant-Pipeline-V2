from __future__ import annotations

from pathlib import Path

import pandas as pd

from .correlation import return_correlations
from .reporting import write_reports


def finalize_existing(root: str | Path, holdout_start: str = "2026-05-01") -> Path:
    root = Path(root)
    summary = pd.read_csv(root / "strategy_summary.csv")
    failed = pd.read_csv(root / "failed_configurations.csv")
    daily = pd.read_parquet(root / "daily_strategy_returns.parquet")
    # Cost copies of the same configuration are not independent sleeves.
    # Use the primary 3-bps promotion stress for correlation analysis.
    ids = set(summary.loc[summary.cost_bps_per_side.eq(3) & summary.sessions.gt(0), "strategy_id"])
    representative = daily.loc[daily.strategy_id.isin(ids)]
    corr_daily, corr_monthly = return_correlations(representative)
    corr_daily.to_csv(root / "daily_return_correlations.csv")
    corr_monthly.to_csv(root / "monthly_return_correlations.csv")
    write_reports(root, summary, failed, holdout_start)
    return root
