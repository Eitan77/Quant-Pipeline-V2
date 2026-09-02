from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from ..holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet

KEYS = ("symbol", "session_date", "bar_start_ts", "decision_ts")


@lru_cache(maxsize=8)
def _column_paths(root: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in (Path(root) / "blocks" / "features").glob("*.parquet"):
        for column in pq.ParquetFile(path).schema.names:
            paths.setdefault(column, path)
    return paths


@lru_cache(maxsize=8)
def _target_paths(root: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in (Path(root) / "blocks" / "targets").glob("*.parquet"):
        for column in pq.ParquetFile(path).schema.names:
            paths.setdefault(column, path)
    return paths


def load_phase1_signal_rows(source_run: str, features: tuple[str, ...], horizon_minutes: int,
                            decision_time_et: str, sealed_holdout_start: str) -> pd.DataFrame:
    """Load only the feature/target columns needed for one causal configuration."""
    feature_map, target_map = _column_paths(source_run), _target_paths(source_run)
    missing = [feature for feature in features if feature not in feature_map]
    target = f"fwd_return_{horizon_minutes}m"
    if missing or target not in target_map:
        raise ValueError(f"Unavailable Phase 1 input: features={missing}, target={target if target not in target_map else None}")
    selected_paths = []
    for path in [*(feature_map[x] for x in features), target_map[target]]:
        if path not in selected_paths:
            assert_pre_holdout_parquet(path, sealed_holdout_start, f"Phase 2 input {path.name}", verify_key_rows=False)
            selected_paths.append(path)
    aliases = {path: f"p{index}" for index, path in enumerate(selected_paths)}
    first = selected_paths[0]
    selects = [f"{aliases[first]}.{column}" for column in KEYS] + [f"{aliases[first]}.bar_end_ts", f"{aliases[first]}.available_at_ts"]
    selects += [f"{aliases[feature_map[feature]]}.{feature} AS {feature}" for feature in features]
    target_alias = aliases[target_map[target]]
    selects += [f"{target_alias}.entry_ts", f"{target_alias}.entry_open_raw", f"{target_alias}.\"exit_ts__{target}\" AS exit_ts", f"{target_alias}.\"exit_close_raw__{target}\" AS exit_close_raw", f"{target_alias}.{target} AS raw_return", f"{target_alias}.beta_at_decision"]
    joins = [f"JOIN read_parquet('{path.as_posix()}') {aliases[path]} USING({','.join(KEYS)})" for path in selected_paths[1:]]
    con = duckdb.connect(); con.execute("SET threads=8")
    try:
        query = f"""SELECT {','.join(selects)} FROM read_parquet('{first.as_posix()}') {aliases[first]}
        {' '.join(joins)} WHERE {aliases[first]}.analysis_eligible
        AND strftime({aliases[first]}.decision_ts AT TIME ZONE 'America/New_York', '%H:%M') = '{decision_time_et}'
        AND CAST({aliases[first]}.session_date AS DATE) < DATE '{sealed_holdout_start}'"""
        frame = con.execute(query).df()
    finally:
        con.close()
    assert_pre_holdout_frame(frame, sealed_holdout_start, "Phase 2 loaded signal rows")
    return frame
