from __future__ import annotations

from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMNS = (
    "session_date", "bar_start_ts", "bar_end_ts", "available_at_ts",
    "decision_ts", "entry_ts", "exit_ts",
)


def assert_pre_holdout_frame(
    frame: pd.DataFrame,
    sealed_holdout_start: str,
    context: str,
) -> None:
    """Fail closed if any known timestamp reaches the sealed holdout."""
    boundary = pd.Timestamp(sealed_holdout_start)
    for column in frame.columns:
        if column not in TIMESTAMP_COLUMNS and not column.startswith("exit_ts__"):
            continue
        values = pd.to_datetime(frame[column], errors="coerce", utc=True)
        compare = boundary.tz_localize("UTC") if getattr(values.dt, "tz", None) is not None else boundary
        if values.ge(compare).any():
            raise ValueError(f"Sealed holdout row detected in {context}: {column}>={sealed_holdout_start}")


def assert_pre_holdout_parquet(
    path: Path,
    sealed_holdout_start: str,
    context: str,
    verify_key_rows: bool = True,
) -> pd.DataFrame | None:
    """Check Parquet statistics before loading cached rows, then verify keys."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    boundary = pd.Timestamp(sealed_holdout_start)
    names = set(parquet.schema.names)
    columns = [c for c in names if c in TIMESTAMP_COLUMNS or c.startswith("exit_ts__")]
    for row_group in range(parquet.num_row_groups):
        metadata = parquet.metadata.row_group(row_group)
        for column in columns:
            index = parquet.schema.names.index(column)
            statistics = metadata.column(index).statistics
            if statistics is None or statistics.max is None:
                continue
            maximum = pd.Timestamp(statistics.max)
            limit = boundary.tz_localize("UTC") if maximum.tzinfo is not None else boundary
            if maximum >= limit:
                raise ValueError(f"Sealed holdout cache rejected in {context}: {column}>={sealed_holdout_start}")

    key_columns = [c for c in ("symbol","session_date","bar_start_ts","decision_ts") if c in names]
    if key_columns and verify_key_rows:
        keys=pd.read_parquet(path,columns=key_columns)
        assert_pre_holdout_frame(keys,sealed_holdout_start,context)
        return keys
    return None
