from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


HOLDOUT_START = pd.Timestamp("2026-05-01", tz="UTC")


def coverage(
    connection: duckdb.DuckDBPyConnection,
    *,
    root: Path,
    timeframe: str,
    target_symbols: pd.DataFrame,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    dataset = root / "feed=sip" / "year=*" / "month=*" / "date=*" / "*.parquet"
    if not list(root.glob("feed=sip/**/*.parquet")):
        raise FileNotFoundError(f"no parquet files under {root}")
    connection.register("target_symbols", target_symbols)
    query = f"""
        SELECT b.symbol,
               TRY_CAST(b.date AS DATE) AS date,
               COUNT(*)::BIGINT AS row_count,
               COUNT(DISTINCT b.timestamp)::BIGINT AS bar_count
        FROM read_parquet('{dataset.as_posix()}', union_by_name=true, hive_partitioning=true) AS b
        JOIN target_symbols AS t ON t.symbol = b.symbol
        WHERE b.feed = 'sip'
          AND b.adjustment = 'raw'
          AND b.timeframe = ?
          AND TRY_CAST(b.date AS DATE) BETWEEN ? AND ?
        GROUP BY 1, 2
    """
    frame = connection.execute(query, [timeframe, start.date(), cutoff.date()]).fetch_df()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "date", "row_count", "bar_count"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["row_count"] = frame["row_count"].astype("int64")
    frame["bar_count"] = frame["bar_count"].astype("int64")
    return frame


def integrity_summary(
    connection: duckdb.DuckDBPyConnection,
    *,
    root: Path,
    timeframe: str,
    target_symbols: pd.DataFrame,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    dataset = root / "feed=sip" / "year=*" / "month=*" / "date=*" / "*.parquet"
    connection.register("target_symbols_integrity", target_symbols)
    row = connection.execute(
        f"""
        SELECT COUNT(*)::BIGINT AS rows,
               COUNT(DISTINCT (b.symbol, TRY_CAST(b.timestamp AS TIMESTAMPTZ)))::BIGINT AS distinct_natural_keys,
               MAX(TRY_CAST(b.timestamp AS TIMESTAMPTZ)) AS max_timestamp,
               SUM(CASE WHEN TRY_CAST(b.timestamp AS TIMESTAMPTZ) >= TIMESTAMPTZ '2026-05-01 00:00:00+00:00' THEN 1 ELSE 0 END)::BIGINT AS holdout_rows,
               SUM(CASE WHEN TRY_CAST(b.date AS DATE) < ? OR TRY_CAST(b.date AS DATE) > ? THEN 1 ELSE 0 END)::BIGINT AS out_of_range_rows
        FROM read_parquet('{dataset.as_posix()}', union_by_name=true, hive_partitioning=true) AS b
        JOIN target_symbols_integrity AS t ON t.symbol = b.symbol
        WHERE b.feed = 'sip'
          AND b.adjustment = 'raw'
          AND b.timeframe = ?
          AND TRY_CAST(b.date AS DATE) BETWEEN ? AND ?
        """,
        [start.date(), cutoff.date(), timeframe, start.date(), cutoff.date()],
    ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "distinct_natural_keys": int(row[1] or 0),
        "duplicate_natural_key_rows": int((row[0] or 0) - (row[1] or 0)),
        "max_timestamp": str(row[2]) if row[2] is not None else None,
        "holdout_rows": int(row[3] or 0),
        "out_of_range_rows": int(row[4] or 0),
    }


def session_summary(expected: pd.DataFrame, observed: pd.DataFrame) -> dict[str, Any]:
    merged = expected.merge(observed, on=["symbol", "date"], how="left")
    merged["bar_count"] = merged["bar_count"].fillna(0).astype("int64")
    merged["row_count"] = merged["row_count"].fillna(0).astype("int64")
    covered = merged["bar_count"] >= 1
    missing = merged.loc[~covered, ["symbol", "date"]]
    counts = merged.loc[covered, "bar_count"]
    return {
        "expected_symbol_sessions": int(len(merged)),
        "covered_symbol_sessions": int(covered.sum()),
        "missing_symbol_sessions": int((~covered).sum()),
        "covered_fraction": float(covered.mean()) if len(covered) else 0.0,
        "missing_examples": [
            {"symbol": str(row.symbol), "date": str(row.date.date())}
            for row in missing.head(20).itertuples(index=False)
        ],
        "bar_count_min_covered": int(counts.min()) if not counts.empty else 0,
        "bar_count_median_covered": float(counts.median()) if not counts.empty else 0.0,
        "bar_count_max_covered": int(counts.max()) if not counts.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-base", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--start", default="2021-05-01")
    parser.add_argument("--cutoff", default="2026-04-30")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize().tz_localize("UTC")
    cutoff = pd.Timestamp(args.cutoff).normalize().tz_localize("UTC")
    if cutoff >= HOLDOUT_START:
        raise ValueError("audit range crosses sealed holdout")
    membership = pd.read_parquet(args.membership)
    required = {"date", "provider_symbol", "is_member"}
    if not required.issubset(membership.columns):
        raise ValueError(f"membership missing {sorted(required - set(membership.columns))}")
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    expected = (
        membership.loc[
            membership["is_member"].astype(bool)
            & membership["date"].between(start.tz_localize(None), cutoff.tz_localize(None)),
            ["date", "provider_symbol"],
        ]
        .drop_duplicates()
        .rename(columns={"provider_symbol": "symbol"})
    )
    expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
    target_symbols = pd.DataFrame({"symbol": sorted(expected["symbol"].unique())})

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = artifact_dir / "duckdb_audit_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.catalog), read_only=True) as connection:
        connection.execute("SET threads=16")
        connection.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
        observed_1m = coverage(
            connection,
            root=args.raw_base / "bars_1m",
            timeframe="1Min",
            target_symbols=target_symbols,
            start=start,
            cutoff=cutoff,
        )
        observed_1d = coverage(
            connection,
            root=args.raw_base / "bars_1d",
            timeframe="1Day",
            target_symbols=target_symbols,
            start=start,
            cutoff=cutoff,
        )
        integrity_1m = integrity_summary(
            connection,
            root=args.raw_base / "bars_1m",
            timeframe="1Min",
            target_symbols=target_symbols,
            start=start,
            cutoff=cutoff,
        )
        integrity_1d = integrity_summary(
            connection,
            root=args.raw_base / "bars_1d",
            timeframe="1Day",
            target_symbols=target_symbols,
            start=start,
            cutoff=cutoff,
        )

    observed_1m.to_parquet(artifact_dir / "final_coverage_1min.parquet", index=False)
    observed_1d.to_parquet(artifact_dir / "final_coverage_1day.parquet", index=False)
    summary_1m = session_summary(expected, observed_1m)
    summary_1d = session_summary(expected, observed_1d)
    keys_1m = set(zip(observed_1m.loc[observed_1m.bar_count >= 1, "symbol"], observed_1m.loc[observed_1m.bar_count >= 1, "date"]))
    keys_1d = set(zip(observed_1d.loc[observed_1d.bar_count >= 1, "symbol"], observed_1d.loc[observed_1d.bar_count >= 1, "date"]))
    report = {
        "status": "passed",
        "requested_start": str(start.date()),
        "requested_cutoff": str(cutoff.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "membership_rows": int(len(membership)),
        "expected_symbol_sessions": int(len(expected)),
        "unique_provider_symbols": int(len(target_symbols)),
        "timeframes": {
            "1Min": {"coverage": summary_1m, "integrity": integrity_1m},
            "1Day": {"coverage": summary_1d, "integrity": integrity_1d},
        },
        "cross_timeframe": {
            "minute_covered_without_daily": int(len(keys_1m - keys_1d)),
            "daily_covered_without_minute": int(len(keys_1d - keys_1m)),
            "both_covered": int(len(keys_1m & keys_1d)),
        },
        "artifacts": {
            "final_coverage_1min": str(artifact_dir / "final_coverage_1min.parquet"),
            "final_coverage_1day": str(artifact_dir / "final_coverage_1day.parquet"),
        },
    }
    if any(
        item["integrity"]["holdout_rows"] or item["integrity"]["out_of_range_rows"]
        for item in report["timeframes"].values()
    ):
        report["status"] = "failed_boundary"
    (artifact_dir / "final_coverage_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
