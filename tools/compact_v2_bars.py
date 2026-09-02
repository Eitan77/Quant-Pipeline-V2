from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and compact the V2 raw SIP minute snapshot.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", default="2019-06-21")
    parser.add_argument("--cutoff", default="2026-04-30")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output snapshot is not empty: {args.output_root.resolve()}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(args.membership, columns=["symbol"])
    symbols = pd.DataFrame({"symbol": sorted(set(map(str, membership.symbol.unique())) | {"SPY", "QQQ"})})
    source_glob = (args.source_root / "bars_1m" / "feed=sip" / "**" / "*.parquet").as_posix().replace("'", "''")
    output = args.output_root.as_posix().replace("'", "''")

    con = duckdb.connect()
    con.execute("SET threads TO 16")
    con.execute("SET preserve_insertion_order=false")
    con.register("wanted_symbols", symbols)
    periods = pd.period_range(pd.Timestamp(args.start), pd.Timestamp(args.cutoff), freq="M")
    for period in periods:
        month_start = max(pd.Timestamp(args.start), period.start_time)
        month_end = min(pd.Timestamp(args.cutoff), period.end_time.normalize())
        partition_start = month_start - pd.Timedelta(days=1)
        partition_end = month_end + pd.Timedelta(days=1)
        session_date = "CAST(TRY_CAST(timestamp AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE)"
        query = f"""
            COPY (
            WITH source AS (
                SELECT r.*
                FROM read_parquet('{source_glob}', union_by_name=true, hive_partitioning=true) r
                JOIN wanted_symbols w USING (symbol)
                WHERE lower(feed) = 'sip' AND lower(adjustment) = 'raw'
                  AND lower(timeframe) IN ('1min', '1m')
                  AND TRY_CAST(date AS DATE) BETWEEN DATE '{partition_start:%Y-%m-%d}' AND DATE '{partition_end:%Y-%m-%d}'
                  AND {session_date} BETWEEN DATE '{month_start:%Y-%m-%d}' AND DATE '{month_end:%Y-%m-%d}'
            )
            SELECT symbol, timestamp, open, high, low, close, volume, trade_count, vwap,
                   lower(feed) AS feed, '1Min' AS timeframe, lower(adjustment) AS adjustment,
                   source_ingestion_id, ingested_at,
                   YEAR({session_date}) AS year,
                   LPAD(CAST(MONTH({session_date}) AS VARCHAR), 2, '0') AS month,
                   CAST({session_date} AS VARCHAR) AS date
            FROM source
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY symbol, timestamp, lower(timeframe), lower(feed), lower(adjustment)
                ORDER BY COALESCE(TRY_CAST(ingested_at AS TIMESTAMP), TIMESTAMP '1900-01-01') DESC,
                         COALESCE(source_ingestion_id, '') DESC
            ) = 1
            ) TO '{output}' (
            FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year, month, date),
            ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE true
            )
        """
        con.execute(query)
        print(f"compacted {period}", flush=True)
    con.close()

    files = list(args.output_root.glob("year=*/month=*/date=*/*.parquet"))
    report = {
        "status": "complete",
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "symbols_requested": len(symbols),
        "start": args.start,
        "cutoff": args.cutoff,
        "output_root": str(args.output_root.resolve()),
    }
    (args.output_root.parent / "compact_snapshot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
