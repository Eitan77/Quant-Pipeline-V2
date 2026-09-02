from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen V2 market-data snapshot.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2019-06-21")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--download-checkpoints", type=Path, action="append", default=[])
    args = parser.parse_args()

    temp_directory = args.output.parent / "audit_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.catalog), read_only=False)
    con.execute("SET threads TO 16")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{temp_directory.resolve().as_posix()}'")
    observed = con.execute("""
        SELECT symbol, session_date, COUNT(*) AS bars,
               COUNT(DISTINCT bar_start_ts_utc) AS natural_keys,
               COUNT(*) - COUNT(volume) AS null_volume,
               COUNT(*) - COUNT(vwap) AS null_vwap,
               COUNT(*) - COUNT(trade_count) AS null_trade_count
        FROM bars_1m_raw
        GROUP BY symbol, session_date
    """).fetchdf()
    expected = con.execute("""
        WITH dates AS (SELECT DISTINCT session_date FROM sp500_pit_membership_daily),
        wanted AS (
            SELECT symbol, session_date FROM sp500_pit_membership_daily WHERE in_universe
            UNION ALL SELECT 'SPY', session_date FROM dates
            UNION ALL SELECT 'QQQ', session_date FROM dates
        )
        SELECT DISTINCT symbol, session_date FROM wanted
    """).fetchdf()
    action_stats = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol),
               SUM(CASE WHEN split_factor > 0 THEN 1 ELSE 0 END)
        FROM corporate_actions
    """).fetchone()
    identity_stats = dict(con.execute("SELECT identity_source, COUNT(*) FROM security_master GROUP BY 1").fetchall())
    security = con.execute("SELECT security_id, symbol FROM security_master").fetchdf()
    con.close()

    observed_symbols = set(observed.symbol)
    rows = int(observed.bars.sum())
    natural_keys = int(observed.natural_keys.sum())
    minimum_date = observed.session_date.min()
    maximum_date = observed.session_date.max()
    security_ids = int(security.loc[security.symbol.isin(observed_symbols), "security_id"].nunique())
    holdout_rows = int(observed.loc[pd.to_datetime(observed.session_date) >= pd.Timestamp("2026-05-01"), "bars"].sum())
    null_volume = int(observed.null_volume.sum())
    null_vwap = int(observed.null_vwap.sum())
    null_trade_count = int(observed.null_trade_count.sum())

    merged = expected.merge(observed, on=["symbol", "session_date"], how="left")
    missing = merged.loc[merged.bars.isna(), ["symbol", "session_date"]].sort_values(["session_date", "symbol"])
    duplicate_rows = rows - natural_keys
    hard_failures = []
    if duplicate_rows:
        hard_failures.append(f"{duplicate_rows} duplicate natural-key rows")
    if holdout_rows:
        hard_failures.append(f"{holdout_rows} sealed-period rows")
    if null_volume or null_vwap or null_trade_count:
        hard_failures.append("null volume, VWAP, or trade-count values")
    provider_unavailable: set[tuple[str, str]] = set()
    for checkpoint_root in args.download_checkpoints:
        for path in checkpoint_root.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                continue
            date = str(payload.get("date"))
            provider_unavailable.update((str(symbol), date) for symbol in payload.get("requested_without_return", []))
    missing_pairs = set(zip(missing.symbol.astype(str), missing.session_date.astype(str)))
    confirmed_unavailable = missing_pairs & provider_unavailable
    unexpected_pairs = missing_pairs - confirmed_unavailable
    if unexpected_pairs:
        hard_failures.append(f"{len(unexpected_pairs)} unexpected missing symbol-sessions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing.to_parquet(args.output.with_name("missing_symbol_sessions.parquet"), index=False)
    report = {
        "status": "passed" if not hard_failures else "failed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_start": str(minimum_date),
        "snapshot_end": str(maximum_date),
        "rows": rows,
        "natural_keys": natural_keys,
        "duplicate_rows": duplicate_rows,
        "security_ids": security_ids,
        "holdout_rows": holdout_rows,
        "null_volume": null_volume,
        "null_vwap": null_vwap,
        "null_trade_count": null_trade_count,
        "expected_symbol_sessions": len(expected),
        "covered_symbol_sessions": int(merged.bars.notna().sum()),
        "missing_symbol_sessions": len(missing),
        "confirmed_provider_unavailable_symbol_sessions": len(confirmed_unavailable),
        "unexpected_missing_symbol_sessions": len(unexpected_pairs),
        "missing_examples": missing.head(20).assign(session_date=lambda x: x.session_date.astype(str)).to_dict("records"),
        "corporate_actions": int(action_stats[0]),
        "corporate_action_symbols": int(action_stats[1]),
        "split_actions": int(action_stats[2] or 0),
        "identity_quality": identity_stats,
        "hard_failures": hard_failures,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
