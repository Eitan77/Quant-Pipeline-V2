from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


FALLBACK_NAMESPACE = uuid.UUID("9d2b4b51-b558-42e4-8f5f-39ad8e44e07f")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_groups(membership: pd.DataFrame, changes: pd.DataFrame) -> dict[str, set[str]]:
    parent: dict[str, str] = {}

    def find(symbol: str) -> str:
        parent.setdefault(symbol, symbol)
        if parent[symbol] != symbol:
            parent[symbol] = find(parent[symbol])
        return parent[symbol]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    member_days = {
        symbol: pd.DatetimeIndex(pd.to_datetime(group.session_date))
        for symbol, group in membership.groupby("symbol", observed=True)
    }
    ticker = r"[A-Z][A-Z.]{0,9}"
    for row in changes.itertuples(index=False):
        old, new = str(row.old_symbol), str(row.new_symbol)
        if not pd.Series([old, new]).str.fullmatch(ticker).all() or old == new or old not in member_days:
            continue
        process_date = pd.Timestamp(row.process_date).normalize()
        prior = member_days[old]
        if not ((prior <= process_date) & (prior >= process_date - pd.Timedelta(days=10))).any():
            continue
        if new in member_days:
            following = member_days[new]
            overlap_after = set(prior[(prior >= process_date) & (prior <= process_date + pd.Timedelta(days=10))]) & set(
                following[(following >= process_date) & (following <= process_date + pd.Timedelta(days=10))]
            )
            if overlap_after:
                continue
        union(old, new)
    groups: dict[str, set[str]] = {}
    for symbol in parent:
        groups.setdefault(find(symbol), set()).add(symbol)
    return {symbol: group for group in groups.values() for symbol in group}


def select_security_ids(
    symbols: list[str], assets: pd.DataFrame, membership: pd.DataFrame, changes: pd.DataFrame
) -> pd.DataFrame:
    assets = assets.copy()
    assets["symbol"] = assets.symbol.astype(str).str.upper()
    rows: list[dict] = []
    groups = _identity_groups(membership, changes)
    for symbol in sorted(set(symbols)):
        aliases = groups.get(symbol, {symbol})
        candidates = assets.loc[assets.symbol.isin(aliases)].copy()
        active = candidates.loc[candidates.status.eq("active")]
        if len(active) == 1:
            chosen = active.iloc[0]
            quality = "alpaca_asset_id_symbol_change_linked" if len(aliases) > 1 else "alpaca_asset_id_active"
        elif len(candidates) == 1:
            chosen = candidates.iloc[0]
            quality = "alpaca_asset_id_symbol_change_linked" if len(aliases) > 1 else "alpaca_asset_id_inactive"
        elif len(candidates):
            chosen = candidates.sort_values(["status", "security_id"], ascending=[True, True]).iloc[0]
            quality = "alpaca_asset_id_ambiguous_symbol"
        else:
            rows.append({
                "security_id": str(uuid.uuid5(FALLBACK_NAMESPACE, f"alpaca-us-equity:{symbol}")),
                "symbol": symbol,
                "asset_status": "historical_or_unlisted",
                "exchange": None,
                "identity_source": "deterministic_symbol_fallback",
            })
            continue
        rows.append({
            "security_id": str(chosen.security_id),
            "symbol": symbol,
            "asset_status": str(chosen.status),
            "exchange": chosen.get("exchange"),
            "identity_source": quality,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone Quant Pipeline V2 data catalog.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--symbol-changes", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--canonical-reference", type=Path, required=True)
    parser.add_argument("--start", default="2019-06-21")
    parser.add_argument("--cutoff", default="2026-04-30")
    args = parser.parse_args()

    membership = pd.read_parquet(args.membership)
    membership["session_date"] = pd.to_datetime(membership.pop("date")).dt.date
    membership["symbol"] = membership.symbol.astype(str).str.upper()
    membership = membership.loc[
        membership.is_member
        & (pd.to_datetime(membership.session_date) >= pd.Timestamp(args.start))
        & (pd.to_datetime(membership.session_date) <= pd.Timestamp(args.cutoff))
    ].copy()

    all_symbols = sorted(set(membership.symbol) | {"SPY", "QQQ"})
    changes = pd.read_parquet(args.symbol_changes)
    security = select_security_ids(all_symbols, pd.read_parquet(args.assets), membership, changes)
    membership = membership.merge(security[["security_id", "symbol"]], on="symbol", how="left", validate="many_to_one")
    membership = membership.rename(columns={"is_member": "in_universe"})

    actions = pd.read_parquet(args.actions).copy()
    actions["symbol"] = actions.symbol.astype(str).str.upper()
    actions["session_date"] = pd.to_datetime(actions.ex_date).dt.date
    actions = actions.loc[
        actions.symbol.isin(all_symbols)
        & (pd.to_datetime(actions.session_date) <= pd.Timestamp(args.cutoff))
    ].merge(security[["security_id", "symbol"]], on="symbol", how="left", validate="many_to_one")
    actions["split_factor"] = pd.to_numeric(actions.split_ratio, errors="coerce")

    args.canonical_reference.mkdir(parents=True, exist_ok=True)
    membership_path = args.canonical_reference / "sp500_pit_membership_daily.parquet"
    security_path = args.canonical_reference / "security_master.parquet"
    actions_path = args.canonical_reference / "corporate_actions.parquet"
    membership.to_parquet(membership_path, index=False)
    security.to_parquet(security_path, index=False)
    actions.to_parquet(actions_path, index=False)

    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    if args.catalog.exists():
        args.catalog.unlink()
    raw_glob = (args.raw_root.resolve() / "bars_1m" / "feed=sip" / "**" / "*.parquet").as_posix().replace("'", "''")
    con = duckdb.connect(str(args.catalog))
    con.execute("SET threads TO 16")
    con.register("membership_frame", membership)
    con.register("security_frame", security)
    con.register("actions_frame", actions)
    con.execute("CREATE TABLE sp500_pit_membership_daily AS SELECT * FROM membership_frame")
    con.execute("CREATE TABLE security_master AS SELECT * FROM security_frame")
    con.execute("CREATE TABLE corporate_actions AS SELECT * FROM actions_frame")
    con.execute("""
        CREATE TABLE split_factors_daily AS
        WITH dates AS (SELECT DISTINCT session_date FROM sp500_pit_membership_daily),
        eligible AS (
            SELECT DISTINCT s.security_id, d.session_date
            FROM security_master s CROSS JOIN dates d
        )
        SELECT e.security_id, e.session_date,
               COALESCE(EXP(SUM(LN(a.split_factor)) FILTER (
                   WHERE a.split_factor > 0 AND a.session_date > e.session_date
               )), 1.0) AS split_factor
        FROM eligible e
        LEFT JOIN corporate_actions a ON a.security_id = e.security_id
        GROUP BY e.security_id, e.session_date
    """)
    con.execute(f"""
        CREATE VIEW bars_1m_raw AS
        WITH source AS (
            SELECT * FROM read_parquet('{raw_glob}', union_by_name=true, hive_partitioning=true)
            WHERE lower(feed) = 'sip' AND lower(adjustment) = 'raw'
              AND lower(timeframe) IN ('1min', '1m')
              AND TRY_CAST(timestamp AS TIMESTAMPTZ) >= TIMESTAMPTZ '{args.start} 00:00:00 America/New_York'
              AND TRY_CAST(timestamp AS TIMESTAMPTZ) < TIMESTAMPTZ '{pd.Timestamp(args.cutoff) + pd.Timedelta(days=1):%Y-%m-%d} 00:00:00 America/New_York'
        )
        SELECT s.security_id, b.symbol,
               TRY_CAST(b.timestamp AS TIMESTAMPTZ) AS bar_start_ts_utc,
               TRY_CAST(b.timestamp AS TIMESTAMPTZ) + INTERVAL 1 MINUTE AS bar_end_ts_utc,
               TRY_CAST(b.timestamp AS TIMESTAMPTZ) + INTERVAL 1 MINUTE AS availability_ts_utc,
               CAST(TRY_CAST(b.timestamp AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS session_date,
               b.open, b.high, b.low, b.close, b.volume, b.vwap, b.trade_count,
               lower(b.feed) AS feed, lower(b.adjustment) AS adjustment,
               b.source_ingestion_id AS ingest_batch_id
        FROM source b JOIN security_master s USING (symbol)
    """)
    con.execute("""
        CREATE VIEW bars_1m_research AS
        SELECT b.security_id, b.bar_start_ts_utc, b.session_date,
               b.open / f.split_factor AS research_open,
               b.high / f.split_factor AS research_high,
               b.low / f.split_factor AS research_low,
               b.close / f.split_factor AS research_close,
               f.split_factor, 'split_consistent' AS price_basis
        FROM bars_1m_raw b
        JOIN split_factors_daily f USING (security_id, session_date)
    """)
    con.close()

    parquet_files = sorted((args.raw_root / "bars_1m" / "feed=sip").glob("year=*/month=*/date=*/*.parquet"))
    parquet_files = [p for p in parquet_files if args.start <= p.parent.name.removeprefix("date=") <= args.cutoff]
    raw_file_records = [
        {
            "relative_path": path.relative_to(args.raw_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in parquet_files
    ]
    raw_files_manifest = args.catalog.parent / "raw_files_manifest.parquet"
    pd.DataFrame(raw_file_records).to_parquet(raw_files_manifest, index=False)
    raw_snapshot_hash = hashlib.sha256(
        json.dumps(raw_file_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standalone_catalog": str(args.catalog.resolve()),
        "raw_source_root": str(args.raw_root.resolve()),
        "raw_partition_files": len(parquet_files),
        "raw_partition_bytes": sum(p.stat().st_size for p in parquet_files),
        "raw_snapshot_sha256": raw_snapshot_hash,
        "raw_files_manifest": str(raw_files_manifest.resolve()),
        "snapshot_start": args.start,
        "snapshot_end": args.cutoff,
        "discovery_start_after_258_warmup_sessions": "2020-07-01",
        "sealed_replication_start": "2026-05-01",
        "reference_sha256": {
            membership_path.name: file_sha256(membership_path),
            security_path.name: file_sha256(security_path),
            actions_path.name: file_sha256(actions_path),
        },
        "identity_quality": security.identity_source.value_counts().to_dict(),
        "membership_rows": len(membership),
        "membership_sessions": int(membership.session_date.nunique()),
        "symbols_including_benchmarks": len(security),
        "corporate_action_rows": len(actions),
    }
    manifest_path = args.catalog.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
