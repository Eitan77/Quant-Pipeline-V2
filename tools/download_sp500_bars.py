from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any
import uuid

import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    import orjson
except ImportError:  # pragma: no cover - the standard-library fallback is tested by execution
    orjson = None


HOLDOUT_START = pd.Timestamp("2026-05-01", tz="UTC")
INGESTION_ID = "CAM-0508_SP500_RAW_SIP_V1"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for key in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        if not values.get(key):
            raise RuntimeError(f"missing credential variable {key}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / float(requests_per_minute)
        self.lock = threading.Lock()
        self.next_request_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_request_at)
            self.next_request_at = scheduled + self.interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


class AlpacaBarsClient:
    def __init__(
        self,
        env: dict[str, str],
        *,
        workers: int,
        requests_per_minute: int,
        timeout_seconds: float = 120.0,
        max_retries: int = 7,
    ) -> None:
        self.base_url = env.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
        self.headers = {
            "APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"],
            "Accept": "application/json",
        }
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.local = threading.local()
        self.workers = workers

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=self.workers,
                pool_maxsize=self.workers,
                max_retries=0,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update(self.headers)
            self.local.session = session
        return session

    def get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            response = self.session().get(
                f"{self.base_url}/v2/stocks/bars",
                params=params,
                timeout=self.timeout_seconds,
            )
            if response.status_code in (401, 403):
                raise RuntimeError(f"Alpaca authentication/permission failure {response.status_code}: {response.text[:500]}")
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(self.retry_delay(response, attempt))
                continue
            response.raise_for_status()
            payload = orjson.loads(response.content) if orjson is not None else response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Alpaca bars response was not a JSON object")
            return payload
        raise AssertionError("retry loop exhausted")

    @staticmethod
    def retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(120.0, max(0.2, float(retry_after)))
            except ValueError:
                pass
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return min(120.0, max(0.2, float(reset) - time.time()))
            except ValueError:
                pass
        return min(60.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.5)


def load_membership(path: Path, start: pd.Timestamp, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "symbol", "is_member"}
    if not required.issubset(frame.columns):
        raise ValueError(f"membership is missing columns: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[
        frame["is_member"].astype(bool)
        & frame["date"].between(start.tz_localize(None), cutoff.tz_localize(None))
    ].copy()
    if frame.empty:
        raise ValueError("membership has no rows in requested range")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("membership contains duplicate date-symbol rows")
    if frame["date"].max() >= HOLDOUT_START.tz_localize(None):
        raise ValueError("membership crosses sealed holdout")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def candidate_probe_dates(group: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
    if len(dates) <= 4:
        selected = list(dates)
    else:
        selected = [dates[0], dates[len(dates) // 2], dates[-1]]
    return [pd.Timestamp(value).normalize() for value in selected]


def fetch_pages(
    client: AlpacaBarsClient,
    *,
    symbols: list[str],
    timeframe: str,
    session_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]], int]:
    start = session_date.date().isoformat() + "T00:00:00Z"
    end = (session_date + pd.Timedelta(days=1)).date().isoformat() + "T00:00:00Z"
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": "raw",
        "feed": "sip",
        "sort": "asc",
        "limit": 10_000,
    }
    token: str | None = None
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    pages = 0
    while True:
        query = dict(params)
        if token:
            query["page_token"] = token
        payload = client.get_json(query)
        pages += 1
        page_rows: list[tuple[str, Any, Any, Any, Any, Any, Any, Any, Any]] = []
        for returned_symbol, values in (payload.get("bars") or {}).items():
            if not isinstance(values, list):
                raise RuntimeError(f"unexpected bars payload for {returned_symbol}")
            for bar in values:
                page_rows.append(
                    (
                        str(returned_symbol).upper(),
                        bar["t"],
                        bar["o"],
                        bar["h"],
                        bar["l"],
                        bar["c"],
                        bar["v"],
                        bar.get("n"),
                        bar.get("vw"),
                    )
                )
        if page_rows:
            # Parsing timestamps once per response is substantially cheaper
            # than invoking pandas for every individual minute bar.
            timestamps = pd.to_datetime([item[1] for item in page_rows], utc=True, errors="coerce")
            local_dates = timestamps.tz_convert("America/New_York").normalize().tz_localize(None)
            for item, timestamp, local_date in zip(page_rows, timestamps, local_dates):
                if pd.isna(timestamp) or local_date != session_date:
                    continue
                rows.append(
                    {
                        "symbol": item[0],
                        "timestamp": timestamp,
                        "open": item[2],
                        "high": item[3],
                        "low": item[4],
                        "close": item[5],
                        "volume": item[6],
                        "trade_count": item[7],
                        "vwap": item[8],
                    }
                )
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        if not isinstance(next_token, str) or next_token in seen:
            raise RuntimeError("Alpaca pagination token did not advance")
        seen.add(next_token)
        token = next_token
    return rows, pages


def resolve_provider_symbols(
    client: AlpacaBarsClient,
    membership: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("membership_sha256") == sha256(MEMBERSHIP_PATH_FOR_HASH):
            return existing

    groups = {
        str(symbol): group.copy()
        for symbol, group in membership.groupby("symbol", sort=True)
    }
    result: dict[str, dict[str, Any]] = {}

    def probe(source_symbol: str, group: pd.DataFrame) -> tuple[str, dict[str, Any]]:
        errors: list[str] = []
        for probe_date in candidate_probe_dates(group):
            try:
                rows, pages = fetch_pages(
                    client,
                    symbols=[source_symbol],
                    timeframe="1Day",
                    session_date=probe_date,
                )
                returned = sorted({str(row["symbol"]).upper() for row in rows})
                if returned:
                    return source_symbol, {
                        "status": "resolved",
                        "provider_symbol": returned[0],
                        "probe_date": str(probe_date.date()),
                        "pages": pages,
                        "returned_symbols": returned,
                    }
                errors.append(f"{probe_date.date()}:empty")
            except Exception as exc:  # preserve the failure for the final audit
                errors.append(f"{probe_date.date()}:{type(exc).__name__}:{exc}")
        return source_symbol, {
            "status": "unresolved",
            "provider_symbol": source_symbol,
            "probe_date": None,
            "pages": 0,
            "returned_symbols": [],
            "errors": errors,
        }

    with ThreadPoolExecutor(max_workers=client.workers, thread_name_prefix="sp500-alias") as executor:
        futures = [executor.submit(probe, symbol, group) for symbol, group in groups.items()]
        for future in as_completed(futures):
            symbol, item = future.result()
            result[symbol] = item
            if len(result) % 25 == 0 or len(result) == len(groups):
                print(f"provider alias probes {len(result)}/{len(groups)}", flush=True)

    report = {
        "status": "passed",
        "membership_sha256": sha256(MEMBERSHIP_PATH_FOR_HASH),
        "source_symbol_count": len(groups),
        "resolved_count": sum(item["status"] == "resolved" for item in result.values()),
        "unresolved_count": sum(item["status"] == "unresolved" for item in result.values()),
        "mapping": dict(sorted(result.items())),
        "credentials_recorded": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def existing_coverage(
    output_root: Path,
    timeframe: str,
    target_symbols: list[str],
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    temp_root: Path,
    threads: int,
) -> pd.DataFrame:
    dataset = output_root / "feed=sip" / "year=*" / "month=*" / "date=*" / "*.parquet"
    if not output_root.exists() or not list(output_root.glob("feed=sip/**/*.parquet")):
        return pd.DataFrame(columns=["symbol", "date", "bar_count"])
    temp_root.mkdir(parents=True, exist_ok=True)
    symbols = pd.DataFrame({"symbol": sorted(set(target_symbols))})
    with duckdb.connect() as connection:
        connection.execute(f"SET threads={max(1, int(threads))}")
        connection.execute(f"SET temp_directory='{temp_root.as_posix()}'")
        connection.register("target_symbols", symbols)
        query = f"""
            SELECT b.symbol,
                   CAST(TRY_CAST(b.timestamp AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS date,
                   COUNT(DISTINCT b.timestamp) AS bar_count
            FROM read_parquet('{dataset.as_posix()}', union_by_name=true, hive_partitioning=true) AS b
            JOIN target_symbols AS t ON t.symbol = b.symbol
            WHERE b.feed = 'sip'
              AND b.adjustment = 'raw'
              AND b.timeframe = ?
              AND CAST(TRY_CAST(b.timestamp AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) BETWEEN ? AND ?
            GROUP BY 1, 2
        """
        frame = connection.execute(
            query,
            [timeframe, start.date(), cutoff.tz_localize(None).date()],
        ).fetch_df()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "date", "bar_count"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["bar_count"] = frame["bar_count"].astype("int64")
    return frame


def write_partition(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    timeframe: str,
    session_date: pd.Timestamp,
    filename_prefix: str = "part-cam0508",
) -> Path | None:
    if frame.empty:
        return None
    frame = frame.drop_duplicates(["symbol", "timestamp"], keep="last").sort_values(["symbol", "timestamp"])
    if frame["timestamp"].max() >= HOLDOUT_START:
        raise RuntimeError("downloaded rows cross sealed holdout")
    frame["feed"] = "sip"
    frame["timeframe"] = timeframe
    frame["adjustment"] = "raw"
    frame["date"] = session_date.date()
    frame["month"] = session_date.strftime("%Y-%m")
    frame["year"] = session_date.year
    frame["source_ingestion_id"] = INGESTION_ID
    frame["ingested_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    destination = (
        output_root
        / "feed=sip"
        / f"year={session_date.year:04d}"
        / f"month={session_date.month:02d}"
        / f"date={session_date.date()}"
        / f"{filename_prefix}-{timeframe.lower()}-{session_date.date()}-{uuid.uuid4().hex}.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def task_signature(timeframe: str, session_date: pd.Timestamp, symbols: list[str], membership_hash: str) -> str:
    body = json.dumps(
        {
            "timeframe": timeframe,
            "date": str(session_date.date()),
            "symbols": symbols,
            "membership_sha256": membership_hash,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def run_timeframe(
    client: AlpacaBarsClient,
    membership: pd.DataFrame,
    *,
    timeframe: str,
    output_root: Path,
    artifact_dir: Path,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    membership_hash: str,
    batch_size: int,
    threads: int,
    min_existing_bars: int,
) -> dict[str, Any]:
    coverage_path = artifact_dir / f"existing_coverage_{timeframe.lower()}.parquet"
    coverage = existing_coverage(
        output_root,
        timeframe,
        sorted(membership["provider_symbol"].unique()),
        start,
        cutoff,
        artifact_dir / "duckdb_tmp",
        min(16, threads),
    )
    coverage.to_parquet(coverage_path, index=False)
    covered = {
        (row.symbol, pd.Timestamp(row.date).normalize())
        for row in coverage.itertuples(index=False)
        if int(row.bar_count) >= min_existing_bars
    }
    expected_by_date: dict[pd.Timestamp, list[str]] = {}
    for row in membership[["date", "provider_symbol"]].drop_duplicates().itertuples(index=False):
        expected_by_date.setdefault(pd.Timestamp(row.date).normalize(), []).append(str(row.provider_symbol))
    tasks: list[tuple[pd.Timestamp, list[str]]] = []
    for session_date, symbols in sorted(expected_by_date.items()):
        missing = sorted({symbol for symbol in symbols if (symbol, session_date) not in covered})
        if missing:
            tasks.append((session_date, missing))
    checkpoint_dir = artifact_dir / "checkpoints" / timeframe.lower()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[pd.Timestamp, list[str], str]] = []
    checkpoint_skips = 0
    for session_date, symbols in tasks:
        signature = task_signature(timeframe, session_date, symbols, membership_hash)
        checkpoint = checkpoint_dir / f"{session_date.date()}.json"
        if checkpoint.exists():
            try:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                saved = {}
            if saved.get("status") == "complete" and saved.get("task_signature") == signature:
                checkpoint_skips += 1
                continue
        pending.append((session_date, symbols, signature))

    summary = {
        "timeframe": timeframe,
        "expected_symbol_sessions": int(len(membership[["date", "provider_symbol"]].drop_duplicates())),
        "existing_covered_symbol_sessions": int(len(covered)),
        "planned_dates_with_missing_coverage": len(tasks),
        "checkpoint_skips": checkpoint_skips,
        "pending_dates": len(pending),
        "requested_symbols": int(len(set(membership["provider_symbol"]))),
        "api_pages": 0,
        "downloaded_rows": 0,
        "completed_dates": 0,
        "empty_dates": 0,
        "failures": [],
    }
    print(
        f"{timeframe}: expected_sessions={summary['expected_symbol_sessions']} existing={summary['existing_covered_symbol_sessions']} "
        f"pending_dates={len(pending)} workers={threads} rpm={client.rate_limiter.interval and round(60.0 / client.rate_limiter.interval)}",
        flush=True,
    )
    if not pending:
        (artifact_dir / f"run_summary_{timeframe.lower()}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    started = time.monotonic()

    def run_one(session_date: pd.Timestamp, symbols: list[str], signature: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        pages = 0
        returned_symbols: set[str] = set()
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset : offset + batch_size]
            batch_rows, batch_pages = fetch_pages(
                client,
                symbols=batch,
                timeframe=timeframe,
                session_date=session_date,
            )
            rows.extend(batch_rows)
            pages += batch_pages
            returned_symbols.update(row["symbol"] for row in batch_rows)
        frame = pd.DataFrame(rows)
        path = write_partition(
            frame,
            output_root=output_root,
            timeframe=timeframe,
            session_date=session_date,
        )
        checkpoint = checkpoint_dir / f"{session_date.date()}.json"
        result = {
            "status": "complete",
            "timeframe": timeframe,
            "date": str(session_date.date()),
            "task_signature": signature,
            "requested_symbols": len(symbols),
            "returned_symbols": sorted(returned_symbols),
            "requested_without_return": sorted(set(symbols) - returned_symbols),
            "pages": pages,
            "rows": int(len(frame)),
            "output": str(path) if path else None,
            "holdout_rows_loaded": 0,
        }
        checkpoint.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix=f"sp500-{timeframe}") as executor:
        futures = {
            executor.submit(run_one, session_date, symbols, signature): (session_date, symbols)
            for session_date, symbols, signature in pending
        }
        for future in as_completed(futures):
            session_date, symbols = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                error = {
                    "date": str(session_date.date()),
                    "symbols": len(symbols),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                summary["failures"].append(error)
                print(f"{timeframe} FAILED {error}", flush=True)
                continue
            summary["api_pages"] += int(result["pages"])
            summary["downloaded_rows"] += int(result["rows"])
            summary["completed_dates"] += 1
            if not result["rows"]:
                summary["empty_dates"] += 1
            completed = summary["completed_dates"]
            elapsed = max(0.1, time.monotonic() - started)
            rate = completed / elapsed
            eta = (len(pending) - completed) / rate if rate else 0.0
            print(
                f"{timeframe} {completed}/{len(pending)} date={session_date.date()} rows={result['rows']} pages={result['pages']} "
                f"eta={eta/60:.1f}m",
                flush=True,
            )
    summary["finished_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    summary["run_seconds"] = time.monotonic() - started
    (artifact_dir / f"run_summary_{timeframe.lower()}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["failures"]:
        raise RuntimeError(f"{timeframe} completed with {len(summary['failures'])} failed dates; see run summary")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\AlgoResearch\data\raw\alpaca\market\stocks"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--start", default="2021-05-01")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--timeframes", default="1Min,1Day")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--requests-per-minute", type=int, default=198)
    parser.add_argument("--batch-1m", type=int, default=25)
    parser.add_argument("--batch-1d", type=int, default=500)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize().tz_localize("UTC")
    cutoff = pd.Timestamp(args.cutoff).normalize().tz_localize("UTC")
    if cutoff >= HOLDOUT_START or start > cutoff:
        raise ValueError("invalid acquisition range or sealed-boundary request")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.requests_per_minute > 200:
        raise ValueError("configured rate exceeds the observed account rate limit")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    membership = load_membership(args.membership, start, cutoff)
    membership_hash = sha256(args.membership)
    global MEMBERSHIP_PATH_FOR_HASH
    MEMBERSHIP_PATH_FOR_HASH = args.membership
    env = load_env(args.env_file)
    client = AlpacaBarsClient(
        env,
        workers=args.workers,
        requests_per_minute=args.requests_per_minute,
    )
    alias_path = args.artifact_dir / "provider_symbol_map.json"
    alias_report = resolve_provider_symbols(client, membership, alias_path)
    mapping = {
        source: item.get("provider_symbol", source)
        for source, item in alias_report["mapping"].items()
    }
    membership["provider_symbol"] = membership["symbol"].map(mapping).fillna(membership["symbol"])
    membership_path = args.artifact_dir / "membership_with_provider_symbols.parquet"
    membership.to_parquet(membership_path, index=False)
    if args.probe_only:
        print(json.dumps({"status": "probe_complete", "alias_report": alias_report, "membership": str(membership_path)}, indent=2))
        return

    timeframes = [value.strip() for value in args.timeframes.split(",") if value.strip()]
    allowed = {"1Min", "1Day"}
    if not set(timeframes).issubset(allowed):
        raise ValueError(f"only raw {sorted(allowed)} are allowed; derive longer bars locally")
    summaries: dict[str, Any] = {}
    for timeframe in timeframes:
        batch_size = args.batch_1m if timeframe == "1Min" else args.batch_1d
        timeframe_root = args.output_root / ("bars_1m" if timeframe == "1Min" else "bars_1d")
        summaries[timeframe] = run_timeframe(
            client,
            membership,
            timeframe=timeframe,
            output_root=timeframe_root,
            artifact_dir=args.artifact_dir,
            start=start,
            cutoff=cutoff,
            membership_hash=membership_hash,
            batch_size=batch_size,
            threads=args.workers,
            min_existing_bars=1,
        )
    report = {
        "status": "passed",
        "requested_start": str(start.date()),
        "requested_cutoff": str(cutoff.date()),
        "holdout_rows_loaded": 0,
        "membership_sha256": membership_hash,
        "alias_report": {
            "source_symbol_count": alias_report["source_symbol_count"],
            "resolved_count": alias_report["resolved_count"],
            "unresolved_count": alias_report["unresolved_count"],
        },
        "timeframes": summaries,
        "credentials_recorded": False,
    }
    report_path = args.artifact_dir / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


MEMBERSHIP_PATH_FOR_HASH = Path(".")


if __name__ == "__main__":
    main()
