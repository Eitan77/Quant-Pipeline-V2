from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


HOLDOUT_START = pd.Timestamp("2026-05-01")


@dataclass(frozen=True)
class Snapshot:
    date: pd.Timestamp
    symbols: frozenset[str]


def normalize_symbol(value: str) -> str:
    value = str(value).strip().upper()
    return value.replace("-", ".")


def read_snapshots(
    path: Path,
    cutoff: pd.Timestamp,
    *,
    allow_duplicate_dates: bool = False,
) -> tuple[list[Snapshot], pd.Timestamp | None]:
    snapshots: list[Snapshot] = []
    first_excluded: pd.Timestamp | None = None
    previous: pd.Timestamp | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "tickers"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            current = pd.Timestamp(row["date"]).normalize()
            if current > cutoff:
                first_excluded = current
                break
            if previous is not None and current < previous:
                raise ValueError(f"{path} dates are not strictly increasing at {current.date()}")
            symbols = frozenset(
                normalize_symbol(token)
                for token in str(row.get("tickers", "")).split(",")
                if str(token).strip()
            )
            if not symbols:
                raise ValueError(f"{path} has an empty snapshot on {current.date()}")
            if previous is not None and current == previous:
                if not allow_duplicate_dates:
                    raise ValueError(f"{path} has duplicate snapshot date at {current.date()}")
                # Some reconstructed histories contain multiple same-day
                # changes.  The final row for that date is the end-of-day
                # membership used by the session-level cross-check.
                snapshots[-1] = Snapshot(current, symbols)
            else:
                snapshots.append(Snapshot(current, symbols))
            previous = current
    if not snapshots:
        raise ValueError(f"{path} produced no cutoff-bounded snapshots")
    return snapshots, first_excluded


def load_sessions(catalog: Path, start: pd.Timestamp, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    with duckdb.connect(str(catalog), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT TRY_CAST(date AS DATE) AS date
            FROM calendar
            WHERE TRY_CAST(date AS DATE) BETWEEN ? AND ?
            ORDER BY 1
            """,
            [start.date(), cutoff.date()],
        ).fetchall()
    sessions = pd.DatetimeIndex(pd.to_datetime([row[0] for row in rows])).normalize()
    if sessions.empty:
        raise ValueError("calendar returned no sessions")
    # The requested start/end may fall on non-trading days; the first session
    # may therefore be later than ``start`` while still being in scope.
    if sessions[0] < start or sessions[-1] > cutoff:
        raise ValueError("calendar session range violates acquisition boundary")
    return sessions


def expand_snapshots(snapshots: list[Snapshot], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    if snapshots[0].date > sessions[0]:
        raise ValueError("primary membership source has no snapshot before acquisition start")
    snapshot_dates = pd.DatetimeIndex([item.date for item in snapshots]).astype("datetime64[ns]")
    session_dates = pd.DatetimeIndex(sessions).astype("datetime64[ns]")
    snapshot_frame = pd.DataFrame(
        {
            "source_snapshot_date": snapshot_dates,
            "source_symbols": [sorted(item.symbols) for item in snapshots],
        }
    ).sort_values("source_snapshot_date")
    calendar_frame = pd.DataFrame({"date": session_dates})
    expanded = pd.merge_asof(
        calendar_frame,
        snapshot_frame,
        left_on="date",
        right_on="source_snapshot_date",
        direction="backward",
    )
    rows: list[dict[str, object]] = []
    for row in expanded.itertuples(index=False):
        for symbol in row.source_symbols:
            rows.append(
                {
                    "date": row.date.date(),
                    "symbol": symbol,
                    "is_member": True,
                    "source_snapshot_date": row.source_snapshot_date.date(),
                    "membership_source": "fja05680/sp500 historical components updated",
                    "membership_source_quality": "open_historical_reconstruction_provisional",
                }
            )
    result = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("expanded membership contains duplicate date-symbol rows")
    return result


def active_sets(snapshots: list[Snapshot], sessions: pd.DatetimeIndex) -> dict[pd.Timestamp, set[str]]:
    pointer = 0
    current: set[str] = set()
    result: dict[pd.Timestamp, set[str]] = {}
    for session in sessions:
        while pointer < len(snapshots) and snapshots[pointer].date <= session:
            current = set(snapshots[pointer].symbols)
            pointer += 1
        if not current:
            raise ValueError(f"no membership snapshot available on {session.date()}")
        result[session] = set(current)
    return result


def compare_sources(
    primary: list[Snapshot],
    secondary: list[Snapshot],
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, object], pd.DataFrame]:
    overlap_end = min(primary[-1].date, secondary[-1].date)
    overlap_sessions = sessions[sessions <= overlap_end]
    primary_sets = active_sets(primary, overlap_sessions)
    secondary_sets = active_sets(secondary, overlap_sessions)
    mismatch_rows: list[dict[str, object]] = []
    for session in overlap_sessions:
        left = primary_sets[session]
        right = secondary_sets[session]
        if left != right:
            mismatch_rows.append(
                {
                    "date": session.date(),
                    "primary_count": len(left),
                    "secondary_count": len(right),
                    "primary_only": ",".join(sorted(left - right)),
                    "secondary_only": ",".join(sorted(right - left)),
                }
            )
    mismatch_frame = pd.DataFrame(mismatch_rows)
    report: dict[str, object] = {
        "overlap_start": str(overlap_sessions[0].date()) if len(overlap_sessions) else None,
        "overlap_end": str(overlap_sessions[-1].date()) if len(overlap_sessions) else None,
        "overlap_sessions": int(len(overlap_sessions)),
        "mismatch_sessions": int(len(mismatch_frame)),
        "mismatch_fraction": float(len(mismatch_frame) / len(overlap_sessions)) if len(overlap_sessions) else None,
        "primary_is_retained": True,
        "decision": "retain_primary_fja_reconstruction_and_preserve_secondary_disagreement",
    }
    if not mismatch_frame.empty:
        report["first_mismatch_date"] = str(mismatch_frame.iloc[0]["date"])
        report["last_mismatch_date"] = str(mismatch_frame.iloc[-1]["date"])
        report["examples"] = mismatch_frame.head(10).astype(str).to_dict("records")
    return report, mismatch_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: object) -> object:
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path(r"D:\AlgoResearch\data\catalog.duckdb"))
    parser.add_argument("--start", default="2021-05-01")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    cutoff = pd.Timestamp(args.cutoff).normalize()
    if cutoff >= HOLDOUT_START or start > cutoff:
        raise ValueError("invalid acquisition range or sealed-boundary request")
    sessions = load_sessions(args.catalog, start, cutoff)
    primary, primary_first_excluded = read_snapshots(args.primary, cutoff)
    secondary, secondary_first_excluded = read_snapshots(
        args.secondary,
        cutoff,
        allow_duplicate_dates=True,
    )
    membership = expand_snapshots(primary, sessions)
    comparison, mismatch_frame = compare_sources(primary, secondary, sessions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = args.output_dir / "sp500_pit_membership_daily.parquet"
    intervals_path = args.output_dir / "sp500_pit_membership_intervals.csv"
    comparison_path = args.output_dir / "secondary_mismatch_sessions.csv"
    report_path = args.output_dir / "membership_report.json"
    membership.to_parquet(membership_path, index=False)

    interval_rows: list[dict[str, object]] = []
    for symbol, group in membership.groupby("symbol", sort=True):
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"])).sort_values()
        if len(dates) == 0:
            continue
        block_start = dates[0]
        previous = dates[0]
        for current in dates[1:]:
            if (current - previous).days > 7:
                interval_rows.append({"symbol": symbol, "start_date": block_start.date(), "end_date": previous.date()})
                block_start = current
            previous = current
        interval_rows.append({"symbol": symbol, "start_date": block_start.date(), "end_date": previous.date()})
    pd.DataFrame(interval_rows).to_csv(intervals_path, index=False)
    mismatch_frame.to_csv(comparison_path, index=False)

    source_hashes = {path.name: sha256(path) for path in (args.primary, args.secondary)}
    counts = membership.groupby("date").size()
    report = {
        "status": "passed",
        "primary_source": str(args.primary),
        "secondary_source": str(args.secondary),
        "source_sha256": source_hashes,
        "requested_start": str(start.date()),
        "requested_cutoff": str(cutoff.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "calendar_sessions": int(len(sessions)),
        "membership_rows": int(len(membership)),
        "membership_dates": int(membership["date"].nunique()),
        "unique_source_symbols": int(membership["symbol"].nunique()),
        "members_per_session_min": int(counts.min()),
        "members_per_session_max": int(counts.max()),
        "members_per_session_mean": float(counts.mean()),
        "primary_snapshot_rows_loaded": len(primary),
        "primary_snapshot_min": str(primary[0].date.date()),
        "primary_snapshot_max": str(primary[-1].date.date()),
        "primary_first_excluded_after_cutoff": str(primary_first_excluded.date()) if primary_first_excluded is not None else None,
        "secondary_snapshot_rows_loaded": len(secondary),
        "secondary_snapshot_min": str(secondary[0].date.date()),
        "secondary_snapshot_max": str(secondary[-1].date.date()),
        "secondary_first_excluded_after_cutoff": str(secondary_first_excluded.date()) if secondary_first_excluded is not None else None,
        "secondary_comparison": comparison,
        "holdout_rows_loaded": 0,
        "artifacts": {
            "membership": str(membership_path),
            "intervals": str(intervals_path),
            "secondary_mismatches": str(comparison_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=json_ready) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
