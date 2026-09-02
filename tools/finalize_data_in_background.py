from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "data_prep" / "finalization_status.json"
LOG = ROOT / "data_prep" / "finalization.log"
CHECKPOINTS = ROOT / "data_prep" / "download" / "checkpoints" / "1min"
EXPECTED_DATES = 507


def write_status(state: str, **details: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **details
    }, indent=2), encoding="utf-8")


def run(label: str, arguments: list[str]) -> None:
    write_status(label)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {label}\n")
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    try:
        deadline = time.monotonic() + 2 * 60 * 60
        while True:
            complete = len(list(CHECKPOINTS.glob("*.json")))
            write_status("waiting_for_download", completed_dates=complete, expected_dates=EXPECTED_DATES)
            if complete >= EXPECTED_DATES:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Download reached only {complete}/{EXPECTED_DATES} dates in two hours")
            time.sleep(30)

        run("compacting_snapshot", [
            "tools/compact_v2_bars.py",
            "--source-root", r"D:\AlgoResearch\data\raw\alpaca\market\stocks",
            "--membership", "data_prep/membership/sp500_pit_membership_daily.parquet",
            "--output-root", "data/raw/alpaca/market/stocks/bars_1m/feed=sip",
            "--start", "2019-06-21", "--cutoff", "2026-04-30",
        ])
        run("building_catalog", [
            "tools/prepare_v2_catalog.py",
            "--raw-root", "data/raw/alpaca/market/stocks",
            "--membership", "data_prep/membership/sp500_pit_membership_daily.parquet",
            "--assets", "reference/alpaca_security_master_20260901.parquet",
            "--actions", "reference/corporate_actions_through_20260430.parquet",
            "--symbol-changes", "reference/alpaca_symbol_changes_through_20260430.parquet",
            "--catalog", "data/catalog.duckdb",
            "--canonical-reference", "data/reference",
            "--start", "2019-06-21", "--cutoff", "2026-04-30",
        ])
        run("auditing_snapshot", [
            "tools/audit_v2_data.py", "--catalog", "data/catalog.duckdb",
            "--output", "data/DATA_VALIDATION.json",
            "--start", "2019-06-21", "--cutoff", "2026-04-30",
            "--download-checkpoints", "data_prep/download/checkpoints/1min",
            "--download-checkpoints", "data_prep/download_patch_20250807/checkpoints/1min",
        ])
        write_status("complete", catalog=str((ROOT / "data" / "catalog.duckdb").resolve()))
    except Exception as error:
        write_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
