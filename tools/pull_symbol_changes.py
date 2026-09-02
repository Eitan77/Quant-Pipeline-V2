from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Alpaca symbol/name-change corporate actions.")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    from alpaca_research.client import AlpacaClient
    from alpaca_research.config import load_settings

    client = AlpacaClient(load_settings())
    rows: list[dict] = []
    params = {"start": args.start, "end": args.end, "types": "name_change", "limit": 1000}
    for page in client.paged_data_get("/v1/corporate-actions", params):
        rows.extend((page.get("corporate_actions") or {}).get("name_changes") or [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Alpaca returned no name-change actions")
    for column in ("old_symbol", "new_symbol"):
        frame[column] = frame[column].astype(str).str.upper()
    frame["process_date"] = pd.to_datetime(frame.process_date, errors="coerce")
    frame = frame.sort_values(["process_date", "old_symbol", "new_symbol", "id"]).drop_duplicates("id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    report = {
        "rows": len(frame),
        "ticker_to_ticker_rows": int(
            frame.old_symbol.str.fullmatch(r"[A-Z][A-Z.]{0,9}").fillna(False)
            .mul(frame.new_symbol.str.fullmatch(r"[A-Z][A-Z.]{0,9}").fillna(False)).sum()
        ),
        "start": args.start,
        "end": args.end,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
