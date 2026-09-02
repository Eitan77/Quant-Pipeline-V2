from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Alpaca's US-equity asset directory.")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership", type=Path)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    from alpaca_research.client import AlpacaClient
    from alpaca_research.config import load_settings

    client = AlpacaClient(load_settings())
    rows: list[dict] = []
    for status in ("active", "inactive"):
        payload = client.trading_get("/v2/assets", {"status": status, "asset_class": "us_equity"})
        if not isinstance(payload, list):
            raise TypeError(f"Unexpected Alpaca assets response: {type(payload).__name__}")
        rows.extend(payload)

    if args.membership:
        membership = pd.read_parquet(args.membership, columns=["symbol"])
        wanted = sorted(set(map(str, membership.symbol.unique())) | {"SPY", "QQQ"})
        present = {str(row.get("symbol", "")).upper() for row in rows}
        for symbol in wanted:
            if symbol in present:
                continue
            try:
                asset = client.trading_get(f"/v2/assets/{symbol}")
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    continue
                raise
            if isinstance(asset, dict) and asset.get("id"):
                rows.append(asset)
                present.add(symbol)

    frame = pd.DataFrame(rows)
    required = {"id", "symbol", "status", "exchange"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Asset directory missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.rename(columns={"id": "security_id", "class": "asset_class"})
    frame["symbol"] = frame.symbol.astype(str).str.upper()
    frame["snapshot_ts_utc"] = datetime.now(timezone.utc).isoformat()
    frame = frame.sort_values(["symbol", "status", "security_id"], kind="mergesort").drop_duplicates(
        ["symbol", "security_id"], keep="last"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    report = {
        "rows": len(frame),
        "symbols": int(frame.symbol.nunique()),
        "security_ids": int(frame.security_id.nunique()),
        "statuses": frame.status.value_counts().to_dict(),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
