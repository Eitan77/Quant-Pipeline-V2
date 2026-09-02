from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--symbols-file")
    parser.add_argument("--catalog")
    parser.add_argument("--membership-table",default="qqq_pit_membership_daily")
    parser.add_argument("--start",required=True)
    parser.add_argument("--end",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    load_dotenv(Path(os.environ.get("ALGO_RESEARCH_ENV",r"D:\AlgoResearch\.env")))
    from alpaca_research.client import AlpacaClient
    from alpaca_research.config import load_settings
    if args.symbols_file:symbols=[s.strip().upper() for s in Path(args.symbols_file).read_text().replace("\n",",").split(",") if s.strip()]
    elif args.catalog:
        import duckdb
        with duckdb.connect(args.catalog,read_only=True) as connection:symbols=[row[0] for row in connection.execute(f"select distinct symbol from {args.membership_table} where is_member order by symbol").fetchall()]
        symbols=sorted(set(symbols)|{"QQQ"})
    else:raise ValueError("Provide --symbols-file or --catalog")
    client=AlpacaClient(load_settings()); rows=[]
    for start in range(0,len(symbols),50):
        params={"start":args.start,"end":args.end,"symbols":",".join(symbols[start:start+50]),"limit":1000}
        for page in client.paged_data_get("/v1/corporate-actions",params):
            groups=page.get("corporate_actions") or {}
            if isinstance(groups,list): groups={"unknown":groups}
            for action_type,items in groups.items():
                for item in items or []:
                    row={"action_type":action_type,**item}
                    if action_type in {"forward_splits","reverse_splits","unit_splits"}:
                        old=float(item.get("old_rate") or 1); new=float(item.get("new_rate") or 1)
                        row["split_ratio"]=new/old if old else None
                    if "dividend" in action_type:
                        row["cash_amount"]=item.get("cash") or item.get("rate") or item.get("amount")
                    rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty: raise RuntimeError("Corporate-action request returned no rows")
    for column in ["ex_date","process_date","effective_date"]:
        if column in frame: frame[column]=pd.to_datetime(frame[column],errors="coerce")
    for column in ["split_ratio","cash_amount","currency"]:
        if column not in frame:frame[column]=None
    ledger=frame.loc[frame.symbol.notna()&frame.ex_date.notna()&((frame.split_ratio.notna())|(frame.cash_amount.notna())),["symbol","ex_date","process_date","action_type","split_ratio","cash_amount","currency","id"]].copy()
    ledger=ledger.sort_values(["symbol","ex_date","id"]).drop_duplicates(["symbol","ex_date","id"])
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); ledger.to_parquet(output,index=False)
    print(f"wrote {len(ledger)} split/dividend actions to {output}")


if __name__=="__main__": main()
