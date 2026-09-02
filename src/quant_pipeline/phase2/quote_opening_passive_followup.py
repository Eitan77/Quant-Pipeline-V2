from pathlib import Path
import importlib.util
import pandas as pd

ROOT=Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2")
OUT=ROOT/r"results\phase2_quote_native_four_lanes_through_20260430"
spec=importlib.util.spec_from_file_location("four",ROOT/r"src\quant_pipeline\phase2\quote_native_four_lane_search.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pd.read_parquet(OUT/"trade_ledger.parquet")
rows=[]
for lane in ("open_momentum","open_reversal"):
    q=d[d.lane.eq(lane)].copy()
    q=q[((q.direction>0)&(q.min_ask_after5<=q.bid5))|((q.direction<0)&(q.max_bid_after5>=q.ask5))]
    q=m.neutral_by_bucket(q); q["raw_ret"]=m.passive_return(q,5)
    for cost in (0.,1.,2.):
        r=m.portfolio(q,cost,1)
        for fold,rr in m.folds(r): rows.append({"lane":lane,"entry":"passive_trade_through","cost_bp_side":cost,"fold":fold,"trades":len(q),**m.metrics(rr)})
pd.DataFrame(rows).to_csv(OUT/"opening_passive_results.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
