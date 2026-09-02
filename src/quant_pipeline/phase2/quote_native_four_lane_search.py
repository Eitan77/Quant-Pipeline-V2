from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

LAKE = r"D:\AlgoResearch\data\raw\alpaca\market\stocks\quotes_sip\schema_v1\quote_lake.duckdb"
CATALOG = r"D:\AlgoResearch\data\catalog.duckdb"
OUT = Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2\results\phase2_quote_native_four_lanes_through_20260430")


def build_snapshots(out: Path) -> None:
    dest = out / "quote_5m_snapshots.parquet"
    if dest.exists():
        print(f"reuse {dest}", flush=True); return
    con = duckdb.connect(LAKE, read_only=True)
    sql = f"""
    COPY (
      WITH q AS (
        SELECT session_date,symbol,quote_ts,date_trunc('minute',quote_ts) bucket,
               bid_price,ask_price,bid_size,ask_size,(bid_price+ask_price)/2 mid
        FROM sip_quotes
        WHERE session_date BETWEEN DATE '2025-05-01' AND DATE '2026-04-30'
          AND bid_price>0 AND ask_price>=bid_price
          AND extract(minute FROM quote_ts)::INTEGER % 5 = 0
          AND extract(second FROM quote_ts) < 15
      )
      SELECT session_date,symbol,bucket,
        arg_min(mid,quote_ts) mid0,
        arg_min(mid,abs(epoch(quote_ts)-epoch(bucket+INTERVAL '5 seconds'))) mid5,
        arg_min(bid_price,abs(epoch(quote_ts)-epoch(bucket+INTERVAL '5 seconds'))) bid5,
        arg_min(ask_price,abs(epoch(quote_ts)-epoch(bucket+INTERVAL '5 seconds'))) ask5,
        arg_min(bid_size,abs(epoch(quote_ts)-epoch(bucket+INTERVAL '5 seconds'))) bid_size5,
        arg_min(ask_size,abs(epoch(quote_ts)-epoch(bucket+INTERVAL '5 seconds'))) ask_size5,
        min(ask_price) FILTER(WHERE quote_ts>=bucket+INTERVAL '5 seconds') min_ask_after5,
        max(bid_price) FILTER(WHERE quote_ts>=bucket+INTERVAL '5 seconds') max_bid_after5,
        count(*) quote_count
      FROM q GROUP BY session_date,symbol,bucket
      HAVING count(*)>=2
    ) TO '{dest.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD)
    """
    con.execute(sql); print(f"built {dest}", flush=True)


def parse_auction(value: str) -> float:
    try:
        rows=json.loads(value or "[]")
        return float(rows[0]["p"]) if rows else np.nan
    except Exception: return np.nan


def load(out: Path) -> pd.DataFrame:
    d=pd.read_parquet(out/"quote_5m_snapshots.parquet")
    d["bucket"]=pd.to_datetime(d.bucket,utc=True).dt.tz_convert("America/Los_Angeles"); d=d.sort_values(["symbol","bucket"])
    d["move5"]=d.mid5/d.mid0-1
    d["imb5"]=(d.bid_size5-d.ask_size5)/(d.bid_size5+d.ask_size5).replace(0,np.nan)
    d["spread_bp"]=(d.ask5-d.bid5)/d.mid5*10000
    return d


def add_future(d: pd.DataFrame, minutes: int) -> pd.DataFrame:
    f=d[["symbol","bucket","bid5","ask5","mid5"]].copy(); f["bucket"]=f.bucket-pd.Timedelta(minutes=minutes)
    return d.merge(f,on=["symbol","bucket"],how="left",suffixes=("",f"_f{minutes}"))


def select_neutral(x: pd.DataFrame, score: str, direction: str, n: int=5) -> pd.DataFrame:
    longs=x[x[direction]>0].nlargest(n,score); shorts=x[x[direction]<0].nlargest(n,score)
    if len(longs)==0 or len(shorts)==0: return x.iloc[0:0]
    k=min(len(longs),len(shorts)); return pd.concat([longs.head(k),shorts.head(k)])


def neutral_by_bucket(x: pd.DataFrame) -> pd.DataFrame:
    parts=[select_neutral(g,"score","direction") for _,g in x.groupby("bucket",sort=False)]
    return pd.concat(parts,ignore_index=True) if parts else x.iloc[0:0].copy()


def market_return(x: pd.DataFrame, horizon: int) -> pd.Series:
    fb=f"bid5_f{horizon}"; fa=f"ask5_f{horizon}"
    return np.where(x.direction>0,x[fb]/x.ask5-1,x.bid5/x[fa]-1)


def passive_return(x: pd.DataFrame, horizon: int) -> pd.Series:
    fb=f"bid5_f{horizon}"; fa=f"ask5_f{horizon}"
    return np.where(x.direction>0,x[fb]/x.bid5-1,x.ask5/x[fa]-1)


def portfolio(trades: pd.DataFrame, extra_bp_side: float, hold_buckets: int) -> pd.Series:
    if trades.empty: return pd.Series(dtype=float)
    t=trades.copy(); t["net_ret"]=t.raw_ret-extra_bp_side/10000*2
    g=t.groupby("bucket").net_ret.mean()/hold_buckets
    idx=pd.date_range(g.index.min(),g.index.max(),freq="5min",tz=g.index.tz)
    return g.reindex(idx,fill_value=0.0)


def metrics(r: pd.Series) -> dict:
    if r.empty: return {"cagr":np.nan,"sharpe":np.nan,"max_dd":np.nan,"pt_days":np.nan}
    eq=(1+r).cumprod(); peak=pd.concat([pd.Series([1.0],index=[r.index[0]-pd.Timedelta(seconds=1)]),eq]).cummax().iloc[1:]; dd=eq/peak-1
    years=max((r.index[-1]-r.index[0]).total_seconds()/31557600,1/252)
    cagr=float(eq.iloc[-1]**(1/years)-1); daily=r.groupby(r.index.date).sum()
    sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0.0
    end=dd.idxmin(); before=eq.loc[:end]; pv=max(1.0,float(before.max())); pdte=r.index[0] if pv==1 else before.idxmax()
    return {"cagr":cagr,"sharpe":sharpe,"max_dd":float(dd.min()),"pt_days":int((end-pdte).total_seconds()/86400)}


def folds(r: pd.Series):
    for name,lo,hi in (("full","2025-05-01","2026-04-30"),("fold1","2025-05-01","2025-08-31"),("fold2","2025-09-01","2025-12-31"),("fold3","2026-01-01","2026-04-30")):
        yield name,r.loc[lo:hi]


def evaluate(out: Path) -> None:
    d=load(out); print(d[["move5","imb5","spread_bp","quote_count"]].quantile([.1,.5,.9,.95,.99]).to_string(),flush=True)
    rows=[]; ledgers=[]
    for horizon in (5,15):
      x=add_future(d,horizon); valid=x[f"bid5_f{horizon}"].notna()&x[f"ask5_f{horizon}"].notna()&(x.spread_bp<=10)&(x.move5.abs()>=.0001)&(x.imb5.abs()>=.15)
      base=x[valid].copy(); base["score"]=base.move5.abs()
      for lane,sgn,passive in (("micro_reversal",-1,True),("micro_momentum",1,False)):
        q=base[np.sign(base.move5)*np.sign(base.imb5)==sgn].copy(); q["direction"]=sgn*np.sign(q.move5)
        q=neutral_by_bucket(q)
        if passive:
            q=q[((q.direction>0)&(q.min_ask_after5<=q.bid5))|((q.direction<0)&(q.max_bid_after5>=q.ask5))].copy(); q["raw_ret"]=passive_return(q,horizon)
        else: q["raw_ret"]=market_return(q,horizon)
        q["lane"]=lane; q["horizon"]=horizon; ledgers.append(q)
        for cost in (0.,1.,2.):
          r=portfolio(q,cost,horizon//5)
          for fold,rr in folds(r): rows.append({"lane":lane,"variant":f"h{horizon}","cost_bp_side":cost,"fold":fold,"trades":len(q),**metrics(rr)})

    # Quote-native component lead/lag: fade dispersion around the live cross-sectional move.
    x=add_future(d,5); leader=x.groupby("bucket").move5.median().rename("leader_move")
    z=x.join(leader,on="bucket"); z["residual_move"]=z.move5-z.leader_move
    z=z[(z.leader_move.abs()>=.0001)&(z.residual_move.abs()>=.0001)&(z.spread_bp<=10)&z.bid5_f5.notna()].copy()
    z["direction"]=-np.sign(z.residual_move); z["score"]=z.residual_move.abs(); z=neutral_by_bucket(z); z["raw_ret"]=market_return(z,5)
    z["lane"]="component_lead_lag"; z["horizon"]=5; ledgers.append(z)
    for cost in (0.,1.,2.):
      r=portfolio(z,cost,1)
      for fold,rr in folds(r): rows.append({"lane":"component_lead_lag","variant":"h5_neutral","cost_bp_side":cost,"fold":fold,"trades":len(z),**metrics(rr)})

    # Opening auction dislocation and causal 12:45->12:50 close-auction signal.
    cat=duckdb.connect(CATALOG,read_only=True); a=cat.execute("select date,symbol,o,c from auctions where date between DATE '2025-05-01' and DATE '2026-04-30'").fetchdf(); a["open_px"]=a.o.map(parse_auction); a["close_px"]=a.c.map(parse_auction)
    a.date=pd.to_datetime(a.date).dt.date
    loc=add_future(d,5); loc["clock"]=loc.bucket.dt.strftime("%H:%M")
    op=loc[loc.clock.eq("06:35")].merge(a,left_on=["session_date","symbol"],right_on=["date","symbol"]); op=op[(op.open_px>0)&(op.spread_bp<=10)&op.bid5_f5.notna()]; op["signal"]=op.mid5/op.open_px-1
    cl45=loc[loc.clock.eq("12:45")][["session_date","symbol","mid5"]].rename(columns={"mid5":"mid45"}); cl=loc[loc.clock.eq("12:50")].merge(cl45,on=["session_date","symbol"]); cl=cl.merge(a,left_on=["session_date","symbol"],right_on=["date","symbol"]); cl=cl[(cl.close_px>0)&(cl.spread_bp<=10)]; cl["signal"]=cl.mid5/cl.mid45-1
    for session,base in (("open",op),("close",cl)):
      for style,sgn in (("reversal",-1),("momentum",1)):
        q=base[base.signal.abs()>=.0005].copy(); q["direction"]=sgn*np.sign(q.signal); q["score"]=q.signal.abs(); q=neutral_by_bucket(q)
        if session=="open": q["raw_ret"]=market_return(q,5)
        else: q["raw_ret"]=np.where(q.direction>0,q.close_px/q.ask5-1,q.bid5/q.close_px-1)
        lane=f"{session}_{style}"; q["lane"]=lane; q["horizon"]=5 if session=="open" else 70; ledgers.append(q)
        for cost in (0.,1.,2.):
          r=portfolio(q,cost,1)
          for fold,rr in folds(r): rows.append({"lane":lane,"variant":"fixed","cost_bp_side":cost,"fold":fold,"trades":len(q),**metrics(rr)})
    pd.DataFrame(rows).to_csv(out/"results.csv",index=False)
    pd.concat(ledgers,ignore_index=True,sort=False).to_parquet(out/"trade_ledger.parquet",index=False)
    rank=pd.DataFrame(rows); full=rank[rank.fold.eq("full")].sort_values(["cost_bp_side","cagr"],ascending=[True,False]); full.to_csv(out/"full_ranking.csv",index=False); print(full.to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("all","build","evaluate"),default="all"); ap.add_argument("--out",default=str(OUT)); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if a.stage in ("all","build"): build_snapshots(out)
    if a.stage in ("all","evaluate"): evaluate(out)

if __name__=="__main__": main()
