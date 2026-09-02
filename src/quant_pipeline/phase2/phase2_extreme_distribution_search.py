from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2");OUT=ROOT/r"results\phase2_holy_shit_execution_first_through_20260430"
sp=importlib.util.spec_from_file_location("hs",ROOT/r"src\quant_pipeline\phase2\phase2_holy_shit_execution_first.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def select_extreme(x:pd.DataFrame,thr:float,max_side:int,direction:str,execution:str,h:int)->pd.DataFrame:
    z=x[x.signal.abs()>=thr].copy();z["direction"]=np.sign(z.signal)*(1 if direction=="continuation" else -1);z["strength"]=z.signal.abs()
    z["side_i"]=z.groupby(["bucket","direction"]).strength.rank(method="first",ascending=False);z=z[z.side_i<=max_side]
    if execution=="passive":z=z[((z.direction>0)&(z.min_ask_after5<=z.bid5))|((z.direction<0)&(z.max_bid_after5>=z.ask5))].copy()
    counts=z.groupby(["bucket","direction"]).size().unstack(fill_value=0)
    if -1 not in counts:counts[-1]=0
    if 1 not in counts:counts[1]=0
    keep=counts[[-1,1]].min(axis=1).clip(upper=max_side).rename("keep_n");z=z.drop(columns="side_i").join(keep,on="bucket");z["side_i"]=z.groupby(["bucket","direction"]).strength.rank(method="first",ascending=False);z=z[(z.keep_n>0)&(z.side_i<=z.keep_n)].copy()
    if execution=="passive":z["raw_ret"]=np.where(z.direction>0,z[f"bid5_f{h}"]/z.bid5-1,z.ask5/z[f"ask5_f{h}"]-1)
    else:z["raw_ret"]=np.where(z.direction>0,z[f"bid5_f{h}"]/z.ask5-1,z.bid5/z[f"ask5_f{h}"]-1)
    return z

def main():
    d=pd.read_parquet(OUT/"execution_feature_matrix.parquet");d.bucket=pd.to_datetime(d.bucket,utc=True);d=d.sort_values(["symbol","bucket"]);d["spread_bp"]=(d.ask5-d.bid5)/d.mid5*10000;d=d[(d.spread_bp<=8)&(d.quote_count>=5)&d.analysis_eligible].copy();fs=m.formulas(d)
    for n,(_,s) in fs.items():d[n]=s
    schedules={"continuous15":(d.minute_of_session>=15)&(d.minutes_until_close>=60)&(d.minute_of_session%15==0),"early":(d.minute_of_session>=15)&(d.minute_of_session<=90),"late":(d.minute_of_session>=240)&(d.minutes_until_close>=60)}
    rows=[];ledgers=[];threshold_rows=[]
    for h in (15,30,60):
      x=m.add_future(d,h);future=x[f"bid5_f{h}"].notna()&x[f"ask5_f{h}"].notna()
      for name,(family,_) in fs.items():
       x["signal"]=x[name]
       for sched,mask in schedules.items():
        keys=set(d.loc[mask,"bucket"].unique());base=x[future&x.bucket.isin(keys)&x.signal.notna()].copy();train=base[(base.session_date>=pd.to_datetime("2025-05-01").date())&(base.session_date<=pd.to_datetime("2025-08-31").date())]
        if len(train)<1000:continue
        for quant in (.90,.95,.99):
         thr=float(train.signal.abs().quantile(quant));threshold_rows.append(dict(feature=name,family=family,schedule=sched,horizon=h,quantile=quant,threshold=thr,train_rows=len(train)))
         if not np.isfinite(thr) or thr<=0:continue
         for direction in ("continuation","reversal"):
          for execution in ("market","passive"):
           q=select_extreme(base,thr,3,direction,execution,h)
           if len(q)<100:continue
           cid=f"{name}__{sched}__h{h}__q{int(quant*100)}__{direction}__{execution}";z=q[["bucket","session_date","symbol","direction","signal","raw_ret","bid5","ask5",f"bid5_f{h}",f"ask5_f{h}","spread_bp"]].copy();z["candidate_id"]=cid;ledgers.append(z)
           for cost in (1.,3.,5.):
            r=m.curve(q,cost,h,sched)
            for fold,lo,hi in (("train","2025-05-01","2025-08-31"),("development","2025-09-01","2025-12-31"),("recent","2026-01-01","2026-04-30"),("full","2025-05-01","2026-04-30")):
             rows.append(dict(candidate_id=cid,feature=name,family=family,schedule=sched,horizon=h,quantile=quant,direction=direction,execution=execution,cost_bp_side=cost,fold=fold,trades=len(q),**m.metrics(r.loc[lo:hi])))
    pd.DataFrame(threshold_rows).to_csv(OUT/"extreme_thresholds.csv",index=False);res=pd.DataFrame(rows);res.to_csv(OUT/"extreme_results.csv",index=False);pd.concat(ledgers,ignore_index=True).to_parquet(OUT/"extreme_trade_ledger.parquet",index=False)
    b=res[res.cost_bp_side.eq(3)].pivot(index="candidate_id",columns="fold",values="cagr");info=res[(res.cost_bp_side.eq(3))&(res.fold.eq("full"))].set_index("candidate_id");five=res[(res.cost_bp_side.eq(5))&(res.fold.eq("full"))].set_index("candidate_id").cagr.rename("cagr_5bp");r=info.join(b.add_prefix("cagr3_")).join(five)
    r["all_folds_positive"]=(r[["cagr3_train","cagr3_development","cagr3_recent"]]>0).all(axis=1);r["gate"]=(r.all_folds_positive)&(r.max_dd>=-.05)&(r.pt_days<=45)&(r.trades>=200)&(r.cagr_5bp>=0);r=r.sort_values(["gate","cagr3_full","sharpe"],ascending=False);r.to_csv(OUT/"extreme_ranking.csv");print(r.head(80).to_string())

if __name__=="__main__":main()
