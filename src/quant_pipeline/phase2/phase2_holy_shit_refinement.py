from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2")
OUT=ROOT/r"results\phase2_holy_shit_execution_first_through_20260430"
sp=importlib.util.spec_from_file_location("hs",ROOT/r"src\quant_pipeline\phase2\phase2_holy_shit_execution_first.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def rebalance(q:pd.DataFrame)->pd.DataFrame:
    if q.empty:return q
    q=q.drop(columns=["keep_n","side_i"],errors="ignore")
    counts=q.groupby(["bucket","direction"]).size().unstack(fill_value=0)
    if -1 not in counts:counts[-1]=0
    if 1 not in counts:counts[1]=0
    keep=counts[[-1,1]].min(axis=1).rename("keep_n");z=q.join(keep,on="bucket");z["side_i"]=z.groupby(["bucket","direction"]).cumcount()
    return z[(z.keep_n>0)&(z.side_i<z.keep_n)].copy()

def main():
    d=pd.read_parquet(OUT/"execution_feature_matrix.parquet");d.bucket=pd.to_datetime(d.bucket,utc=True);d=d.sort_values(["symbol","bucket"]);d["spread_bp"]=(d.ask5-d.bid5)/d.mid5*10000;d=d[(d.spread_bp<=8)&(d.quote_count>=5)&d.analysis_eligible].copy()
    fs=m.formulas(d)
    for name,(_,sig) in fs.items():d[name]=sig
    rank=pd.read_csv(OUT/"execution_ranking.csv");pool=rank[(rank.all_folds_positive)&(rank.trades>=300)&(rank.cagr_full>0)&(rank.max_dd>=-.10)].copy()
    pool=pool.sort_values(["family","execution","cagr_full"],ascending=[True,True,False]).groupby(["family","execution"]).head(2)
    rows=[];rets={};ledgers=[]
    for _,mrow in pool.iterrows():
        h=int(mrow.horizon);x=m.add_future(d,h);x=x[x[f"bid5_f{h}"].notna()&x[f"ask5_f{h}"].notna()&(x.spread_bp<=8)&(x.quote_count>=5)].copy();x["signal"]=x[mrow.feature]
        if mrow.schedule=="early":x=x[(x.minute_of_session>=15)&(x.minute_of_session<=90)]
        elif mrow.schedule=="late":x=x[(x.minute_of_session>=240)&(x.minutes_until_close>=60)]
        else:x=x[(x.minute_of_session>=15)&(x.minutes_until_close>=60)&(x.minute_of_session%15==0)]
        grp=x.groupby("bucket");x["signal_zgap"]=(grp.signal.transform("max")-grp.signal.transform("min"))/grp.signal.transform("std").replace(0,np.nan)
        selected=m.selected_base(x,int(mrow.n));q=m.execute_selected(selected,mrow.direction,mrow.execution,h);q["vwap_agree"]=q.direction*(q.close_adjusted/q.vwap_adjusted-1)>0;q["move_agree"]=q.direction*q.return_3>0
        event_rv=q.groupby("bucket").tod_relative_volume_20.transform("mean");event_z=q.groupby("bucket").signal_zgap.transform("first")
        masks={"all":pd.Series(True,index=q.index),"signal_z3":event_z>=3,"high_volume":event_rv>=1.25,"vwap_agree":q.vwap_agree,"move_agree":q.move_agree,"high_market_vol":q.high_market_vol.astype(bool),"low_market_vol":~q.high_market_vol.astype(bool),"tight_spread":q.spread_bp<=4,"strong_move":q.return_3.abs()>=.0015,"volume_vwap":(q.tod_relative_volume_20>=1)&q.vwap_agree,"signal_volume":(event_z>=3)&(event_rv>=1.25)}
        for flt,mask in masks.items():
            z=rebalance(q[mask].copy())
            if len(z)<200:continue
            cid=f"{mrow.candidate_id}__filter_{flt}";led=z[["bucket","session_date","symbol","direction","signal","raw_ret","bid5","ask5",f"bid5_f{h}",f"ask5_f{h}","spread_bp"]].copy();led["candidate_id"]=cid;ledgers.append(led)
            for cost in (1.,3.,5.):
                r=m.curve(z,cost,h,mrow.schedule);rets[(cid,cost)]=r
                for fold,lo,hi in (("train","2025-05-01","2025-08-31"),("development","2025-09-01","2025-12-31"),("recent","2026-01-01","2026-04-30"),("full","2025-05-01","2026-04-30")):
                    rows.append(dict(candidate_id=cid,base_candidate=mrow.candidate_id,feature=mrow.feature,family=mrow.family,schedule=mrow.schedule,horizon=h,n=int(mrow.n),direction=mrow.direction,execution=mrow.execution,filter=flt,cost_bp_side=cost,fold=fold,trades=len(z),**m.metrics(r.loc[lo:hi])))
    res=pd.DataFrame(rows);res.to_csv(OUT/"refinement_results.csv",index=False);pd.concat(ledgers,ignore_index=True).to_parquet(OUT/"refinement_trade_ledger.parquet",index=False)
    b=res[res.cost_bp_side.eq(3)].pivot(index="candidate_id",columns="fold",values="cagr");info=res[(res.cost_bp_side.eq(3))&(res.fold.eq("full"))].set_index("candidate_id");five=res[(res.cost_bp_side.eq(5))&(res.fold.eq("full"))].set_index("candidate_id").cagr.rename("cagr_5bp")
    r=info.join(b.add_prefix("cagr3_")).join(five);r["all_folds_positive"]=(r[["cagr3_train","cagr3_development","cagr3_recent"]]>0).all(axis=1);r["gate"]=(r.all_folds_positive)&(r.max_dd>=-.05)&(r.pt_days<=45)&(r.trades>=300)&(r.cagr_5bp>=-.01);r=r.sort_values(["gate","cagr3_full","sharpe"],ascending=False);r.to_csv(OUT/"refinement_ranking.csv");print(r.head(60).to_string())

if __name__=="__main__":main()
