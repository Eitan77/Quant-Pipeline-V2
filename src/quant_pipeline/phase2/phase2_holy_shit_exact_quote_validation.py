from __future__ import annotations
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor,as_completed
from hashlib import sha1
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

from quant_pipeline.alpha_discovery.data.alpaca_ingest import AlpacaHistoricalClient
from quant_pipeline.alpha_discovery.data.quote_lake import load_quote_windows,quote_window_coverage
from quant_pipeline.phase2.phase2_holy_shit_execution_first import formulas

QP=Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2");SEARCH=QP/r"results\phase2_holy_shit_execution_first_through_20260430";ROOT=Path(r"D:\AlgoResearch\research_pipeline\terra_reports\phase2_holy_shit_exact_quote_1y_v2");BATCH_ROOT=Path(r"D:\AlgoResearch\research_pipeline\terra_reports\phase2_holy_shit_exact_quote_1y\quote_batches")
FB=QP/r"runs\phase1_final_discovery_through_20260430\blocks\features";START="2025-05-01";END="2026-04-30";HOLDOUT="2026-05-01";WINDOW=10;DELAY=5;MAX_SPREAD_BP=10
SPECS=[
 dict(id="S01_session_momentum",feature="since_open",schedule="early",h=15,q=None,direction="continuation",execution="passive",max_side=1),
 dict(id="S02_range_location",feature="range_location",schedule="early",h=30,q=None,direction="continuation",execution="passive",max_side=1),
 dict(id="S03_opening_continuation",feature="opening_15m",schedule="late",h=15,q=.95,direction="continuation",execution="passive",max_side=3),
 dict(id="S04_extreme_momentum",feature="move_15m",schedule="early",h=30,q=.95,direction="continuation",execution="market",max_side=3),
 dict(id="S05_volume_confirmed",feature="volume_move_15m",schedule="early",h=30,q=.95,direction="continuation",execution="market",max_side=3),
 dict(id="S06_vwap_displacement",feature="vwap_distance",schedule="continuous15",h=15,q=.99,direction="continuation",execution="market",max_side=3),
 dict(id="S07_market_residual",feature="residual_15m",schedule="early",h=30,q=.95,direction="continuation",execution="market",max_side=3),
 dict(id="S08_breadth_reversal",feature="market_lag",schedule="early",h=60,q=.95,direction="reversal",execution="passive",max_side=3),
 dict(id="S09_gap_reaction",feature="gap_reaction",schedule="early",h=60,q=.95,direction="continuation",execution="market",max_side=3),
 dict(id="S10_bar_structure",feature="bar_close_quality",schedule="early",h=15,q=.95,direction="continuation",execution="passive",max_side=3),
]
def fp(n):return str(next(FB.glob(f"feature_{n:03d}_*.parquet"))).replace("\\","/")
def iso(ts):return pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00","Z")

def build_matrix():
    out=ROOT/"full_universe_matrix.parquet";ROOT.mkdir(parents=True,exist_ok=True)
    if out.exists():
      try:pd.read_parquet(out);return out
      except Exception:pass
    c=duckdb.connect();c.execute("PRAGMA threads=16");c.execute("PRAGMA memory_limit='20GB'");tmp=ROOT/"duckdb_tmp";tmp.mkdir(exist_ok=True);c.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
    c.execute(f"""COPY(SELECT b.symbol,b.session_date,b.decision_ts bucket,b.close_adjusted,b.vwap_adjusted,b.analysis_eligible,
      x2.return_3,x2.relative_volume_3,x21.return_since_open,x21.overnight_gap,x21.close_location,
      x22.session_range_position,x22.minute_of_session,x22.minutes_until_close,x24.opening_return_15m,x24.opening_close_location_15m,
      x32.unreacted_market_move_5,x32.stock_minus_market_return_3
      FROM read_parquet('{fp(0)}') b
      JOIN read_parquet('{fp(2)}') x2 ON b.symbol=x2.symbol AND b.decision_ts=x2.decision_ts
      JOIN read_parquet('{fp(21)}') x21 ON b.symbol=x21.symbol AND b.decision_ts=x21.decision_ts
      JOIN read_parquet('{fp(22)}') x22 ON b.symbol=x22.symbol AND b.decision_ts=x22.decision_ts
      JOIN read_parquet('{fp(24)}') x24 ON b.symbol=x24.symbol AND b.decision_ts=x24.decision_ts
      JOIN read_parquet('{fp(32)}') x32 ON b.symbol=x32.symbol AND b.decision_ts=x32.decision_ts
      WHERE b.session_date BETWEEN DATE '{START}' AND DATE '{END}' AND b.analysis_eligible)
      TO '{out.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD)""");return out

def balance(z,max_side):
    z=z.sort_values(["bucket","symbol"],kind="mergesort").copy();z["strength"]=z.signal.abs();z["side_i"]=z.groupby(["bucket","side"]).strength.rank(method="first",ascending=False);z=z[z.side_i<=max_side]
    n=z.groupby(["bucket","side"]).size().unstack(fill_value=0)
    if -1 not in n:n[-1]=0
    if 1 not in n:n[1]=0
    keep=n[[-1,1]].min(axis=1).clip(upper=max_side).rename("keep");z=z.join(keep,on="bucket");return z[(z.keep>0)&(z.side_i<=z.keep)].copy()

def build():
    d=pd.read_parquet(build_matrix());d.bucket=pd.to_datetime(d.bucket,utc=True);d["mid0"]=d.close_adjusted;d["mid5"]=d.close_adjusted;d["bid_size5"]=1;d["ask_size5"]=1;d["vwap_slope"]=np.nan;d["tod_relative_volume_20"]=d.relative_volume_3;d["tod_cumulative_relative_volume_20"]=d.relative_volume_3;d["return_6"]=np.nan;d["stock_minus_market_return_6"]=np.nan;d["volatility_acceleration_3"]=np.nan;d["return_vol_ratio_3"]=np.nan;d["return_vol_ratio_6"]=np.nan;d["return_outlier_score_3"]=np.nan;d["return_outlier_score_6"]=np.nan;d["return_consistency_3"]=np.nan;d["return_consistency_6"]=np.nan;d["range_position_3"]=np.nan;d["range_position_6"]=np.nan;d["breakout_magnitude_3"]=np.nan;d["breakdown_magnitude_3"]=np.nan;d["breakout_magnitude_6"]=np.nan;d["breakdown_magnitude_6"]=np.nan;d["volume_acceleration_3"]=np.nan;d["market_residual_return_20"]=np.nan;d["market_residual_return_60"]=np.nan;d["return_10"]=np.nan;d["relative_volume_6"]=np.nan;d["opening_return_30m"]=np.nan;d["opening_breakout_15m"]=np.nan;d["opening_breakdown_15m"]=np.nan;d["opening_breakout_30m"]=np.nan;d["opening_breakdown_30m"]=np.nan;d["high_market_vol"]=False;d["universe_breadth_positive"]=np.nan;d["universe_return_dispersion"]=np.nan;d["market_up"]=False;d["unreacted_market_move_3"]=np.nan;d["market_return_3"]=np.nan;d["previous_session_return"]=np.nan;d["cumulative_volume"]=np.nan;d["current_volume_share"]=np.nan;d["vwap_cross"]=0
    fs=formulas(d);rows=[];threshold_rows=[]
    for s in SPECS:
        z=d.copy();z["signal"]=fs[s["feature"]][1]
        if s["schedule"]=="early":z=z[(z.minute_of_session>=15)&(z.minute_of_session<=90)]
        elif s["schedule"]=="late":z=z[(z.minute_of_session>=240)&(z.minutes_until_close>=60)]
        else:z=z[(z.minute_of_session>=15)&(z.minutes_until_close>=60)&(z.minute_of_session%15==0)]
        z=z[z.signal.notna()].sort_values(["bucket","symbol"],kind="mergesort").copy()
        quantiles=[None] if s["q"] is None else sorted(set([s["q"]]+([.99] if s["q"]<.99 else [])))
        for quantile in quantiles:
            v=z.copy();variant=s["id"] if quantile is None else f"{s['id']}_q{int(quantile*100)}"
            if quantile is None:
                g=v.groupby("bucket");hi=g.signal.rank(method="first",ascending=False)<=s["max_side"];lo=g.signal.rank(method="first",ascending=True)<=s["max_side"];v=pd.concat([v[hi].assign(side=1),v[lo].assign(side=-1)])
            else:
                train=v[v.session_date<=pd.Timestamp("2025-08-31")];threshold=float(train.signal.abs().quantile(quantile));threshold_rows.append(dict(candidate_id=variant,mechanism_id=s["id"],feature=s["feature"],schedule=s["schedule"],quantile=quantile,threshold=threshold,train_rows=len(train)));v=v[v.signal.abs()>=threshold];v["side"]=np.sign(v.signal)*(1 if s["direction"]=="continuation" else -1)
            v=balance(v,s["max_side"]);cadence=15 if s["schedule"]=="continuous15" else 5;cohort=max(1,int(np.ceil(s["h"]/cadence)));v["target_weight"]=(.5/v.groupby(["bucket","side"]).symbol.transform("size"))/cohort
            v["mechanism_id"]=s["id"];v["horizon"]=s["h"];v["entry_request_ts"]=v.bucket;v["exit_request_ts"]=v.bucket+pd.to_timedelta(s["h"],unit="m")
            for execution in ("market","passive"):
                q=v.copy();q["candidate_id"]=variant+"__"+execution;q["execution"]=execution;rows.append(q[["candidate_id","mechanism_id","symbol","session_date","bucket","entry_request_ts","exit_request_ts","side","target_weight","signal","execution","horizon"]])
    orders=pd.concat(rows,ignore_index=True);orders.to_parquet(ROOT/"source_orders.parquet",index=False);pd.DataFrame(threshold_rows).to_csv(ROOT/"frozen_full_universe_thresholds.csv",index=False)
    ep=pd.concat([orders[["session_date","symbol","entry_request_ts"]].rename(columns={"entry_request_ts":"request_ts"}),orders[["session_date","symbol","exit_request_ts"]].rename(columns={"exit_request_ts":"request_ts"})]).drop_duplicates();ep.to_parquet(ROOT/"quote_endpoints.parquet",index=False)
    audit={"candidates":orders.candidate_id.nunique(),"orders":len(orders),"windows":len(ep),"sessions":orders.session_date.nunique(),"symbols":orders.symbol.nunique(),"min":str(orders.session_date.min()),"max":str(orders.session_date.max())};(ROOT/"build_audit.json").write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

def fetch(client,session,ts,symbols,chunk):
    folder=BATCH_ROOT/f"session={session}";folder.mkdir(parents=True,exist_ok=True);out=folder/f"{ts.value}_{chunk}_{sha1(','.join(symbols).encode()).hexdigest()[:10]}.parquet"
    if out.exists():
      try:
        pd.read_parquet(out)
        return out
      except Exception:
        pass
    pages,_=client._paged_json("/v2/stocks/quotes",{"symbols":",".join(symbols),"start":iso(ts),"end":iso(ts+pd.Timedelta(seconds=WINDOW)),"feed":"sip","limit":10000,"sort":"asc"});rows=[]
    for p in pages:
      for sym,vals in p.get("quotes",{}).items():
       for v in vals:rows.append(dict(request_ts=ts,symbol=sym,quote_ts=v.get("t"),bid_price=v.get("bp"),ask_price=v.get("ap"),bid_size=v.get("bs"),ask_size=v.get("as")))
    x=pd.DataFrame(rows,columns=["request_ts","symbol","quote_ts","bid_price","ask_price","bid_size","ask_size"])
    if len(x):x.request_ts=pd.to_datetime(x.request_ts,utc=True);x.quote_ts=pd.to_datetime(x.quote_ts,utc=True);x=x[(x.bid_price>0)&(x.ask_price>x.bid_price)&(x.bid_size>=0)&(x.ask_size>=0)]
    tmp=out.with_name(f"{out.stem}.{threading.get_ident()}.tmp.parquet");x.to_parquet(tmp,index=False)
    for attempt in range(20):
      try:os.replace(tmp,out);break
      except PermissionError:
        if attempt==19:raise
        time.sleep(.25*(attempt+1))
    return out

def download(workers=16):
    ep=pd.read_parquet(ROOT/"quote_endpoints.parquet");ep.request_ts=pd.to_datetime(ep.request_ts,utc=True);daily=ROOT/"daily_quotes";daily.mkdir(exist_ok=True);client=AlpacaHistoricalClient.from_env(r"D:\AlgoResearch\.env");client.max_retries=20;client.requests_per_minute=200
    sessions=sorted(ep.session_date.astype(str).unique())
    for i,session in enumerate(sessions,1):
      out=daily/f"quotes_{session}.parquet"
      if out.exists():print(f"quotes {i}/{len(sessions)} cached {session}",flush=True);continue
      day=ep[ep.session_date.astype(str)==session];cov=quote_window_coverage(day,window_seconds=WINDOW);covered=day[cov];missing=day[~cov];lake=load_quote_windows(covered,window_seconds=WINDOW) if len(covered) else pd.DataFrame();tasks=[]
      for ts,g in missing.groupby("request_ts"):
       syms=sorted(g.symbol.unique())
       for j in range(0,len(syms),100):tasks.append((ts,tuple(syms[j:j+100]),j//100))
      files=[]
      with ThreadPoolExecutor(max_workers=workers) as pool:
       fut=[pool.submit(fetch,client,session,*t) for t in tasks]
       for z in as_completed(fut):files.append(z.result())
      frames=[x for x in [lake]+[pd.read_parquet(p) for p in files] if x is not None and len(x)];x=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["request_ts","symbol","quote_ts","bid_price","ask_price","bid_size","ask_size"]);x=x.sort_values(["request_ts","symbol","quote_ts"]);tmp=out.with_suffix(".tmp.parquet");x.to_parquet(tmp,index=False);tmp.replace(out);print(f"quotes {i}/{len(sessions)} {session} covered={len(covered)} missing={len(missing)} tasks={len(tasks)}",flush=True)

def replay():
    orders=pd.read_parquet(ROOT/"source_orders.parquet");orders.entry_request_ts=pd.to_datetime(orders.entry_request_ts,utc=True);orders.exit_request_ts=pd.to_datetime(orders.exit_request_ts,utc=True);rows=[]
    for i,(session,g) in enumerate(orders.groupby(orders.session_date.astype(str)),1):
      q=pd.read_parquet(ROOT/"daily_quotes"/f"quotes_{session}.parquet");q.request_ts=pd.to_datetime(q.request_ts,utc=True);q.quote_ts=pd.to_datetime(q.quote_ts,utc=True);q=q[(q.bid_price>0)&(q.ask_price>q.bid_price)&(((q.ask_price-q.bid_price)/((q.ask_price+q.bid_price)/2))*10000<=MAX_SPREAD_BP)&(q.bid_size>=0)&(q.ask_size>=0)];paths={(s,t):x for (s,t),x in q.groupby(["symbol","request_ts"])}
      for o in g.itertuples(index=False):
        ep=paths.get((o.symbol,o.entry_request_ts));xp=paths.get((o.symbol,o.exit_request_ts))
        for delay in (0,5):
          base=dict(candidate_id=f"{o.candidate_id}__d{delay}",source_candidate_id=o.candidate_id,mechanism_id=o.mechanism_id,symbol=o.symbol,session_date=o.session_date,bucket=o.bucket,side=o.side,target_weight=o.target_weight,execution=o.execution,horizon=o.horizon,delay_seconds=delay)
          if ep is None or xp is None:rows.append({**base,"status":"missing_path"});continue
          et=o.entry_request_ts+pd.Timedelta(seconds=delay);xt=o.exit_request_ts+pd.Timedelta(seconds=delay);ea=ep[ep.quote_ts>=et];xa=xp[xp.quote_ts>=xt]
          if ea.empty or xa.empty:rows.append({**base,"status":"missing_quote"});continue
          eq=ea.iloc[0];xq=xa.iloc[0]
          if o.execution=="passive":
            limit=float(eq.bid_price if o.side==1 else eq.ask_price);later=ea[ea.quote_ts>=eq.quote_ts];cross=later[later.ask_price<=limit] if o.side==1 else later[later.bid_price>=limit]
            if cross.empty:rows.append({**base,"status":"unfilled"});continue
            entry=limit;fill_ts=cross.iloc[0].quote_ts
          else:entry=float(eq.ask_price if o.side==1 else eq.bid_price);fill_ts=eq.quote_ts
          exitpx=float(xq.bid_price if o.side==1 else xq.ask_price);gross=(exitpx/entry-1)*o.side
          rows.append({**base,"status":"filled","entry_price":entry,"exit_price":exitpx,"entry_fill_ts":fill_ts,"exit_fill_ts":xq.quote_ts,"gross_return":gross})
      if i%25==0:print(f"replay {i}",flush=True)
    led=pd.DataFrame(rows)
    paired=led.status.eq("filled")
    if paired.any():
      p=led[paired].sort_values(["candidate_id","bucket","side","symbol"],kind="mergesort").copy();p["side_rank"]=p.groupby(["candidate_id","bucket","side"]).cumcount();counts=p.groupby(["candidate_id","bucket","side"]).size().unstack(fill_value=0)
      if -1 not in counts:counts[-1]=0
      if 1 not in counts:counts[1]=0
      keep=counts[[-1,1]].min(axis=1).rename("keep_n");p=p.join(keep,on=["candidate_id","bucket"]);cancel=p[p.side_rank>=p.keep_n].index;led.loc[cancel,"status"]="cancelled_unpaired";led.loc[cancel,"gross_return"]=np.nan
    led.to_parquet(ROOT/"quote_replay_ledger.parquet",index=False);led.groupby(["candidate_id","mechanism_id","execution","status"]).size().rename("orders").reset_index().to_csv(ROOT/"fill_status_counts.csv",index=False);summ=[];period=[];daily_columns={}
    session_index=pd.Index(sorted(pd.to_datetime(orders.session_date).dt.date.unique()),name="session_date")
    def stats(r):
      r=r.astype(float);eq=(1+r).cumprod();peaks=np.maximum.accumulate(np.r_[1.,eq.to_numpy()]);dd=eq.to_numpy()/peaks[1:]-1;end_i=int(np.argmin(dd));prior=np.r_[1.,eq.to_numpy()[:end_i+1]];peak_i=int(np.argmax(prior));peak_date=r.index[0] if peak_i==0 else r.index[peak_i-1];end_date=r.index[end_i];years=max(((pd.Timestamp(r.index[-1])-pd.Timestamp(r.index[0])).days+1)/365.25,1/252);sd=r.std();return dict(cagr=float(eq.iloc[-1]**(1/years)-1),total_return=float(eq.iloc[-1]-1),sharpe=float(np.sqrt(252)*r.mean()/sd) if sd>0 else 0.,max_dd=float(dd.min()),pt_days=int((pd.Timestamp(end_date)-pd.Timestamp(peak_date)).days),positive_day_rate=float((r>0).mean()))
    for cid,g in led.groupby("candidate_id"):
      for cost in (-1.,-.5,0.,1.,3.,5.):
        z=g.copy();z["pnl"]=np.where(z.status.eq("filled"),z.target_weight*(z.gross_return-2*cost/10000),0);daily=z.groupby(pd.to_datetime(z.session_date).dt.date).pnl.sum();r=daily.reindex(session_index,fill_value=0);m=stats(r);row=dict(candidate_id=cid,mechanism_id=g.mechanism_id.iloc[0],execution=g.execution.iloc[0],delay_seconds=int(g.delay_seconds.iloc[0]),cost_bp_side=cost,orders=len(g),events=g.bucket.nunique(),sessions=g.session_date.nunique(),symbols=g.symbol.nunique(),filled=int(g.status.eq("filled").sum()),fill_rate=float(g.status.eq("filled").mean()),coverage=float((~g.status.str.startswith("missing")).mean()),**m);summ.append(row);daily_columns[f"{cid}__cost{cost:g}"]=r
        for name,lo,hi in (("train","2025-05-01","2025-08-31"),("development","2025-09-01","2025-12-31"),("recent","2026-01-01","2026-04-30")):
          rr=r[(pd.to_datetime(r.index)>=lo)&(pd.to_datetime(r.index)<=hi)];period.append({"candidate_id":cid,"mechanism_id":g.mechanism_id.iloc[0],"execution":g.execution.iloc[0],"delay_seconds":int(g.delay_seconds.iloc[0]),"cost_bp_side":cost,"period":name,**stats(rr)})
    pd.DataFrame(summ).to_csv(ROOT/"summary.csv",index=False);pd.DataFrame(period).to_csv(ROOT/"period_stats.csv",index=False);pd.DataFrame(daily_columns,index=session_index).to_parquet(ROOT/"daily_returns.parquet");print(pd.DataFrame(summ).to_string(index=False))

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("all","build","download","replay"),default="all");p.add_argument("--workers",type=int,default=16);a=p.parse_args()
    if a.stage in ("all","build"):build()
    if a.stage in ("all","download"):download(a.workers)
    if a.stage in ("all","replay"):replay()
if __name__=="__main__":main()
