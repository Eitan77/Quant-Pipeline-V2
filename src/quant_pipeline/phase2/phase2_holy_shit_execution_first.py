from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2")
RUN=ROOT/r"results\phase2_holy_shit_execution_first_through_20260430"
SNAP=ROOT/r"results\phase2_quote_native_four_lanes_through_20260430\quote_5m_snapshots.parquet"
FB=ROOT/r"runs\phase1_final_discovery_through_20260430\blocks\features"
KEY="symbol,session_date,decision_ts"


def f(n:int)->str:
    return str(next(FB.glob(f"feature_{n:03d}_*.parquet"))).replace("\\","/")


def build(out:Path)->None:
    dest=out/"execution_feature_matrix.parquet"
    if dest.exists(): print(f"reuse {dest}",flush=True); return
    out.mkdir(parents=True,exist_ok=True); tmp=out/"duckdb_tmp"; tmp.mkdir(exist_ok=True)
    c=duckdb.connect(); c.execute("PRAGMA threads=16"); c.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'"); c.execute("PRAGMA memory_limit='20GB'")
    sql=f"""
    COPY(
      SELECT s.*, b.close_adjusted,b.open_adjusted,b.vwap_adjusted,b.volume,b.analysis_eligible,
        x2.return_3,x2.realized_vol_3,x2.return_vol_ratio_3,x2.relative_volume_3,
        x3.return_consistency_3,x3.range_position_3,x3.breakout_magnitude_3,x3.breakdown_magnitude_3,x3.volume_acceleration_3,x3.volatility_acceleration_3,x3.return_outlier_score_3,
        x6.return_6,x6.realized_vol_6,x6.return_vol_ratio_6,x6.relative_volume_6,x6.return_consistency_6,
        x7.range_position_6,x7.breakout_magnitude_6,x7.breakdown_magnitude_6,x7.volume_acceleration_6,x7.volatility_acceleration_6,x7.return_outlier_score_6,
        x8.return_10,x9.relative_volume_10,x9.return_consistency_10,x9.range_position_10,x9.return_outlier_score_10,
        x21.return_since_open,x21.overnight_gap,x21.previous_session_return,x21.close_location,
        x22.session_range_position,x22.cumulative_volume,x22.current_volume_share,x22.vwap_slope,x22.vwap_cross,x22.consecutive_positive_bars,x22.consecutive_negative_bars,x22.minute_of_session,x22.minutes_until_close,
        x24.opening_return_15m,x24.opening_range_15m,x24.opening_close_location_15m,x24.opening_breakout_15m,x24.opening_breakdown_15m,
        x25.opening_return_30m,x25.opening_range_30m,x25.opening_close_location_30m,x25.opening_breakout_30m,x25.opening_breakdown_30m,
        x27.range_position_10_z_4680,x27.distance_session_vwap_z_4680,x27.tod_relative_volume_20,x27.tod_cumulative_relative_volume_20,
        x31.universe_breadth_positive,x31.universe_return_dispersion,x31.market_up,x31.high_market_vol,x31.unreacted_market_move_1,
        x32.unreacted_market_move_3,x32.unreacted_market_move_5,x32.market_return_3,x32.stock_minus_market_return_3,
        x33.stock_minus_market_return_6,x33.stock_minus_market_return_10,x33.market_residual_return_20,x34.market_residual_return_60
      FROM read_parquet('{SNAP.as_posix()}') s
      JOIN read_parquet('{f(0)}') b ON s.symbol=b.symbol AND s.bucket=b.decision_ts
      JOIN read_parquet('{f(2)}') x2 ON s.symbol=x2.symbol AND s.bucket=x2.decision_ts
      JOIN read_parquet('{f(3)}') x3 ON s.symbol=x3.symbol AND s.bucket=x3.decision_ts
      JOIN read_parquet('{f(6)}') x6 ON s.symbol=x6.symbol AND s.bucket=x6.decision_ts
      JOIN read_parquet('{f(7)}') x7 ON s.symbol=x7.symbol AND s.bucket=x7.decision_ts
      JOIN read_parquet('{f(8)}') x8 ON s.symbol=x8.symbol AND s.bucket=x8.decision_ts
      JOIN read_parquet('{f(9)}') x9 ON s.symbol=x9.symbol AND s.bucket=x9.decision_ts
      JOIN read_parquet('{f(21)}') x21 ON s.symbol=x21.symbol AND s.bucket=x21.decision_ts
      JOIN read_parquet('{f(22)}') x22 ON s.symbol=x22.symbol AND s.bucket=x22.decision_ts
      JOIN read_parquet('{f(24)}') x24 ON s.symbol=x24.symbol AND s.bucket=x24.decision_ts
      JOIN read_parquet('{f(25)}') x25 ON s.symbol=x25.symbol AND s.bucket=x25.decision_ts
      JOIN read_parquet('{f(27)}') x27 ON s.symbol=x27.symbol AND s.bucket=x27.decision_ts
      JOIN read_parquet('{f(31)}') x31 ON s.symbol=x31.symbol AND s.bucket=x31.decision_ts
      JOIN read_parquet('{f(32)}') x32 ON s.symbol=x32.symbol AND s.bucket=x32.decision_ts
      JOIN read_parquet('{f(33)}') x33 ON s.symbol=x33.symbol AND s.bucket=x33.decision_ts
      JOIN read_parquet('{f(34)}') x34 ON s.symbol=x34.symbol AND s.bucket=x34.decision_ts
      WHERE s.session_date<=DATE '2026-04-30' AND b.analysis_eligible
    ) TO '{dest.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD)
    """
    c.execute(sql); print(c.execute(f"select count(*),min(session_date),max(session_date) from read_parquet('{dest.as_posix()}')").fetchall(),flush=True)


def formulas(d:pd.DataFrame)->dict[str,tuple[str,pd.Series]]:
    clip=lambda s:s.clip(.05,10).pow(.5)
    signed_vacc=np.sign(d.return_3)*d.volatility_acceleration_3
    prior3=(1+d.return_6)/(1+d.return_3)-1
    prior_resid3=d.stock_minus_market_return_6-d.stock_minus_market_return_3
    micro_move=d.mid5/d.mid0-1
    imbalance=(d.bid_size5-d.ask_size5)/(d.bid_size5+d.ask_size5).replace(0,np.nan)
    return {
      "move_15m":("price_momentum",d.return_3),"move_30m":("price_momentum",d.return_6),"move_50m":("price_momentum",d.return_10),
      "volnorm_15m":("volnorm_momentum",d.return_vol_ratio_3),"volnorm_30m":("volnorm_momentum",d.return_vol_ratio_6),
      "outlier_15m":("exhaustion",d.return_outlier_score_3),"outlier_30m":("exhaustion",d.return_outlier_score_6),
      "volume_move_15m":("volume_confirmed",d.return_3*clip(d.relative_volume_3)),"volume_move_30m":("volume_confirmed",d.return_6*clip(d.relative_volume_6)),
      "persistent_15m":("persistence",d.return_3*d.return_consistency_3.clip(0,1)),"persistent_30m":("persistence",d.return_6*d.return_consistency_6.clip(0,1)),
      "vwap_distance":("vwap",(d.close_adjusted/d.vwap_adjusted-1)),"vwap_slope":("vwap",d.vwap_slope),
      "vwap_volume":("vwap_volume",(d.close_adjusted/d.vwap_adjusted-1)*clip(d.tod_relative_volume_20)),
      "range_location":("range",d.session_range_position-.5),"range_break_30m":("range_breakout",d.breakout_magnitude_6-d.breakdown_magnitude_6),
      "range_volume":("range_breakout",(d.range_position_6-.5)*clip(d.relative_volume_6)),
      "since_open":("session_momentum",d.return_since_open),"session_volume":("session_momentum",d.return_since_open*clip(d.tod_cumulative_relative_volume_20)),
      "opening_15m":("opening",d.opening_return_15m),"opening_break_15m":("opening",d.opening_breakout_15m-d.opening_breakdown_15m),
      "opening_30m":("opening",d.opening_return_30m),"opening_break_30m":("opening",d.opening_breakout_30m-d.opening_breakdown_30m),
      "residual_15m":("market_residual",d.stock_minus_market_return_3),"residual_30m":("market_residual",d.stock_minus_market_return_6),
      "residual_volume":("market_residual_volume",d.stock_minus_market_return_3*clip(d.relative_volume_3)),
      "beta_residual_20":("beta_residual",d.market_residual_return_20),"beta_residual_60":("beta_residual",d.market_residual_return_60),
      "gap_confirm":("gap",d.overnight_gap+d.return_since_open),"gap_reaction":("gap",d.overnight_gap*d.opening_close_location_15m),
      "volatility_expansion":("volatility",signed_vacc),"volume_acceleration":("volume_acceleration",np.sign(d.return_3)*d.volume_acceleration_3),
      "breadth_lag":("breadth",d.unreacted_market_move_3),"market_lag":("breadth",d.unreacted_market_move_5),
      "price_acceleration":("price_acceleration",d.return_3-prior3),"residual_acceleration":("residual_acceleration",d.stock_minus_market_return_3-prior_resid3),
      "quote_impulse":("micro_momentum",micro_move),"quote_imbalance":("order_book",imbalance),"quote_confirmed":("micro_confirmed",micro_move*(1+np.sign(micro_move)*imbalance)),
      "bar_close_quality":("bar_structure",d.return_3*(2*d.close_location-1).abs()),"opening_quality":("opening_quality",d.opening_return_15m*(2*d.opening_close_location_15m-1).abs()),
      "vwap_cross_impulse":("vwap_cross",np.sign(d.close_adjusted/d.vwap_adjusted-1)*d.vwap_cross*d.return_3.abs()),
    }


def add_future(d:pd.DataFrame,h:int)->pd.DataFrame:
    f=d[["symbol","bucket","bid5","ask5"]].copy(); f.bucket=f.bucket-pd.Timedelta(minutes=h)
    return d.merge(f,on=["symbol","bucket"],how="left",suffixes=("",f"_f{h}"))


def select_event(g:pd.DataFrame,n:int)->pd.DataFrame:
    lo=g.nsmallest(n,"signal"); hi=g.nlargest(n,"signal"); k=min(len(lo),len(hi),n)
    if k<1:return g.iloc[:0]
    hi=hi.head(k).copy();lo=lo.head(k).copy();hi["direction"]=1;lo["direction"]=-1
    return pd.concat([hi,lo],ignore_index=True)


def make_trades(x:pd.DataFrame,n:int,style:str,h:int)->pd.DataFrame:
    z=x.copy();grp=z.groupby("bucket",sort=False);enough=grp.signal.transform("size")>=max(6,2*n)
    hi=grp.signal.rank(method="first",ascending=False)<=n;lo=grp.signal.rank(method="first",ascending=True)<=n
    a=z[enough&hi].copy();b=z[enough&lo].copy();a["direction"]=1;b["direction"]=-1;q=pd.concat([a,b],ignore_index=True)
    if style.startswith("reversal"):q.direction*=-1
    if style.endswith("passive"):
        q=q[((q.direction>0)&(q.min_ask_after5<=q.bid5))|((q.direction<0)&(q.max_bid_after5>=q.ask5))].copy()
        # restore side balance after heterogeneous fill rates
        side=q.groupby(["bucket","direction"]).cumcount();counts=q.groupby(["bucket","direction"]).direction.transform("size");mins=q.groupby("bucket").direction.transform(lambda s:min((s>0).sum(),(s<0).sum()))
        q=q[(mins>0)&(side<mins)].copy()
        q["raw_ret"]=np.where(q.direction>0,q[f"bid5_f{h}"]/q.bid5-1,q.ask5/q[f"ask5_f{h}"]-1)
    else:q["raw_ret"]=np.where(q.direction>0,q[f"bid5_f{h}"]/q.ask5-1,q.bid5/q[f"ask5_f{h}"]-1)
    return q


def selected_base(x:pd.DataFrame,n:int)->pd.DataFrame:
    z=x.copy();grp=z.groupby("bucket",sort=False);enough=grp.signal.transform("size")>=max(6,2*n)
    hi=grp.signal.rank(method="first",ascending=False)<=n;lo=grp.signal.rank(method="first",ascending=True)<=n
    a=z[enough&hi].copy();b=z[enough&lo].copy();a["direction"]=1;b["direction"]=-1
    return pd.concat([a,b],ignore_index=True)


def execute_selected(q:pd.DataFrame,direction:str,execution:str,h:int)->pd.DataFrame:
    z=q.copy()
    if direction=="reversal":z.direction*=-1
    if execution=="passive":
        z=z[((z.direction>0)&(z.min_ask_after5<=z.bid5))|((z.direction<0)&(z.max_bid_after5>=z.ask5))].copy()
        counts=z.groupby(["bucket","direction"]).size().unstack(fill_value=0)
        if -1 not in counts:counts[-1]=0
        if 1 not in counts:counts[1]=0
        keep=counts[[-1,1]].min(axis=1).rename("keep_n");z=z.join(keep,on="bucket");z["side_i"]=z.groupby(["bucket","direction"]).cumcount();z=z[(z.keep_n>0)&(z.side_i<z.keep_n)].copy()
        z["raw_ret"]=np.where(z.direction>0,z[f"bid5_f{h}"]/z.bid5-1,z.ask5/z[f"ask5_f{h}"]-1)
    else:z["raw_ret"]=np.where(z.direction>0,z[f"bid5_f{h}"]/z.ask5-1,z.bid5/z[f"ask5_f{h}"]-1)
    return z


def metrics(r:pd.Series)->dict:
    if len(r)<2:return dict(cagr=np.nan,sharpe=np.nan,max_dd=np.nan,pt_days=np.nan,total_return=np.nan)
    eq=(1+r).cumprod();peak=pd.concat([pd.Series([1.],index=[r.index[0]-pd.Timedelta(microseconds=1)]),eq]).cummax().iloc[1:];dd=eq/peak-1
    yrs=max((r.index[-1]-r.index[0]).total_seconds()/31557600,1/252);daily=r.groupby(r.index.date).sum();end=dd.idxmin();prior=eq.loc[:end];pv=max(1.,prior.max());pdt=r.index[0] if pv==1 else prior.idxmax()
    return dict(cagr=float(eq.iloc[-1]**(1/yrs)-1),sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0,max_dd=float(dd.min()),pt_days=int((end-pdt).total_seconds()/86400),total_return=float(eq.iloc[-1]-1))


def curve(q:pd.DataFrame,cost:float,h:int,schedule:str)->pd.Series:
    if q.empty:return pd.Series(dtype=float)
    z=q.copy();z["net"]=z.raw_ret-2*cost/10000;g=z.groupby("bucket").net.mean()
    cadence=15 if schedule=="continuous15" else 5
    cohort=max(1,int(np.ceil(h/cadence)));g/=cohort
    day=g.groupby(g.index.normalize()).sum()
    idx=pd.date_range(day.index.min(),day.index.max(),freq="D",tz=day.index.tz);return day.reindex(idx,fill_value=0)


def screen(out:Path)->None:
    d=pd.read_parquet(out/"execution_feature_matrix.parquet");d.bucket=pd.to_datetime(d.bucket,utc=True);d=d.sort_values(["symbol","bucket"])
    d["spread_bp"]=(d.ask5-d.bid5)/d.mid5*10000; d=d[(d.spread_bp<=8)&(d.quote_count>=5)&d.analysis_eligible]
    fs=formulas(d)
    for name,(_,sig) in fs.items():d[name]=sig
    rows=[];daily_store={}
    schedules={"continuous15":(d.minute_of_session>=15)&(d.minutes_until_close>=60)&(d.minute_of_session%15==0),"early":(d.minute_of_session>=15)&(d.minute_of_session<=90),"late":(d.minute_of_session>=240)&(d.minutes_until_close>=60)}
    for h in (5,15,30,60):
      xh=add_future(d,h);ok=xh[f"bid5_f{h}"].notna()&xh[f"ask5_f{h}"].notna()
      for name,(family,_) in fs.items():
        xh["signal"]=xh[name]
        for sched,mask0 in schedules.items():
          keys=set(d.loc[mask0,"bucket"].unique());base=xh[ok&xh.bucket.isin(keys)&xh.signal.notna()].copy()
          if len(base)<1000:continue
          for n in (1,3):
            selected=selected_base(base,n)
            for direction in ("continuation","reversal"):
              for execution in ("market","passive"):
                qq=execute_selected(selected,direction,execution,h)
                if len(qq)<100:continue
                cid=f"{name}__{sched}__h{h}__n{n}__{direction}__{execution}"
                for cost in (0.,1.,3.,5.):
                  r=curve(qq,cost,h,sched)
                  for fold,lo,hi in (("train","2025-05-01","2025-08-31"),("development","2025-09-01","2025-12-31"),("internal_validation","2026-01-01","2026-04-30"),("full","2025-05-01","2026-04-30")):
                    rr=r.loc[lo:hi];rows.append(dict(candidate_id=cid,feature=name,family=family,schedule=sched,horizon=h,n=n,direction=direction,execution=execution,cost_bp_side=cost,fold=fold,trades=len(qq),**metrics(rr)))
                daily_store[cid]=curve(qq,1.,h,sched)
    res=pd.DataFrame(rows);res.to_csv(out/"execution_screen.csv",index=False)
    base=res[res.cost_bp_side.eq(1)].pivot(index="candidate_id",columns="fold",values="cagr");stress=res[(res.cost_bp_side.eq(3))&(res.fold.eq("full"))].set_index("candidate_id").cagr.rename("cagr_3bp")
    info=res[(res.cost_bp_side.eq(1))&(res.fold.eq("full"))].set_index("candidate_id");rank=info.join(base.add_prefix("cagr_")).join(stress)
    rank["all_folds_positive"]=(rank[["cagr_train","cagr_development","cagr_internal_validation"]]>0).all(axis=1);rank["gate"]=(rank.all_folds_positive)&(rank.cagr_3bp>0)&(rank.max_dd>=-.05)&(rank.pt_days<=45)&(rank.trades>=300)
    rank=rank.sort_values(["gate","cagr_full","sharpe"],ascending=False);rank.to_csv(out/"execution_ranking.csv")
    pd.DataFrame(daily_store).to_parquet(out/"daily_returns_1bp.parquet");print(rank.head(40).to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("all","build","screen"),default="all");p.add_argument("--out",default=str(RUN));a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.stage in ("all","build"):build(out)
    if a.stage in ("all","screen"):screen(out)

if __name__=="__main__":main()
