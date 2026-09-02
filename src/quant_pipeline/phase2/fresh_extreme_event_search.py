"""Sparse extreme-move continuation and exhaustion-reversal discovery."""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from quant_pipeline.phase2.data import _column_paths, _target_paths
from quant_pipeline.phase2.fresh_cleanroom_leverage_search import metrics


SOURCE="C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs/phase1_final_discovery_through_20260430"
ROOT=Path("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_extreme_event_search_through_20260430")
HOLDOUT="2026-05-01"
LOOKBACKS=(6,12);HORIZONS=(15,30,60);VOLUME_THRESHOLDS=(0.0,1.5,2.0);PERSISTENCE=("none","same_sign_3")
SIGNALS={"return_outlier_score":(1.5,2.0,2.5),"return_vol_ratio":(5.0,8.0)}
COSTS=(1.0,2.0,3.0)


def lit(x):return "'"+str(x).replace("'","''")+"'"


def build():
    ROOT.mkdir(parents=True,exist_ok=True);blocks=ROOT/"event_blocks";blocks.mkdir(exist_ok=True);features=_column_paths(SOURCE);targets=_target_paths(SOURCE);files=[]
    con=duckdb.connect();con.execute("SET threads=16");con.execute(f"SET temp_directory={lit((ROOT/'duckdb_tmp').as_posix())}")
    try:
        mx=con.execute(f"select max(session_date) from read_parquet({lit((Path(SOURCE)/'canonical_bars.parquet').as_posix())})").fetchone()[0]
        if pd.Timestamp(mx)>=pd.Timestamp(HOLDOUT):raise RuntimeError("holdout contamination")
    finally:con.close()
    for lookback in LOOKBACKS:
        for horizon in HORIZONS:
            target=f"fwd_return_{horizon}m";target_path=targets[target];dest=blocks/f"l{lookback}_h{horizon}.parquet";files.append(dest)
            if dest.exists():continue
            columns=[f"return_outlier_score_{lookback}",f"return_vol_ratio_{lookback}",f"relative_volume_{lookback}","return_3"]
            paths=[]
            for path in [*(features[name] for name in columns),target_path]:
                if path not in paths:paths.append(path)
            aliases={path:f"p{i}" for i,path in enumerate(paths)};first=paths[0]
            joins=" ".join(f"JOIN read_parquet({lit(path.as_posix())}) {aliases[path]} USING(symbol,session_date,bar_start_ts,decision_ts)" for path in paths[1:])
            feature_select=",".join(f'{aliases[features[name]]}."{name}" AS "{name}"' for name in columns)
            target_alias=aliases[target_path]
            specs=[]
            for prefix,thresholds in SIGNALS.items():
                signal=f"{prefix}_{lookback}"
                for threshold in thresholds:
                    for volume in VOLUME_THRESHOLDS:
                        for persistence in PERSISTENCE:
                            volume_clause="TRUE" if volume==0 else f"relative_volume_{lookback}>={volume}"
                            persistence_clause="TRUE" if persistence=="none" else f"return_3*{signal}>0"
                            spec=f"{signal}__thr{threshold:g}__rv{volume:g}__{persistence}"
                            qualify=f"({volume_clause}) AND ({persistence_clause})"
                            specs.append(f"""SELECT {lit(spec)} spec,session_date,decision_ts,minute_of_session,
                              {lit(prefix)} signal_family,{lookback} lookback,{threshold} threshold,{volume} volume_threshold,{lit(persistence)} persistence,{horizon} horizon,
                              count(*) FILTER(WHERE {qualify} AND {signal}>={threshold}) n_long,
                              count(*) FILTER(WHERE {qualify} AND {signal}<=-{threshold}) n_short,
                              avg(target_return) FILTER(WHERE {qualify} AND {signal}>={threshold}) long_return,
                              avg(target_return) FILTER(WHERE {qualify} AND {signal}<=-{threshold}) short_return
                            FROM base GROUP BY session_date,decision_ts,minute_of_session""")
            union=" UNION ALL ".join(specs);con=duckdb.connect();con.execute("SET threads=16");con.execute(f"SET temp_directory={lit((ROOT/'duckdb_tmp').as_posix())}")
            try:
                con.execute(f"""COPY (WITH base AS (
                  SELECT CAST({aliases[first]}.session_date AS DATE) session_date,{aliases[first]}.decision_ts,CAST(date_diff('minute',{aliases[first]}.scheduled_open,{aliases[first]}.decision_ts) AS INTEGER) minute_of_session,
                    {feature_select},CAST({target_alias}.\"{target}\" AS DOUBLE) target_return
                  FROM read_parquet({lit(first.as_posix())}) {aliases[first]} {joins}
                  WHERE {aliases[first]}.analysis_eligible AND {target_alias}.analysis_eligible AND CAST({aliases[first]}.session_date AS DATE)<DATE {lit(HOLDOUT)} AND {target_alias}.\"{target}\" IS NOT NULL
                    AND date_diff('minute',{aliases[first]}.scheduled_open,{aliases[first]}.decision_ts)>=30 AND date_diff('minute',{aliases[first]}.scheduled_open,{aliases[first]}.decision_ts)%15=0
                ), events AS ({union})
                SELECT *,least(0.5,0.1*least(n_long,n_short)) side_gross,
                  least(0.5,0.1*least(n_long,n_short))*(long_return-short_return) spread_return,
                  n_long+n_short positions
                FROM events WHERE n_long>=3 AND n_short>=3 ORDER BY decision_ts,spec
                ) TO {lit(dest.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)""")
            finally:con.close()
            print(f"built l{lookback} h{horizon}",flush=True)
    con=duckdb.connect();fs=",".join(lit(x.as_posix()) for x in files)
    try:con.execute(f"COPY (SELECT * FROM read_parquet([{fs}]) ORDER BY decision_ts,spec) TO {lit((ROOT/'events.parquet').as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)")
    finally:con.close()
    audit={"rows":duckdb.sql(f"select count(*) from read_parquet({lit((ROOT/'events.parquet').as_posix())})").fetchone()[0],"max_source_session":str(mx),"holdout_access":False};(ROOT/"build_audit.json").write_text(json.dumps(audit,indent=2));print(audit,flush=True)


def daily_return(g,calendar,direction,cost,high_dispersion):
    x=g[g.high_dispersion] if high_dispersion else g
    net=(direction*x.spread_return-(2*cost/10000)*(2*x.side_gross))/np.ceil(x.horizon/15)
    return pd.Series(net.to_numpy(),index=x.session_date).groupby(level=0).sum().reindex(calendar,fill_value=0.0)


def evaluate():
    events=pd.read_parquet(ROOT/"events.parquet");events["session_date"]=pd.to_datetime(events.session_date)
    for column in ("threshold","volume_threshold","side_gross","spread_return","horizon"):
        events[column]=pd.to_numeric(events[column],errors="raise").astype(float)
    context=pd.read_parquet("C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results/phase2_cleanroom_leverage_search_through_20260430/context.parquet");context["session_date"]=pd.to_datetime(context.session_date)
    events=events.merge(context[["decision_ts","dispersion","dispersion_prior_median"]],on="decision_ts",how="left",validate="many_to_one");events["high_dispersion"]=events.dispersion>events.dispersion_prior_median
    calendar=pd.DatetimeIndex(context.session_date.drop_duplicates().sort_values());folds=(("fold_2019_2021","2019-06-21","2021-12-31"),("fold_2022_2023","2022-01-01","2023-12-31"),("fold_2024_2026","2024-01-01","2026-04-30"),("full","2019-06-21","2026-04-30"));rows=[]
    for spec,g in events.groupby("spec",sort=False):
        meta=g.iloc[0]
        for direction in (1,-1):
            for regime in ("all","high_dispersion"):
                high=regime=="high_dispersion"
                for cost in COSTS:
                    row={"candidate_id":f"{spec}__h{int(meta.horizon)}__dir{direction:+d}__{regime}","spec":spec,"signal_family":meta.signal_family,"lookback":int(meta.lookback),"horizon":int(meta.horizon),"threshold":float(meta.threshold),"volume_threshold":float(meta.volume_threshold),"persistence":meta.persistence,"direction":direction,"regime":regime,"cost_bps_per_side":cost,"events":len(g[g.high_dispersion]) if high else len(g),"positions":int((g[g.high_dispersion] if high else g).positions.sum()),"average_gross":float((2*(g[g.high_dispersion] if high else g).side_gross).mean())}
                    for name,start,end in folds:
                        cal=calendar[(calendar>=start)&(calendar<=end)];d=daily_return(g,cal,direction,cost,high);row.update({f"{name}_{k}":v for k,v in metrics(d).items()})
                    rows.append(row)
    out=pd.DataFrame(rows);primary=out[np.isclose(out.cost_bps_per_side,2.0)].copy();primary["min_fold_cagr"]=primary[["fold_2019_2021_cagr","fold_2022_2023_cagr","fold_2024_2026_cagr"]].min(axis=1);primary["all_folds_positive"]=primary.min_fold_cagr>0
    neighbor_keys=["signal_family","lookback","volume_threshold","persistence","direction","regime"]
    primary["positive_neighbor_count"]=primary.all_folds_positive.groupby([primary[k] for k in neighbor_keys]).transform("sum")
    stress=out[np.isclose(out.cost_bps_per_side,3.0)][["candidate_id","full_cagr"]].rename(columns={"full_cagr":"full_cagr_3bps"});primary=primary.merge(stress,on="candidate_id",how="left")
    primary["holy_shit_gate"]=primary.all_folds_positive&(primary.positive_neighbor_count>=2)&(primary.full_cagr>=.10)&(primary.full_cagr_3bps>0)&(primary.full_maximum_drawdown>=-.05)&(primary.full_peak_to_trough_calendar_days<=45)&(primary.events>=300)
    out.to_csv(ROOT/"cost_grid.csv",index=False);primary.sort_values(["holy_shit_gate","min_fold_cagr","full_cagr"],ascending=False).to_csv(ROOT/"ranking.csv",index=False)
    print(primary.sort_values(["holy_shit_gate","min_fold_cagr","full_cagr"],ascending=False)[["candidate_id","min_fold_cagr","full_cagr","full_cagr_3bps","full_maximum_drawdown","full_peak_to_trough_calendar_days","events","positive_neighbor_count","holy_shit_gate"]].head(30).to_string(index=False),flush=True)


if __name__=="__main__":
    import argparse;p=argparse.ArgumentParser();p.add_argument("--stage",choices=("all","build","evaluate"),default="all");a=p.parse_args()
    if a.stage in ("all","build"):build()
    if a.stage in ("all","evaluate"):evaluate()
