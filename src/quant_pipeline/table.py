from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .config import ScanConfig
from .holdout import assert_pre_holdout_frame
from .registry import TargetSpec

ET = "America/New_York"
ROW_KEYS=["symbol","session_date","bar_start_ts","decision_ts"]


def apply_analysis_eligibility(frame:pd.DataFrame)->pd.DataFrame:
    out=frame.copy()
    out["analysis_eligible"]=(out.symbol_role.eq("tradable")&out.pit_member.fillna(False)&out.scan_eligible.fillna(False)&out.session_grid_eligible.fillna(False))
    return out


def benchmark_valid_mask(frame:pd.DataFrame,benchmark_symbol:str)->pd.Series:
    return (frame.symbol.eq(benchmark_symbol)&frame.symbol_role.eq("benchmark")&frame.pit_member.fillna(True)&frame.scan_eligible.fillna(False)&frame.session_grid_eligible.fillna(False))


def load_canonical_bars(config: ScanConfig) -> pd.DataFrame:
    """Load raw execution bars and attach causal research metadata."""
    if not config.allow_holdout_access and pd.Timestamp(config.discovery_end) >= pd.Timestamp(config.sealed_holdout_start):
        raise ValueError(f"Discovery end {config.discovery_end} reaches sealed holdout {config.sealed_holdout_start}")
    con=duckdb.connect(config.catalog_path,read_only=True)
    try:
        tables={row[0] for row in con.execute("show tables").fetchall()}
        if config.require_membership and config.membership_table not in tables:
            raise ValueError(f"Required membership table is missing: {config.membership_table}")
        symbol_clause="" if not config.universe else "and symbol in ("+",".join("?" for _ in config.universe)+")"
        bars=con.execute(f"""
          select symbol,bar_start_ts,bar_end_ts,available_at_ts,open,high,low,close,volume,vwap,
                 session_date,ingested_at,source_ingestion_id
          from {config.source_table}
          where feed=? and adjustment=? and bar_complete
            and cast(session_date as date)>=cast(? as date) and cast(session_date as date)<cast(? as date)
            {symbol_clause}
          order by symbol,bar_start_ts
        """,[config.feed,config.adjustment,config.start,config.sealed_holdout_start,*config.universe]).fetchdf()
        membership=(con.execute(f"""
          select cast(date as date) session_date,symbol,is_member,source_ingestion_id,ingested_at
          from {config.membership_table}
          where cast(date as date)>=cast(? as date) and cast(date as date)<cast(? as date)
        """,[config.start,config.sealed_holdout_start]).fetchdf() if config.membership_table in tables else pd.DataFrame())
    finally:
        con.close()
    if bars.empty: raise ValueError("No canonical bars matched the configuration")
    assert_pre_holdout_frame(bars,config.sealed_holdout_start,"raw bars")
    if not membership.empty:assert_pre_holdout_frame(membership.rename(columns={"date":"session_date"}),config.sealed_holdout_start,"universe membership")
    for column in ["bar_start_ts","bar_end_ts","available_at_ts"]: bars[column]=pd.to_datetime(bars[column],utc=True)
    bars["_ingested_at"]=pd.to_datetime(bars.pop("ingested_at"),utc=True,errors="coerce")
    bars=bars.sort_values(["symbol","bar_start_ts","_ingested_at"]).drop_duplicates(["symbol","bar_start_ts"],keep="last")
    bars["session_date"]=pd.to_datetime(bars.session_date).dt.normalize()
    bars=_attach_calendar(bars,config)
    bars=_attach_membership(bars,membership,config)
    bars["pit_member"]=np.where(bars.symbol_role.eq("benchmark"),True,bars.is_member.fillna(False))
    bars["membership_source_quality"]=config.membership_source_quality
    bars["shortened_session"]=bars.is_shortened_session
    bars=_attach_adjusted_prices(bars,config)
    bars["decision_ts"]=bars.available_at_ts
    bars["scan_eligible"]=bars.bar_grid_valid&bars.available_at_ts.ge(bars.bar_end_ts)
    bars=apply_analysis_eligibility(bars); bars["benchmark_valid"]=benchmark_valid_mask(bars,config.benchmark_symbol)
    return bars.sort_values(["symbol","bar_start_ts"]).reset_index(drop=True)


def _attach_calendar(bars: pd.DataFrame,config: ScanConfig) -> pd.DataFrame:
    calendar=xcals.get_calendar(config.exchange_calendar)
    sessions=calendar.sessions_in_range(pd.Timestamp(config.start),pd.Timestamp(config.discovery_end))
    schedule=pd.DataFrame({
        "session_date":pd.to_datetime([s.date() for s in sessions]),
        "scheduled_open":[calendar.session_open(s) for s in sessions],
        "scheduled_close":[calendar.session_close(s) for s in sessions],
    })
    out=bars.merge(schedule,on="session_date",how="inner",validate="many_to_one")
    out["session_length_minutes"]=(out.scheduled_close-out.scheduled_open).dt.total_seconds().div(60).astype(int)
    out["is_shortened_session"]=out.session_length_minutes.lt(390)
    offset=(out.bar_start_ts-out.scheduled_open).dt.total_seconds().div(60)
    out["bar_grid_valid"]=offset.ge(0)&offset.mod(5).eq(0)&out.bar_end_ts.eq(out.bar_start_ts+pd.Timedelta(minutes=5))&out.bar_end_ts.le(out.scheduled_close)
    out=out[out.bar_grid_valid].copy()
    grouped=out.groupby(["symbol","session_date"],sort=False)
    out["bar_gap"]=grouped.bar_start_ts.diff().ne(pd.Timedelta(minutes=5))
    out.loc[grouped.head(1).index,"bar_gap"]=False
    out["gap_segment"]=out.bar_gap.groupby([out.symbol,out.session_date],sort=False).cumsum().astype(int)
    actual=grouped.bar_start_ts.transform("nunique")
    expected=(out.session_length_minutes//5).astype(int)
    out["expected_bars_in_session"]=expected
    out["missing_bars_in_session"]=(expected-actual).clip(lower=0).astype(int)
    out["session_grid_eligible"]=out.missing_bars_in_session.le(config.maximum_missing_bars_per_session)
    return out


def _attach_membership(bars: pd.DataFrame,membership: pd.DataFrame,config: ScanConfig) -> pd.DataFrame:
    benchmark=set(config.benchmark_symbols)|{config.benchmark_symbol}
    if membership.empty:
        if config.require_membership: raise ValueError("Membership data is required but empty")
        bars["is_member"]=~bars.symbol.isin(benchmark)
        bars["membership_source_id"]="unverified"
    else:
        membership=membership.copy(); membership["session_date"]=pd.to_datetime(membership.session_date).dt.normalize()
        membership=membership.sort_values(["symbol","session_date","ingested_at"]).drop_duplicates(["symbol","session_date"],keep="last")
        membership=membership.rename(columns={"source_ingestion_id":"membership_source_id"})
        bars=bars.merge(membership[["session_date","symbol","is_member","membership_source_id"]],on=["session_date","symbol"],how="left")
    bars["symbol_role"]=np.where(bars.symbol.isin(benchmark),"benchmark",np.where(bars.is_member.fillna(False),"tradable","ineligible"))
    return bars[bars.symbol_role.ne("ineligible")].copy()


def _attach_adjusted_prices(bars: pd.DataFrame,config: ScanConfig) -> pd.DataFrame:
    path=Path(config.corporate_actions_path)
    if not path.exists():
        if config.require_corporate_actions: raise ValueError(f"Corporate-action ledger is required: {path}")
        actions=pd.DataFrame(columns=["symbol","ex_date","split_ratio"])
    else:
        actions=pd.read_parquet(path)
        if "ex_date" in actions and pd.to_datetime(actions.ex_date,errors="coerce").ge(pd.Timestamp(config.sealed_holdout_start)).any():
            raise ValueError("Corporate-action ledger reaches sealed holdout")
    out=bars.copy()
    for column in ["open","high","low","close","vwap"]: out[f"{column}_raw"]=pd.to_numeric(out[column],errors="coerce")
    out["split_factor"]=1.0
    if not actions.empty:
        actions=actions.copy(); actions.ex_date=pd.to_datetime(actions.ex_date).dt.normalize()
        for symbol,index in out.groupby("symbol",sort=False).groups.items():
            a=actions.loc[actions.symbol.eq(symbol)&actions.split_ratio.gt(0),["ex_date","split_ratio"]].sort_values("ex_date")
            if a.empty: continue
            dates=out.loc[index,"session_date"].to_numpy(dtype="datetime64[ns]")
            ex=a.ex_date.to_numpy(dtype="datetime64[ns]"); ratios=a.split_ratio.to_numpy(float)
            future=np.cumprod(ratios[::-1])[::-1]
            positions=np.searchsorted(ex,dates,side="right")
            factors=np.where(positions<len(ex),future[np.minimum(positions,len(ex)-1)],1.0)
            out.loc[index,"split_factor"]=factors
    for column in ["open","high","low","close","vwap"]:
        out[f"{column}_adjusted"]=out[f"{column}_raw"]/out.split_factor
    out["dividend_factor"]=1.0
    if not actions.empty and "cash_amount" in actions:
        dividends=actions.loc[actions.cash_amount.fillna(0).gt(0)&actions.ex_date.notna(),["symbol","ex_date","cash_amount"]]
        for symbol,index in out.groupby("symbol",sort=False).groups.items():
            factor=np.ones(len(index)); locations=np.asarray(list(index)); dates=out.loc[index,"session_date"].to_numpy(dtype="datetime64[ns]")
            for action in dividends.loc[dividends.symbol.eq(symbol)].sort_values("ex_date").itertuples():
                ex=np.datetime64(pd.Timestamp(action.ex_date).normalize()); prior=np.flatnonzero(dates<ex)
                if not len(prior):continue
                previous_close=float(out.loc[locations[prior[-1]],"close_adjusted"]); adjustment=1-float(action.cash_amount)/previous_close if previous_close>float(action.cash_amount) else np.nan
                if np.isfinite(adjustment):factor[dates<ex]*=adjustment
            out.loc[index,"dividend_factor"]=factor
    for column in ["open","high","low","close","vwap"]:out[f"{column}_total_return_adjusted"]=out[f"{column}_adjusted"]*out.dividend_factor
    return out


def filter_decision_rows(frame: pd.DataFrame,config: ScanConfig) -> pd.DataFrame:
    out=frame
    if config.decision_times_et:
        local=pd.to_datetime(out.decision_ts,utc=True).dt.tz_convert(ET)
        out=out.loc[local.dt.strftime("%H:%M").isin(config.decision_times_et)]
    return out.loc[out.analysis_eligible.fillna(False)].reset_index(drop=True)


def add_targets(frame: pd.DataFrame,targets: list[TargetSpec],benchmark_symbol: str="QQQ",config:ScanConfig|None=None) -> pd.DataFrame:
    """Locate the first genuinely actionable entry bar and explicit exits."""
    cfg=config or ScanConfig(); assert_pre_holdout_frame(frame,cfg.sealed_holdout_start,"target input")
    out=frame.sort_values(["symbol","session_date","bar_start_ts"],kind="stable").reset_index(drop=True).copy()
    if "open_raw" not in out:out["open_raw"]=out.open
    if "close_raw" not in out:out["close_raw"]=out.close
    out["decision_ts"]=out.available_at_ts
    n=len(out); entry_ts=np.full(n,np.datetime64("NaT"),dtype="datetime64[ns]"); entry_open=np.full(n,np.nan)
    values={t.name:np.full(n,np.nan,dtype=np.float32) for t in targets if t.classification=="raw"}
    exit_meta={t.name:np.full(n,np.datetime64("NaT"),dtype="datetime64[ns]") for t in targets if t.classification=="raw"}
    exit_price_meta={t.name:np.full(n,np.nan,dtype=np.float64) for t in targets if t.classification=="raw"}
    actual_meta={t.name:np.full(n,np.nan,dtype=np.float32) for t in targets if t.classification=="raw"}
    for _,idx in out.groupby(["symbol","session_date"],sort=False).groups.items():
        loc=np.asarray(list(idx),dtype=int); group=out.loc[loc]
        starts=group.bar_start_ts.to_numpy(dtype="datetime64[ns]"); ends=group.bar_end_ts.to_numpy(dtype="datetime64[ns]")
        decisions=group.decision_ts.to_numpy(dtype="datetime64[ns]"); opens=group.open_raw.to_numpy(float); closes=group.close_raw.to_numpy(float)
        entries=np.searchsorted(starts,decisions,side="left"); valid_entry=entries<len(starts)
        entry_ts[loc[valid_entry]]=starts[entries[valid_entry]]; entry_open[loc[valid_entry]]=opens[entries[valid_entry]]
        for target in [t for t in targets if t.classification=="raw"]:
            if target.horizon_minutes is None:
                exits=np.full(len(group),len(group)-1)
            else:
                desired=starts[np.minimum(entries,len(starts)-1)]+np.timedelta64(target.horizon_minutes,"m")
                exits=np.searchsorted(ends,desired,side="left")
            valid=valid_entry&(exits<len(ends))&(exits>=entries)
            rows=loc[valid]; values[target.name][rows]=(closes[exits[valid]]/opens[entries[valid]]-1).astype(np.float32)
            exit_meta[target.name][rows]=ends[exits[valid]]
            exit_price_meta[target.name][rows]=closes[exits[valid]]
            actual_meta[target.name][rows]=((ends[exits[valid]]-starts[entries[valid]])/np.timedelta64(1,"m")).astype(np.float32)
    row_valid=(out.analysis_eligible.fillna(False)|out.benchmark_valid.fillna(False)).to_numpy() if {"analysis_eligible","benchmark_valid"}.issubset(out) else np.ones(n,dtype=bool)
    entry_ts[~row_valid]=np.datetime64("NaT"); entry_open[~row_valid]=np.nan
    for name in values:values[name][~row_valid]=np.nan; exit_meta[name][~row_valid]=np.datetime64("NaT"); exit_price_meta[name][~row_valid]=np.nan; actual_meta[name][~row_valid]=np.nan
    additions={"entry_ts":pd.to_datetime(entry_ts,utc=True),"entry_open_raw":entry_open}
    for name,array in values.items():additions[name]=array; additions[f"exit_ts__{name}"]=pd.to_datetime(exit_meta[name],utc=True); additions[f"exit_close_raw__{name}"]=exit_price_meta[name]; additions[f"actual_horizon_minutes__{name}"]=actual_meta[name]
    out=pd.concat([out,pd.DataFrame(additions,index=out.index)],axis=1)
    adjusted={}; benchmark_mask=out.benchmark_valid.fillna(False) if "benchmark_valid" in out else out.symbol.eq(benchmark_symbol)
    for target in [t for t in targets if t.classification=="benchmark_adjusted"]:
        raw=target.name.removesuffix("_benchmark_adjusted")
        benchmark=out.loc[benchmark_mask,["decision_ts",raw]].drop_duplicates("decision_ts").set_index("decision_ts")[raw]
        adjusted[target.name]=(out[raw]-out.decision_ts.map(benchmark)).astype(np.float32)
    beta_specs=[t for t in targets if t.classification=="beta_residual"]
    if beta_specs:
        close_column="close_total_return_adjusted" if "close_total_return_adjusted" in out else "close_adjusted" if "close_adjusted" in out else "close_raw"
        daily=out.groupby(["symbol","session_date"],sort=False)[close_column].last().unstack("symbol"); returns=daily.pct_change(); market=returns.get(benchmark_symbol)
        beta_lookup={}
        if market is not None:
            for symbol in returns:
                covariance=returns[symbol].rolling(cfg.beta_window_sessions,min_periods=cfg.beta_min_observations).cov(market); variance=market.rolling(cfg.beta_window_sessions,min_periods=cfg.beta_min_observations).var(); beta_lookup[symbol]=(covariance/variance.replace(0,np.nan)).shift(1)
        beta_values=pd.Series(np.nan,index=out.index,dtype=float)
        for symbol,indices in out.groupby("symbol",sort=False).groups.items():
            if symbol in beta_lookup:beta_values.loc[indices]=out.loc[indices,"session_date"].map(beta_lookup[symbol]).to_numpy()
        adjusted["beta_at_decision"]=beta_values
        for target in beta_specs:
            raw=target.name.removesuffix("_beta_residual"); benchmark=out.loc[benchmark_mask,["decision_ts",raw]].drop_duplicates("decision_ts").set_index("decision_ts")[raw]
            adjusted[target.name]=(out[raw]-beta_values*out.decision_ts.map(benchmark)).astype(np.float32)
    if adjusted:out=pd.concat([out,pd.DataFrame(adjusted,index=out.index)],axis=1)
    assert_pre_holdout_frame(out,cfg.sealed_holdout_start,"target output")
    return out


def validate_point_in_time(frame: pd.DataFrame,targets: list[TargetSpec],sealed_holdout_start: str|None=None) -> dict[str,int|float]:
    if sealed_holdout_start:assert_pre_holdout_frame(frame,sealed_holdout_start,"point-in-time validation")
    required={"symbol","session_date","decision_ts","bar_start_ts","bar_end_ts","available_at_ts","entry_ts"}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"Feature-table identifiers missing: {sorted(missing)}")
    duplicate=int(frame.duplicated(["symbol","session_date","decision_ts"]).sum())
    source_duplicate=int(frame.duplicated(["symbol","session_date","bar_start_ts"]).sum())
    availability=int((frame.available_at_ts<frame.bar_end_ts).sum())
    entry_before=int((frame.entry_ts<frame.decision_ts).fillna(False).sum())
    exit_before=0
    cross_session=0
    horizon_mismatch=0; missing_entry_price=0; missing_exit_price=0
    for target in [t for t in targets if t.classification=="raw"]:
        column=f"exit_ts__{target.name}"
        if column not in frame: continue
        valid=frame.entry_ts.notna()&frame[column].notna()
        exit_before+=int((frame.loc[valid,column]<=frame.loc[valid,"entry_ts"]).sum())
        exit_local=frame.loc[valid,column].dt.tz_convert(ET).dt.date
        cross_session+=int((exit_local!=pd.to_datetime(frame.loc[valid,"session_date"]).dt.date).sum())
        price_column=f"exit_close_raw__{target.name}"; horizon_column=f"actual_horizon_minutes__{target.name}"
        target_valid=frame[target.name].notna() if target.name in frame else pd.Series(False,index=frame.index)
        missing_entry_price+=int((target_valid&frame.entry_open_raw.isna()).sum()) if "entry_open_raw" in frame else int(target_valid.sum())
        missing_exit_price+=int((target_valid&frame[price_column].isna()).sum()) if price_column in frame else int(target_valid.sum())
        if target.horizon_minutes is not None and horizon_column in frame:horizon_mismatch+=int((target_valid&frame[horizon_column].ne(target.horizon_minutes)).sum())
    holdout=int((pd.to_datetime(frame.session_date)>=pd.Timestamp(sealed_holdout_start)).sum()) if sealed_holdout_start else 0
    universe=int((frame.symbol_role.eq("tradable")&frame.is_member.ne(True)).sum()) if "is_member" in frame else 0
    grid=int((~frame.bar_grid_valid.fillna(False)).sum()) if "bar_grid_valid" in frame else 0
    benchmark_alignment=0
    benchmark_missing_rows=int((frame.symbol_role.eq("benchmark")&~frame.benchmark_valid.fillna(False)).sum()) if {"symbol_role","benchmark_valid"}.issubset(frame) else 0
    benchmark_invalid_sessions=int(frame.loc[frame.symbol_role.eq("benchmark")&~frame.benchmark_valid.fillna(False),"session_date"].nunique()) if {"symbol_role","benchmark_valid"}.issubset(frame) else 0
    benchmark_present=("symbol_role" in frame and frame.symbol_role.eq("benchmark").any()) or frame.symbol.eq("QQQ").any()
    for target in [t for t in targets if t.classification=="benchmark_adjusted"] if benchmark_present else []:
        raw=target.name.removesuffix("_benchmark_adjusted")
        if raw in frame and target.name in frame:
            eligible=frame.symbol_role.eq("tradable") if "symbol_role" in frame else pd.Series(True,index=frame.index)
            benchmark_alignment+=int((eligible&frame[raw].notna()&frame[target.name].isna()).sum())
    beta_history_excluded=0
    for target in [t for t in targets if t.classification=="beta_residual"]:
        raw=target.name.removesuffix("_beta_residual")
        if raw in frame and target.name in frame:
            eligible=frame.symbol_role.eq("tradable") if "symbol_role" in frame else pd.Series(True,index=frame.index)
            beta_history_excluded+=int((eligible&frame[raw].notna()&frame[target.name].isna()).sum())
    failures=duplicate+source_duplicate+availability+entry_before+exit_before+cross_session+horizon_mismatch+missing_entry_price+missing_exit_price+holdout+universe+grid+benchmark_alignment
    if failures: raise ValueError(f"PIT validation failed: duplicates={duplicate}, source_duplicates={source_duplicate}, availability={availability}, entry={entry_before}, exit={exit_before}, cross_session={cross_session}, horizon={horizon_mismatch}, entry_price={missing_entry_price}, exit_price={missing_exit_price}, holdout={holdout}, universe={universe}, grid={grid}, benchmark_alignment={benchmark_alignment}")
    sessions=frame.drop_duplicates(["symbol","session_date"])
    expected=int(sessions.expected_bars_in_session.sum()) if "expected_bars_in_session" in sessions else 0; missing_bars=int(sessions.missing_bars_in_session.sum()) if "missing_bars_in_session" in sessions else 0
    incomplete=sessions.loc[sessions.missing_bars_in_session.gt(0)] if "missing_bars_in_session" in sessions else sessions.iloc[0:0]
    excluded=sessions.loc[~sessions.session_grid_eligible.fillna(False)] if "session_grid_eligible" in sessions else sessions.iloc[0:0]
    excessive=sorted(excluded.symbol.unique().tolist()) if len(excluded) else []
    return {"rows":len(frame),"duplicate_identifiers":duplicate,"duplicate_source_bars":source_duplicate,"availability_violations":availability,"target_timing_violations":entry_before+exit_before,"entry_before_decision_violations":entry_before,"exit_before_entry_violations":exit_before,"target_cross_session_violations":cross_session,"horizon_mismatch_violations":horizon_mismatch,"missing_entry_price_rows":missing_entry_price,"missing_exit_price_rows":missing_exit_price,"holdout_rows":holdout,"universe_violations":universe,"grid_violations":grid,"missing_benchmark_rows":benchmark_alignment,"benchmark_source_rows_invalid":benchmark_missing_rows,"benchmark_sessions_invalid":benchmark_invalid_sessions,"beta_residual_rows_excluded_insufficient_history":beta_history_excluded,"expected_bars":expected,"missing_bars":missing_bars,"missing_bar_rate":missing_bars/expected if expected else 0.0,"incomplete_sessions":len(incomplete),"sessions_excluded":len(excluded),"symbols_with_excessive_missingness":excessive,"target_columns":len(targets)}


def source_provenance(config: ScanConfig) -> dict:
    con=duckdb.connect(config.catalog_path,read_only=True)
    try:
        schema=con.execute(f"describe {config.source_table}").fetchall()
        stats=con.execute(f"""select count(*),min(bar_start_ts),max(bar_end_ts),count(distinct symbol),max(ingested_at)
          from {config.source_table} where feed=? and adjustment=? and cast(session_date as date)>=cast(? as date) and cast(session_date as date)<cast(? as date)""",[config.feed,config.adjustment,config.start,config.sealed_holdout_start]).fetchone()
        partitions=con.execute(f"""select year(cast(session_date as date)),count(*),min(bar_start_ts),max(bar_end_ts),max(ingested_at) from {config.source_table} where feed=? and adjustment=? and cast(session_date as date)>=cast(? as date) and cast(session_date as date)<cast(? as date) group by 1 order by 1""",[config.feed,config.adjustment,config.start,config.sealed_holdout_start]).fetchall()
        membership=con.execute(f"""select count(*),min(date),max(date),count(distinct symbol),sum(case when is_member then 1 else 0 end) from {config.membership_table} where cast(date as date)>=cast(? as date) and cast(date as date)<cast(? as date)""",[config.start,config.sealed_holdout_start]).fetchone()
    finally: con.close()
    return {"catalog_path":config.catalog_path,"source_table":config.source_table,"row_count":stats[0],"minimum_timestamp":str(stats[1]),"maximum_timestamp":str(stats[2]),"symbol_count":stats[3],"latest_ingestion_timestamp":str(stats[4]),"feed":config.feed,"adjustment":config.adjustment,"bar_frequency":"5m","holdout_boundary":config.sealed_holdout_start,"source_schema_hash":hashlib.sha256(json.dumps(schema,default=str).encode()).hexdigest(),"source_partition_fingerprint":hashlib.sha256(json.dumps(partitions,default=str).encode()).hexdigest(),"membership_fingerprint":hashlib.sha256(json.dumps(membership,default=str).encode()).hexdigest(),"duplicate_resolution_policy":"latest ingested row per symbol and bar_start_ts","corporate_action_source":config.corporate_actions_path,"universe_source":config.membership_table,"sector_source":config.sector_map_path,"exchange_calendar":config.exchange_calendar}
