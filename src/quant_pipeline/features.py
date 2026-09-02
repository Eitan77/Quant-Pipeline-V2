from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ScanConfig
from .registry import FeatureSpec
from .table import apply_analysis_eligibility,benchmark_valid_mask


def build_features(
    bars: pd.DataFrame,
    config: ScanConfig,
    specs: list[FeatureSpec],
    *,
    symbol_local: bool = False,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build the clean-room core library, always from past/completed bars."""
    wanted={s.name for s in specs if s.status=="requested"}
    active_lookbacks={int(s.lookback) for s in specs if s.lookback in config.lookbacks}
    if any(s.family=="normalization" for s in specs): active_lookbacks.update({1,10})
    active_lead_lags={int(s.lookback) for s in specs if s.family=="lead_lag" and s.lookback is not None}
    active_lookbacks.update(active_lead_lags)
    # All later rolling/grouped results align on this canonical row order.
    # Filtered Parquet shards retain arbitrary source indexes; resetting here
    # prevents pandas from silently realigning values to stale row labels.
    x = bars.sort_values(["symbol", "bar_start_ts"], kind="stable").reset_index(drop=True).copy()
    if "decision_ts" not in x:x["decision_ts"]=x["available_at_ts"]
    if "gap_segment" not in x:x["gap_segment"]=0
    if "scheduled_open" not in x:x["scheduled_open"]=x.groupby(["symbol","session_date"]).bar_start_ts.transform("min")
    if "scheduled_close" not in x:x["scheduled_close"]=x.groupby(["symbol","session_date"]).bar_end_ts.transform("max")
    if "session_length_minutes" not in x:x["session_length_minutes"]=(x.scheduled_close-x.scheduled_open).dt.total_seconds()/60
    if "symbol_role" not in x:x["symbol_role"]=np.where(x.symbol.eq(config.benchmark_symbol),"benchmark","tradable")
    defaults={"pit_member":True,"scan_eligible":True,"session_grid_eligible":True,"shortened_session":False,"bar_gap":False,"membership_source_quality":"synthetic_or_prevalidated","split_factor":1.0}
    for column,value in defaults.items():
        if column not in x:x[column]=value
    x=apply_analysis_eligibility(x); x["benchmark_valid"]=benchmark_valid_mask(x,config.benchmark_symbol)
    for column in ["open","high","low","close","vwap"]:
        if f"{column}_raw" not in x:x[f"{column}_raw"]=x[column]
        if f"{column}_adjusted" not in x:x[f"{column}_adjusted"]=x[column]
    # Research features use a split-adjusted price basis. Raw prices remain in
    # the table for target entry/exit and later execution modeling.
    for column in ["open","high","low","close","vwap"]:
        adjusted=f"{column}_adjusted"
        if adjusted in x: x[column]=x[adjusted]
    g=x.groupby("symbol",sort=False)
    sg=x.groupby(["symbol","session_date"],sort=False)
    rg=x.groupby(["symbol","session_date","gap_segment"],sort=False)
    total_close=x["close_total_return_adjusted"] if "close_total_return_adjusted" in x else x.close
    x["_intraday_ret"]=rg["close"].pct_change(); x["_continuous_ret"]=total_close.groupby(x.symbol,sort=False).pct_change()
    x["_ret"]=x._intraday_ret; x["intraday_return_1"]=x._intraday_ret; x["continuous_return_1"]=x._continuous_ret
    x["_prev_close"] = g["close"].shift(); x["_range"] = x.high - x.low
    x["_session_open"] = x.groupby(["symbol", "session_date"], sort=False)["open"].transform("first")
    x["_session_high"] = x.groupby(["symbol", "session_date"], sort=False)["high"].cummax(); x["_session_low"] = x.groupby(["symbol", "session_date"], sort=False)["low"].cummin()
    x["_cum_pv"] = (x.close * x.volume).groupby([x.symbol, x.session_date], sort=False).cumsum(); x["_cum_vol"] = x.volume.groupby([x.symbol, x.session_date], sort=False).cumsum()
    x["return_since_open"] = x.close / x._session_open - 1; x["distance_session_vwap"] = x.close / (x._cum_pv / x._cum_vol.replace(0,np.nan)) - 1
    x["distance_session_high"] = x.close / x._session_high - 1; x["distance_session_low"] = x.close / x._session_low - 1
    x["bar_range_pct"] = x._range / x.close.replace(0,np.nan); x["body_to_range"] = (x.close-x.open).abs()/x._range.replace(0,np.nan)
    x["upper_wick_to_range"] = (x.high-x[["open","close"]].max(axis=1)) / x._range.replace(0,np.nan); x["lower_wick_to_range"] = (x[["open","close"]].min(axis=1)-x.low) / x._range.replace(0,np.nan)
    x["close_location"] = (x.close-x.low)/x._range.replace(0,np.nan)
    x["bar_return"] = x.close / x.open.replace(0, np.nan) - 1; x["bar_log_return"] = np.log(x.close / x.open.replace(0, np.nan))
    x["body_pct"] = (x.close-x.open)/x.open.replace(0,np.nan); x["absolute_body_pct"] = x.body_pct.abs()
    prev_high=rg.high.shift(); prev_low=rg.low.shift(); has_previous=prev_high.notna()&prev_low.notna()
    x["inside_bar"]=((x.high<=prev_high)&(x.low>=prev_low)).astype(float).where(has_previous); x["outside_bar"]=((x.high>=prev_high)&(x.low<=prev_low)).astype(float).where(has_previous)
    x["higher_high"]=(x.high>prev_high).astype(float).where(has_previous); x["higher_low"]=(x.low>prev_low).astype(float).where(has_previous); x["lower_high"]=(x.high<prev_high).astype(float).where(has_previous); x["lower_low"]=(x.low<prev_low).astype(float).where(has_previous)
    x["session_range_position"]=(x.close-x._session_low)/(x._session_high-x._session_low).replace(0,np.nan)
    x["cumulative_volume"]=x._cum_vol; x["current_volume_share"]=x.volume/x._cum_vol.replace(0,np.nan)
    x["_session_vwap"]=x._cum_pv/x._cum_vol.replace(0,np.nan); x["vwap_slope"]=x._session_vwap.groupby([x.symbol,x.session_date,x.gap_segment],sort=False).diff()
    side=np.sign(x.close-x._session_vwap); previous_side=side.groupby([x.symbol,x.session_date,x.gap_segment],sort=False).shift(); x["vwap_cross"]=((previous_side<=0)&(side>0)|(previous_side>=0)&(side<0)).astype(float).where(previous_side.notna())
    x["consecutive_positive_bars"]=_session_run_length(x._ret.gt(0),x.symbol,x.session_date,x.gap_segment); x["consecutive_negative_bars"]=_session_run_length(x._ret.lt(0),x.symbol,x.session_date,x.gap_segment)
    local=x.available_at_ts.dt.tz_convert("America/New_York"); x["bar_start_minute"]=((x.bar_start_ts-x.scheduled_open).dt.total_seconds()/60).astype(int); x["minutes_since_open"]=(x.available_at_ts-x.scheduled_open).dt.total_seconds()/60; x["minute_of_session"]=x.minutes_since_open; x["minutes_until_close"]=(x.scheduled_close-x.available_at_ts).dt.total_seconds()/60
    x["day_of_week"]=local.dt.dayofweek; x["month"]=local.dt.month; x["quarter"]=local.dt.quarter; x["decision_time_bucket"]=(x.minutes_since_open//30).astype(int)
    phase=2*np.pi*x.minutes_since_open/x.session_length_minutes.replace(0,np.nan); x["decision_time_sin"]=np.sin(phase); x["decision_time_cos"]=np.cos(phase)
    for n in sorted(active_lookbacks):
        columns_before=set(x.columns)
        prior_close=rg.close.shift(n); r=x.close/prior_close-1; roll_ret=rg["_ret"].rolling(n,min_periods=n)
        x[f"return_{n}"]=r.reset_index(level=0,drop=True); x[f"log_return_{n}"]=np.log1p(r)
        rv=roll_ret.std().reset_index(level=[0,1,2],drop=True); x[f"realized_vol_{n}"]=rv; tail_min=max(1,min(n,max(2,n//3))); x[f"downside_vol_{n}"]=x._ret.where(x._ret<0).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=tail_min).std().reset_index(level=[0,1,2],drop=True); x[f"upside_vol_{n}"]=x._ret.where(x._ret>0).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=tail_min).std().reset_index(level=[0,1,2],drop=True); x[f"return_vol_ratio_{n}"]=x[f"return_{n}"]/rv.replace(0,np.nan); x[f"return_consistency_{n}"]=(x._ret>0).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).mean().reset_index(level=[0,1,2],drop=True)
        x[f"positive_return_sum_{n}"]=x._ret.clip(lower=0).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).sum().reset_index(level=[0,1,2],drop=True); x[f"negative_return_sum_{n}"]=x._ret.clip(upper=0).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).sum().reset_index(level=[0,1,2],drop=True)
        x[f"volume_sum_{n}"]=rg.volume.rolling(n,min_periods=n).sum().reset_index(level=[0,1,2],drop=True); x[f"volume_mean_{n}"]=rg.volume.shift(1).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).mean().reset_index(level=[0,1,2],drop=True)
        inclusive_volume_mean=rg.volume.rolling(n,min_periods=n).mean().reset_index(level=[0,1,2],drop=True)
        x[f"relative_volume_prior_{n}"]=x.volume/x[f"volume_mean_{n}"].replace(0,np.nan); x[f"relative_volume_inclusive_{n}"]=x.volume/inclusive_volume_mean.replace(0,np.nan)
        x[f"relative_volume_{n}"]=x.volume/x[f"volume_mean_{n}"].replace(0,np.nan); x[f"dollar_volume_mean_{n}"]=(x.close*x.volume).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).mean().reset_index(level=[0,1,2],drop=True)
        x[f"largest_positive_bar_{n}"]=rg._ret.rolling(n,min_periods=n).max().reset_index(level=[0,1,2],drop=True); x[f"largest_negative_bar_{n}"]=rg._ret.rolling(n,min_periods=n).min().reset_index(level=[0,1,2],drop=True)
        high=rg.high.rolling(n,min_periods=n).max().reset_index(level=[0,1,2],drop=True); low=rg.low.rolling(n,min_periods=n).min().reset_index(level=[0,1,2],drop=True); prior_high=high.groupby([x.symbol,x.session_date,x.gap_segment]).shift(); prior_low=low.groupby([x.symbol,x.session_date,x.gap_segment]).shift()
        x[f"distance_rolling_high_{n}"]=x.close/high-1; x[f"distance_rolling_low_{n}"]=x.close/low-1; x[f"range_position_{n}"]=(x.close-low)/(high-low).replace(0,np.nan)
        range_mean=rg._range.rolling(n,min_periods=n).mean().reset_index(level=[0,1,2],drop=True); x[f"range_mean_{n}"]=range_mean; x[f"range_ratio_{n}"]=x._range/range_mean.replace(0,np.nan)
        x[f"breakout_magnitude_{n}"]=(x.close/prior_high-1).clip(lower=0); x[f"breakdown_magnitude_{n}"]=(x.close/prior_low-1).clip(upper=0)
        half=max(1,n//2); short_vol=rg.volume.rolling(half,min_periods=half).mean().reset_index(level=[0,1,2],drop=True); short_rv=rg._ret.rolling(half,min_periods=max(1,half)).std().reset_index(level=[0,1,2],drop=True)
        x[f"volume_acceleration_{n}"]=short_vol/x[f"volume_mean_{n}"].replace(0,np.nan)-1; x[f"volatility_acceleration_{n}"]=short_rv/rv.replace(0,np.nan)-1
        pv=(x.close*x.volume).groupby([x.symbol,x.session_date,x.gap_segment]).rolling(n,min_periods=n).sum().reset_index(level=[0,1,2],drop=True); vv=rg.volume.rolling(n,min_periods=n).sum().reset_index(level=[0,1,2],drop=True)
        if symbol_local:x=x.copy()
        x[f"rolling_vwap_distance_{n}"]=x.close/(pv/vv.replace(0,np.nan))-1
        x[f"volume_range_ratio_{n}"]=x.volume/(x._range.replace(0,np.nan)); x[f"return_volume_product_{n}"]=x._ret*x.volume; x[f"return_outlier_score_{n}"]=x._ret/rv.replace(0,np.nan)
        if not symbol_local:
            eligible=x.analysis_eligible
            for base in ["return","relative_volume","realized_vol","range_position","return_vol_ratio"]: x[f"{base}_rank_{n}"]=x[f"{base}_{n}"].where(eligible).groupby(x.decision_ts).rank(pct=True)
        for column in set(x.columns)-columns_before:
            if pd.api.types.is_float_dtype(x[column]): x[column]=x[column].astype(np.float32)
        # Consolidate small symbol shards between lookbacks. A full-universe
        # consolidation duplicates several GiB at once and can exhaust RAM.
        if symbol_local:x=x.copy()
        g=x.groupby("symbol",sort=False); sg=x.groupby(["symbol","session_date"],sort=False); rg=x.groupby(["symbol","session_date","gap_segment"],sort=False)
    total_open=x["open_total_return_adjusted"] if "open_total_return_adjusted" in x else x.open; total_close=x["close_total_return_adjusted"] if "close_total_return_adjusted" in x else x.close
    daily=pd.DataFrame({"symbol":x.symbol,"session_date":x.session_date,"total_open":total_open,"total_close":total_close}).groupby(["symbol","session_date"],sort=False).agg(session_open=("total_open","first"),session_close=("total_close","last"))
    daily["previous_close"]=daily.groupby(level=0).session_close.shift()
    completed_return=daily.groupby(level=0).session_close.pct_change()
    # At any decision in session D, only returns through D-1 are known.
    # The unshifted value uses D's final close and is future leakage.
    daily["previous_session_return"]=completed_return.groupby(level=0).shift(1)
    x=x.join(daily[["previous_close","previous_session_return"]],on=["symbol","session_date"])
    x["overnight_gap"]=x._session_open/x.previous_close-1
    for sessions in [2,5]:
        if f"session_return_{sessions}" not in wanted: continue
        daily[f"session_return_{sessions}"]=daily.groupby(level=0).session_close.shift(1)/daily.groupby(level=0).session_close.shift(sessions+1)-1
        x=x.join(daily[[f"session_return_{sessions}"]],on=["symbol","session_date"])
    if not symbol_local and config.benchmark_symbol in set(x.symbol):
        market=x.loc[x.benchmark_valid,["decision_ts","_ret"]].drop_duplicates("decision_ts").set_index("decision_ts")["_ret"]
        x["market_return_1"]=x.decision_ts.map(market); x["stock_minus_market_return_1"]=x._ret-x.market_return_1
    elif not symbol_local: x["market_return_1"]=np.nan; x["stock_minus_market_return_1"]=np.nan
    if not symbol_local:
        eligible=x.analysis_eligible
        cross_valid=eligible&x._ret.notna(); breadth=x.loc[cross_valid].groupby("decision_ts")._ret.apply(lambda s: float((s>0).mean()))
        dispersion=x.loc[cross_valid].groupby("decision_ts")._ret.std()
        x["universe_breadth_positive"]=x.decision_ts.map(breadth).where(x.analysis_eligible); x["universe_return_dispersion"]=x.decision_ts.map(dispersion).where(x.analysis_eligible)
        for n in sorted(active_lookbacks):
            benchmark=x.loc[x.benchmark_valid,["decision_ts",f"return_{n}"]].drop_duplicates("decision_ts").set_index("decision_ts")[f"return_{n}"]
            x[f"market_return_{n}"]=x.decision_ts.map(benchmark); x[f"stock_minus_market_return_{n}"]=x[f"return_{n}"]-x[f"market_return_{n}"]
        x["market_up"]=(x.market_return_1>0).astype(float).where(x.market_return_1.notna()); market_vol=x.loc[x.benchmark_valid,["bar_start_ts","realized_vol_10"]].rename(columns={"realized_vol_10":"_market_vol"}) if "realized_vol_10" in x else pd.DataFrame()
        if not market_vol.empty:
            market_vol=market_vol.sort_values("bar_start_ts").drop_duplicates("bar_start_ts")
            market_vol["_market_vol_median"]=market_vol._market_vol.expanding(min_periods=100).median()
            x=x.merge(market_vol,on="bar_start_ts",how="left"); x["high_market_vol"]=(x._market_vol>x._market_vol_median).astype(float)
        else: x["high_market_vol"]=np.nan
        for n in [20,60]:
            if f"market_residual_return_{n}" not in wanted:continue
            prior_stock=x.groupby("symbol",sort=False)._ret.shift(1); prior_market=x.groupby("symbol",sort=False).market_return_1.shift(1)
            beta=pd.Series(np.nan,index=x.index,dtype=float)
            for _,indices in x.groupby("symbol",sort=False).groups.items():
                stock=prior_stock.loc[indices]; market_series=prior_market.loc[indices]
                beta.loc[indices]=(stock.rolling(n,min_periods=n).cov(market_series)/market_series.rolling(n,min_periods=n).var().replace(0,np.nan)).to_numpy()
            x[f"market_residual_return_{n}"]=x._ret-beta*x.market_return_1
    for minutes in config.opening_windows_minutes:
        if not any(name.endswith(f"_{minutes}m") and name.startswith("opening_") or name.startswith("distance_opening_") and name.endswith(f"_{minutes}m") for name in wanted): continue
        if minutes < 5:  # source is 5-minute bars; 1m/3m windows are not fabricated
            continue
        bars_n=max(1,int(np.ceil(minutes/5))); grp=x.groupby(["symbol","session_date"],sort=False)
        open0=grp.open.transform("first"); close_n=grp.close.transform(lambda s:s.iloc[min(bars_n-1,len(s)-1)]); high_n=grp.high.transform(lambda s:s.iloc[:bars_n].max()); low_n=grp.low.transform(lambda s:s.iloc[:bars_n].min()); vol_n=grp.volume.transform(lambda s:s.iloc[:bars_n].sum()); ret_n=grp._ret.transform(lambda s:s.iloc[:bars_n].std())
        offset=((x.bar_start_ts-x.scheduled_open).dt.total_seconds()/60).astype(int)
        exact_prefix=(offset.lt(minutes)&offset.mod(5).eq(0)).groupby([x.symbol,x.session_date]).transform("sum").eq(bars_n)
        starts_at_open=grp.bar_start_ts.transform("first").eq(x.scheduled_open)
        valid=exact_prefix & starts_at_open & x.available_at_ts.ge(x.scheduled_open+pd.to_timedelta(minutes,unit="m"))
        prefix=f"_{minutes}m"; x[f"opening_return{prefix}"]=(close_n/open0-1).where(valid); x[f"opening_range{prefix}"]=((high_n-low_n)/open0).where(valid); x[f"opening_volume{prefix}"]=vol_n.where(valid); x[f"opening_realized_vol{prefix}"]=ret_n.where(valid); x[f"opening_close_location{prefix}"]=((close_n-low_n)/(high_n-low_n).replace(0,np.nan)).where(valid); x[f"distance_opening_high{prefix}"]=(x.close/high_n-1).where(valid); x[f"distance_opening_low{prefix}"]=(x.close/low_n-1).where(valid); x[f"opening_breakout{prefix}"]=(x.close>high_n).where(valid).astype(float); x[f"opening_breakdown{prefix}"]=(x.close<low_n).where(valid).astype(float)
    for sessions in [20,60]:
        if not ({f"tod_relative_volume_{sessions}",f"tod_cumulative_relative_volume_{sessions}"}&wanted): continue
        keys=[x.symbol,x.minutes_since_open]
        baseline=x.volume.groupby(keys,sort=False).transform(lambda s:s.shift().rolling(sessions,min_periods=max(5,sessions//4)).mean())
        cumulative_baseline=x.cumulative_volume.groupby(keys,sort=False).transform(lambda s:s.shift().rolling(sessions,min_periods=max(5,sessions//4)).mean())
        x[f"tod_relative_volume_{sessions}"]=x.volume/baseline.replace(0,np.nan); x[f"tod_cumulative_relative_volume_{sessions}"]=x.cumulative_volume/cumulative_baseline.replace(0,np.nan)
    if not symbol_local:
        breadth_change=x.groupby("decision_ts").universe_breadth_positive.first().diff(); dispersion_change=x.groupby("decision_ts").universe_return_dispersion.first().diff()
        for lag in sorted(active_lead_lags):
            benchmark_lag=x.groupby("decision_ts").market_return_1.first().shift(lag)
            x[f"lagged_market_return_{lag}"]=x.decision_ts.map(benchmark_lag); x[f"lagged_breadth_change_{lag}"]=x.decision_ts.map(breadth_change.shift(lag)); x[f"lagged_dispersion_change_{lag}"]=x.decision_ts.map(dispersion_change.shift(lag)); x[f"unreacted_market_move_{lag}"]=x[f"lagged_market_return_{lag}"]-x[f"return_{lag}"]
    for base in ["return_1","relative_volume_10","realized_vol_10","range_position_10","distance_session_vwap"]:
        if base not in x: continue
        for window in [1560,4680]:
            if not ({f"{base}_z_{window}",f"{base}_mean_ratio_{window}",f"{base}_median_diff_{window}"}&wanted): continue
            prior=x.groupby("symbol",sort=False)[base].shift(1)
            mean=prior.groupby(x.symbol).rolling(window,min_periods=max(100,window//5)).mean().reset_index(level=0,drop=True); std=prior.groupby(x.symbol).rolling(window,min_periods=max(100,window//5)).std().reset_index(level=0,drop=True); median=prior.groupby(x.symbol).rolling(window,min_periods=max(100,window//5)).median().reset_index(level=0,drop=True); x[f"{base}_z_{window}"]=(x[base]-mean)/std.replace(0,np.nan)
            if base in {"relative_volume_10","realized_vol_10","range_position_10"}: x[f"{base}_mean_ratio_{window}"]=x[base]/mean.replace(0,np.nan)
            x[f"{base}_median_diff_{window}"]=x[base]-median
    requested={s.name:s for s in specs}; built=[]
    for name,spec in requested.items():
        if spec.status=="requested" and name in x: built.append(spec)
    identifiers=["symbol","session_date","bar_start_ts","bar_end_ts","available_at_ts","decision_ts","scheduled_open","scheduled_close","open","high","low","close","vwap","volume","open_raw","high_raw","low_raw","close_raw","vwap_raw","open_adjusted","high_adjusted","low_adjusted","close_adjusted","vwap_adjusted","split_factor","symbol_role","pit_member","scan_eligible","session_grid_eligible","analysis_eligible","benchmark_valid","shortened_session","bar_gap","gap_segment","membership_source_quality"]
    identifiers += [column for column in ["close_total_return_adjusted","sector","industry","market_cap"] if column in x]
    keep=identifiers+[s.name for s in built]
    # Projection is the memory boundary: temporary dependencies and unrelated
    # features never enter target construction or statistical scanning.
    result=x.loc[:,list(dict.fromkeys(keep))].copy()
    for column in [s.name for s in built]:
        if pd.api.types.is_float_dtype(result[column]): result[column]=result[column].astype(np.float32)
    return result, built


def _session_run_length(mask: pd.Series, symbols: pd.Series, sessions: pd.Series,gaps:pd.Series) -> pd.Series:
    blocks=(mask.ne(mask.shift())|symbols.ne(symbols.shift())|sessions.ne(sessions.shift())|gaps.ne(gaps.shift())).cumsum()
    return mask.astype(int).groupby(blocks).cumsum()
