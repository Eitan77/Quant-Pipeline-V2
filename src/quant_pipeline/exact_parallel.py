from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScanConfig
from .diagnostics import exact_time_diagnostics,recent_period_diagnostics,recency_weighted_diagnostics,regime_diagnostics,scope_diagnostics,symbol_and_concentration_diagnostics
from .holdout import assert_pre_holdout_frame
from .registry import FeatureSpec
from .scanner import scan


def exact_pair(
    feature_path: Path,
    target_path: Path,
    spec: FeatureSpec,
    target: str,
    config: ScanConfig,
    direction_hint: float | None = None,
    diagnostic_context_path: Path | None = None,
) -> tuple[dict | None, list[dict], dict[str,list[dict]]]:
    import pyarrow.parquet as pq
    identifiers=["symbol","session_date","bar_start_ts","decision_ts","analysis_eligible"]
    feature_schema=set(pq.ParquetFile(feature_path).schema.names); optional=[c for c in ["close_raw","volume","sector","industry","market_cap"] if c in feature_schema]
    feature_frame=pd.read_parquet(feature_path,columns=[*identifiers,*optional,spec.name])
    target_schema=set(pq.ParquetFile(target_path).schema.names); raw=target.removesuffix("_benchmark_adjusted").removesuffix("_beta_residual"); adjusted=f"{raw}_benchmark_adjusted"; target_columns=["symbol","session_date","bar_start_ts","decision_ts",target]
    for column in [raw,adjusted]:
        if column in target_schema and column not in target_columns:target_columns.append(column)
    target_frame=pd.read_parquet(target_path,columns=target_columns)
    frame=feature_frame.merge(target_frame,on=["symbol","session_date","bar_start_ts","decision_ts"],how="inner",validate="one_to_one")
    if len(frame)!=len(feature_frame):raise ValueError("Feature-target cache key mismatch")
    if raw in frame and adjusted in frame:frame["market_context_return"]=frame[raw]-frame[adjusted]
    if diagnostic_context_path is not None and diagnostic_context_path.exists():
        context=pd.read_parquet(diagnostic_context_path); assert_pre_holdout_frame(context,config.sealed_holdout_start,"diagnostic context")
        context=context[["decision_ts",*[column for column in context.columns if column!="decision_ts" and column not in frame]]]
        frame=frame.merge(context,on="decision_ts",how="left",validate="many_to_one")
    assert_pre_holdout_frame(frame,config.sealed_holdout_start,"exact pair")
    eligible=frame.analysis_eligible.fillna(False)
    dates=pd.to_datetime(frame.session_date); full=frame.loc[eligible&dates.le(pd.Timestamp(config.discovery_end))]
    if config.use_separate_confirmation_period:
        if not config.selection_end or not config.confirmation_start:raise ValueError("Separate confirmation mode requires selection_end and confirmation_start")
        discovery=full.loc[pd.to_datetime(full.session_date).le(pd.Timestamp(config.selection_end))]
        confirmation=full.loc[pd.to_datetime(full.session_date).ge(pd.Timestamp(config.confirmation_start))]
    else:
        discovery=full; confirmation=full.iloc[0:0]
    exact,tables=scan(discovery,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint)
    row=exact.iloc[0].to_dict() if not exact.empty else None
    confirmed,_=scan(confirmation,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint) if config.use_separate_confirmation_period else (pd.DataFrame(),{})
    diagnostic_tables={}
    if row is not None:
        table=tables.get((spec.name,target),pd.DataFrame()).to_dict("records")
        if config.use_separate_confirmation_period and not confirmed.empty:
            c=confirmed.iloc[0]
            for column in ["n","valid_sessions","valid_symbols","spearman","top_bottom_spread","two_way_cluster_t","two_way_cluster_p","session_bootstrap_ci_low","session_bootstrap_ci_high","symbol_breadth","time_stability","outlier_worst_signed_spread"]:
                if column in c:row[f"confirmation_{column}"]=c[column]
            expected=np.sign(direction_hint or row.get("spearman",0)); row.update(_confirmation_gate(float(row.get("top_bottom_spread",np.nan)),c.to_dict(),expected,config))
        elif config.use_separate_confirmation_period:row.update(_confirmation_gate(float(row.get("top_bottom_spread",np.nan)),{},np.sign(direction_hint or row.get("spearman",0)),config))
        direction=direction_hint or row.get("spearman",0)
        # The legacy descriptive routines are decile-based. Binary candidates
        # use on/off diagnostics instead of being coerced into artificial
        # deciles.
        if spec.dtype == "binary":
            direction=_binary_direction(row,direction_hint)
            if config.run_historical_walk_forward_diagnostics:row.update(_binary_historical_subperiod_diagnostics(full,spec.name,target,direction))
            if config.run_recent_period_diagnostics:
                summary,recent=_binary_recent_period_diagnostics(full,spec.name,target,direction,config);row.update(summary);diagnostic_tables["recent_periods"]=recent.to_dict("records")
            summary,tables=_binary_symbol_and_concentration_diagnostics(full,spec.name,target,direction,config);row.update(summary);diagnostic_tables.update({name:value.to_dict("records") for name,value in tables.items()})
            summary,exact_time=_binary_exact_time_diagnostics(full,spec.name,target,direction,config);row.update(summary);diagnostic_tables["exact_time"]=exact_time.to_dict("records")
            row.update({"regime_summary_label":"insufficient_regime_evidence","volatility_regime_status":"unavailable","breadth_regime_status":"unavailable","dispersion_regime_status":"unavailable","trend_regime_status":"unavailable","gap_regime_status":"unavailable","sector_scope_status":"unavailable_missing_point_in_time_sector_data","industry_scope_status":"unavailable_missing_point_in_time_industry_data","scope_classification":"insufficient_scope_evidence","binary_diagnostics_status":"on_off_effect_with_robustness_diagnostics"})
            return row,table,diagnostic_tables
        if config.run_historical_walk_forward_diagnostics:row.update(_historical_subperiod_diagnostics(full,spec.name,target,direction))
        if config.run_recent_period_diagnostics:
            summary,table=recent_period_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["recent_periods"]=table.to_dict("records")
        if config.run_recency_weighted_diagnostics:
            summary,table=recency_weighted_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["recency_weighted"]=table.to_dict("records")
        summary,diagnostics=symbol_and_concentration_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables.update({name:table.to_dict("records") for name,table in diagnostics.items()})
        summary,table=regime_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["regime"]=table.to_dict("records")
        summary,tables=scope_diagnostics(full,spec.name,target,direction,row.get("symbol_breadth_classification","insufficient_evidence"),config); row.update(summary); diagnostic_tables.update({name:table.to_dict("records") for name,table in tables.items()})
        summary,table=exact_time_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["exact_time"]=table.to_dict("records")
    table=tables.get((spec.name,target),pd.DataFrame()).to_dict("records")
    return row,table,diagnostic_tables


def _binary_direction(row:dict,direction_hint:float|None)->int:
    hint=pd.to_numeric(direction_hint,errors="coerce")
    if not np.isfinite(hint) or hint==0:hint=pd.to_numeric(row.get("top_bottom_spread",row.get("effect_value")),errors="coerce")
    return 1 if not np.isfinite(hint) or hint>=0 else -1


def _binary_effect(frame:pd.DataFrame,feature:str,target:str,direction:int)->float:
    z=frame[[feature,target]].replace([np.inf,-np.inf],np.nan).dropna();on=z[feature].astype(float).gt(.5)
    if not on.any() or on.all():return np.nan
    return float((z.loc[on,target].mean()-z.loc[~on,target].mean())*direction)


def _binary_historical_subperiod_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:int)->dict:
    years=pd.to_datetime(frame.session_date).dt.year;effects=[]
    for year in (2022,2023,2024,2025,2026):
        sub=frame.loc[years.eq(year)]
        if year==2026:sub=sub.loc[pd.to_datetime(sub.session_date).le(pd.Timestamp("2026-04-30"))]
        effect=_binary_effect(sub,feature,target,direction)
        if np.isfinite(effect):effects.append(effect)
    return {"diagnostic_evidence_label":"historical_subperiod_stability","historical_subperiod_folds":len(effects),"historical_subperiod_mean_signed_effect":float(np.mean(effects)) if effects else np.nan,"historical_subperiod_positive_fold_pct":float(np.mean(np.asarray(effects)>0)) if effects else np.nan,"historical_subperiod_worst_signed_effect":float(np.min(effects)) if effects else np.nan}


def _binary_recent_period_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:int,config:ScanConfig)->tuple[dict,pd.DataFrame]:
    dates=pd.to_datetime(frame.session_date);end=pd.Timestamp(config.discovery_end);starts={"full_discovery":pd.Timestamp(config.start),"recent_5y":end-pd.DateOffset(years=5),"recent_3y":end-pd.DateOffset(years=3),"recent_2y":end-pd.DateOffset(years=2),"recent_12m":end-pd.DateOffset(months=12),"jan_apr_2026":pd.Timestamp("2026-01-01")};rows=[]
    for label,start in starts.items():
        sub=frame.loc[dates.ge(start)&dates.le(end)];rows.append({"period":label,"start":str(max(start,pd.Timestamp(config.start)).date()),"end":str(end.date()),"effect":_binary_effect(sub,feature,target,direction),"observations":len(sub),"sessions":sub.session_date.nunique(),"symbols":sub.symbol.nunique()})
    table=pd.DataFrame(rows);effects=table.set_index("period").effect;full=effects.get("full_discovery",np.nan);recent=effects.get("recent_12m",np.nan);ratio=recent/full if np.isfinite(full) and full else np.nan;recent_row=table.loc[table.period.eq("recent_12m")].iloc[0];enough=recent_row.sessions>=min(config.min_sessions,100) and recent_row.symbols>=min(config.min_symbols,20)
    if not enough:classification="insufficient_recent_data"
    elif not np.isfinite(recent) or recent<=0:classification="historically_strong_but_currently_weak" if np.isfinite(full) and full>0 else "regime_unstable"
    elif np.isfinite(ratio) and ratio>=1.5:classification="strengthening_recently"
    elif np.isfinite(ratio) and ratio<.5:classification="weakening_recently"
    else:classification="persistent"
    summary={"recent_classification":classification,"recent_to_full_effect_ratio":ratio,"recent_12m_effect":recent,**{f"{name}_effect":effects.get(name,np.nan) for name in ("recent_5y","recent_3y","recent_2y","jan_apr_2026")}}
    return summary,table


def _binary_symbol_and_concentration_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:int,config:ScanConfig)->tuple[dict,dict[str,pd.DataFrame]]:
    z=frame[["symbol","session_date","decision_ts",feature,target]].replace([np.inf,-np.inf],np.nan).dropna();rows=[]
    for symbol,group in z.groupby("symbol",sort=False):
        effect=_binary_effect(group,feature,target,direction);rows.append({"symbol":symbol,"valid_observations":len(group),"unique_sessions":group.session_date.nunique(),"signed_on_minus_off_effect":effect,"direction_matches":bool(np.isfinite(effect) and effect>0),"passes_minimum_sample":group.session_date.nunique()>=50})
    symbols=pd.DataFrame(rows)
    if symbols.empty:return {"symbol_breadth_classification":"insufficient_evidence"},{"symbol":symbols}
    contribution=(symbols.signed_on_minus_off_effect.fillna(0)*symbols.valid_observations).abs();symbols["effect_contribution_pct"]=contribution/contribution.sum() if contribution.sum() else 0;ordered=symbols.sort_values("effect_contribution_pct",ascending=False);shares=ordered.effect_contribution_pct;reliable=symbols.loc[symbols.passes_minimum_sample]
    def removed(k:int)->float:return _binary_effect(z.loc[~z.symbol.isin(set(ordered.head(k).symbol))],feature,target,direction)
    expected=float(reliable.direction_matches.mean()) if len(reliable) else np.nan;best=float(shares.head(1).sum());top3=float(shares.head(3).sum());hhi=float((shares**2).sum())
    if len(reliable)<3:breadth="insufficient_evidence"
    elif best>=.60:breadth="single_symbol_dominated"
    elif top3>=.75 or hhi>=.25:breadth="highly_concentrated"
    elif top3>=.50 or expected<.60:breadth="moderately_concentrated"
    else:breadth="broad_across_symbols"
    summary={"best_symbol_effect_pct":best,"top3_symbol_effect_pct":top3,"top5_symbol_effect_pct":float(shares.head(5).sum()),"symbol_effect_hhi":hhi,"effect_remove_best_symbol":removed(1),"effect_remove_top3_symbols":removed(3),"effect_remove_top5_symbols":removed(5),"eligible_symbols_expected_direction_pct":expected,"eligible_symbols_meaningful_effect_pct":float((reliable.signed_on_minus_off_effect>=config.minimum_effect_bps/10000).mean()) if len(reliable) else np.nan,"symbol_breadth_classification":breadth}
    local=pd.to_datetime(z.decision_ts,utc=True).dt.tz_convert("America/New_York");z["time_bucket"]=pd.cut(local.dt.hour*60+local.dt.minute,[0,630,720,900,1440],labels=["open","morning","midday","close"]);time=z.groupby("time_bucket",observed=True).apply(lambda q:pd.Series({"observations":len(q),"signed_effect":_binary_effect(q,feature,target,direction)}),include_groups=False).reset_index();summary["time_concentration_classification"]=(str(time.loc[time.signed_effect.idxmax(),"time_bucket"])+"_concentrated") if len(time) else "insufficient_evidence";summary["phase2_recommendation_seed"]="advance_as_broad_cross_sectional_candidate" if breadth=="broad_across_symbols" else "advance_for_conditional_phase2_testing" if breadth in {"moderately_concentrated","highly_concentrated"} else "retain_for_monitoring_only"
    return summary,{"symbol":symbols,"time_of_day":time}


def _binary_exact_time_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:int,config:ScanConfig)->tuple[dict,pd.DataFrame]:
    z=frame[["symbol","session_date","decision_ts",feature,target]].replace([np.inf,-np.inf],np.nan).dropna().copy();z["decision_time"]=pd.to_datetime(z.decision_ts,utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M");rows=[]
    for decision_time,group in z.groupby("decision_time",sort=True):
        observations=len(group);sessions=group.session_date.nunique();symbols=group.symbol.nunique();enough=observations>=config.exact_time_min_observations and sessions>=config.exact_time_min_sessions and symbols>=config.exact_time_min_symbols;rows.append({"candidate_id":f"{feature}__{target}","feature":feature,"target":target,"expected_direction":direction,"decision_time":decision_time,"valid_observations":observations,"unique_sessions":sessions,"unique_symbols":symbols,"signed_on_minus_off_effect":_binary_effect(group,feature,target,direction) if enough else np.nan,"minimum_sample_status":"sufficient" if enough else "insufficient_data"})
    table=pd.DataFrame(rows);sufficient=table.loc[table.minimum_sample_status.eq("sufficient")].dropna(subset=["signed_on_minus_off_effect"])
    if sufficient.empty:return {"strongest_exact_decision_time":None,"weakest_exact_decision_time":None,"effect_after_removing_strongest_exact_time":np.nan,"time_concentration_label":"insufficient_time_evidence"},table
    strongest=sufficient.loc[sufficient.signed_on_minus_off_effect.idxmax()];weakest=sufficient.loc[sufficient.signed_on_minus_off_effect.idxmin()];after=_binary_effect(z.loc[z.decision_time.ne(strongest.decision_time)],feature,target,direction);weights=sufficient.signed_on_minus_off_effect.clip(lower=0)*sufficient.valid_observations;top_share=float((weights/weights.sum()).max()) if weights.sum() else np.nan;preserving=float(sufficient.signed_on_minus_off_effect.gt(0).mean())
    label="insufficient_time_evidence" if len(sufficient)<3 else "time_unstable" if preserving<.60 else "persistent_through_session" if not np.isfinite(top_share) or top_share<.60 else "opening_only"
    return {"strongest_exact_decision_time":strongest.decision_time,"weakest_exact_decision_time":weakest.decision_time,"effect_after_removing_strongest_exact_time":after,"time_concentration_label":label},table


def _confirmation_gate(discovery_effect:float,confirmation:dict,expected:float,config:ScanConfig)->dict:
    effect=float(confirmation.get("top_bottom_spread",np.nan)); ratio=abs(effect/discovery_effect) if np.isfinite(discovery_effect) and discovery_effect else np.nan; direction_ok=np.isfinite(effect) and np.sign(effect)==expected
    economic=direction_ok and abs(effect)>=config.confirmation_min_effect_bps/10000; ratio_ok=np.isfinite(ratio) and config.confirmation_min_discovery_ratio<=ratio<=config.confirmation_max_discovery_ratio; coverage=confirmation.get("valid_sessions",0)>=config.confirmation_min_sessions and confirmation.get("valid_symbols",0)>=config.confirmation_min_symbols
    bootstrap=confirmation.get("session_bootstrap_ci_low",np.nan)>0 if expected>0 else confirmation.get("session_bootstrap_ci_high",np.nan)<0; statistical=confirmation.get("two_way_cluster_p",1)<config.confirmation_alpha and bootstrap; robust=confirmation.get("outlier_worst_signed_spread",-np.inf)>0
    status="not_replicated"
    if direction_ok:status="directionally_replicated"
    if direction_ok and economic and ratio_ok:status="economically_replicated"
    if direction_ok and economic and ratio_ok and coverage and statistical:status="statistically_confirmed"
    return {"discovery_effect":discovery_effect,"confirmation_effect":effect,"confirmation_to_discovery_ratio":ratio,"confirmation_clustered_t":confirmation.get("two_way_cluster_t"),"confirmation_p_value":confirmation.get("two_way_cluster_p"),"confirmation_bootstrap_lower":confirmation.get("session_bootstrap_ci_low"),"confirmation_bootstrap_upper":confirmation.get("session_bootstrap_ci_high"),"confirmation_sessions":confirmation.get("valid_sessions"),"confirmation_symbols":confirmation.get("valid_symbols"),"internal_confirmation_direction_match":bool(direction_ok),"internal_confirmation_effect_ratio":ratio,"confirmation_status":status,"confirmation_robust":bool(robust)}


def _historical_subperiod_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:float)->dict:
    years=pd.to_datetime(frame.session_date).dt.year; folds=[]
    for test_year in [2022,2023,2024,2025,2026]:
        train=frame.loc[years.lt(test_year),feature].dropna()
        test=frame.loc[years.eq(test_year)]
        if test_year==2026:test=test.loc[pd.to_datetime(test.session_date).le(pd.Timestamp("2026-04-30"))]
        if len(test)<100 or len(train)<100:continue
        edges=np.unique(train.quantile(np.linspace(.1,.9,9)).to_numpy())
        if len(edges)<2:continue
        bins=pd.Series(np.searchsorted(edges,test[feature].to_numpy(),side="right"),index=test.index)
        means=test.groupby(bins,observed=True)[target].mean(); spread=float(means.iloc[-1]-means.iloc[0]) if len(means)>1 else np.nan
        folds.append({"year":test_year,"spread":spread})
    signed=[x["spread"]*np.sign(direction) for x in folds if np.isfinite(x["spread"])]
    return {"diagnostic_evidence_label":"historical_subperiod_stability","historical_subperiod_folds":len(signed),"historical_subperiod_mean_signed_effect":float(np.mean(signed)) if signed else np.nan,"historical_subperiod_positive_fold_pct":float(np.mean(np.asarray(signed)>0)) if signed else np.nan,"historical_subperiod_worst_signed_effect":float(np.min(signed)) if signed else np.nan}


def _walk_forward(frame:pd.DataFrame,feature:str,target:str,direction:float)->dict:
    """Backward-compatible name; results are explicitly not OOS evidence."""
    return _historical_subperiod_diagnostics(frame,feature,target,direction)
