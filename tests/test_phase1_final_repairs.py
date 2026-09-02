from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_pipeline.cache import CacheFingerprintMismatch,assert_key_alignment,validate_cache,write_cache_metadata
from quant_pipeline.config import ScanConfig
from quant_pipeline.features import build_features
from quant_pipeline.exact_parallel import _confirmation_gate,_walk_forward
from quant_pipeline.registry import FeatureSpec,target_registry
from quant_pipeline.bulk_scan import assert_valid_screen_results,cuda_screen
from quant_pipeline.scanner import _categorical_clustered_test,_cluster_covariance,_cross_sectional_ic,_outlier_diagnostics,_quantiles,_two_way_clustered_covariance,benjamini_hochberg
from quant_pipeline.table import _attach_adjusted_prices,add_targets,apply_analysis_eligibility
from quant_pipeline.run import _cluster_candidates


def panel_bars():
    rows=[]
    for symbol,role,pit,grid,mult in [("AAA","tradable",True,True,1),("BBB","tradable",True,False,50),("CCC","tradable",False,True,-20),("QQQ","benchmark",True,True,10)]:
        for i,start in enumerate(pd.date_range("2024-01-02 09:30",periods=3,freq="5min",tz="America/New_York").tz_convert("UTC")):
            close=100+mult*i
            rows.append({"symbol":symbol,"session_date":pd.Timestamp("2024-01-02"),"bar_start_ts":start,"bar_end_ts":start+pd.Timedelta(minutes=5),"available_at_ts":start+pd.Timedelta(minutes=5),"decision_ts":start+pd.Timedelta(minutes=5),"open":close,"high":close+1,"low":close-1,"close":close,"vwap":close,"volume":100,"symbol_role":role,"pit_member":pit,"scan_eligible":True,"session_grid_eligible":grid,"gap_segment":0})
    return apply_analysis_eligibility(pd.DataFrame(rows))


def test_universal_eligibility_controls_ranks_breadth_and_dispersion():
    bars=panel_bars(); specs=[FeatureSpec("return_rank_1","rank","cross_sectional",1,"cross_sectional"),FeatureSpec("universe_breadth_positive","breadth","breadth"),FeatureSpec("universe_return_dispersion","dispersion","breadth")]
    frame,_=build_features(bars,ScanConfig(lookbacks=[1]),specs)
    valid=frame.analysis_eligible
    assert set(frame.loc[valid,"symbol"])=={"AAA"}
    assert frame.loc[~valid,["return_rank_1","universe_breadth_positive","universe_return_dispersion"]].isna().all().all()
    assert frame.loc[valid,"return_rank_1"].dropna().eq(1).all()
    assert frame.loc[valid,"universe_breadth_positive"].dropna().eq(1).all()


def test_gap_resets_every_sequential_feature():
    bars=panel_bars().loc[lambda z:z.symbol.eq("AAA")].copy(); bars.loc[bars.index[-1],"gap_segment"]=1
    names=["intraday_return_1","higher_high","outside_bar","consecutive_positive_bars","vwap_cross","vwap_slope"]
    frame,_=build_features(bars,ScanConfig(lookbacks=[1]),[FeatureSpec(n,n,"test") for n in names],symbol_local=True); row=frame.iloc[-1]
    assert pd.isna(row.intraday_return_1) and pd.isna(row.higher_high) and pd.isna(row.outside_bar)
    assert row.consecutive_positive_bars==0 and pd.isna(row.vwap_cross) and pd.isna(row.vwap_slope)


def test_invalid_benchmark_invalidates_market_features_and_adjusted_targets():
    bars=panel_bars(); bad_time=bars.loc[bars.symbol.eq("QQQ"),"decision_ts"].iloc[-1]; bars.loc[bars.symbol.eq("QQQ")&bars.decision_ts.eq(bad_time),"session_grid_eligible"]=False
    bars=apply_analysis_eligibility(bars); specs=[FeatureSpec("market_return_1","market","market_context",1),FeatureSpec("stock_minus_market_return_1","relative","market_context",1)]
    frame,_=build_features(bars,ScanConfig(lookbacks=[1]),specs); affected=frame.symbol.eq("AAA")&frame.decision_ts.eq(bad_time)
    assert frame.loc[affected,["market_return_1","stock_minus_market_return_1"]].isna().all().all()
    targets=add_targets(frame,target_registry([5],[5]),"QQQ"); assert targets.loc[affected,"fwd_return_5m_benchmark_adjusted"].isna().all()


def test_two_way_covariance_matches_statsmodels():
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
    rng=np.random.default_rng(7); dates=np.repeat(np.arange(12),24); symbols=np.tile(np.repeat(np.arange(6),4),12); x=rng.normal(size=len(dates)); y=.4*x+rng.normal(size=12)[dates]+rng.normal(size=6)[symbols]+rng.normal(size=len(x)); X=sm.add_constant(x); fit=sm.OLS(y,X).fit(); residual=fit.resid
    ours=_two_way_clustered_covariance(X,residual,dates.astype(str),symbols.astype(str)); reference=cov_cluster_2groups(fit,dates,symbols)[0]
    assert np.allclose(ours,reference,rtol=1e-10,atol=1e-12)


def test_categorical_cluster_test_resists_duplicate_rows():
    rng=np.random.default_rng(2); base=pd.DataFrame({"session_date":np.repeat(pd.date_range("2024-01-01",periods=30),4),"symbol":np.tile(list("ABCD"),30),"category":np.tile([0,1,0,1],30),"target":rng.normal(size=120)})
    first=_categorical_clustered_test(base,"category","target")["raw_p"]; duplicated=pd.concat([base]*5,ignore_index=True); second=_categorical_clustered_test(duplicated,"category","target")["raw_p"]
    assert abs(first-second)<.02


def test_fast_categorical_cluster_test_matches_dense_design():
    rng=np.random.default_rng(7); n=240
    frame=pd.DataFrame({"session_date":np.repeat(pd.date_range("2024-01-01",periods=30),8),"symbol":np.tile(list("ABCDEFGH"),30),"category":np.tile([0,1,2,0,1,2,0,1],30),"target":rng.normal(size=n)})
    design=pd.get_dummies(frame.category,drop_first=True,dtype=float); X=np.column_stack([np.ones(n),design.to_numpy()]); y=frame.target.to_numpy(); beta=np.linalg.pinv(X.T@X)@(X.T@y); residual=y-X@beta
    date_cov=_cluster_covariance(X,residual,frame.session_date.astype(str).to_numpy()); two_cov=_two_way_clustered_covariance(X,residual,frame.session_date.astype(str).to_numpy(),frame.symbol.astype(str).to_numpy())
    def wald(cov):
        b=beta[1:]; v=cov[1:,1:]; return float(stats.chi2.sf(float(b@np.linalg.pinv(v)@b),len(b)))
    actual=_categorical_clustered_test(frame,"category","target")
    assert actual["raw_p"]==pytest.approx(wald(date_cov),rel=1e-10,abs=1e-12)
    assert actual["two_way_cluster_p"]==pytest.approx(wald(two_cov),rel=1e-10,abs=1e-12)


def test_screen_journal_survives_different_row_schemas(tmp_path):
    n=24; feature=pd.DataFrame({"session_date":np.repeat(pd.date_range("2021-01-01",periods=12),2),"symbol":["A","B"]*12,"decision_ts":pd.date_range("2021-01-01",periods=n,freq="h"),"constant":1.0,"variable":np.arange(n,dtype=float)});target=pd.DataFrame({"y":np.random.default_rng(9).normal(size=n)})
    cfg=ScanConfig(use_cuda=False,min_observations=2,min_sessions=2,min_symbols=1,min_decision_timestamps=2,min_years=1);journal=tmp_path/"screen.jsonl"
    prior=cuda_screen(feature,target,[FeatureSpec("constant","constant","test")],["y"],cfg,pd.DataFrame(),journal)
    cuda_screen(feature,target,[FeatureSpec("variable","variable","test")],["y"],cfg,prior,journal)
    persisted=pd.read_json(journal,lines=True).set_index("feature")
    assert persisted.loc["constant","status"]=="constant_feature"
    assert 0<=persisted.loc["variable","raw_p"]<=1


def test_invalid_probabilities_fail_before_promotion():
    bad=pd.DataFrame({"feature":["x"],"target":["y"],"raw_p":[-0.1]})
    with pytest.raises(RuntimeError,match="Invalid probability"):assert_valid_screen_results(bad,"test")
    with pytest.raises(ValueError,match="\[0, 1\]"):benjamini_hochberg(bad.raw_p)


def test_cache_reordering_and_key_mismatch_are_rejected(tmp_path):
    frame=panel_bars().sort_values(["symbol","session_date","bar_start_ts","decision_ts"]).reset_index(drop=True); path=tmp_path/"block.parquet"; frame.to_parquet(path,index=False); write_cache_metadata(path,frame,"abc"); validate_cache(path,"abc")
    reordered=frame.sample(frac=1,random_state=1).reset_index(drop=True); reordered.to_parquet(path,index=False)
    with pytest.raises(CacheFingerprintMismatch):validate_cache(path,"abc")
    with pytest.raises(CacheFingerprintMismatch):assert_key_alignment(frame,reordered)


def test_cash_dividend_does_not_create_cross_session_return(tmp_path):
    bars=panel_bars().loc[lambda z:z.symbol.eq("AAA")].iloc[:1].copy(); second=bars.copy(); second["session_date"]=pd.Timestamp("2024-01-03"); second[["bar_start_ts","bar_end_ts","available_at_ts","decision_ts"]]+=pd.Timedelta(days=1); second[["open","high","low","close","vwap"]]-=1; bars=pd.concat([bars,second],ignore_index=True)
    ledger=pd.DataFrame({"symbol":["AAA"],"ex_date":[pd.Timestamp("2024-01-03")],"split_ratio":[np.nan],"cash_amount":[1.0]}); path=tmp_path/"actions.parquet"; ledger.to_parquet(path,index=False); adjusted=_attach_adjusted_prices(bars,ScanConfig(corporate_actions_path=str(path)))
    frame,_=build_features(adjusted,ScanConfig(lookbacks=[]),[FeatureSpec("continuous_return_1","continuous","momentum")],symbol_local=True); first=frame.loc[frame.session_date.eq(pd.Timestamp("2024-01-03"))].iloc[0]
    assert abs(first.continuous_return_1)<1e-8


def test_session_quantile_ci_is_stable_to_within_day_duplicates():
    rng=np.random.default_rng(3); sessions=pd.Series(np.repeat(pd.date_range("2024-01-01",periods=40),20)); x=pd.Series(rng.normal(size=800)); y=pd.Series(.01*x+rng.normal(size=800)); first,_=_quantiles(x,y,sessions,5,5,100)
    second,_=_quantiles(pd.concat([x,x],ignore_index=True),pd.concat([y,y],ignore_index=True),pd.concat([sessions,sessions],ignore_index=True),5,5,100)
    assert abs(first["daily_spread_hac_se"]-second["daily_spread_hac_se"])<1e-10


def test_signed_robustness_is_symmetric():
    rng=np.random.default_rng(4); frame=pd.DataFrame({"session_date":np.repeat(pd.date_range("2024-01-01",periods=30),20),"symbol":np.tile([f"S{i}" for i in range(20)],30),"x":rng.normal(size=600)}); frame["positive"]=.01*frame.x+rng.normal(0,.01,600); frame["negative"]=-frame.positive
    positive=_outlier_diagnostics(frame,"x","positive",1); negative=_outlier_diagnostics(frame,"x","negative",-1)
    assert positive["outlier_worst_signed_spread"]==pytest.approx(negative["outlier_worst_signed_spread"])


def test_cross_sectional_ic_requires_minimum_symbols():
    frame=pd.DataFrame({"decision_ts":pd.date_range("2024-01-01",periods=10,freq="h").repeat(3),"x":np.arange(30),"y":np.arange(30)})
    result=_cross_sectional_ic(frame,"x","y",5); assert np.isnan(result["ic_mean"])


@pytest.mark.parametrize("sessions,symbols",[(99,30),(120,19)])
def test_confirmation_rejects_negligible_or_undercovered_effects(sessions,symbols):
    cfg=ScanConfig(); confirmation={"top_bottom_spread":1e-8,"valid_sessions":sessions,"valid_symbols":symbols,"session_bootstrap_ci_low":1e-9,"session_bootstrap_ci_high":2e-8,"two_way_cluster_p":.01,"outlier_worst_signed_spread":1e-9}
    result=_confirmation_gate(.001,confirmation,1,cfg)
    assert result["confirmation_status"]=="directionally_replicated"


def test_beta_at_decision_uses_prior_sessions_only():
    rows=[]; qqq=100.0; stock=100.0
    for day_index,day in enumerate(pd.bdate_range("2023-01-02",periods=70)):
        qqq*=1+.001*np.sin(day_index/3); stock*=1+.002*np.sin(day_index/3)
        for symbol,price in [("QQQ",qqq),("AAA",stock)]:
            for bar in range(2):
                start=(pd.Timestamp(day.date()).tz_localize("America/New_York")+pd.Timedelta(hours=9,minutes=30+5*bar)).tz_convert("UTC")
                rows.append({"symbol":symbol,"session_date":pd.Timestamp(day),"bar_start_ts":start,"bar_end_ts":start+pd.Timedelta(minutes=5),"available_at_ts":start+pd.Timedelta(minutes=5),"decision_ts":start+pd.Timedelta(minutes=5),"open_raw":price,"close_raw":price,"close_total_return_adjusted":price,"symbol_role":"benchmark" if symbol=="QQQ" else "tradable","pit_member":True,"scan_eligible":True,"session_grid_eligible":True,"analysis_eligible":symbol=="AAA","benchmark_valid":symbol=="QQQ"})
    frame=pd.DataFrame(rows); specs=target_registry([5],[5]); cfg=ScanConfig(beta_window_sessions=40,beta_min_observations=30); first=add_targets(frame,specs,"QQQ",cfg); target_day=pd.Timestamp(pd.bdate_range("2023-01-02",periods=70)[-1]); before=first.loc[first.symbol.eq("AAA")&first.session_date.eq(target_day),"beta_at_decision"].iloc[0]
    frame.loc[frame.symbol.eq("AAA")&frame.session_date.eq(target_day),"close_total_return_adjusted"]*=10; second=add_targets(frame,specs,"QQQ",cfg); after=second.loc[second.symbol.eq("AAA")&second.session_date.eq(target_day),"beta_at_decision"].iloc[0]
    assert np.isfinite(before) and before==pytest.approx(after)


def test_historical_subperiod_diagnostic_covers_recent_years_without_oos_claim():
    rows=[]
    for year in range(2021,2027):
        for i in range(120):rows.append({"session_date":pd.Timestamp(year=year,month=1,day=2),"x":i,"y":i/1000})
    result=_walk_forward(pd.DataFrame(rows),"x","y",1)
    assert result["historical_subperiod_folds"]==5
    assert result["diagnostic_evidence_label"]=="historical_subperiod_stability"


def test_redundant_candidates_are_capped_and_targets_are_tiered():
    frame=pd.DataFrame({"feature":[f"return_{i}" for i in range(1,7)],"target":["fwd_return_5m"]*6,"raw_p":np.linspace(.001,.006,6),"primary_global_fdr":np.linspace(.01,.06,6),"anomaly_score":np.linspace(1,.5,6),"feature_family":["momentum"]*6,"redundancy_group":["return_LOOKBACK"]*6,"monotonicity":[1.0]*6,"spearman":[.1]*6})
    promoted=_cluster_candidates(frame,None,ScanConfig(max_candidates_per_cluster=3),250)
    assert len(promoted)<=3
    specs=target_registry([5,10,15],[5,15]); tiers={s.name:s.tier for s in specs}
    assert tiers["fwd_return_5m"]=="primary" and tiers["fwd_return_10m"]=="exploratory"
