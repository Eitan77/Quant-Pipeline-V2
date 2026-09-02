from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.bulk_scan import build_cuda_feature_context,cuda_screen
from quant_pipeline.config import ScanConfig
from quant_pipeline.features import build_features
from quant_pipeline.fingerprint import run_fingerprint
from quant_pipeline.registry import FeatureSpec,feature_registry,target_registry
from quant_pipeline.scanner import scan
from quant_pipeline.table import _attach_adjusted_prices,_attach_calendar,_attach_membership,add_targets,filter_decision_rows,validate_point_in_time


def bars_for_days(days=("2024-01-02","2024-01-03"),symbols=("AAA",),bars=4):
    rows=[]
    for symbol in symbols:
        for day_index,day in enumerate(days):
            starts=pd.date_range(f"{day} 09:30",periods=bars,freq="5min",tz="America/New_York").tz_convert("UTC")
            for i,start in enumerate(starts):
                close=(100 if day_index==0 else 200)+i
                rows.append({"symbol":symbol,"session_date":pd.Timestamp(day),"bar_start_ts":start,"bar_end_ts":start+pd.Timedelta(minutes=5),"available_at_ts":start+pd.Timedelta(minutes=5),"open":close-.25,"high":close+.5,"low":close-.5,"close":close,"vwap":close,"volume":100+i})
    return pd.DataFrame(rows)


def build(names,bars=None):
    specs=[s for s in feature_registry([1,2,3]) if s.name in set(names)]
    return build_features(bars if bars is not None else bars_for_days(),ScanConfig(lookbacks=[1,2,3],opening_windows_minutes=[5,15]),specs,symbol_local=True)[0]


def test_intraday_and_sequence_features_reset_between_sessions():
    frame=build(["intraday_return_1","continuous_return_1","realized_vol_2","return_since_open","higher_high","outside_bar","consecutive_positive_bars","vwap_cross","vwap_slope"])
    second=frame.loc[frame.session_date.eq(pd.Timestamp("2024-01-03"))].reset_index(drop=True)
    assert pd.isna(second.loc[0,"intraday_return_1"])
    assert second.loc[0,"continuous_return_1"]>.9
    assert pd.isna(second.loc[0,"realized_vol_2"])
    assert second.loc[0,"return_since_open"]==pytest.approx(200/199.75-1)
    assert pd.isna(second.loc[0,"higher_high"]) and pd.isna(second.loc[0,"outside_bar"])
    assert second.loc[0,"consecutive_positive_bars"]==0
    assert pd.isna(second.loc[0,"vwap_cross"]) and pd.isna(second.loc[0,"vwap_slope"])
    assert second.loc[1,"intraday_return_1"]==pytest.approx(201/200-1)


def test_split_adjustment_removes_false_return_but_targets_remain_raw(tmp_path):
    source=bars_for_days(bars=2); price_columns=["open","high","low","close","vwap"]; source[price_columns]=source[price_columns].astype(float); source.loc[source.session_date.eq(pd.Timestamp("2024-01-02")),price_columns]*=2; source.loc[source.session_date.eq(pd.Timestamp("2024-01-03")),price_columns]/=2
    actions=pd.DataFrame({"symbol":["AAA"],"ex_date":[pd.Timestamp("2024-01-03")],"split_ratio":[2.0]}); path=tmp_path/"actions.parquet"; actions.to_parquet(path,index=False)
    adjusted=_attach_adjusted_prices(source,ScanConfig(corporate_actions_path=str(path)))
    frame=build(["continuous_return_1"],adjusted)
    first_post=frame.loc[frame.session_date.eq(pd.Timestamp("2024-01-03"))].iloc[0]
    assert abs(first_post.continuous_return_1)<.02
    targets=add_targets(adjusted,target_registry([5],[5]))
    assert targets.entry_open_raw.dropna().iloc[0]==adjusted.open_raw.iloc[1]


def test_membership_and_benchmark_roles_are_point_in_time():
    bars=bars_for_days(symbols=("AAA","BBB","QQQ"),bars=1)
    membership=pd.DataFrame({"session_date":[pd.Timestamp("2024-01-02"),pd.Timestamp("2024-01-03"),pd.Timestamp("2024-01-03")],"symbol":["AAA","AAA","BBB"],"is_member":[True,True,True],"source_ingestion_id":["x","x","x"],"ingested_at":["2024-01-01"]*3})
    out=_attach_membership(bars,membership,ScanConfig())
    assert not ((out.symbol.eq("BBB"))&out.session_date.eq(pd.Timestamp("2024-01-02"))).any()
    assert out.loc[out.symbol.eq("QQQ"),"symbol_role"].eq("benchmark").all()
    assert out.loc[out.symbol.eq("AAA"),"symbol_role"].eq("tradable").all()


def test_benchmark_is_excluded_from_cross_sectional_rank_and_breadth():
    bars=bars_for_days(days=("2024-01-02",),symbols=("AAA","BBB","QQQ"),bars=3)
    bars.loc[bars.symbol.eq("QQQ"),"close"]*=10
    bars["symbol_role"]=np.where(bars.symbol.eq("QQQ"),"benchmark","tradable"); bars["scan_eligible"]=bars.symbol.ne("QQQ")
    frame=build_features(bars,ScanConfig(lookbacks=[1]),[FeatureSpec("return_rank_1","rank","cross_sectional",1,"cross_sectional"),FeatureSpec("universe_breadth_positive","breadth","breadth")])[0]
    assert frame.loc[frame.symbol.eq("QQQ"),"return_rank_1"].isna().all()
    decision=frame.decision_ts.iloc[-1]; expected=(frame.loc[frame.decision_ts.eq(decision)&frame.symbol.ne("QQQ"),"intraday_return_1"]>0).mean() if "intraday_return_1" in frame else 1.0
    assert frame.loc[frame.decision_ts.eq(decision),"universe_breadth_positive"].dropna().eq(expected).all()


def test_missing_open_and_gap_invalidate_opening_and_rolling_windows():
    bars=bars_for_days(days=("2024-01-02",),bars=5).drop(index=[0,2]).reset_index(drop=True)
    bars["gap_segment"]=[0,1,1]
    frame=build(["opening_return_15m","return_2"],bars)
    assert frame.opening_return_15m.isna().all()
    assert pd.isna(frame.return_2.iloc[1])


def test_decision_time_is_actionable_and_filtering_happens_after_build():
    bars=bars_for_days(days=("2024-01-02",),bars=8)
    frame=build(["return_3","minutes_since_open","opening_return_15m"],bars)
    filtered=filter_decision_rows(frame,ScanConfig(decision_times_et=["10:00"]))
    assert len(filtered)==1 and filtered.return_3.notna().all() and filtered.opening_return_15m.notna().all()
    assert filtered.minutes_since_open.iloc[0]==30


def test_exchange_calendar_handles_early_close_and_holiday():
    starts=pd.to_datetime(["2024-07-03 13:30:00+00:00","2024-07-04 13:30:00+00:00"])
    bars=pd.DataFrame({"symbol":"AAA","session_date":pd.to_datetime(["2024-07-03","2024-07-04"]),"bar_start_ts":starts,"bar_end_ts":starts+pd.Timedelta(minutes=5),"available_at_ts":starts+pd.Timedelta(minutes=5),"open":100,"high":101,"low":99,"close":100,"vwap":100,"volume":100})
    out=_attach_calendar(bars,ScanConfig(start="2024-07-03",discovery_end="2024-07-04",maximum_missing_bars_per_session=100))
    assert out.session_date.nunique()==1 and out.is_shortened_session.all() and out.session_length_minutes.eq(210).all()
    frame=build(["minutes_until_close"],out); assert frame.minutes_until_close.iloc[0]==205


def test_delayed_availability_selects_first_actionable_bar_and_stays_intraday():
    bars=bars_for_days(days=("2024-01-02",),bars=5); bars.loc[0,"available_at_ts"]=bars.loc[0,"bar_start_ts"]+pd.Timedelta(minutes=12)
    out=add_targets(bars,target_registry([5],[5])); assert out.loc[0,"entry_ts"]==bars.loc[3,"bar_start_ts"]
    assert out.filter(like="fwd_return_5m").iloc[-1].isna().all()


def test_pairwise_coverage_is_computed_after_missing_values(tmp_path):
    feature=pd.DataFrame({"session_date":np.repeat(pd.date_range("2021-01-01",periods=6),2),"symbol":["A","B"]*6,"decision_ts":pd.date_range("2021-01-01",periods=12,freq="h"),"x":np.arange(12,dtype=float)})
    target=pd.DataFrame({"y":np.r_[np.arange(6,dtype=float),np.full(6,np.nan)]})
    cfg=ScanConfig(use_cuda=False,min_observations=2,min_sessions=2,min_symbols=1,min_decision_timestamps=2,min_years=1)
    result=cuda_screen(feature,target,[FeatureSpec("x","x","test")],["y"],cfg,pd.DataFrame(),tmp_path/"j.csv").iloc[0]
    assert result.valid_observations==6 and result.valid_sessions==3 and result.valid_symbols==2 and result.valid_decision_timestamps==6


def test_reused_feature_context_matches_standalone_screen(tmp_path):
    feature=pd.DataFrame({"session_date":np.repeat(pd.date_range("2021-01-01",periods=6),2),"symbol":["A","B"]*6,"decision_ts":pd.date_range("2021-01-01",periods=12,freq="h"),"x":np.arange(12,dtype=float)})
    target=pd.DataFrame({"y":np.arange(12,dtype=float)})
    spec=FeatureSpec("x","x","test"); cfg=ScanConfig(use_cuda=False,min_observations=2,min_sessions=2,min_symbols=1,min_decision_timestamps=2,min_years=1)
    standalone=cuda_screen(feature,target,[spec],["y"],cfg,pd.DataFrame(),tmp_path/"standalone.csv")
    context=build_cuda_feature_context(feature,[spec],cfg)
    reused=cuda_screen(feature,target,[spec],["y"],cfg,pd.DataFrame(),tmp_path/"reused.csv",context)
    pd.testing.assert_frame_equal(standalone,reused)


def test_categorical_features_never_receive_pearson_interpretation():
    n=200; frame=pd.DataFrame({"session_date":np.repeat(pd.date_range("2021-01-01",periods=20),10),"symbol":[f"S{i%5}" for i in range(n)],"decision_ts":pd.date_range("2021-01-01",periods=n,freq="h"),"day_of_week":np.arange(n)%5,"y":np.random.default_rng(1).normal(size=n)})
    cfg=ScanConfig(min_observations=20,min_sessions=2,min_symbols=2,min_decision_timestamps=20,min_years=1,min_bin_observations=5)
    result,_=scan(frame,[FeatureSpec("day_of_week","weekday","calendar",classification="categorical",dtype="categorical")],["y"],cfg)
    assert result.iloc[0].status in {"categorical_screened","no_meaningful_relationship","statistically_interesting"}
    assert "pearson" not in result or pd.isna(result.iloc[0].pearson)


def test_fingerprint_changes_with_formula_source(monkeypatch,tmp_path):
    monkeypatch.setattr("quant_pipeline.fingerprint.source_provenance",lambda config:{"rows":1})
    cfg=ScanConfig(corporate_actions_path=str(tmp_path/"none")); specs=[FeatureSpec("x","one","test")]; targets=target_registry([5],[5])
    first=run_fingerprint(cfg,specs,targets,"abc")["sha256"]
    second=run_fingerprint(cfg,[FeatureSpec("x","changed","test")],targets,"abc")["sha256"]
    assert first!=second


def test_holdout_rows_are_rejected_when_seal_is_enforced():
    bars=bars_for_days(days=("2026-05-01",),bars=3); bars["symbol_role"]="tradable"; bars["bar_grid_valid"]=True; bars["is_member"]=True
    with pytest.raises(ValueError,match="Sealed holdout row"):
        add_targets(bars,target_registry([5],[5]))
