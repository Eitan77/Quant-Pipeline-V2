import warnings

import numpy as np
import pandas as pd

from quant_pipeline.config import ScanConfig
from quant_pipeline.bulk_scan import _clustered_inference_from_moments, _pair_coverage_counts, _pair_moments_stable
from quant_pipeline.features import build_features
from quant_pipeline.gpu import CorrelationBackend
from quant_pipeline.parallel_features import build_parallel_blocks
from quant_pipeline.registry import feature_registry, target_registry
from quant_pipeline.run import _classify_detailed_candidates
from quant_pipeline.scanner import _clustered_slope, scan
from quant_pipeline.table import add_targets, validate_point_in_time


def fixture_bars(days=4):
    parts=[]
    for day in pd.bdate_range("2026-04-01", periods=days):
        ts=pd.date_range(pd.Timestamp(day.date()).tz_localize("America/New_York")+pd.Timedelta(hours=9,minutes=30),periods=78,freq="5min").tz_convert("UTC")
        for j,symbol in enumerate(["AAA","BBB","QQQ"]):
            z=np.arange(78); close=100+j+z*.01+np.sin(z/7+j)*.1
            parts.append(pd.DataFrame({"symbol":symbol,"bar_start_ts":ts,"bar_end_ts":ts+pd.Timedelta(minutes=5),"available_at_ts":ts+pd.Timedelta(minutes=5),"open":close-.01,"high":close+.05,"low":close-.05,"close":close,"volume":1000+z*7+j,"vwap":close,"session_date":day.date()}))
    return pd.concat(parts,ignore_index=True)


def test_full_core_registry_builds_and_accounts_for_skips():
    cfg=ScanConfig(lookbacks=[1,2,3,5,10])
    specs=feature_registry(cfg.lookbacks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        frame,built=build_features(fixture_bars(),cfg,specs)
    assert len(specs)>=290
    skipped={s.name for s in specs}-{s.name for s in built}
    assert {"sector_return","stock_minus_sector_return","sector_rank","sector_breadth_positive"}.issubset(skipped)
    assert all(("_1m" in name or "_3m" in name or name.startswith("sector") or name=="stock_minus_sector_return") for name in skipped)
    frame=add_targets(frame,target_registry(),"QQQ")
    assert validate_point_in_time(frame,target_registry())["target_timing_violations"]==0
    assert frame.filter(like="benchmark_adjusted").shape[1]==sum(t.classification=="benchmark_adjusted" for t in target_registry())
    assert frame.filter(like="beta_residual").shape[1]==sum(t.classification=="beta_residual" for t in target_registry())


def test_signed_features_do_not_register_unstable_mean_ratios():
    names={spec.name for spec in feature_registry([1,2,3,5,10])}
    assert "return_1_mean_ratio_1560" not in names
    assert "distance_session_vwap_mean_ratio_4680" not in names
    assert "range_position_10_mean_ratio_1560" in names


def test_symbol_worker_builds_multiple_scan_blocks_in_one_pass(tmp_path):
    bars=fixture_bars(days=2)
    for column in ["open","high","low","close","vwap","volume"]:bars[column]=pd.to_numeric(bars[column],downcast="float")
    canonical=tmp_path/"bars.parquet"; bars.to_parquet(canonical,index=False)
    cfg=ScanConfig(lookbacks=[1,2],feature_workers=2)
    registry={spec.name:spec for spec in feature_registry(cfg.lookbacks)}
    chunks=[[registry["return_1"]],[registry["realized_vol_2"]]]
    paths=[tmp_path/"first.parquet",tmp_path/"second.parquet"]
    built=build_parallel_blocks(canonical,list(zip(paths,chunks)),cfg,tmp_path/"progress.json")
    first=pd.read_parquet(paths[0]); second=pd.read_parquet(paths[1])
    expected,_=build_features(bars,cfg,[*chunks[0],*chunks[1]],symbol_local=True)
    assert "return_1" in first and "realized_vol_2" not in first
    assert "realized_vol_2" in second and "return_1" not in second
    assert [[spec.name for spec in group] for group in built]==[["return_1"],["realized_vol_2"]]
    first_check=first[["symbol","bar_start_ts","return_1"]].merge(expected[["symbol","bar_start_ts","return_1"]],on=["symbol","bar_start_ts"],suffixes=("_actual","_expected"),validate="one_to_one")
    second_check=second[["symbol","bar_start_ts","realized_vol_2"]].merge(expected[["symbol","bar_start_ts","realized_vol_2"]],on=["symbol","bar_start_ts"],suffixes=("_actual","_expected"),validate="one_to_one")
    np.testing.assert_allclose(first_check.return_1_actual,first_check.return_1_expected,equal_nan=True)
    np.testing.assert_allclose(second_check.realized_vol_2_actual,second_check.realized_vol_2_expected,equal_nan=True)
    aaa=first.loc[(first.symbol.eq("AAA"))&(first.session_date.astype(str).eq("2026-04-02"))].sort_values("bar_start_ts")
    expected_second=float(aaa.close.iloc[1]/aaa.close.iloc[0]-1)
    assert np.isclose(aaa.return_1.iloc[1],expected_second)


def test_combined_target_build_matches_separate_batches():
    cfg=ScanConfig(lookbacks=[1,2],beta_window_sessions=2,beta_min_observations=2)
    frame,_=build_features(fixture_bars(days=4),cfg,feature_registry(cfg.lookbacks))
    targets=target_registry([5,15],[5,15])
    batch_5=[target for target in targets if target.name.startswith("fwd_return_5m")]
    batch_15=[target for target in targets if target.name.startswith("fwd_return_15m")]
    combined=add_targets(frame,[*batch_5,*batch_15],"QQQ",cfg)
    separate_5=add_targets(frame,batch_5,"QQQ",cfg); separate_15=add_targets(frame,batch_15,"QQQ",cfg)
    for batch,separate in [(batch_5,separate_5),(batch_15,separate_15)]:
        columns=[target.name for target in batch]+[f"exit_ts__{target.name}" for target in batch if target.classification=="raw"]+[f"exit_close_raw__{target.name}" for target in batch if target.classification=="raw"]+[f"actual_horizon_minutes__{target.name}" for target in batch if target.classification=="raw"]
        pd.testing.assert_frame_equal(combined[columns],separate[columns])


def test_scanner_outputs_fdr_ranking_and_cuda_backend(tmp_path):
    cfg=ScanConfig(lookbacks=[1,2,3,5,10],min_observations=100,min_sessions=2,min_symbols=2,min_bin_observations=5,quantiles=5,checkpoint_every_pairs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame,built=build_features(fixture_bars(),cfg,feature_registry(cfg.lookbacks))
    frame=add_targets(frame,target_registry(),"QQQ")
    wanted=[s for s in built if s.name in {"return_1","relative_volume_5","return_rank_5"}]
    result,_=scan(frame,wanted,["fwd_return_5m","fwd_return_15m"],cfg,tmp_path/"pairs.csv")
    assert len(result)==6
    assert {"bh_fdr_p","anomaly_score","redundancy_group","cluster_t"}.issubset(result.columns)
    assert (tmp_path/"pairs.csv").exists()
    assert CorrelationBackend(True).name.startswith("torch:")


def test_batched_clustered_inference_matches_exact_pair_regressions():
    import torch

    rng=np.random.default_rng(12); sessions=np.repeat(np.arange(30),20)
    session_shock=np.repeat(rng.normal(0,.8,30),20)
    x=np.column_stack([rng.normal(size=len(sessions)),rng.normal(size=len(sessions))])
    y=np.column_stack([.4*x[:,0]+session_shock+rng.normal(size=len(sessions)),
                       -.25*x[:,1]+.5*session_shock+rng.normal(size=len(sessions))])
    x[::17,1]=np.nan; y[::23,0]=np.nan
    tx=torch.as_tensor(x,dtype=torch.float64); ty=torch.as_tensor(y,dtype=torch.float64)
    corr,n,mx,my,vx,vy,cov=_pair_moments_stable(tx,ty,feature_block=2)
    se,t=_clustered_inference_from_moments(
        tx,ty,torch.as_tensor(sessions,dtype=torch.long),n,mx,my,vx,cov,feature_block=2
    )
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            valid=np.isfinite(x[:,i])&np.isfinite(y[:,j])
            _,expected_se,expected_t,_=_clustered_slope(x[valid,i],y[valid,j],sessions[valid])
            assert np.isclose(se[i,j].item(),expected_se,rtol=1e-10,atol=1e-12)
            assert np.isclose(t[i,j].item(),expected_t,rtol=1e-10,atol=1e-12)


def test_batched_pair_coverage_matches_exact_unique_counts():
    import torch

    finite_x=np.array([[1,1],[1,0],[0,1],[1,1],[1,1],[0,1]],dtype=bool)
    finite_y=np.array([[1,1],[1,1],[1,0],[0,1],[1,1],[1,1]],dtype=bool)
    groups={"sessions":np.array([0,0,0,1,1,1]),"symbols":np.array([0,1,0,1,0,1])}
    device="cuda" if torch.cuda.is_available() else "cpu"
    actual=_pair_coverage_counts(torch.as_tensor(finite_x,device=device),torch.as_tensor(finite_y,device=device),groups)
    for name,codes in groups.items():
        expected=np.zeros((finite_x.shape[1],finite_y.shape[1]),dtype=int)
        for i in range(finite_x.shape[1]):
            for j in range(finite_y.shape[1]):expected[i,j]=np.unique(codes[finite_x[:,i]&finite_y[:,j]]).size
        np.testing.assert_array_equal(actual[name],expected)


def test_robust_candidate_status_requires_global_fdr():
    frame=pd.DataFrame({
        "status":["statistically_interesting"]*2,
        "raw_p":[1e-6,1e-6],
        "screen_bh_fdr_p_global":[.2,.01],
        "year_consistency":[1.,1.],
        "symbol_breadth":[.8,.8],
        "n":[1000,1000],"valid_sessions":[200,200],"valid_symbols":[30,30],
        "two_way_cluster_p":[1e-6,1e-6],"top_bottom_spread":[.001,.001],
        "monotonicity":[1.,1.],"outlier_worst_signed_spread":[.0005,.0005],
        "symbol_breadth_classification":["broad_across_symbols"]*2,
    })
    classified=_classify_detailed_candidates(frame)
    assert classified.status.tolist()==["exploratory_relationship","robust_phase1_anomaly_candidate"]
