from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.config import ScanConfig
from quant_pipeline.dual_features import _materialize, centered_oriented_rank, eligible_cross_sectional_rank
from quant_pipeline.dual_registry import compile_dual_plan, plan_hash
from quant_pipeline.effects import effect_value, feature_scan_kind
from quant_pipeline.registry import FeatureSpec, feature_registry
from quant_pipeline.scanner import binary_scan_batch, scan
from quant_pipeline.phase1b_run import merge_combined_results
from quant_pipeline.registry import TargetSpec


def test_phase1b_manifest_is_deterministic_and_uses_existing_parents():
    config=ScanConfig(dual_factor_enabled=True,dual_factor_manifest_path="configs/phase1b_dual_factors.yaml")
    first=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)
    second=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)
    assert len(first)==5
    assert [item.spec.name for item in first] == [item.spec.name for item in second]
    assert plan_hash(first)==plan_hash(second)
    assert all(item.spec.discovery_phase=="1B" and item.spec.arity==2 for item in first)


def test_dual_intersection_keeps_missing_as_missing():
    config=ScanConfig(dual_factor_enabled=True,dual_factor_manifest_path="configs/phase1b_dual_factors.yaml")
    item=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)[0]
    timestamp=pd.Timestamp("2026-04-01 14:35",tz="UTC")
    frame=pd.DataFrame({"symbol":["A","B","C"],"session_date":["2026-04-01"]*3,"decision_ts":[timestamp]*3,"analysis_eligible":True,"session_range_position":[1.,.2,np.nan],"tod_relative_volume_20":[2.,.2,2.]})
    result=_materialize(frame,item,ScanConfig(cross_sectional_min_symbols=2))
    assert result.tolist()[:2]==[1.0,0.0]
    assert np.isnan(result.iloc[2])


def test_binary_effect_is_on_minus_off_not_decile_spread():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    frame=pd.DataFrame({"signal":[0,0,1,1],"target":[-.0005,-.0005,.001,.001]})
    assert feature_scan_kind(spec)=="binary"
    assert np.isclose(effect_value(frame,spec,"target"),.0015)


def test_binary_screen_reports_exact_on_minus_off_effect():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    dates=pd.date_range("2025-01-01",periods=8,freq="D")
    frame=pd.DataFrame({"signal":[0,1]*8,"target":[-.0005,.001]*8,"session_date":np.repeat(dates,2),"symbol":["A","B"]*8,"decision_ts":pd.date_range("2025-01-01",periods=16,freq="5min",tz="UTC"),"analysis_eligible":True})
    config=ScanConfig(min_observations=4,min_sessions=2,min_symbols=2,min_decision_timestamps=2,min_years=1,binary_min_on_observations=2,binary_min_off_observations=2,binary_min_on_sessions=2,binary_min_off_sessions=2,binary_min_on_symbols=1,binary_min_off_symbols=1)
    result=binary_scan_batch(frame,[spec],["target"],config).iloc[0]
    assert result["scan_kind"]=="binary"
    assert np.isclose(result["top_bottom_spread"],.0015)
    assert np.isnan(result["monotonicity"])
    assert result["screen_inference"]=="two_way_date_symbol"
    assert result["raw_p"]==pytest.approx(result["two_way_cluster_p"])
    assert "date_cluster_p" in result.index


def test_exact_binary_scan_does_not_construct_deciles():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    frame=pd.DataFrame({"signal":[0,1]*8,"target":[-.0005,.001]*8,"session_date":np.repeat(pd.date_range("2025-01-01",periods=8,freq="D"),2),"symbol":["A","B"]*8,"decision_ts":pd.date_range("2025-01-01",periods=16,freq="5min",tz="UTC"),"analysis_eligible":True})
    config=ScanConfig(min_observations=4,min_sessions=2,min_symbols=2,min_decision_timestamps=2,min_years=1,use_cuda=False,binary_min_on_observations=2,binary_min_off_observations=2,binary_min_on_sessions=2,binary_min_off_sessions=2,binary_min_on_symbols=1,binary_min_off_symbols=1)
    result,tables=scan(frame,[spec],["target"],config)
    assert result.iloc[0]["effect_kind"]=="binary_on_minus_off"
    assert list(tables[("signal","target")]["signal"]) == [0,1]


def test_production_yaml_loads_with_phase1b_fields():
    config=ScanConfig.from_yaml("configs/discovery_5m.yaml")
    assert not config.dual_factor_enabled
    assert config.discovery_end=="2026-04-30"


def test_cross_sectional_rank_excludes_ineligible_extremes_and_small_groups():
    ts=pd.Timestamp("2026-04-01 14:35",tz="UTC")
    frame=pd.DataFrame({"session_date":["2026-04-01"]*4,"decision_ts":[ts]*4,"analysis_eligible":[True,True,False,True]})
    rank=eligible_cross_sectional_rank(frame,pd.Series([1.,2.,999.,2.]),min_symbols=3)
    assert np.isclose(rank.iloc[0],1/3) and np.isclose(rank.iloc[1],5/6)
    assert np.isnan(rank.iloc[2])
    assert eligible_cross_sectional_rank(frame,pd.Series([1.,2.,999.,2.]),min_symbols=4).isna().all()


def test_centered_orientation_is_bounded_and_symmetric():
    rank=pd.Series([.1,.5,.9])
    assert np.allclose(centered_oriented_rank(rank,1),[-.8,0,.8])
    assert np.allclose(centered_oriented_rank(rank,-1),[.8,0,-.8])


def test_manifest_rejects_invalid_definition_before_data_access(tmp_path):
    path=tmp_path/"bad.yaml"
    path.write_text("schema_version: phase1b_manifest_v1\ndefinitions:\n  - id: bad\n    feature_a: session_range_position\n    feature_b: tod_relative_volume_20\n    operator: intersection\n    condition_a: {transform: raw, comparator: nope, threshold: 1}\n    condition_b: {transform: raw, comparator: eq, threshold: 1}\n",encoding="utf-8")
    with pytest.raises(ValueError,match="Invalid dual condition"):
        compile_dual_plan(path,feature_registry(ScanConfig().lookbacks),ScanConfig())


def test_combined_fdr_recomputes_primary_and_preserves_explicit_redundancy():
    base=pd.DataFrame({"feature":["base"],"target":["t"],"raw_p":[.01],"feature_family":["base"],"target_family":["t"],"top_bottom_spread":[.001],"monotonicity":[.5],"status":["cuda_screened"],"redundancy_group":["explicit_base"]})
    dual=pd.DataFrame({"feature":["dual"],"target":["t"],"raw_p":[.02],"feature_family":["dual_factor"],"target_family":["t"],"top_bottom_spread":[.002],"monotonicity":[np.nan],"status":["binary_screened"],"redundancy_group":["explicit_dual"]})
    targets=[TargetSpec("t","",5,"","",tier="primary")]
    result=merge_combined_results(base,dual,targets)
    assert set(result.primary_global_fdr.dropna())=={.02}
    assert set(result.redundancy_group)=={"explicit_base","explicit_dual"}


def test_swapped_commutative_definition_with_swapped_conditions_is_rejected(tmp_path):
    path=tmp_path/"duplicates.yaml"
    path.write_text("""schema_version: phase1b_manifest_v1
definitions:
  - id: first
    feature_a: session_range_position
    feature_b: tod_relative_volume_20
    operator: intersection
    condition_a: {transform: raw, comparator: ge, threshold: 0.8}
    condition_b: {transform: raw, comparator: ge, threshold: 1.2}
    output_dtype: binary
  - id: swapped
    feature_a: tod_relative_volume_20
    feature_b: session_range_position
    operator: intersection
    condition_a: {transform: raw, comparator: ge, threshold: 1.2}
    condition_b: {transform: raw, comparator: ge, threshold: 0.8}
    output_dtype: binary
""",encoding="utf-8")
    with pytest.raises(ValueError,match="Duplicate commutative"):
        compile_dual_plan(path,feature_registry(ScanConfig().lookbacks),ScanConfig())


def test_binary_pair_coverage_uses_target_complete_on_and_off_states():
    spec=FeatureSpec("signal","","test",dtype="binary")
    dates=pd.date_range("2025-01-01",periods=6,freq="D")
    frame=pd.DataFrame({
        "signal":[0,1]*6,
        "target":[0.0,np.nan,0.0,np.nan,0.0,np.nan,0.0,.001,0.0,.001,0.0,.001],
        "session_date":np.repeat(dates,2),"symbol":["A","B"]*6,
        "decision_ts":pd.date_range("2025-01-01",periods=12,freq="5min",tz="UTC"),
        "analysis_eligible":True,
    })
    config=ScanConfig(min_observations=4,min_sessions=2,min_symbols=2,min_decision_timestamps=2,min_years=1,binary_min_on_observations=4,binary_min_off_observations=4,binary_min_on_sessions=2,binary_min_off_sessions=2,binary_min_on_symbols=1,binary_min_off_symbols=1)
    row=binary_scan_batch(frame,[spec],["target"],config).iloc[0]
    assert row.status=="insufficient_data"
    assert row.coverage_reason=="insufficient_on_observations"
