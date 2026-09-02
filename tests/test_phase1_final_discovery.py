from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.cache import CacheFingerprintMismatch,assert_cache_key_alignment,validate_cache,write_cache_metadata
from quant_pipeline.config import ScanConfig
from quant_pipeline.diagnostics import recent_period_diagnostics,recency_weighted_diagnostics
from quant_pipeline.report import write_reports
from quant_pipeline.run import _classify_detailed_candidates,apply_confirmation_fdr


def diagnostic_panel(start="2024-01-02",periods=520):
    rows=[]; rng=np.random.default_rng(17)
    for day in pd.bdate_range(start,periods=periods):
        for index,symbol in enumerate(["AAA","BBB","CCC","DDD","EEE"]):
            decision=(pd.Timestamp(day.date()).tz_localize("America/New_York")+pd.Timedelta(hours=10)).tz_convert("UTC")
            x=rng.normal(); rows.append({"symbol":symbol,"session_date":day.normalize(),"bar_start_ts":decision-pd.Timedelta(minutes=5),"decision_ts":decision,"x":x,"y":.002*x+rng.normal(0,.002),"analysis_eligible":True})
    return pd.DataFrame(rows)


def test_final_config_uses_complete_pre_holdout_discovery():
    config=ScanConfig(); config.validate()
    assert config.discovery_end=="2026-04-30"
    assert config.sealed_holdout_start=="2026-05-01"
    assert not config.use_separate_confirmation_period and config.selection_end is None


def test_feature_build_batch_size_must_be_positive():
    with pytest.raises(ValueError,match="feature_build_batch_chunks"):
        ScanConfig(feature_build_batch_chunks=0).validate()


def test_target_build_batch_size_must_be_positive():
    with pytest.raises(ValueError,match="target_build_batch_chunks"):
        ScanConfig(target_build_batch_chunks=0).validate()


def test_empty_confirmation_configuration_does_not_apply_confirmation_fdr():
    frame=pd.DataFrame({"raw_p":[.001,.01],"confirmation_p_value":[.001,.02]})
    result=apply_confirmation_fdr(frame,ScanConfig(use_separate_confirmation_period=False))
    assert result.confirmation_fdr.isna().all() and not result.confirmation_fdr_confirmed.any()
    enabled=apply_confirmation_fdr(frame,ScanConfig(use_separate_confirmation_period=True,selection_end="2023-12-31",confirmation_start="2024-01-01"))
    assert enabled.confirmation_fdr.notna().all()
    assert not enabled.confirmation_fdr.equals(enabled.raw_p)


def test_recent_diagnostics_stay_pre_holdout_and_include_full_sample():
    frame=diagnostic_panel(); config=ScanConfig(start="2024-01-01",min_sessions=20,min_symbols=3)
    summary,table=recent_period_diagnostics(frame,"x","y",1,config)
    assert "full_discovery" in set(table.period)
    assert pd.to_datetime(table.end).max()<pd.Timestamp(config.sealed_holdout_start)
    assert "recent_classification" in summary


def test_recency_weighting_reports_session_effective_sample_size():
    frame=diagnostic_panel(); summary,table=recency_weighted_diagnostics(frame,"x","y",1,ScanConfig(start="2024-01-01"))
    assert table.effective_sessions.between(1,table.actual_sessions).all()
    assert summary["recency_weighted_exploratory"] is True


def test_recency_weighted_evidence_cannot_create_strongest_status():
    frame=pd.DataFrame({"status":["statistically_interesting"],"raw_p":[.001],"screen_bh_fdr_p_global":[.5],"year_consistency":[1.],"symbol_breadth":[1.],"n":[10000],"valid_sessions":[1000],"valid_symbols":[100],"two_way_cluster_p":[.001],"top_bottom_spread":[.01],"monotonicity":[1.],"outlier_worst_signed_spread":[.01],"symbol_breadth_classification":["broad_across_symbols"],"recency_6m_effect":[.1]})
    assert _classify_detailed_candidates(frame).status.iloc[0]=="exploratory_relationship"


def test_statuses_never_claim_independent_confirmation():
    frame=pd.DataFrame({"status":["insufficient_data"]})
    result=_classify_detailed_candidates(frame)
    forbidden=("independent_confirmation","final_confirmation","holdout_confirmed","fully_confirmed")
    assert not any(token in value for value in result.status.astype(str) for token in forbidden)


def test_cache_write_and_resume_reject_may_first(tmp_path):
    frame=diagnostic_panel(periods=2).head(2); frame.loc[frame.index[-1],["session_date","bar_start_ts","decision_ts"]]=[pd.Timestamp("2026-05-01"),pd.Timestamp("2026-05-01 14:30",tz="UTC"),pd.Timestamp("2026-05-01 14:35",tz="UTC")]
    path=tmp_path/"holdout.parquet"; frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match="holdout"):
        write_cache_metadata(path,frame,"fingerprint")
    path.with_suffix(path.suffix+".meta.json").write_text('{"fingerprint":"fingerprint"}')
    with pytest.raises(ValueError,match="holdout"):
        validate_cache(path,"fingerprint")


def test_cache_metadata_alignment_uses_validated_key_hashes(tmp_path):
    feature=tmp_path/"feature.parquet"; target=tmp_path/"target.parquet"
    feature.with_suffix(".parquet.meta.json").write_text('{"row_count": 10, "row_key_hash": "same"}')
    target.with_suffix(".parquet.meta.json").write_text('{"row_count": 10, "row_key_hash": "same"}')
    assert_cache_key_alignment(feature,target)
    target.with_suffix(".parquet.meta.json").write_text('{"row_count": 10, "row_key_hash": "different"}')
    with pytest.raises(CacheFingerprintMismatch):assert_cache_key_alignment(feature,target)


def test_report_rejects_holdout_rows(tmp_path):
    results=pd.DataFrame({"feature":["x"],"target":["y"],"session_date":[pd.Timestamp("2026-05-01")]})
    with pytest.raises(ValueError,match="holdout"):
        write_reports(results,{},tmp_path,config=ScanConfig())
