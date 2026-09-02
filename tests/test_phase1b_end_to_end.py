from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_pipeline.cache import (
    CacheFingerprintMismatch, file_sha256, row_key_hash, schema_hash,
    write_cache_metadata,
)
from quant_pipeline.config import ScanConfig
from quant_pipeline.phase1b_run import run_phase1b, validate_phase1a_source
from quant_pipeline.registry import FeatureSpec, TargetSpec, registry_frame
from quant_pipeline.candidate_selection import select_exact_candidate_queue
from quant_pipeline.run import _cluster_candidates


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source(tmp_path: Path, *, contaminate: bool = False) -> tuple[Path, Path, ScanConfig]:
    source = tmp_path / "phase1a"
    features_root = source / "blocks" / "features"
    targets_root = source / "blocks" / "targets"
    features_root.mkdir(parents=True)
    targets_root.mkdir(parents=True)
    rows = []
    for day in pd.date_range("2026-04-20", periods=4, freq="D"):
        for symbol_number, symbol in enumerate(("AAA", "BBB", "CCC")):
            for minute in (35, 40):
                decision = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=minute)
                if contaminate and day == pd.Timestamp("2026-04-23") and symbol == "CCC" and minute == 40:
                    decision = pd.Timestamp("2026-05-01 09:40", tz="America/New_York")
                a = float((symbol_number + minute // 5 + day.day) % 2)
                b = float((symbol_number + day.day) % 3 != 0)
                rows.append({
                    "symbol": symbol, "session_date": decision.date(),
                    "bar_start_ts": decision.tz_convert("UTC") - pd.Timedelta(minutes=5),
                    "decision_ts": decision.tz_convert("UTC"), "analysis_eligible": True,
                    "parent_a": a, "parent_b": b,
                    "target_5m": .002 * float(a and b) + .0001 * (symbol_number - 1),
                })
    frame = pd.DataFrame(rows).sort_values(
        ["symbol", "session_date", "bar_start_ts", "decision_ts"], kind="stable"
    ).reset_index(drop=True)
    feature_frame = frame.drop(columns="target_5m")
    target_frame = frame[["symbol", "session_date", "bar_start_ts", "decision_ts", "target_5m"]]
    feature_path = features_root / "feature_000.parquet"
    target_path = targets_root / "target_000.parquet"
    feature_frame.to_parquet(feature_path, index=False)
    target_frame.to_parquet(target_path, index=False)
    source_fingerprint = "source-fingerprint"
    if not contaminate:
        write_cache_metadata(feature_path, feature_frame, source_fingerprint)
        write_cache_metadata(target_path, target_frame, source_fingerprint)
    else:
        # Deliberately create structurally valid metadata around a forbidden
        # row so source validation, not the fixture writer, rejects it.
        for path, payload in ((feature_path, feature_frame), (target_path, target_frame)):
            metadata = {
                "fingerprint": source_fingerprint, "row_count": len(payload),
                "row_key_hash": row_key_hash(payload), "column_schema_hash": schema_hash(payload),
                "file_size": path.stat().st_size, "file_sha256": file_sha256(path),
            }
            path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    feature_specs = [
        FeatureSpec("parent_a", "parent a", "test", dtype="binary"),
        FeatureSpec("parent_b", "parent b", "test", dtype="binary"),
    ]
    target_specs = [TargetSpec("target_5m", "target", 5, "next open", "close", tier="primary")]
    registry_frame(feature_specs).to_csv(source / "feature_registry.csv", index=False)
    registry_frame(target_specs).to_csv(source / "target_registry.csv", index=False)
    pd.DataFrame([{"feature":"parent_a","status":"built","reason":""},{"feature":"parent_b","status":"built","reason":""}]).to_csv(source/"scan_coverage.csv",index=False)
    pd.DataFrame([
        {"feature": "parent_a", "target": "target_5m", "raw_p": .04, "top_bottom_spread": .001, "monotonicity": .5, "status": "cuda_screened"},
        {"feature": "parent_b", "target": "target_5m", "raw_p": .08, "top_bottom_spread": .0005, "monotonicity": .3, "status": "cuda_screened"},
    ]).to_csv(source / "master_results.csv", index=False)
    pd.DataFrame([
        {"feature":"parent_a","target":"target_5m","raw_p":.04,"top_bottom_spread":.001,"status":"statistically_significant_discovery"},
        {"feature":"parent_b","target":"target_5m","raw_p":.08,"top_bottom_spread":.0005,"status":"exploratory_relationship"},
    ]).to_csv(source/"detailed_candidates.csv",index=False)
    (source / "manifest.json").write_text(json.dumps({
        "discovery_end": "2026-04-30", "sealed_holdout_start": "2026-05-01",
        "allow_holdout_access": False,
    }), encoding="utf-8")
    (source / "fingerprint.json").write_text(json.dumps({"sha256": source_fingerprint}), encoding="utf-8")
    (source / "progress.json").write_text(json.dumps({"stage": "complete"}), encoding="utf-8")
    manifest = tmp_path / "dual.yaml"
    manifest.write_text("""schema_version: phase1b_manifest_v1
definitions:
  - id: a_and_b
    feature_a: parent_a
    feature_b: parent_b
    operator: intersection
    condition_a: {transform: raw, comparator: eq, threshold: 1}
    condition_b: {transform: raw, comparator: eq, threshold: 1}
    output_dtype: binary
    expected_direction: 1
""", encoding="utf-8")
    config = ScanConfig(
        output_root=str(tmp_path / "derived"), experiment_id="phase1b_test",
        dual_factor_enabled=True, dual_factor_manifest_path=str(manifest),
        use_cuda=False, resume=True, min_observations=4, min_sessions=2,
        min_symbols=2, min_decision_timestamps=2, min_years=1,
        cross_sectional_min_symbols=2, dual_factor_min_signal_observations=2,
        dual_factor_min_signal_sessions=1, dual_factor_min_signal_symbols=1,
        binary_min_on_observations=2, binary_min_off_observations=2,
        binary_min_on_sessions=1, binary_min_off_sessions=1,
        binary_min_on_symbols=1, binary_min_off_symbols=1,
        require_corporate_actions=False,
    )
    return source, manifest, config


def test_standalone_phase1b_executes_without_rebuilding_source(tmp_path, monkeypatch):
    source, _, config = _source(tmp_path)
    before = _tree_hash(source)
    monkeypatch.setattr("quant_pipeline.table.source_provenance", lambda *_: (_ for _ in ()).throw(AssertionError("raw source access")))
    monkeypatch.setattr("quant_pipeline.run.load_canonical_bars", lambda *_: (_ for _ in ()).throw(AssertionError("Phase 1A rebuild")))
    monkeypatch.setattr("quant_pipeline.run.build_features", lambda *_: (_ for _ in ()).throw(AssertionError("Phase 1A rebuild")))
    monkeypatch.setattr("quant_pipeline.run.add_targets", lambda *_: (_ for _ in ()).throw(AssertionError("Phase 1A rebuild")))
    root = run_phase1b(source, config)
    assert _tree_hash(source) == before
    dual = pd.read_csv(root / "phase1b" / "dual_screen_results.csv")
    combined = pd.read_csv(root / "master_results.csv")
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(dual) == 1
    assert dual.loc[0, "screen_inference"] == "two_way_date_symbol"
    assert dual.loc[0, "raw_p"] == pytest.approx(dual.loc[0, "two_way_cluster_p"])
    assert len(combined) == 3
    assert {"primary_global_fdr", "family_fdr", "cluster_fdr"}.issubset(combined)
    assert manifest["promotion_ready"] is True
    assert manifest["evidence_stage"] == "exact_diagnostics_complete"
    assert (root / "detailed_candidates.csv").exists()
    assert manifest["source_immutable"] is True
    assert manifest["phase1a_rebuilt"] is False
    assert (root / "phase1b_readiness_report.txt").exists()


def test_source_row_key_mismatch_is_rejected(tmp_path):
    source, _, config = _source(tmp_path)
    target_meta = next((source / "blocks" / "targets").glob("*.meta.json"))
    payload = json.loads(target_meta.read_text())
    payload["row_key_hash"] = "wrong"
    target_meta.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CacheFingerprintMismatch, match="do not align"):
        validate_phase1a_source(source, config)


def test_source_holdout_row_is_rejected(tmp_path):
    source, _, config = _source(tmp_path, contaminate=True)
    with pytest.raises(ValueError, match="holdout"):
        validate_phase1a_source(source, config)


def test_resume_rejects_changed_phase1b_fingerprint(tmp_path):
    source, _, config = _source(tmp_path)
    run_phase1b(source, config)
    changed = replace(config, binary_min_on_observations=3)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        run_phase1b(source, changed)


def test_resume_reuses_compatible_dual_cache_and_journal(tmp_path):
    source, _, config = _source(tmp_path)
    first = run_phase1b(source, config)
    first_results = pd.read_csv(first / "phase1b" / "dual_screen_results.csv")
    second = run_phase1b(source, config)
    pd.testing.assert_frame_equal(first_results, pd.read_csv(second / "phase1b" / "dual_screen_results.csv"))


def test_incomplete_source_is_rejected(tmp_path):
    source, _, config = _source(tmp_path)
    (source / "progress.json").write_text(json.dumps({"stage": "exact_diagnostics"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        validate_phase1a_source(source, config)


def test_empty_plan_completes_with_explicit_reason(tmp_path):
    source, manifest, config = _source(tmp_path)
    manifest.write_text("schema_version: phase1b_manifest_v1\ndefinitions: []\n", encoding="utf-8")
    root = run_phase1b(source, config)
    progress = json.loads((root / "progress.json").read_text())
    assert progress["stage"] == "complete"
    assert progress["compiled_dual_features"] == 0
    assert progress["empty_plan_reason"] == "manifest_compiled_zero_features"


def test_phase1b_cli_calls_execution_path(tmp_path, monkeypatch, capsys):
    from quant_pipeline import phase1b_launcher
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(ScanConfig(dual_factor_enabled=True).as_dict()), encoding="utf-8")
    produced = tmp_path / "produced"
    called = {}
    monkeypatch.setattr(phase1b_launcher, "run_phase1b", lambda source, config: called.update(source=source) or produced)
    monkeypatch.setattr("sys.argv", ["quant-pipeline-phase1b", "--source-run", str(tmp_path / "source"), "--config", str(config_path)])
    phase1b_launcher.main()
    assert called["source"] == str(tmp_path / "source")
    assert str(produced) in capsys.readouterr().out


def test_requested_but_skipped_source_feature_is_accepted_unless_required_parent(tmp_path):
    source,manifest,config=_source(tmp_path)
    registry=pd.read_csv(source/"feature_registry.csv")
    registry=pd.concat([registry,registry.iloc[[0]].assign(name="opening_return_1m")],ignore_index=True)
    registry.to_csv(source/"feature_registry.csv",index=False)
    coverage=pd.read_csv(source/"scan_coverage.csv")
    pd.concat([coverage,pd.DataFrame([{"feature":"opening_return_1m","status":"skipped","reason":"source bars are 5m"}])],ignore_index=True).to_csv(source/"scan_coverage.csv",index=False)
    validate_phase1a_source(source,config)
    manifest.write_text("""schema_version: phase1b_manifest_v1
definitions:
  - id: missing_parent
    feature_a: opening_return_1m
    feature_b: parent_a
    operator: intersection
    condition_a: {transform: raw, comparator: gt, threshold: 0}
    condition_b: {transform: raw, comparator: eq, threshold: 1}
    output_dtype: binary
""",encoding="utf-8")
    with pytest.raises(FileNotFoundError,match="required dual parents"):
        run_phase1b(source,config)


def test_nonempty_plan_with_all_dual_features_failing_coverage_errors(tmp_path):
    source,_,config=_source(tmp_path)
    config=replace(config,dual_factor_min_signal_observations=10_000)
    with pytest.raises(RuntimeError,match="none passed construction and coverage"):
        run_phase1b(source,config)


def test_standalone_and_normal_exact_candidate_queues_match():
    results=pd.DataFrame({
        "feature":["a_5","a_10","b_5"],"target":["fwd_return_5m"]*3,
        "raw_p":[.001,.01,.20],"primary_global_fdr":[.003,.015,.20],
        "anomaly_score":[.9,.8,.4],"feature_family":["a","a","b"],
        "redundancy_group":["a_LOOKBACK","a_LOOKBACK","b_LOOKBACK"],
        "monotonicity":[.8,.7,-.4],"spearman":[.1,.1,-.1],
    })
    config=ScanConfig(max_candidates_per_cluster=2)
    normal=set(map(tuple,_cluster_candidates(results,None,config,250)[["feature","target"]].to_numpy()))
    standalone=set(map(tuple,select_exact_candidate_queue(results,config,None,250)[["feature","target"]].to_numpy()))
    assert normal==standalone


def test_built_dual_rows_without_valid_inference_fail(tmp_path,monkeypatch):
    source,_,config=_source(tmp_path)
    invalid=pd.DataFrame({"feature":["dual_test_feature"],"target":["target_5m"],"raw_p":[np.nan],"status":["insufficient_data"]})
    monkeypatch.setattr("quant_pipeline.phase1b_run.screen_feature_blocks_against_target_blocks",lambda **_:invalid)
    with pytest.raises(RuntimeError,match="no feature-target pair produced valid statistical inference"):
        run_phase1b(source,config)
    output=Path(config.output_root)/config.experiment_id/"manifest.json"
    assert not output.exists() or json.loads(output.read_text()).get("promotion_ready") is not True


def test_systematic_only_end_to_end_keeps_source_immutable_and_separates_fdr(tmp_path):
    source,_,config=_source(tmp_path);before=_tree_hash(source)
    master=pd.read_csv(source/"master_results.csv")
    master["target_tier"]="primary";master["target_family"]="target_5m";master["primary_global_fdr"]=[.05,.10];master["anomaly_score"]=[1.,.8]
    master["valid_observations"]=24;master["valid_sessions"]=4;master["valid_symbols"]=3;master["valid_decision_timestamps"]=8
    master.to_csv(source/"master_results.csv",index=False)
    # The source fixture is intentionally immutable during execution; update its
    # declared artifact hash before taking the byte-for-byte baseline.
    config=replace(config,phase1b_mode="systematic_only",experiment_id="systematic_test")
    parent=replace(config.systematic_phase1b.parent_selection,minimum_valid_observations=4,minimum_sessions=2,minimum_symbols=2,minimum_decision_timestamps=2)
    pair=replace(config.systematic_phase1b.pair_generation,minimum_joint_observations=4,minimum_joint_sessions=2,minimum_joint_symbols=2,minimum_joint_decision_timestamps=2)
    binary=replace(config.systematic_phase1b.binary_state_coverage,minimum_on_observations=2,minimum_off_observations=2,minimum_on_sessions=1,minimum_off_sessions=1,minimum_on_symbols=1,minimum_off_symbols=1,minimum_activation_rate=0,maximum_activation_rate=1)
    limits=replace(config.systematic_phase1b.limits,feature_chunk_size=1)
    config=replace(config,systematic_phase1b=replace(config.systematic_phase1b,parent_selection=parent,pair_generation=pair,binary_state_coverage=binary,limits=limits))
    before=_tree_hash(source);root=run_phase1b(source,config)
    assert _tree_hash(source)==before
    systematic=pd.read_csv(root/"phase1b"/"systematic"/"screen_results.csv")
    assert len(systematic)>0 and systematic.target_tier.eq("exploratory").all()
    assert systematic.systematic_test_count.iloc[0]==systematic.raw_p.notna().sum()
    combined=pd.read_csv(root/"master_results.csv")
    assert combined.loc[combined.discovery_phase.eq("1B_systematic"),"primary_global_fdr"].isna().all()
    assert (root/"phase1b"/"systematic"/"parent_selection.csv").exists()
    assert (root/"phase1b"/"combined"/"dual_parent_comparison.csv").exists()
