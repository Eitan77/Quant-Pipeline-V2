"""Cache-consuming broad screening shared by integrated and derived runs."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .bulk_scan import (
    assert_valid_screen_results,
    build_cuda_binary_context,
    build_cuda_feature_context,
    cuda_binary_scan_batch,
    cuda_screen,
)
from .cache import assert_cache_key_alignment
from .config import ScanConfig
from .registry import FeatureSpec, TargetSpec
from .scanner import binary_scan_batch, categorical_scan_batch


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    # A complete JSON object occupies each line. A truncated final line is
    # ignored on resume; completed pairs are never inferred from partial CSV.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(frame.to_json(orient="records", lines=True, double_precision=15))
        handle.flush()
        os.fsync(handle.fileno())


def _resume(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates(["feature", "target"], keep="last")
        assert_valid_screen_results(result, "resumed Phase 1B journal")
    return result


def screen_feature_blocks_against_target_blocks(
    *,
    feature_paths: list[Path],
    feature_specs_by_path: list[list[FeatureSpec]],
    target_paths: list[Path],
    target_specs_by_path: list[list[TargetSpec]],
    config: ScanConfig,
    run_root: Path,
    resume_results: pd.DataFrame | None = None,
    journal_path: Path | None = None,
) -> pd.DataFrame:
    """Screen validated aligned caches without constructing any source data."""
    if len(feature_paths) != len(feature_specs_by_path):
        raise ValueError("Feature path/spec chunk counts differ")
    if len(target_paths) != len(target_specs_by_path):
        raise ValueError("Target path/spec chunk counts differ")
    run_root.mkdir(parents=True, exist_ok=True)
    journal = journal_path or (run_root / "journals" / "screen_journal.jsonl")
    journal.parent.mkdir(parents=True, exist_ok=True)
    results = _resume(journal) if config.resume else pd.DataFrame()
    if resume_results is not None and not resume_results.empty:
        results = pd.concat([results, resume_results], ignore_index=True).drop_duplicates(
            ["feature", "target"], keep="last"
        )
    target_groups=[]
    for start in range(0,len(target_paths),config.cuda_target_batch_group_size):
        target_groups.append((target_paths[start:start+config.cuda_target_batch_group_size],target_specs_by_path[start:start+config.cuda_target_batch_group_size]))
    total = len(feature_paths) * len(target_groups)
    completed_batches = 0
    for feature_path, specs in zip(feature_paths, feature_specs_by_path):
        if not specs:
            completed_batches += len(target_groups)
            continue
        feature_frame = pd.read_parquet(feature_path)
        screen_end = config.selection_end if config.use_separate_confirmation_period else config.discovery_end
        if not screen_end:
            raise ValueError("A discovery screen end is required")
        selection = pd.to_datetime(feature_frame.session_date).le(pd.Timestamp(screen_end))
        selection &= feature_frame.analysis_eligible.fillna(False)
        if config.decision_times_et:
            local = pd.to_datetime(feature_frame.decision_ts, utc=True).dt.tz_convert("America/New_York")
            selection &= local.dt.strftime("%H:%M").isin(config.decision_times_et)
        selected_features = feature_frame.loc[selection].reset_index(drop=True)
        continuous = [item for item in specs if item.classification != "categorical" and item.dtype not in {"categorical", "binary"}]
        binary = [item for item in specs if item.dtype == "binary"]
        categorical = [item for item in specs if item.classification == "categorical" or item.dtype == "categorical"]
        context = build_cuda_feature_context(selected_features, continuous, config) if continuous else None
        binary_context = build_cuda_binary_context(selected_features, binary, config) if binary and config.use_cuda else None
        for grouped_paths, grouped_specs in target_groups:
            for target_path in grouped_paths:assert_cache_key_alignment(feature_path, target_path)
            targets=[item for chunk in grouped_specs for item in chunk]
            names = [item.name for item in targets]
            completed = {(row.feature, row.target) for row in results.itertuples()}
            if all((spec.name, target) in completed for spec in specs for target in names):
                completed_batches += 1
                continue
            target_values = pd.concat([pd.read_parquet(path,columns=[item.name for item in specs]).loc[selection].reset_index(drop=True) for path,specs in zip(grouped_paths,grouped_specs)],axis=1)
            prior_count = len(results)
            completed = {(row.feature, row.target) for row in results.itertuples()}
            pending_continuous=[spec for spec in continuous if any((spec.name,name) not in completed for name in names)]
            active_context=context if context is not None and tuple(spec.name for spec in pending_continuous)==context.feature_names else (build_cuda_feature_context(selected_features,pending_continuous,config) if pending_continuous else None)
            results = cuda_screen(
                selected_features, target_values, pending_continuous, names, config,
                results, journal, active_context,
            )
            combined = pd.concat([selected_features, target_values], axis=1)
            for scanner in (binary_scan_batch, categorical_scan_batch):
                active = binary if scanner is binary_scan_batch else categorical
                if not active:
                    continue
                completed = {(row.feature, row.target) for row in results.itertuples()}
                active = [spec for spec in active if any((spec.name, name) not in completed for name in names)]
                if not active:
                    continue
                if scanner is binary_scan_batch and config.use_cuda:
                    active_binary_context = binary_context if tuple(spec.name for spec in active) == binary_context.feature_names else build_cuda_binary_context(selected_features, active, config)
                    additions = cuda_binary_scan_batch(combined, active, names, config, active_binary_context)
                else:
                    additions = scanner(combined, active, names, config)
                completed = {(row.feature, row.target) for row in results.itertuples()}
                additions = additions.loc[
                    [((row.feature, row.target) not in completed) for row in additions.itertuples()]
                ]
                assert_valid_screen_results(additions, f"{scanner.__name__} batch")
                _append_jsonl(additions, journal)
                results = pd.concat([results, additions], ignore_index=True)
            results = results.drop_duplicates(["feature", "target"], keep="last")
            completed_batches += 1
            _atomic_json({
                "stage": "broad_screen",
                "completed_batches": completed_batches,
                "total_batches": total,
                "completed_pairs": len(results),
                "new_pairs": len(results) - prior_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, run_root / "progress.json")
    if results.empty:
        results = pd.DataFrame(columns=["feature", "target", "raw_p"])
    assert_valid_screen_results(results, "completed broad screen")
    return results.reset_index(drop=True)
