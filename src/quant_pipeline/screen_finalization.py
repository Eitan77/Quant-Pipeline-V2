"""One authoritative broad-screen finalization path for Phase 1A and 1B."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from .bulk_scan import assert_valid_screen_results, finalize_screen
from .config import ScanConfig
from .registry import FeatureSpec, TargetSpec, registry_frame
from .scanner import benjamini_hochberg


DERIVED_SCREEN_COLUMNS = {
    "bh_fdr_p", "bh_fdr_p_global", "bh_fdr_p_group",
    "primary_global_fdr", "family_fdr", "cluster_fdr",
    "exploratory_family_fdr", "candidate_cluster", "test_count",
    "primary_test_count", "exploratory_test_count", "anomaly_score",
}


@dataclass(frozen=True)
class FinalizedScreen:
    master: pd.DataFrame
    primary: pd.DataFrame
    exploratory: pd.DataFrame


def target_horizon_family(target: str) -> str:
    if "eod" in target:
        return "long_intraday"
    match = re.search(r"_(\d+)m(?:_|$)", target)
    if not match:
        return target
    minutes = int(match.group(1))
    if minutes <= 30:
        return "short_intraday"
    if minutes <= 120:
        return "medium_intraday"
    return "long_intraday"


def finalize_phase1_screen(
    raw_results: pd.DataFrame,
    *,
    feature_registry: list[FeatureSpec],
    target_registry: list[TargetSpec],
    config: ScanConfig,
) -> FinalizedScreen:
    """Recompute every active broad-screen statistic over one hypothesis set."""
    result = raw_results.drop(columns=list(DERIVED_SCREEN_COLUMNS), errors="ignore").copy()
    required = {"feature", "target", "raw_p"}
    if not required.issubset(result):
        raise ValueError(f"Screen results missing columns: {sorted(required - set(result))}")
    duplicated = result.duplicated(["feature", "target"], keep=False)
    if duplicated.any():
        pairs = result.loc[duplicated, ["feature", "target"]].to_dict("records")
        raise ValueError(f"Duplicate feature-target hypotheses: {pairs}")
    for column in ("raw_p", "top_bottom_spread", "monotonicity"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    features = registry_frame(feature_registry).rename(
        columns={"name": "feature", "family": "feature_family", "classification": "feature_classification"}
    )
    authoritative = [
        "feature", "feature_family", "feature_classification", "dtype",
        "discovery_phase", "arity", "operator", "parent_features",
        "redundancy_group", "lineage_hash", "complexity_units",
        "expected_direction",
        "evidence_class",
    ]
    features = features[authoritative]
    result = result.drop(columns=[column for column in authoritative if column != "feature" and column in result], errors="ignore")
    result = result.merge(features, on="feature", how="left", validate="many_to_one")
    missing_features = sorted(result.loc[result.feature_family.isna(), "feature"].unique())
    if missing_features:
        raise ValueError(f"Unknown result features: {missing_features}")

    targets = registry_frame(target_registry).rename(
        columns={"name": "target", "classification": "target_classification"}
    )
    target_columns = ["target", "tier", "target_classification", "horizon_minutes"]
    result = result.drop(columns=["target_tier", "target_classification", "horizon_minutes"], errors="ignore")
    result = result.merge(targets[target_columns], on="target", how="left", validate="many_to_one")
    result = result.rename(columns={"tier": "target_tier"})
    missing_targets = sorted(result.loc[result.target_tier.isna(), "target"].unique())
    if missing_targets:
        raise ValueError(f"Unknown result targets: {missing_targets}")
    result["target_family"] = result["target"].map(target_horizon_family)
    result["redundancy_group"] = result["redundancy_group"].fillna(result["feature"])

    result = finalize_screen(result)
    # finalize_screen preserves registry groups but can reorder rows.
    result["scan_kind"] = np.where(
        result.dtype.eq("binary"), "binary",
        np.where(result.dtype.eq("categorical"), "categorical", "continuous"),
    )
    primary = result.loc[result.target_tier.eq("primary")].copy()
    exploratory = result.loc[result.target_tier.eq("exploratory")].copy()
    primary["primary_global_fdr"] = benjamini_hochberg(primary.raw_p)
    primary["family_fdr"] = primary.groupby(
        ["feature_family", "target_family"], dropna=False
    ).raw_p.transform(benjamini_hochberg)
    primary["candidate_cluster"] = (
        primary.redundancy_group.fillna(primary.feature)
        + "__" + primary.target.map(target_horizon_family)
    )
    primary["cluster_fdr"] = primary.groupby(
        "candidate_cluster", dropna=False
    ).raw_p.transform(benjamini_hochberg)
    exploratory["exploratory_family_fdr"] = exploratory.groupby(
        ["feature_family", "target_family"], dropna=False
    ).raw_p.transform(benjamini_hochberg)

    for column in ("primary_global_fdr", "family_fdr", "candidate_cluster", "cluster_fdr"):
        result = result.merge(primary[["feature", "target", column]], on=["feature", "target"], how="left")
    result = result.merge(
        exploratory[["feature", "target", "exploratory_family_fdr"]],
        on=["feature", "target"], how="left",
    )
    result["primary_test_count"] = int(primary.raw_p.notna().sum())
    result["exploratory_test_count"] = int(exploratory.raw_p.notna().sum())
    result = result.sort_values(["feature", "target"], kind="mergesort").reset_index(drop=True)
    assert_valid_screen_results(result, "finalized combined screen", check_fdr=True)
    expected = {
        "primary_global_fdr", "family_fdr", "candidate_cluster", "cluster_fdr",
        "exploratory_family_fdr", "primary_test_count", "exploratory_test_count",
    }
    if not expected.issubset(result):
        raise RuntimeError(f"Finalized screen missing fields: {sorted(expected - set(result))}")
    return FinalizedScreen(result, primary, exploratory)
