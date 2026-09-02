"""Deterministic, exploratory Phase 1B systematic interaction registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_selection import target_horizon_family
from .config import ScanConfig
from .dual_registry import CompiledDualFeature, ConditionSpec, DualFeatureDefinition, TransformSpec, _spec
from .registry import FeatureSpec
from .scanner import benjamini_hochberg


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _number(frame: pd.DataFrame, names: tuple[str, ...], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


@dataclass(frozen=True)
class SystematicParent:
    feature: str
    feature_family: str
    redundancy_group: str | None
    best_primary_global_fdr: float
    best_absolute_effect: float
    best_anomaly_score: float
    valid_primary_target_count: int
    primary_target_family_count: int
    parent_score: float
    diversity_backfill: bool
    orientation: int


@dataclass(frozen=True)
class SystematicPair:
    feature_a: str
    feature_b: str
    family_a: str
    family_b: str
    redundancy_group_a: str | None
    redundancy_group_b: str | None
    parent_a_score: float
    parent_b_score: float
    sampled_parent_spearman: float
    joint_valid_observations: int
    joint_sessions: int
    joint_symbols: int
    joint_decision_timestamps: int
    pair_score: float
    orientation_a: int
    orientation_b: int


@dataclass(frozen=True)
class SystematicDualPlan:
    selected_parents: tuple[SystematicParent, ...]
    selected_pairs: tuple[SystematicPair, ...]
    compiled_features: tuple[CompiledDualFeature, ...]
    parent_selection_hash: str
    pair_selection_hash: str
    plan_hash: str
    parent_selection_ledger: tuple[dict[str, Any], ...]
    pair_selection_ledger: tuple[dict[str, Any], ...]


def select_systematic_parents(
    source_results: pd.DataFrame,
    source_features: list[FeatureSpec],
    feature_cache_index: dict[str, Path],
    config: ScanConfig,
    scan_coverage: pd.DataFrame | None = None,
) -> tuple[tuple[SystematicParent, ...], tuple[dict[str, Any], ...]]:
    cfg = config.systematic_phase1b.parent_selection
    metadata = {item.name: item for item in source_features}
    built = set(feature_cache_index)
    if scan_coverage is not None and not scan_coverage.empty:
        key = "feature" if "feature" in scan_coverage else "name"
        built &= set(scan_coverage.loc[scan_coverage.status.eq("built"), key].astype(str))
    rows = source_results.copy()
    if "target_tier" in rows:
        rows = rows.loc[rows.target_tier.eq("primary")]
    fdr = _number(rows, ("primary_global_fdr", "bh_fdr_p_global", "bh_fdr_p"))
    effect = _number(rows, ("top_bottom_spread", "binary_mean_difference"))
    observations = _number(rows, ("valid_observations", "n"), 0)
    sessions = _number(rows, ("valid_sessions", "sessions"), 0)
    symbols = _number(rows, ("valid_symbols", "symbols"), 0)
    decisions = _number(rows, ("valid_decision_timestamps",), 0)
    coverage_ok = observations.ge(cfg.minimum_valid_observations) & sessions.ge(cfg.minimum_sessions) & symbols.ge(cfg.minimum_symbols) & decisions.ge(cfg.minimum_decision_timestamps)
    valid = rows.loc[coverage_ok & fdr.notna() & effect.notna()].copy()
    valid["_fdr"] = fdr.loc[valid.index]
    valid["_effect"] = effect.loc[valid.index]
    valid["_target_family"] = valid.get("target_family", valid.target.map(target_horizon_family))
    summaries: dict[str, dict[str, Any]] = {}
    for feature, group in valid.groupby("feature", sort=True):
        qualifying=group.loc[group._fdr.le(cfg.primary_global_fdr_max)&group._effect.abs().ge(cfg.minimum_absolute_effect_bps/10_000)]
        orientation_pool=qualifying if not qualifying.empty else group
        signs = np.sign(orientation_pool.loc[orientation_pool._effect.abs().eq(orientation_pool._effect.abs().max()), "_effect"].dropna().to_numpy())
        orientation = int(signs[0]) if len(signs) and np.all(signs == signs[0]) else 0
        summaries[str(feature)] = {
            "best_primary_global_fdr": float(group._fdr.min()),
            "best_absolute_effect": float(group._effect.abs().max()),
            "best_anomaly_score": float(_number(group, ("anomaly_score",), 0).max()),
            "valid_primary_target_count": int(len(group)),
            "primary_target_family_count": int(group._target_family.nunique()),
            "orientation": orientation,
            "target_families": tuple(sorted(group._target_family.dropna().astype(str).unique())),
            "strict_qualifies": bool(len(qualifying)),
        }
    strict = {name for name, item in summaries.items() if item["strict_qualifies"]}
    backfill: set[str] = set()
    pool = pd.DataFrame([{"feature": name, "family": metadata[name].family if name in metadata else "unknown", **item} for name, item in summaries.items() if name in built and name in metadata])
    if not pool.empty:
        pool = pool.sort_values(["best_absolute_effect", "best_primary_global_fdr", "feature"], ascending=[False, True, True], kind="mergesort")
        remaining=pool.loc[~pool.feature.isin(strict)]
        for _, group in remaining.groupby("family", sort=True):
            backfill.update(group.head(cfg.include_top_per_family_when_threshold_not_met).feature)
        exploded = pool.explode("target_families")
        exploded=exploded.loc[~exploded.feature.isin(strict)]
        for _, group in exploded.groupby("target_families", sort=True):
            backfill.update(group.head(cfg.include_top_per_target_family_when_threshold_not_met).feature)
    candidates = sorted((strict | backfill) & built & set(metadata) & set(summaries))
    score_frame = pd.DataFrame([{"feature": name, **summaries[name]} for name in candidates])
    if not score_frame.empty:
        significance = -np.log10(score_frame.best_primary_global_fdr.clip(lower=1e-300))
        score_frame["parent_score"] = .40 * significance.rank(pct=True) + .30 * score_frame.best_absolute_effect.rank(pct=True) + .20 * score_frame.best_anomaly_score.fillna(0).rank(pct=True) + .10 * score_frame.primary_target_family_count.rank(pct=True)
        score_map = score_frame.set_index("feature").parent_score.to_dict()
    else:
        score_map = {}
    ranked = sorted(candidates, key=lambda name: (-score_map[name], name))
    family_counts: dict[str, int] = {}; redundancy_counts: dict[str, int] = {}; selected: list[SystematicParent] = []; reasons: dict[str, str] = {}
    for name in ranked:
        spec = metadata[name]; redundancy = spec.redundancy_group
        if family_counts.get(spec.family, 0) >= cfg.max_per_feature_family:
            reasons[name] = "feature_family_cap"; continue
        if redundancy and redundancy_counts.get(redundancy, 0) >= cfg.max_per_redundancy_group:
            reasons[name] = "redundancy_group_cap"; continue
        if len(selected) >= cfg.max_parent_features:
            reasons[name] = "total_parent_cap"; continue
        item = summaries[name]
        selected.append(SystematicParent(name, spec.family, redundancy, item["best_primary_global_fdr"], item["best_absolute_effect"], item["best_anomaly_score"], item["valid_primary_target_count"], item["primary_target_family_count"], float(score_map[name]), name not in strict, item["orientation"]))
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
        if redundancy: redundancy_counts[redundancy] = redundancy_counts.get(redundancy, 0) + 1
    selected_names = {item.feature for item in selected}
    ledger = []
    for name in sorted(set(metadata) | set(source_results.feature.astype(str))):
        spec = metadata.get(name)
        summary = summaries.get(name, {})
        if name not in feature_cache_index: reason = "missing_feature_cache"
        elif name not in built: reason = "not_marked_built"
        elif name not in summaries: reason = "insufficient_primary_coverage_or_no_valid_primary_result"
        elif name not in candidates: reason = "fdr_or_effect_threshold"
        else: reason = reasons.get(name, "")
        ledger.append({"feature": name, "feature_family": spec.family if spec else None, "redundancy_group": spec.redundancy_group if spec else None, **{k: summary.get(k) for k in ("best_primary_global_fdr", "best_absolute_effect", "best_anomaly_score", "valid_primary_target_count", "primary_target_family_count")}, "parent_score": score_map.get(name), "diversity_backfill": name in backfill and name not in strict, "selected": name in selected_names, "rejection_reason": "" if name in selected_names else reason})
    return tuple(selected), tuple(ledger)


def _load_pair_inputs(parents: tuple[SystematicParent, ...], paths: dict[str, Path]):
    """Load only validity masks and a deterministic correlation sample.

    Pair construction never needs all parent values resident as float64.  The
    dense boolean matrix is about one eighth the size and identical missingness
    patterns share one cached joint-coverage calculation.
    """
    ordered=tuple(sorted(parents,key=lambda item:item.feature));first=paths[ordered[0].feature]
    keys=pd.read_parquet(first,columns=["session_date","decision_ts","symbol","analysis_eligible"])
    eligible=keys.analysis_eligible.fillna(False).to_numpy(dtype=bool,copy=False);n=len(keys)
    codes=[]
    for column in ("session_date","symbol","decision_ts"):
        code,_=pd.factorize(keys[column],sort=False);codes.append(code.astype(np.int32,copy=False))
    del keys
    sample_index=np.arange(0,n,max(1,n//20_000),dtype=np.int64)[:20_000]
    valid=np.empty((len(ordered),n),dtype=bool);sample=np.full((len(ordered),len(sample_index)),np.nan,dtype=np.float64)
    position={parent.feature:index for index,parent in enumerate(ordered)};by_path:dict[Path,list[str]]={}
    for parent in ordered:by_path.setdefault(paths[parent.feature],[]).append(parent.feature)
    for path,names in sorted(by_path.items(),key=lambda item:str(item[0])):
        frame=pd.read_parquet(path,columns=sorted(names))
        for name in sorted(names):
            values=pd.to_numeric(frame[name],errors="coerce").to_numpy(copy=False);index=position[name]
            valid[index]=eligible&np.isfinite(values);sample[index]=values[sample_index]
        del frame
    packed=np.packbits(valid,axis=1)
    mask_ids=[hashlib.sha1(packed[index].tobytes()).hexdigest() for index in range(len(ordered))]
    return ordered,position,valid,sample,tuple(codes),mask_ids


def generate_systematic_pairs(parents: tuple[SystematicParent, ...], feature_cache_index: dict[str, Path], config: ScanConfig) -> tuple[tuple[SystematicPair, ...], tuple[dict[str, Any], ...]]:
    cfg = config.systematic_phase1b.pair_generation
    if len(parents)<2:return (),()
    ordered,position,valid_matrix,samples,group_codes,mask_ids=_load_pair_inputs(parents,feature_cache_index)
    coverage_cache:dict[tuple[str,str],tuple[int,int,int,int]]={};accepted: list[SystematicPair] = []; ledger: list[dict[str, Any]] = []
    for left, right in combinations(ordered, 2):
        common = {"feature_a": left.feature, "feature_b": right.feature, "family_a": left.feature_family, "family_b": right.feature_family, "redundancy_group_a": left.redundancy_group, "redundancy_group_b": right.redundancy_group, "parent_a_score": left.parent_score, "parent_b_score": right.parent_score}
        if cfg.forbid_same_redundancy_group and left.redundancy_group and left.redundancy_group == right.redundancy_group:
            ledger.append({**common, "selected": False, "rejection_reason": "same_redundancy_group"}); continue
        ia,ib=position[left.feature],position[right.feature];av,bv=samples[ia],samples[ib];sample_valid=np.isfinite(av)&np.isfinite(bv)
        corr=float(pd.Series(av[sample_valid]).corr(pd.Series(bv[sample_valid]),method="spearman")) if sample_valid.sum()>1 else np.nan
        mask_key=tuple(sorted((mask_ids[ia],mask_ids[ib])))
        if mask_key not in coverage_cache:
            joint=valid_matrix[ia]&valid_matrix[ib];observations=int(np.count_nonzero(joint))
            counts=[]
            for codes in group_codes:
                counts.append(int(np.count_nonzero(np.bincount(codes[joint],minlength=int(codes.max())+1))))
            coverage_cache[mask_key]=(observations,*counts)
        observations,sessions,symbols,decisions=coverage_cache[mask_key]
        values = {**common, "sampled_parent_spearman": corr, "joint_valid_observations": observations, "joint_sessions": sessions, "joint_symbols": symbols, "joint_decision_timestamps": decisions}
        if np.isfinite(corr) and abs(corr) >= cfg.maximum_absolute_parent_spearman: reason = "parent_correlation"
        elif observations < cfg.minimum_joint_observations: reason = "insufficient_joint_observations"
        elif sessions < cfg.minimum_joint_sessions: reason = "insufficient_joint_sessions"
        elif symbols < cfg.minimum_joint_symbols: reason = "insufficient_joint_symbols"
        elif decisions < cfg.minimum_joint_decision_timestamps: reason = "insufficient_joint_decision_timestamps"
        else: reason = ""
        score = .40 * left.parent_score + .40 * right.parent_score + .20 * (1 - abs(corr) if np.isfinite(corr) else 0)
        values["pair_score"] = score
        if reason: ledger.append({**values, "selected": False, "rejection_reason": reason}); continue
        accepted.append(SystematicPair(**{k: values[k] for k in SystematicPair.__dataclass_fields__ if k in values}, orientation_a=left.orientation, orientation_b=right.orientation))
    accepted.sort(key=lambda item: (-item.pair_score, item.feature_a, item.feature_b))
    counts: dict[tuple[str, str], int] = {}; selected=[]
    for item in accepted:
        bucket=tuple(sorted((item.family_a,item.family_b)))
        reason=""
        if counts.get(bucket,0)>=cfg.maximum_pairs_per_family_pair:reason="family_pair_cap"
        elif len(selected)>=cfg.max_parent_pairs:reason="total_pair_cap"
        else:selected.append(item);counts[bucket]=counts.get(bucket,0)+1
        values={key:value for key,value in asdict(item).items() if key not in {"orientation_a","orientation_b"}};ledger.append({**values,"selected":not reason,"rejection_reason":reason})
    ledger.sort(key=lambda row:(row["feature_a"],row["feature_b"]))
    return tuple(selected),tuple(ledger)


def _compiled(definition: DualFeatureDefinition, parents: tuple[FeatureSpec, FeatureSpec], pair: SystematicPair) -> CompiledDualFeature:
    spec=_spec(definition,parents)
    lineage=spec.lineage_hash or ""
    name=f"dual_auto__{definition.operator}__{pair.feature_a}__{pair.feature_b}__{lineage[:8]}"
    spec=replace(spec,name=name,family="dual_factor_systematic",discovery_phase="1B_systematic",evidence_class="systematic_generated_interaction",redundancy_group=f"dual_auto::{pair.feature_a}::{pair.feature_b}")
    return CompiledDualFeature(definition,spec)


def compile_systematic_dual_plan(source_results: pd.DataFrame, source_features: list[FeatureSpec], feature_cache_index: dict[str, Path], config: ScanConfig, scan_coverage: pd.DataFrame | None = None) -> SystematicDualPlan:
    parents,parent_ledger=select_systematic_parents(source_results,source_features,feature_cache_index,config,scan_coverage)
    pairs,pair_ledger=generate_systematic_pairs(parents,feature_cache_index,config)
    parent_specs={item.name:item for item in source_features}; compiled=[]; operators=config.systematic_phase1b.operators
    for pair in pairs:
        specs=(parent_specs[pair.feature_a],parent_specs[pair.feature_b])
        if operators.aligned_rank_mean and pair.orientation_a and pair.orientation_b:
            definition=DualFeatureDefinition(f"auto_aligned__{pair.feature_a}__{pair.feature_b}",pair.feature_a,pair.feature_b,"aligned_rank_mean",transform_a=TransformSpec("cross_sectional_rank",pair.orientation_a),transform_b=TransformSpec("cross_sectional_rank",pair.orientation_b),output_dtype="continuous",expected_direction=1)
            compiled.append(_compiled(definition,specs,pair))
        if operators.directional_intersection and pair.orientation_a and pair.orientation_b:
            a_strong=pair.parent_a_score>=pair.parent_b_score
            def condition(orientation:int,strong:bool)->ConditionSpec:return ConditionSpec("cross_sectional_rank","ge" if orientation>0 else "le",.90 if strong and orientation>0 else .10 if strong else .75 if orientation>0 else .25)
            definition=DualFeatureDefinition(f"auto_intersection__{pair.feature_a}__{pair.feature_b}",pair.feature_a,pair.feature_b,"intersection",condition_a=condition(pair.orientation_a,a_strong),condition_b=condition(pair.orientation_b,not a_strong),output_dtype="binary",expected_direction=1)
            compiled.append(_compiled(definition,specs,pair))
        if operators.gated_anchor:
            anchor_a=pair.parent_a_score>=pair.parent_b_score
            anchor,gate=(pair.feature_a,pair.feature_b) if anchor_a else (pair.feature_b,pair.feature_a)
            anchor_o,gate_o=(pair.orientation_a,pair.orientation_b) if anchor_a else (pair.orientation_b,pair.orientation_a)
            if anchor_o and gate_o:
                ordered_specs=(parent_specs[anchor],parent_specs[gate])
                gate_condition=ConditionSpec("cross_sectional_rank","ge" if gate_o>0 else "le",.75 if gate_o>0 else .25)
                definition=DualFeatureDefinition(f"auto_gate__{anchor}__{gate}",anchor,gate,"gated_anchor",condition_b=gate_condition,transform_a=TransformSpec("raw",anchor_o),output_dtype="continuous",expected_direction=anchor_o)
                compiled.append(_compiled(definition,ordered_specs,pair))
    compiled=compiled[:config.systematic_phase1b.limits.max_generated_features]
    parent_hash=_hash(parent_ledger);pair_hash=_hash(pair_ledger);plan_hash=_hash([item.spec for item in compiled])
    return SystematicDualPlan(parents,pairs,tuple(compiled),parent_hash,pair_hash,plan_hash,parent_ledger,pair_ledger)


def finalize_systematic_results(results: pd.DataFrame, specs: list[FeatureSpec], source_features: list[FeatureSpec] | None = None) -> pd.DataFrame:
    out=results.copy(); metadata={item.name:item for item in specs};families={item.name:item.family for item in (source_features or [])}
    out["target_tier"]="exploratory";out["exploratory_family"]="phase1b_systematic_interactions";out["discovery_phase"]="1B_systematic";out["evidence_class"]="systematic_generated_interaction"
    out["operator"]=out.feature.map(lambda name:metadata[name].operator);out["redundancy_group"]=out.feature.map(lambda name:metadata[name].redundancy_group)
    out["parent_family_pair"]=out.feature.map(lambda name:"::".join(sorted(families.get(parent,parent) for parent in metadata[name].parent_features)))
    valid=pd.to_numeric(out.raw_p,errors="coerce").between(0,1,inclusive="both")
    out["systematic_test_count"]=int(valid.sum());out["systematic_global_fdr"]=np.nan
    out.loc[valid,"systematic_global_fdr"]=benjamini_hochberg(pd.to_numeric(out.loc[valid,"raw_p"]))
    for group,column in (("operator","systematic_operator_fdr"),("parent_family_pair","systematic_family_pair_fdr"),("redundancy_group","systematic_pair_cluster_fdr"),("target_family","systematic_target_family_fdr")):
        out[column]=np.nan
        out.loc[valid,column]=out.loc[valid].groupby(group,dropna=False).raw_p.transform(benjamini_hochberg)
    return out


def classify_systematic_candidate(row: pd.Series | dict[str, Any], config: ScanConfig) -> str:
    value=dict(row);cfg=config.systematic_phase1b.promotion
    required=("systematic_global_fdr","two_way_cluster_p","top_bottom_spread","best_parent_absolute_effect","historical_subperiod_positive_fold_pct","historical_subperiod_worst_signed_effect","recent_to_full_effect_ratio","eligible_symbols_expected_direction_pct","symbol_effect_hhi","top5_symbol_effect_pct","effect_remove_top5_symbols")
    if any(not np.isfinite(pd.to_numeric(value.get(name),errors="coerce")) for name in required):return "systematic_interaction_requires_more_diagnostics"
    effect=abs(float(value["top_bottom_spread"]));parent=abs(float(value["best_parent_absolute_effect"]));increment=effect-parent
    tolerance=1e-15
    statistical=float(value["systematic_global_fdr"])<=.05 and float(value["two_way_cluster_p"])<=.05 and effect+tolerance>=cfg.minimum_absolute_effect_bps/10_000
    incremental=effect+tolerance>=cfg.minimum_effect_ratio_vs_best_parent*parent and increment+tolerance>=cfg.minimum_effect_increment_bps_vs_best_parent/10_000
    recent_ratio=float(value["recent_to_full_effect_ratio"])
    if recent_ratio>cfg.maximum_recent_to_full_effect_ratio:return "recently_emerged_high_instability"
    robust=(float(value["historical_subperiod_positive_fold_pct"])+tolerance>=cfg.minimum_positive_historical_fold_fraction and float(value["historical_subperiod_worst_signed_effect"])>0 and recent_ratio+tolerance>=cfg.minimum_recent_to_full_effect_ratio and float(value["eligible_symbols_expected_direction_pct"])+tolerance>=cfg.minimum_expected_direction_symbol_fraction and float(value["symbol_effect_hhi"])<=cfg.maximum_symbol_effect_hhi+tolerance and float(value["top5_symbol_effect_pct"])<=cfg.maximum_top5_symbol_effect_share+tolerance and float(value["effect_remove_top5_symbols"])+tolerance>=cfg.minimum_remove_top5_effect_retention*effect)
    if statistical and robust and not incremental:return "significant_but_not_incremental_to_parent"
    return "exploratory_generated_interaction_candidate" if statistical and incremental and robust else "systematic_interaction_not_robust"
