"""Phase 1B-only source-run validation and combined-FDR primitives.

This module deliberately treats Phase 1A caches as immutable inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .cache import assert_cache_key_alignment, validate_cache
from .config import ScanConfig
from .dual_features import build_dual_feature_chunks
from .dual_registry import compile_dual_plan, plan_hash
from .fingerprint import enforce_fingerprint, phase1b_run_fingerprint
from .holdout import assert_pre_holdout_parquet
from .registry import (
    FeatureSpec, TargetSpec, load_feature_registry, load_target_registry,
    registry_frame, write_registry_json,
)
from .screen_finalization import finalize_phase1_screen
from .screen_orchestration import screen_feature_blocks_against_target_blocks
from .bulk_scan import assert_valid_screen_results
from .candidate_selection import select_exact_candidate_queue
from .systematic_dual_registry import (
    classify_systematic_candidate, compile_systematic_dual_plan,
    finalize_systematic_results,
)


@dataclass(frozen=True)
class Phase1ASource:
    root: Path
    manifest: dict
    fingerprint: dict
    feature_registry: list[FeatureSpec]
    target_registry: list[TargetSpec]
    feature_cache_index: dict[str,Path]
    target_cache_index: dict[str,Path]
    master_results_path: Path
    source_manifest_hash: str
    tree_hash: str
    source_results_hash: str


def _index(paths:list[Path], names:set[str]) -> dict[str,Path]:
    import pyarrow.parquet as pq
    result={}
    for path in paths:
        for name in set(pq.ParquetFile(path).schema.names)&names:
            if name in result:raise RuntimeError(f"Cache column has ambiguous ownership: {name}")
            result[name]=path
    return result


def _tree_hash(root:Path)->str:
    digest=hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).replace("\\","/").encode()); digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _atomic_csv(frame:pd.DataFrame,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp");frame.to_csv(temporary,index=False);temporary.replace(path)


def _atomic_json(payload:dict,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8");temporary.replace(path)


def validate_phase1a_source(source_run: str | Path, config: ScanConfig) -> Phase1ASource:
    root=Path(source_run)
    manifest_path=root/"manifest.json"; fingerprint_path=root/"fingerprint.json"
    progress_path=root/"progress.json"; results_path=root/"master_results.csv"; feature_registry_path=root/"feature_registry.csv"; target_registry_path=root/"target_registry.csv"; coverage_path=root/"scan_coverage.csv"
    if not all(path.exists() for path in (manifest_path,fingerprint_path,progress_path,results_path,feature_registry_path,target_registry_path,coverage_path)):raise FileNotFoundError("Source Phase 1A is incomplete")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); fingerprint=json.loads(fingerprint_path.read_text(encoding="utf-8"))
    progress=json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("stage")!="complete":raise ValueError("Source Phase 1A run is not complete")
    if manifest.get("allow_holdout_access") or manifest.get("holdout_access") or manifest.get("sealed_holdout_start")!=config.sealed_holdout_start or manifest.get("discovery_end")!=config.discovery_end:raise ValueError("Source run has incompatible discovery or holdout boundaries")
    source_hash=fingerprint.get("sha256")
    if not source_hash:raise ValueError("Source fingerprint is missing its digest")
    feature_paths=sorted((root/"blocks"/"features").glob("*.parquet")); target_paths=sorted((root/"blocks"/"targets").glob("*.parquet"))
    if not feature_paths or not target_paths:raise FileNotFoundError("Source feature or target caches are missing")
    for path in [*feature_paths,*target_paths]:
        validate_cache(path,source_hash,config.sealed_holdout_start); assert_pre_holdout_parquet(path,config.sealed_holdout_start,"phase1b source validation",verify_key_rows=False)
    features=load_feature_registry(feature_registry_path); targets=load_target_registry(target_registry_path)
    source_results=pd.read_csv(results_path)
    required={"feature","target","raw_p"}
    if not required.issubset(source_results):raise ValueError(f"Source master results missing columns: {sorted(required-set(source_results))}")
    if source_results.duplicated(["feature","target"]).any():raise ValueError("Source master results contain duplicate hypotheses")
    assert_valid_screen_results(source_results,"source Phase 1A master results",check_fdr=True)
    unknown_result_features=sorted(set(source_results.feature)-{item.name for item in features})
    unknown_result_targets=sorted(set(source_results.target)-{item.name for item in targets})
    if unknown_result_features:raise ValueError(f"Source results contain unknown features: {unknown_result_features}")
    if unknown_result_targets:raise ValueError(f"Source results contain unknown targets: {unknown_result_targets}")
    feature_index=_index(feature_paths,{item.name for item in features}); target_index=_index(target_paths,{item.name for item in targets})
    coverage=pd.read_csv(coverage_path)
    feature_column="feature" if "feature" in coverage else "name" if "name" in coverage else None
    if feature_column is None or "status" not in coverage:raise ValueError("Source scan_coverage.csv is missing feature/name or status")
    built_features=set(coverage.loc[coverage.status.eq("built"),feature_column].dropna())
    missing_features=sorted(built_features-set(feature_index))
    missing_targets=sorted({item.name for item in targets}-set(target_index))
    if missing_features:raise FileNotFoundError(f"Features marked built but missing from the cache index: {missing_features}")
    if missing_targets:raise FileNotFoundError(f"Source registry targets missing from caches: {missing_targets}")
    anchor=target_paths[0]
    for path in [*feature_paths,*target_paths[1:]]:assert_cache_key_alignment(path,anchor)
    return Phase1ASource(root,manifest,fingerprint,features,targets,feature_index,target_index,results_path,hashlib.sha256(manifest_path.read_bytes()).hexdigest(),_tree_hash(root),hashlib.sha256(results_path.read_bytes()).hexdigest())


def merge_combined_results(base: pd.DataFrame, dual: pd.DataFrame, targets: list[TargetSpec] | None=None, features:list[FeatureSpec]|None=None, config:ScanConfig|None=None) -> pd.DataFrame:
    base=base.copy();dual=dual.copy()
    if "discovery_phase" not in base:base["discovery_phase"]="1A"
    if "arity" not in base:base["arity"]=1
    result=pd.concat([base,dual],ignore_index=True)
    if result.duplicated(["feature","target"]).any():raise ValueError("Duplicate feature-target rows in combined Phase 1 results")
    if targets is None:return result
    if features is None:
        features=[]
        for row in result.drop_duplicates("feature").itertuples():
            features.append(FeatureSpec(name=row.feature,description="persisted screen feature",family=getattr(row,"feature_family","unknown"),discovery_phase=getattr(row,"discovery_phase","1A") if pd.notna(getattr(row,"discovery_phase","1A")) else "1A",arity=int(getattr(row,"arity",1)) if pd.notna(getattr(row,"arity",1)) else 1,redundancy_group=getattr(row,"redundancy_group",None)))
    return finalize_phase1_screen(result,feature_registry=features,target_registry=targets,config=config or ScanConfig()).master


def _parent_comparison(combined:pd.DataFrame,compiled)->pd.DataFrame:
    rows=[]
    lookup=combined.set_index(["feature","target"])
    for item in compiled:
        for target in combined.loc[combined.feature.eq(item.spec.name),"target"]:
            dual=lookup.loc[(item.spec.name,target)]
            parents=[]
            for parent in item.spec.parent_features:
                parents.append(lookup.loc[(parent,target)] if (parent,target) in lookup.index else pd.Series(dtype=object))
            if len(parents)!=2:continue
            de=float(dual.get("top_bottom_spread",float("nan"))); effects=[float(parent.get("top_bottom_spread",float("nan"))) for parent in parents]
            finite=[abs(value) for value in effects if pd.notna(value)]; best=max(finite) if finite else float("nan")
            increment=abs(de)-best if pd.notna(de) and pd.notna(best) else float("nan")
            ratio=abs(de)/best if pd.notna(de) and best else float("nan")
            dual_fdr=dual.get("systematic_global_fdr",dual.get("primary_global_fdr"))
            rows.append({"dual_feature":item.spec.name,"operator":item.spec.operator,"evidence_class":item.spec.evidence_class,"target":target,"parent_a":item.spec.parent_features[0],"parent_b":item.spec.parent_features[1],"dual_effect":de,"parent_a_effect":effects[0],"parent_b_effect":effects[1],"best_parent_absolute_effect":best,"dual_incremental_effect":increment,"dual_to_best_parent_effect_ratio":ratio,"dual_fdr":dual_fdr,"parent_a_fdr":parents[0].get("primary_global_fdr"),"parent_b_fdr":parents[1].get("primary_global_fdr"),"same_direction_as_parent_a":bool(pd.notna(de) and pd.notna(effects[0]) and de*effects[0]>0),"same_direction_as_parent_b":bool(pd.notna(de) and pd.notna(effects[1]) and de*effects[1]>0),"incremental_gate_passed":bool(pd.notna(increment) and pd.notna(ratio) and increment>=.000025 and ratio>=1.25)})
    return pd.DataFrame(rows)


def _run_dual_exact(
    source:Phase1ASource,
    combined:pd.DataFrame,
    dual_paths:list[Path],
    chunks:list[list[FeatureSpec]],
    config:ScanConfig,
    root:Path,
    candidate_queue:pd.DataFrame,
) -> pd.DataFrame:
    """Run authoritative CPU exact diagnostics for newly introduced features."""
    from .exact_parallel import exact_pair
    dual_path_by_name={spec.name:path for path,specs in zip(dual_paths,chunks) for spec in specs}
    specs={spec.name:spec for specs_chunk in chunks for spec in specs_chunk}
    queue=candidate_queue.loc[candidate_queue.discovery_phase.astype(str).str.startswith("1B")]
    exact_root=root/"exact_journal"/"cpu";exact_root.mkdir(parents=True,exist_ok=True)
    rows=[];diagnostic_root=root/"candidate_diagnostics";diagnostic_root.mkdir(exist_ok=True)
    for queued in queue.itertuples(index=False):
        key=(queued.feature,queued.target);digest=hashlib.sha1("\0".join(key).encode()).hexdigest();journal=exact_root/f"{digest}.pkl"
        if journal.exists():
            payload=pd.read_pickle(journal)
            if tuple(payload.get("key",()))!=key:raise RuntimeError(f"Exact journal key mismatch: {journal}")
        else:payload={}
        if not payload or (specs[queued.feature].dtype=="binary" and payload.get("diagnostics_version",1)<2):
            row,table,diagnostics=exact_pair(dual_path_by_name[queued.feature],source.target_cache_index[queued.target],specs[queued.feature],queued.target,config,float(getattr(queued,"spearman",0) or 0),None)
            payload={"key":key,"row":row,"table":table,"diagnostics":diagnostics,"diagnostics_version":2};temporary=journal.with_suffix(".pkl.tmp");pd.to_pickle(payload,temporary);temporary.replace(journal)
        row=payload.get("row")
        if row is None:continue
        row=dict(row);row.update({"feature":queued.feature,"target":queued.target,"screen_bh_fdr_p_global":queued.primary_global_fdr,"primary_global_fdr":queued.primary_global_fdr,"family_fdr":getattr(queued,"family_fdr",float("nan")),"cluster_fdr":getattr(queued,"cluster_fdr",float("nan")),"candidate_cluster":queued.candidate_cluster,"target_tier":getattr(queued,"target_tier","primary"),"discovery_phase":queued.discovery_phase,"evidence_class":getattr(queued,"evidence_class",None),"systematic_global_fdr":getattr(queued,"systematic_global_fdr",float("nan")),"promotion_inference":"exact_cpu_two_way_date_symbol"});rows.append(row)
        stem=f"{queued.feature}__{queued.target}".replace("/","_")
        for name,records in payload.get("diagnostics",{}).items():
            if records:_atomic_csv(pd.DataFrame(records),diagnostic_root/f"{stem}__{name}.csv")
    detailed=pd.DataFrame(rows)
    source_detail=source.root/"detailed_candidates.csv"
    if source_detail.exists():
        prior=pd.read_csv(source_detail)
        refresh=combined.copy()
        for column in ("primary_global_fdr","family_fdr","cluster_fdr","candidate_cluster"):
            if column not in refresh:refresh[column]=pd.NA
        refresh=refresh[["feature","target","primary_global_fdr","family_fdr","cluster_fdr","candidate_cluster"]]
        prior=prior.drop(columns=[column for column in refresh.columns if column not in {"feature","target"} and column in prior],errors="ignore").merge(refresh,on=["feature","target"],how="inner")
        prior["discovery_phase"]="1A";prior["screen_bh_fdr_p_global"]=prior.primary_global_fdr
        detailed=pd.concat([prior,detailed],ignore_index=True,sort=False)
    if not detailed.empty:
        from .run import _classify_detailed_candidates
        detailed=_classify_detailed_candidates(detailed,config)
        detailed=detailed.sort_values(["primary_global_fdr","feature","target"],kind="mergesort")
    _atomic_csv(detailed if not detailed.empty else pd.DataFrame(columns=["feature","target","status"]),root/"detailed_candidates.csv")
    return detailed


def run_phase1b(source_run: str | Path, config: ScanConfig) -> Path:
    """Build curated and/or systematic factors from immutable Phase 1A caches."""
    if not config.dual_factor_enabled:raise ValueError("Phase 1B-only execution requires dual_factor_enabled=true")
    source=validate_phase1a_source(source_run,config);base=pd.read_csv(source.master_results_path);coverage_source=pd.read_csv(source.root/"scan_coverage.csv")
    run_curated=config.phase1b_mode in {"curated_only","curated_plus_systematic"};run_systematic=config.phase1b_mode in {"systematic_only","curated_plus_systematic"}
    if run_systematic and not config.systematic_phase1b.enabled:raise ValueError("Systematic Phase 1B mode requires systematic_phase1b.enabled=true")
    manifest_path=Path(config.dual_factor_manifest_path or "configs/phase1b_dual_factors.yaml")
    if not manifest_path.exists():manifest_path=Path(__file__).resolve().parents[2]/manifest_path
    curated=compile_dual_plan(manifest_path,source.feature_registry,config) if run_curated else []
    systematic=compile_systematic_dual_plan(base,source.feature_registry,source.feature_cache_index,config,coverage_source) if run_systematic else None
    generated=list(systematic.compiled_features) if systematic else []
    all_compiled=[*curated,*generated]
    missing=sorted({parent for item in all_compiled for parent in item.spec.parent_features if parent not in source.feature_cache_index})
    if missing:raise FileNotFoundError(f"Source caches lack required dual parents: {missing}")
    root=(Path(config.output_root)/config.experiment_id).resolve();source_root=source.root.resolve()
    if root==source_root or source_root in root.parents or root in source_root.parents:raise ValueError("Phase 1B output must be separate from the immutable Phase 1A source tree")
    extra={"phase1b_mode":config.phase1b_mode,"source_fingerprint":source.fingerprint["sha256"],"source_manifest_hash":source.source_manifest_hash,"source_results_hash":source.source_results_hash,"source_feature_registry_hash":hashlib.sha256((source.root/"feature_registry.csv").read_bytes()).hexdigest(),"source_tree_hash":source.tree_hash,"curated_manifest_hash":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"curated_plan_hash":plan_hash(curated),"systematic_parent_hash":systematic.parent_selection_hash if systematic else None,"systematic_pair_hash":systematic.pair_selection_hash if systematic else None,"systematic_plan_hash":systematic.plan_hash if systematic else None}
    combined_registry=source.feature_registry+[item.spec for item in all_compiled]
    fingerprint=phase1b_run_fingerprint(config,combined_registry,source.target_registry,extra);enforce_fingerprint(root,fingerprint,config.resume);root.mkdir(parents=True,exist_ok=True)
    phase=root/"phase1b";curated_root=phase/"curated";systematic_root=phase/"systematic";combined_root=phase/"combined"
    for path in (phase,curated_root,systematic_root,combined_root):path.mkdir(exist_ok=True)
    _atomic_json({"source_run":str(source.root),"fingerprint":source.fingerprint["sha256"],"manifest_sha256":source.source_manifest_hash,"master_results_sha256":source.source_results_hash,"tree_sha256":source.tree_hash},phase/"source_artifact_hashes.json")

    def build_and_screen(compiled,build_config,output,cache_name,primary_only=False):
        if not compiled:return [],[],[],pd.DataFrame()
        _atomic_json({"plan_hash":plan_hash(list(compiled)),"features":[{"definition":item.definition,"spec":item.spec} for item in compiled]},output/"compiled_feature_plan.json")
        coverage_path=output/"feature_coverage.csv"
        resume_coverage=pd.read_csv(coverage_path) if config.resume and coverage_path.exists() else None
        paths,chunks,feature_coverage=build_dual_feature_chunks(list(compiled),source.feature_cache_index,build_config,root/cache_name,fingerprint["sha256"],resume_coverage)
        _atomic_csv(pd.DataFrame(feature_coverage),output/"feature_coverage.csv");_atomic_csv(registry_frame([item.spec for item in compiled]),output/"feature_registry.csv");write_registry_json([item.spec for item in compiled],output/"feature_registry.json")
        if compiled and not sum(map(len,chunks)):raise RuntimeError(f"{output.name} compiled features, but none passed construction and coverage")
        targets=[item for item in source.target_registry if not primary_only or item.tier=="primary"]
        target_paths=sorted({source.target_cache_index[item.name] for item in targets},key=str);target_chunks=[[item for item in targets if source.target_cache_index[item.name]==path] for path in target_paths]
        screened=screen_feature_blocks_against_target_blocks(feature_paths=paths,feature_specs_by_path=chunks,target_paths=target_paths,target_specs_by_path=target_chunks,config=build_config,run_root=root/output.name)
        if sum(map(len,chunks)) and not pd.to_numeric(screened.get("raw_p",pd.Series(dtype=float)),errors="coerce").notna().any():raise RuntimeError(f"{output.name} features were built, but no feature-target pair produced valid statistical inference")
        return paths,chunks,feature_coverage,screened

    if run_curated:
        copy=curated_root/"manifest_source.yaml";tmp=copy.with_suffix(".yaml.tmp");tmp.write_bytes(manifest_path.read_bytes());tmp.replace(copy)
    curated_paths,curated_chunks,curated_coverage,curated_results=build_and_screen(curated,config,curated_root,"phase1b_curated_features")
    if run_curated:
        _atomic_csv(curated_results.sort_values(["feature","target"]) if not curated_results.empty else pd.DataFrame(columns=["feature","target","raw_p"]),curated_root/"dual_screen_results.csv")
        for generic,required in (("feature_coverage.csv","dual_feature_coverage.csv"),("feature_registry.csv","dual_feature_registry.csv"),("feature_registry.json","dual_feature_registry.json")):
            origin=curated_root/generic
            if origin.exists():origin.replace(curated_root/required)
    systematic_paths=[];systematic_chunks=[];systematic_coverage=[];systematic_results=pd.DataFrame()
    if systematic:
        _atomic_json(config.as_dict()["systematic_phase1b"],systematic_root/"systematic_config.json");_atomic_csv(pd.DataFrame(systematic.parent_selection_ledger),systematic_root/"parent_selection.csv");_atomic_csv(pd.DataFrame(systematic.pair_selection_ledger),systematic_root/"parent_pairs.csv")
        binary=config.systematic_phase1b.binary_state_coverage;limits=config.systematic_phase1b.limits
        systematic_config=replace(config,dual_factor_feature_chunk_size=limits.feature_chunk_size,dual_factor_max_generated_features=limits.max_generated_features,dual_factor_min_signal_observations=binary.minimum_on_observations,dual_factor_min_signal_sessions=binary.minimum_on_sessions,dual_factor_min_signal_symbols=binary.minimum_on_symbols,dual_factor_min_activation_rate=binary.minimum_activation_rate,dual_factor_max_activation_rate=binary.maximum_activation_rate,binary_min_on_observations=binary.minimum_on_observations,binary_min_off_observations=binary.minimum_off_observations,binary_min_on_sessions=binary.minimum_on_sessions,binary_min_off_sessions=binary.minimum_off_sessions,binary_min_on_symbols=binary.minimum_on_symbols,binary_min_off_symbols=binary.minimum_off_symbols)
        systematic_paths,systematic_chunks,systematic_coverage,raw=build_and_screen(generated,systematic_config,systematic_root,"phase1b_systematic_features",config.systematic_phase1b.screening.primary_targets_only)
        _atomic_json({"parent_selection_hash":systematic.parent_selection_hash,"pair_selection_hash":systematic.pair_selection_hash,"plan_hash":systematic.plan_hash,"features":[{"definition":item.definition,"spec":item.spec} for item in generated]},systematic_root/"compiled_feature_plan.json")
        systematic_results=finalize_systematic_results(raw,[item.spec for item in generated],source.feature_registry) if not raw.empty else raw
        _atomic_csv(systematic_results.sort_values(["feature","target"]),systematic_root/"screen_results.csv")

    curated_combined=merge_combined_results(base,curated_results,source.target_registry,source.feature_registry+[item.spec for item in curated],config) if run_curated else base.copy()
    combined=pd.concat([curated_combined,systematic_results],ignore_index=True,sort=False)
    if combined.duplicated(["feature","target"]).any():raise RuntimeError("Duplicate hypotheses in Phase 1B combined results")
    _atomic_csv(combined,root/"master_results.csv");_atomic_csv(registry_frame(combined_registry),root/"feature_registry.csv");_atomic_csv(registry_frame(source.target_registry),root/"target_registry.csv");write_registry_json(combined_registry,root/"feature_registry.json");write_registry_json(source.target_registry,root/"target_registry.json")
    parent_comparison=_parent_comparison(combined,all_compiled);_atomic_csv(parent_comparison,combined_root/"dual_parent_comparison.csv")
    fdr_summary=pd.DataFrame([{"evidence_class":"curated_predeclared_interaction","valid_tests":int(pd.to_numeric(curated_results.get("raw_p",pd.Series(dtype=float)),errors="coerce").notna().sum())},{"evidence_class":"systematic_generated_interaction","valid_tests":int(pd.to_numeric(systematic_results.get("raw_p",pd.Series(dtype=float)),errors="coerce").notna().sum())}]);_atomic_csv(fdr_summary,combined_root/"fdr_summary.csv")
    all_paths=[*curated_paths,*systematic_paths];all_chunks=[*curated_chunks,*systematic_chunks];feature_paths_for_queue={**source.feature_cache_index,**{spec.name:path for path,specs in zip(all_paths,all_chunks) for spec in specs}}
    queues=[]
    if run_curated:
        queues.append(select_exact_candidate_queue(curated_combined.loc[curated_combined.target_tier.eq("primary")],config,feature_paths_for_queue,limit=250))
    if not systematic_results.empty:
        effect=pd.to_numeric(systematic_results.get("top_bottom_spread"),errors="coerce").abs();gate=systematic_results.raw_p.notna()&systematic_results.systematic_global_fdr.le(config.systematic_phase1b.screening.exploratory_family_fdr_max)&effect.ge(config.systematic_phase1b.promotion.minimum_absolute_effect_bps/10_000)
        queue_input=systematic_results.loc[gate].copy();queue_input["primary_global_fdr"]=queue_input.systematic_global_fdr;queue_input["family_fdr"]=queue_input.systematic_operator_fdr;queue_input["cluster_fdr"]=queue_input.systematic_pair_cluster_fdr;queue_input["feature_family"]=queue_input.operator
        if "anomaly_score" not in queue_input:queue_input["anomaly_score"]=pd.to_numeric(queue_input.get("top_bottom_spread"),errors="coerce").abs()
        selector_config=replace(config,max_candidates_per_feature_family=20,max_candidates_per_cluster=3,max_candidates_per_target_family=30)
        queues.append(select_exact_candidate_queue(queue_input,selector_config,feature_paths_for_queue,limit=config.systematic_phase1b.screening.exact_candidate_limit))
    candidate_queue=pd.concat([queue for queue in queues if not queue.empty],ignore_index=True,sort=False) if any(not queue.empty for queue in queues) else pd.DataFrame()
    detailed=_run_dual_exact(source,combined,all_paths,all_chunks,config,root,candidate_queue) if not candidate_queue.empty else pd.read_csv(source.root/"detailed_candidates.csv") if (source.root/"detailed_candidates.csv").exists() else pd.DataFrame()
    if not detailed.empty and not systematic_results.empty:
        fields=["feature","target","systematic_global_fdr"]
        detailed=detailed.drop(columns=[name for name in fields[2:] if name in detailed],errors="ignore").merge(systematic_results[fields],on=["feature","target"],how="left")
        detailed=detailed.merge(parent_comparison[["dual_feature","target","best_parent_absolute_effect"]].rename(columns={"dual_feature":"feature"}),on=["feature","target"],how="left")
        mask=detailed.discovery_phase.eq("1B_systematic")
        detailed.loc[mask,"status"]=detailed.loc[mask].apply(lambda row:classify_systematic_candidate(row,config),axis=1)
        from .diagnostics import phase2_recommendation
        recommendations=pd.DataFrame([phase2_recommendation(row) for _,row in detailed.loc[mask].iterrows()],index=detailed.index[mask])
        for column in recommendations:detailed.loc[mask,column]=recommendations[column]
        _atomic_csv(detailed,root/"detailed_candidates.csv");_atomic_csv(detailed.loc[mask],systematic_root/"promoted_candidates.csv")
    elif systematic is not None:
        _atomic_csv(pd.DataFrame(columns=["feature","target","status"]),systematic_root/"promoted_candidates.csv")
    expected=set(map(tuple,candidate_queue[["feature","target"]].to_numpy())) if not candidate_queue.empty else set();observed=set(map(tuple,detailed[["feature","target"]].to_numpy())) if not detailed.empty else set()
    built_count=sum(map(len,all_chunks));valid_count=int(pd.to_numeric(pd.concat([curated_results,systematic_results],ignore_index=True).get("raw_p",pd.Series(dtype=float)),errors="coerce").notna().sum());promotion_ready=bool(built_count and valid_count and expected.issubset(observed))
    limitations=[] if promotion_ready else ["One or more enabled Phase 1B evidence classes lacks valid construction, inference, or exact diagnostics."]
    status={"stage":"complete","run_type":"phase1b_derived","phase1b_mode":config.phase1b_mode,"source_phase1a_run":str(source.root),"source_phase1a_fingerprint":source.fingerprint["sha256"],"phase1b_fingerprint":fingerprint["sha256"],"discovery_end":config.discovery_end,"sealed_holdout_start":config.sealed_holdout_start,"holdout_access":False,"compiled_dual_features":len(all_compiled),"built_dual_features":built_count,"dual_pairs_screened":len(curated_results)+len(systematic_results),"dual_pairs_with_valid_inference":valid_count,"empty_plan_reason":"manifest_compiled_zero_features" if run_curated and not all_compiled else None,"exact_candidate_count":len(detailed),"promotion_ready":promotion_ready,"evidence_stage":"exact_diagnostics_complete" if promotion_ready else "exact_diagnostics_incomplete","source_immutable":True,"phase1a_rebuilt":False,"known_limitations":limitations}
    _atomic_json(status,root/"progress.json");_atomic_json({**status,"config":config.as_dict(),"multiple_testing":"curated combined with Phase 1A; systematic isolated exploratory BH"},root/"manifest.json")
    if run_curated:
        for source_name,legacy_name in (("dual_screen_results.csv","dual_screen_results.csv"),("dual_feature_coverage.csv","dual_feature_coverage.csv"),("dual_feature_registry.csv","dual_feature_registry.csv"),("compiled_feature_plan.json","compiled_feature_plan.json"),("manifest_source.yaml","manifest_source.yaml")):
            target=phase/legacy_name
            origin=curated_root/source_name
            if origin.exists() and target!=origin:target.write_bytes(origin.read_bytes())
    _atomic_csv(parent_comparison,phase/"dual_parent_comparison.csv")
    if _tree_hash(source.root)!=source.tree_hash:raise RuntimeError("Phase 1A source changed during derived Phase 1B execution")
    readiness=[f"Phase 1B readiness: {'GO' if promotion_ready else 'NO-GO'}",f"Mode: {config.phase1b_mode}",f"Discovery end: {config.discovery_end}",f"Sealed holdout start: {config.sealed_holdout_start}","Holdout access: false","Source immutable: true",f"Compiled dual features: {len(all_compiled)}",f"Built dual features: {built_count}",f"Valid broad-screen tests: {valid_count}",f"Exact candidates: {len(detailed)}",f"Known limitations: {limitations or 'none'}"]
    report=root/"phase1b_readiness_report.txt";tmp=report.with_suffix(".txt.tmp");tmp.write_text("\n".join(readiness)+"\n",encoding="utf-8");tmp.replace(report)
    return root
