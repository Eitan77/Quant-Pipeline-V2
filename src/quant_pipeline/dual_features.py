"""Materialize frozen Phase 1B features from already validated base caches."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ROW_KEYS, assert_cache_key_alignment, validate_cache, write_aligned_derived_cache_metadata
from .config import ScanConfig
from .dual_registry import CompiledDualFeature, ConditionSpec, TransformSpec
from .registry import FeatureSpec
from .binary_coverage import BinaryCoverage, build_status


def build_feature_cache_index(feature_paths: list[Path], specs_by_chunk: list[list[FeatureSpec]]) -> dict[str, Path]:
    return {spec.name: path for path, specs in zip(feature_paths, specs_by_chunk) for spec in specs}


def eligible_cross_sectional_rank(frame: pd.DataFrame, value: pd.Series, *, min_symbols: int) -> pd.Series:
    """Rank only tradable analysis rows at one decision timestamp."""
    if "analysis_eligible" not in frame:
        raise KeyError("analysis_eligible is required for cross-sectional ranks")
    numeric=pd.to_numeric(value,errors="coerce")
    eligible=frame["analysis_eligible"].fillna(False).astype(bool)&numeric.notna()&np.isfinite(numeric)
    result=pd.Series(np.nan,index=frame.index,dtype="float64")
    if not eligible.any():return result
    work=pd.DataFrame({"session_date":frame.loc[eligible,"session_date"],"decision_ts":frame.loc[eligible,"decision_ts"],"value":numeric.loc[eligible]})
    counts=work.groupby(["session_date","decision_ts"],sort=False)["value"].transform("count")
    ranked=work.loc[counts.ge(min_symbols)].groupby(["session_date","decision_ts"],sort=False)["value"].rank(pct=True,method="average")
    result.loc[ranked.index]=ranked
    return result


def centered_oriented_rank(rank: pd.Series, orientation: int) -> pd.Series:
    if orientation not in {-1,1}:raise ValueError("orientation must be -1 or 1")
    return orientation*(2.0*rank-1.0)


def _transform(frame: pd.DataFrame, value: pd.Series, transform: TransformSpec | None, config: ScanConfig | None=None, *, centered: bool=False) -> pd.Series:
    transform = transform or TransformSpec()
    config=config or ScanConfig()
    if transform.kind == "raw":
        out = pd.to_numeric(value, errors="coerce")
    elif transform.kind == "cross_sectional_rank":
        cached=f"__dual_rank__{value.name}"
        out = pd.to_numeric(frame[cached],errors="coerce") if cached in frame else eligible_cross_sectional_rank(frame,pd.to_numeric(value,errors="coerce"),min_symbols=config.cross_sectional_min_symbols)
    else:
        raise ValueError(f"Unsupported dual transform: {transform.kind}")
    return centered_oriented_rank(out,transform.orientation) if centered and transform.kind=="cross_sectional_rank" else out*transform.orientation


def _condition(frame: pd.DataFrame, value: pd.Series, condition: ConditionSpec | None, config: ScanConfig | None=None) -> pd.Series:
    if condition is None:
        raise ValueError("A dual condition is required for this operator")
    transformed = _transform(frame, value, TransformSpec(condition.transform, 1),config)
    ops = {"ge": transformed.ge, "gt": transformed.gt, "le": transformed.le, "lt": transformed.lt, "eq": transformed.eq}
    if condition.comparator not in ops:
        raise ValueError(f"Unsupported dual comparator: {condition.comparator}")
    result = ops[condition.comparator](condition.threshold).astype(float)
    return result.where(transformed.notna())


def _persistence(frame: pd.DataFrame, left: pd.Series, right: pd.Series, a: ConditionSpec, b: ConditionSpec, config: ScanConfig) -> pd.Series:
    if a.lag_bars < 0 or b.lag_bars < 0:
        raise ValueError("persistence lag_bars must be nonnegative")
    ordered = frame[["symbol", "session_date", "decision_ts"]].copy()
    ordered["left"] = _condition(frame, left, a, config); ordered["right"] = _condition(frame, right, b, config)
    ordered["row"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["symbol", "session_date", "decision_ts"])
    grouped = ordered.groupby(["symbol", "session_date"], sort=False)
    # A five-minute source has no valid persistence link across a bar gap.
    previous = grouped["decision_ts"].shift(a.lag_bars)
    expected = pd.to_timedelta(config.bar_interval_minutes * a.lag_bars, unit="m")
    left_value = grouped["left"].shift(a.lag_bars) if a.lag_bars else ordered["left"]
    if a.lag_bars:
        left_value = left_value.where(pd.to_datetime(ordered["decision_ts"], utc=True).sub(pd.to_datetime(previous, utc=True)).eq(expected))
    right_value = grouped["right"].shift(b.lag_bars) if b.lag_bars else ordered["right"]
    if b.lag_bars:
        previous_b = grouped["decision_ts"].shift(b.lag_bars)
        expected_b = pd.to_timedelta(config.bar_interval_minutes * b.lag_bars, unit="m")
        right_value = right_value.where(pd.to_datetime(ordered["decision_ts"], utc=True).sub(pd.to_datetime(previous_b, utc=True)).eq(expected_b))
    output = (left_value.eq(1) & right_value.eq(1)).astype(float).where(left_value.notna() & right_value.notna())
    return output.set_axis(ordered["row"]).reindex(range(len(frame))).reset_index(drop=True)


def _materialize(frame: pd.DataFrame, compiled: CompiledDualFeature, config: ScanConfig | None=None) -> pd.Series:
    config=config or ScanConfig()
    item = compiled.definition
    left, right = frame[item.feature_a], frame[item.feature_b]
    if item.operator == "intersection":
        a, b = _condition(frame, left, item.condition_a,config), _condition(frame, right, item.condition_b,config)
        return (a.eq(1) & b.eq(1)).astype(float).where(a.notna() & b.notna())
    if item.operator == "gated_anchor":
        gate = _condition(frame, right, item.condition_b,config)
        anchor = _transform(frame, left, item.transform_a,config)
        return anchor.where(gate.eq(1) & gate.notna())
    if item.operator == "aligned_rank_mean":
        a = _transform(frame, left, item.transform_a,config,centered=True); b = _transform(frame, right, item.transform_b,config,centered=True)
        return (a+b)/2
    if item.operator == "persistence_intersection":
        return _persistence(frame, left, right, item.condition_a or ConditionSpec(), item.condition_b or ConditionSpec(),config)
    raise ValueError(f"Unsupported dual operator: {item.operator}")


def _materialize_group_from_cached_ranks(
    frame: pd.DataFrame,
    compiled: list[CompiledDualFeature],
    config: ScanConfig,
) -> dict[str, np.ndarray]:
    """Materialize one parent pair while reusing its arrays and predicates."""
    arrays: dict[tuple[str, str], np.ndarray] = {}
    transforms: dict[tuple[str, str, int, bool], np.ndarray] = {}
    conditions: dict[tuple[str, str, str, float], tuple[np.ndarray, np.ndarray]] = {}

    def values(name: str, kind: str) -> np.ndarray:
        key = (name, kind)
        if key not in arrays:
            column = f"__dual_rank__{name}" if kind == "cross_sectional_rank" else name
            arrays[key] = frame[column].to_numpy(dtype=float, copy=False)
        return arrays[key]

    def transform(name: str, spec: TransformSpec | None, centered: bool = False) -> np.ndarray:
        spec = spec or TransformSpec()
        key = (name, spec.kind, spec.orientation, centered)
        if key not in transforms:
            source = values(name, spec.kind)
            if centered and spec.kind == "cross_sectional_rank":
                transforms[key] = spec.orientation * (2.0 * source - 1.0)
            elif spec.orientation == 1:
                transforms[key] = source
            else:
                transforms[key] = -source
        return transforms[key]

    def condition(name: str, spec: ConditionSpec | None) -> tuple[np.ndarray, np.ndarray]:
        if spec is None:
            raise ValueError("A dual condition is required")
        key = (name, spec.transform, spec.comparator, spec.threshold)
        if key not in conditions:
            source = values(name, spec.transform)
            finite = np.isfinite(source)
            operations = {
                "ge": np.greater_equal,
                "gt": np.greater,
                "le": np.less_equal,
                "lt": np.less,
                "eq": np.equal,
            }
            conditions[key] = (operations[spec.comparator](source, spec.threshold), finite)
        return conditions[key]

    output: dict[str, np.ndarray] = {}
    for item in compiled:
        definition = item.definition
        if definition.operator == "intersection":
            left, left_finite = condition(definition.feature_a, definition.condition_a)
            right, right_finite = condition(definition.feature_b, definition.condition_b)
            result = np.full(len(frame), np.nan, dtype=float)
            valid = left_finite & right_finite
            result[valid] = left[valid] & right[valid]
        elif definition.operator == "gated_anchor":
            anchor = transform(definition.feature_a, definition.transform_a)
            gate, gate_finite = condition(definition.feature_b, definition.condition_b)
            result = np.full(len(frame), np.nan, dtype=float)
            active = gate_finite & gate
            result[active] = anchor[active]
        elif definition.operator == "aligned_rank_mean":
            left = transform(definition.feature_a, definition.transform_a, True)
            right = transform(definition.feature_b, definition.transform_b, True)
            result = (left + right) / 2.0
        else:
            result = _materialize(frame, item, config).to_numpy()
        output[item.spec.name] = result
    return output


def _binary_coverage_from_codes(frame:pd.DataFrame,names:list[str],group_codes:tuple[np.ndarray,np.ndarray,np.ndarray])->dict[str,BinaryCoverage]:
    eligible=frame.get("analysis_eligible",pd.Series(True,index=frame.index)).fillna(False).to_numpy(dtype=bool,copy=False);result={}
    group_sizes=[int(codes.max())+1 for codes in group_codes]
    def distinct(mask:np.ndarray,codes:np.ndarray,size:int)->int:
        return int(np.count_nonzero(np.bincount(codes[mask],minlength=size)))
    for name in names:
        values=pd.to_numeric(frame[name],errors="coerce").to_numpy(dtype=float,copy=False);valid=eligible&np.isfinite(values)
        if not np.isin(values[valid],[0.0,1.0]).all():raise ValueError(f"Binary feature contains values outside 0/1: {name}")
        on=valid&(values==1);off=valid&(values==0);on_groups=[distinct(on,codes,size) for codes,size in zip(group_codes,group_sizes)];off_groups=[distinct(off,codes,size) for codes,size in zip(group_codes,group_sizes)];n=int(np.count_nonzero(valid));on_count=int(np.count_nonzero(on));off_count=int(np.count_nonzero(off))
        result[name]=BinaryCoverage(valid_observations=n,signal_on_count=on_count,signal_off_count=off_count,signal_on_sessions=on_groups[0],signal_off_sessions=off_groups[0],signal_on_symbols=on_groups[1],signal_off_symbols=off_groups[1],signal_on_decision_timestamps=on_groups[2],signal_off_decision_timestamps=off_groups[2],activation_rate=float(on_count/n) if n else np.nan)
    return result


def binary_build_status(metrics: dict, config: ScanConfig) -> tuple[str,str|None]:
    from .binary_coverage import BinaryCoverage
    return build_status(BinaryCoverage(**metrics),config)


def build_dual_feature_chunks(
    compiled: list[CompiledDualFeature],
    feature_path_by_name: dict[str, Path],
    config: ScanConfig,
    output_root: Path,
    fingerprint_sha: str,
    resume_coverage: pd.DataFrame | None = None,
) -> tuple[list[Path], list[list[FeatureSpec]], list[dict]]:
    """Create resume-safe Phase 1B cache chunks from aligned base caches."""
    output_root.mkdir(parents=True, exist_ok=True)
    chunks = [compiled[i:i + config.dual_factor_feature_chunk_size] for i in range(0, len(compiled), config.dual_factor_feature_chunk_size)]
    paths = [output_root / f"dual_{index:03d}.parquet" for index in range(len(chunks))]
    specs_by_chunk: list[list[FeatureSpec]] = []; coverage: list[dict] = []
    coverage_by_feature = {
        str(row["feature"]): row
        for row in (resume_coverage.to_dict("records") if resume_coverage is not None else [])
    }
    resumable_paths = [
        path for path, chunk in zip(paths, chunks)
        if path.exists() and all(item.spec.name in coverage_by_feature for item in chunk)
    ]
    if resumable_paths:
        with ThreadPoolExecutor(max_workers=min(4, len(resumable_paths))) as executor:
            list(executor.map(lambda path: validate_cache(path, fingerprint_sha, config.sealed_holdout_start), resumable_paths))
    memo: dict[Path, pd.DataFrame] = {}
    rank_memo: dict[str, np.ndarray] = {}
    group_codes: tuple[np.ndarray,np.ndarray,np.ndarray]|None = None
    required_by_path: dict[Path, set[str]] = {}
    for item in compiled:
        for parent in item.spec.parent_features:
            required_by_path.setdefault(feature_path_by_name[parent],set()).add(parent)
    def parent_frame(path:Path)->pd.DataFrame:
        if path not in memo:
            columns=list(dict.fromkeys([*ROW_KEYS,"session_date","analysis_eligible","symbol","bar_start_ts","decision_ts",*sorted(required_by_path[path])]))
            memo[path]=pd.read_parquet(path,columns=columns)
        return memo[path]
    for index, chunk in enumerate(chunks):
        path = paths[index]; specs = [item.spec for item in chunk]
        specs_by_chunk.append(specs)
        if path in resumable_paths:
            records = [coverage_by_feature[spec.name] for spec in specs]
            coverage.extend(records)
            specs_by_chunk[-1] = [spec for spec, record in zip(specs, records) if record.get("status") == "built"]
            continue
        if path.exists():
            validate_cache(path, fingerprint_sha, config.sealed_holdout_start)
            frame = pd.read_parquet(path)
        else:
            parents = sorted({parent for item in chunk for parent in item.spec.parent_features})
            parent_paths = {feature_path_by_name[parent] for parent in parents}
            first = next(iter(parent_paths))
            for parent_path in parent_paths:
                assert_cache_key_alignment(first, parent_path)
            base = parent_frame(first)
            frame = base.loc[:, [col for col in base.columns if col in ROW_KEYS or col in {"session_date", "analysis_eligible", "symbol", "bar_start_ts", "decision_ts"}]].copy()
            if group_codes is None:group_codes=tuple(pd.factorize(frame[column],sort=False)[0].astype(np.int32,copy=False) for column in ("session_date","symbol","decision_ts"))
            for parent in parents:
                parent_path = feature_path_by_name[parent]
                source_frame = parent_frame(parent_path)
                frame[parent] = source_frame[parent].to_numpy()
            ranked_parents=set()
            for item in chunk:
                definition=item.definition
                if (definition.transform_a and definition.transform_a.kind=="cross_sectional_rank") or (definition.condition_a and definition.condition_a.transform=="cross_sectional_rank"):ranked_parents.add(definition.feature_a)
                if (definition.transform_b and definition.transform_b.kind=="cross_sectional_rank") or (definition.condition_b and definition.condition_b.transform=="cross_sectional_rank"):ranked_parents.add(definition.feature_b)
            for parent in sorted(ranked_parents):
                if parent not in rank_memo:rank_memo[parent]=eligible_cross_sectional_rank(frame,frame[parent],min_symbols=config.cross_sectional_min_symbols).to_numpy()
                frame[f"__dual_rank__{parent}"]=rank_memo[parent]
            groups: dict[str, list[CompiledDualFeature]] = {}
            for item in chunk:
                groups.setdefault(item.spec.redundancy_group, []).append(item)
            workers = min(8, len(groups), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pieces = executor.map(
                    lambda group: _materialize_group_from_cached_ranks(frame, group, config),
                    groups.values(),
                )
                generated = {name: values for piece in pieces for name, values in piece.items()}
            generated_frame = pd.DataFrame(generated, index=frame.index)
            keys = list(dict.fromkeys([
                col for col in frame.columns
                if col in ROW_KEYS or col in {"session_date", "analysis_eligible", "symbol", "bar_start_ts", "decision_ts"}
            ]))
            frame = pd.concat(
                [frame.loc[:, keys], generated_frame],
                axis=1,
                copy=False,
            )
            temporary=path.with_suffix(path.suffix+".tmp");frame.to_parquet(temporary,index=False);temporary.replace(path)
            source_metadata=json.loads(first.with_suffix(first.suffix+".meta.json").read_text(encoding="utf-8"))
            write_aligned_derived_cache_metadata(path,frame,fingerprint_sha,source_metadata,config.sealed_holdout_start)
        built_specs=[];binary_names=[spec.name for spec in specs if spec.dtype=="binary"]
        if group_codes is None:group_codes=tuple(pd.factorize(frame[column],sort=False)[0].astype(np.int32,copy=False) for column in ("session_date","symbol","decision_ts"))
        binary_metrics_by_name=_binary_coverage_from_codes(frame,binary_names,group_codes) if binary_names else {}
        for spec in specs:
            signal = pd.to_numeric(frame[spec.name], errors="coerce")
            eligible=frame.get("analysis_eligible",pd.Series(True,index=frame.index)).fillna(False).astype(bool)
            valid=signal.notna()&eligible
            if spec.dtype=="binary":
                binary_metrics=binary_metrics_by_name[spec.name];metrics=binary_metrics.as_dict();status,reason=build_status(binary_metrics,config)
            else:status,reason=("built",None) if valid.any() else ("skipped","insufficient_valid_observations")
            if spec.dtype!="binary":metrics={"valid_observations":int(valid.sum()),"activation_rate":np.nan,"signal_on_count":0,"signal_off_count":0,"signal_on_sessions":0,"signal_off_sessions":0,"signal_on_symbols":0,"signal_off_symbols":0,"signal_on_decision_timestamps":0,"signal_off_decision_timestamps":0}
            coverage.append({"feature":spec.name,"status":status,"skip_reason":reason,**metrics})
            if status=="built":built_specs.append(spec)
        specs_by_chunk[-1]=built_specs
    return paths, specs_by_chunk, coverage
