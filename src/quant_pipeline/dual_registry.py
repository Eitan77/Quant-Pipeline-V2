"""Frozen, target-independent Phase 1B dual-feature plans."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ScanConfig
from .registry import FeatureSpec


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class TransformSpec:
    kind: str = "raw"
    orientation: int = 1


@dataclass(frozen=True)
class ConditionSpec:
    transform: str = "raw"
    comparator: str = "ge"
    threshold: float = 0.0
    lag_bars: int = 0


@dataclass(frozen=True)
class DualFeatureDefinition:
    identifier: str
    feature_a: str
    feature_b: str
    operator: str
    condition_a: ConditionSpec | None = None
    condition_b: ConditionSpec | None = None
    transform_a: TransformSpec | None = None
    transform_b: TransformSpec | None = None
    output_dtype: str = "binary"
    expected_direction: int = 0


@dataclass(frozen=True)
class CompiledDualFeature:
    definition: DualFeatureDefinition
    spec: FeatureSpec
    condition_specs: tuple[FeatureSpec, ...] = ()


def _condition(raw: dict[str, Any] | None) -> ConditionSpec | None:
    return None if raw is None else ConditionSpec(**{k: raw[k] for k in ("transform", "comparator", "threshold", "lag_bars") if k in raw})


def _transform(raw: dict[str, Any] | None) -> TransformSpec | None:
    if raw is None:
        return None
    return TransformSpec(kind=raw.get("kind", raw.get("transform", "raw")), orientation=int(raw.get("orientation", 1)))


def _feature_name(definition: DualFeatureDefinition) -> tuple[str, str]:
    payload = {"id": definition.identifier, "a": definition.feature_a, "b": definition.feature_b,
               "operator": definition.operator, "condition_a": definition.condition_a,
               "condition_b": definition.condition_b, "transform_a": definition.transform_a,
               "transform_b": definition.transform_b, "dtype": definition.output_dtype,
               "direction": definition.expected_direction}
    lineage = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    stem = f"dual__{definition.feature_a}__{definition.feature_b}__{definition.identifier}"
    return f"{stem}__{lineage[:8]}", lineage


def _classification(definition: DualFeatureDefinition, parents: tuple[FeatureSpec,FeatureSpec]) -> str:
    transforms=[definition.transform_a,definition.transform_b]
    transforms += [TransformSpec(c.transform) for c in (definition.condition_a,definition.condition_b) if c]
    if any(item and item.kind in {"cross_sectional_rank","centered_cross_sectional_rank"} for item in transforms):return "cross_sectional"
    if any(parent.classification in {"context","contextual"} for parent in parents):return "context"
    return "time_series"


def _combine_eligibility(left:str,right:str)->str:
    return left if left==right else "derived_requires_"+"_and_".join(sorted({left,right}))


def _spec(definition: DualFeatureDefinition, parents: tuple[FeatureSpec,FeatureSpec]) -> FeatureSpec:
    name, lineage = _feature_name(definition)
    params = _canonical({"condition_a": definition.condition_a, "condition_b": definition.condition_b,
                         "transform_a": definition.transform_a, "transform_b": definition.transform_b})
    return FeatureSpec(
        name=name, description=f"Frozen dual factor: {definition.identifier}", family="dual_factor",
        classification=_classification(definition,parents), dtype=definition.output_dtype, discovery_phase="1B", arity=2,
        parent_features=(definition.feature_a, definition.feature_b), operator=definition.operator,
        operator_parameters_json=params, lineage_hash=lineage, redundancy_group=f"dual::{definition.identifier}",
        complexity_units=2, expected_direction=definition.expected_direction,
        required_columns=tuple(sorted({definition.feature_a,definition.feature_b})),
        minimum_history_bars=max(parents[0].minimum_history_bars+(definition.condition_a.lag_bars if definition.condition_a else 0),parents[1].minimum_history_bars+(definition.condition_b.lag_bars if definition.condition_b else 0)),
        session_reset=parents[0].session_reset or parents[1].session_reset or definition.operator=="persistence_intersection",
        uses_current_bar=parents[0].uses_current_bar or parents[1].uses_current_bar,
        uses_previous_sessions=parents[0].uses_previous_sessions or parents[1].uses_previous_sessions,
        feature_available_offset_minutes=max(parents[0].feature_available_offset_minutes,parents[1].feature_available_offset_minutes),
        eligibility_requirement=_combine_eligibility(parents[0].eligibility_requirement,parents[1].eligibility_requirement),
        price_basis=parents[0].price_basis if parents[0].price_basis==parents[1].price_basis else "derived_mixed_parent_price_basis",
        evidence_class="curated_predeclared_interaction",
    )


def compile_dual_plan(path: str | Path, base_features: list[FeatureSpec], config: ScanConfig) -> list[CompiledDualFeature]:
    """Parse a strict, deterministic and target-independent manifest."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != "phase1b_manifest_v1":
        raise ValueError("Unsupported Phase 1B manifest schema_version")
    base_by_name = {spec.name:spec for spec in base_features if spec.status == "requested"}
    defaults = raw.get("defaults", {})
    compiled: list[CompiledDualFeature] = []
    identifiers=set(); canonical_definitions=set()
    for item in raw.get("definitions", []):
        unknown = set(item) - {"id", "feature_a", "feature_b", "operator", "condition_a", "condition_b", "transform_a", "transform_b", "output_dtype", "expected_direction"}
        if unknown:
            raise ValueError(f"Unknown dual definition keys: {sorted(unknown)}")
        if item.get("id") in identifiers:raise ValueError(f"Duplicate dual manifest id: {item.get('id')}")
        identifiers.add(item.get("id"))
        definition = DualFeatureDefinition(
            identifier=item["id"], feature_a=item["feature_a"], feature_b=item["feature_b"], operator=item["operator"],
            condition_a=_condition(item.get("condition_a")), condition_b=_condition(item.get("condition_b")),
            transform_a=_transform(item.get("transform_a")), transform_b=_transform(item.get("transform_b")),
            output_dtype=item.get("output_dtype", "binary"),
            expected_direction=int(item.get("expected_direction", defaults.get("expected_direction", 0))),
        )
        if definition.operator not in config.dual_factor_allowed_operators:
            raise ValueError(f"Unsupported dual operator: {definition.operator}")
        if definition.feature_a not in base_by_name or definition.feature_b not in base_by_name:
            raise ValueError(f"Dual definition {definition.identifier} references unavailable base parent")
        if definition.output_dtype not in {"binary", "continuous"}:
            raise ValueError("Dual output_dtype must be binary or continuous")
        if definition.operator in {"intersection", "persistence_intersection"} and definition.output_dtype != "binary":
            raise ValueError(f"{definition.operator} must emit binary output")
        if definition.operator in {"gated_anchor", "aligned_rank_mean"} and definition.output_dtype != "continuous":
            raise ValueError(f"{definition.operator} must emit continuous output")
        if definition.expected_direction not in {-1,0,1}:raise ValueError("expected_direction must be -1, 0, or 1")
        for transform in (definition.transform_a,definition.transform_b):
            if transform and (transform.kind not in {"raw","cross_sectional_rank"} or transform.orientation not in {-1,1}):raise ValueError("Unsupported transform or orientation")
        for condition in (definition.condition_a,definition.condition_b):
            if condition and (condition.transform not in {"raw","cross_sectional_rank"} or condition.comparator not in {"ge","gt","le","lt","eq"} or not isinstance(condition.lag_bars,int) or condition.lag_bars<0):raise ValueError("Invalid dual condition")
        if definition.operator=="intersection" and (not definition.condition_a or not definition.condition_b):raise ValueError("intersection requires both conditions")
        if definition.operator=="gated_anchor" and (not definition.transform_a or not definition.condition_b):raise ValueError("gated_anchor requires anchor transform and gate condition")
        if definition.operator=="aligned_rank_mean" and (not definition.transform_a or not definition.transform_b):raise ValueError("aligned_rank_mean requires both transforms")
        if definition.operator=="persistence_intersection" and (not definition.condition_a or not definition.condition_b or max(definition.condition_a.lag_bars,definition.condition_b.lag_bars)<=0):raise ValueError("persistence_intersection requires conditions and a positive lag")
        parents=(base_by_name[definition.feature_a],base_by_name[definition.feature_b])
        side_a=(definition.feature_a,_canonical(definition.condition_a),_canonical(definition.transform_a))
        side_b=(definition.feature_b,_canonical(definition.condition_b),_canonical(definition.transform_b))
        sides=tuple(sorted((side_a,side_b))) if definition.operator in {"intersection","persistence_intersection","aligned_rank_mean"} else (side_a,side_b)
        canonical=(definition.operator,sides,definition.output_dtype,definition.expected_direction)
        if canonical in canonical_definitions:raise ValueError("Duplicate commutative dual definition")
        canonical_definitions.add(canonical)
        compiled.append(CompiledDualFeature(definition, _spec(definition,parents)))
    if len(compiled) > config.dual_factor_max_generated_features:
        raise ValueError("Compiled dual plan exceeds dual_factor_max_generated_features")
    names = [item.spec.name for item in compiled]
    if len(names) != len(set(names)):
        raise ValueError("Dual plan contains duplicate generated feature names")
    return compiled


def plan_hash(compiled: list[CompiledDualFeature]) -> str:
    return hashlib.sha256(_canonical([item.spec for item in compiled]).encode()).hexdigest()
