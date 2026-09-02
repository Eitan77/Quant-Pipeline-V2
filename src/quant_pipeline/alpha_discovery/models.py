from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Literal


def stable_hash(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class TimeScale:
    kind: Literal["minutes", "sessions", "session_to_date", "context", "paired", "bucket"]
    value: int | None
    label: str


@dataclass(frozen=True)
class FeatureConceptSpec:
    concept_id: str
    family: str
    description: str
    valid_scales: tuple[TimeScale, ...]
    required_columns: tuple[str, ...]
    decision_grids: tuple[str, ...]
    representations: tuple[str, ...]
    min_observations: int
    builder_key: str
    priority: str = "core"
    active: bool = True
    unavailable_reason: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledFeatureSpec:
    feature_id: str
    concept_id: str
    family: str
    scale: TimeScale
    representation: str
    decision_grid: str
    availability_rule: str
    minimum_history: int
    redundancy_group: str
    builder_key: str
    price_basis: str
    definition_hash: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledTargetSpec:
    target_id: str
    family: str
    decision_grid: str
    horizon: TimeScale
    entry_rule: str
    exit_rule: str
    return_basis: str
    price_basis: str
    definition_hash: str


@dataclass(frozen=True)
class AlphaSpec:
    alpha_id: str
    source_type: Literal["single", "dual", "triple", "formula", "ml_distilled"]
    feature_ids: tuple[str, ...]
    target_id: str
    scoring_rule: str
    direction: int
    plateau_definition: dict[str, Any]
    decision_grid: str
    expected_holding_period: str
    definition_hash: str


class ResearchState(StrEnum):
    BUILD_ONLY = "BUILD_ONLY"
    DISCOVERY_OPEN = "DISCOVERY_OPEN"
    DISCOVERY_FROZEN = "DISCOVERY_FROZEN"
    REPLICATION_OPEN = "REPLICATION_OPEN"
    REPLICATION_FROZEN = "REPLICATION_FROZEN"
    PORTFOLIO_FROZEN = "PORTFOLIO_FROZEN"
    FINAL_HOLDOUT_OPEN = "FINAL_HOLDOUT_OPEN"
    FINAL_COMPLETE = "FINAL_COMPLETE"


STATE_ORDER = tuple(ResearchState)
