from __future__ import annotations

from hashlib import sha256
from itertools import combinations

from ..eligibility import EligibilityMatrix
from ..models import CompiledFeatureSpec


def pair_id(left: str, right: str) -> str:
    a, b = sorted((left, right))
    return sha256(f"{a}\0{b}".encode()).hexdigest()[:24]


def enumerate_pair_trials(features: tuple[CompiledFeatureSpec, ...], eligibility: EligibilityMatrix):
    for left, right in combinations(features, 2):
        if left.redundancy_group == right.redundancy_group:
            yield {"pair_id": pair_id(left.feature_id, right.feature_id), "feature_a": left.feature_id,
                   "feature_b": right.feature_id, "target_id": None, "eligible": False, "reason": "exact_alias"}
            continue
        shared = eligibility.active_targets(left.feature_id) & eligibility.active_targets(right.feature_id)
        if not shared:
            yield {"pair_id": pair_id(left.feature_id, right.feature_id), "feature_a": left.feature_id,
                   "feature_b": right.feature_id, "target_id": None, "eligible": False, "reason": "no_joint_target"}
            continue
        for target_id in sorted(shared):
            yield {"pair_id": pair_id(left.feature_id, right.feature_id), "feature_a": left.feature_id,
                   "feature_b": right.feature_id, "target_id": target_id, "eligible": True, "reason": None}
