from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import pandas as pd

from .models import CompiledFeatureSpec, CompiledTargetSpec


@dataclass(frozen=True)
class EligibilityMatrix:
    frame: pd.DataFrame

    @classmethod
    def build(cls, features: tuple[CompiledFeatureSpec, ...], targets: tuple[CompiledTargetSpec, ...]) -> "EligibilityMatrix":
        rows = []
        for feature in features:
            for target in targets:
                active = feature.decision_grid == target.decision_grid
                reason = None if active else "decision_grid_mismatch"
                rows.append({"feature_id": feature.feature_id, "decision_grid": feature.decision_grid,
                             "target_id": target.target_id, "active": active, "reason": reason})
        return cls(pd.DataFrame(rows))

    def active_targets(self, feature_id: str) -> set[str]:
        subset = self.frame[(self.frame.feature_id == feature_id) & self.frame.active]
        return set(subset.target_id)

    def eligible_pairs(self, features: tuple[CompiledFeatureSpec, ...]):
        for left, right in combinations(features, 2):
            if left.redundancy_group == right.redundancy_group:
                continue
            shared = self.active_targets(left.feature_id) & self.active_targets(right.feature_id)
            if shared:
                yield left, right, tuple(sorted(shared))
