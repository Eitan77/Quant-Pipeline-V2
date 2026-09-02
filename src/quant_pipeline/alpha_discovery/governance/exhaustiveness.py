from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ExhaustivenessAudit:
    compiled_concepts: int
    compiled_feature_specs: int
    active_feature_specs: int
    unavailable_feature_specs: int
    compiled_targets: int
    expected_single_tests: int
    attempted_single_tests: int
    unavailable_single_tests: int
    structurally_invalid_single_tests: int
    raw_dual_pairs: int
    dual_exclusions_by_reason: dict[str, int]
    eligible_dual_pairs: int
    expected_pair_target_tests: int
    attempted_pair_target_tests: int
    coarse_completed: int
    fine_5x5_completed: int
    failed_blocks: int
    exhaustive_5x5_required: bool

    @property
    def status(self) -> str:
        singles = self.expected_single_tests == self.attempted_single_tests + self.unavailable_single_tests + self.structurally_invalid_single_tests
        pairs = self.expected_pair_target_tests == self.attempted_pair_target_tests and self.coarse_completed == self.expected_pair_target_tests
        fine = not self.exhaustive_5x5_required or self.fine_5x5_completed == self.expected_pair_target_tests
        return "PASS" if singles and pairs and fine and self.failed_blocks == 0 else "INCOMPLETE"

    def write(self, path: str | Path) -> None:
        payload = asdict(self) | {"exhaustiveness_status": self.status}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
