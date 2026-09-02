from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from ..models import ResearchState, STATE_ORDER


@dataclass
class ResearchStateMachine:
    path: Path
    state: ResearchState = ResearchState.BUILD_ONLY

    @classmethod
    def load(cls, path: str | Path) -> "ResearchStateMachine":
        resolved = Path(path)
        if not resolved.exists():
            return cls(resolved)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(resolved, ResearchState(payload["state"]))

    def transition(self, target: ResearchState, prerequisites: dict[str, bool] | None = None) -> None:
        current_index, target_index = STATE_ORDER.index(self.state), STATE_ORDER.index(target)
        if target_index != current_index + 1:
            raise ValueError(f"Illegal research-state transition {self.state} -> {target}")
        failed = sorted(name for name, passed in (prerequisites or {}).items() if not passed)
        if failed:
            raise RuntimeError(f"State transition blocked; failed prerequisites: {failed}")
        self.state = target
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"state": target.value, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        temporary.replace(self.path)
