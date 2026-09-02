from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class ContentManifest:
    stage: str
    config_hash: str
    data_fingerprint: str
    definition_hashes: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def write(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"digest": self.digest}
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load_verified(cls, path: str | Path) -> "ContentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8")); digest = payload.pop("digest")
        manifest = cls(**payload)
        if manifest.digest != digest:
            raise ValueError("Content manifest hash mismatch")
        return manifest
