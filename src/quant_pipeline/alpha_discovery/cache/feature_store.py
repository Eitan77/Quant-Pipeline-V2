from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class ArrayStore:
    """Memory-mapped feature/target blocks with atomic metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, values: np.ndarray, columns: list[str]) -> Path:
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError("Array shape and columns disagree")
        target = self.root / f"{name}.npy"; temporary = self.root / f"{name}.tmp.npy"
        np.save(temporary, np.asarray(values, dtype=np.float32), allow_pickle=False)
        temporary.replace(target)
        meta = target.with_suffix(".json"); tmp_meta = meta.with_suffix(".tmp.json")
        tmp_meta.write_text(json.dumps({"shape": values.shape, "dtype": "float32", "columns": columns}, indent=2), encoding="utf-8")
        tmp_meta.replace(meta)
        return target

    def read(self, name: str) -> tuple[np.memmap, list[str]]:
        target = self.root / f"{name}.npy"; metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        values = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != metadata["shape"]:
            raise ValueError("Array store shape mismatch")
        return values, list(metadata["columns"])
