from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pandas as pd

from ..models import AlphaSpec


class AlphaRegistry:
    def __init__(self, specs: list[AlphaSpec] | None = None) -> None:
        self.specs = list(specs or [])

    def add(self, spec: AlphaSpec) -> None:
        if any(item.alpha_id == spec.alpha_id for item in self.specs):
            raise ValueError(f"Duplicate alpha_id: {spec.alpha_id}")
        self.specs.append(spec)

    def write(self, path: str | Path) -> None:
        pd.DataFrame([asdict(item) for item in self.specs]).to_parquet(path, index=False)
