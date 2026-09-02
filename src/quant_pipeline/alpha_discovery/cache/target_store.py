from __future__ import annotations

import numpy as np

from .feature_store import ArrayStore


class TargetStore(ArrayStore):
    """Float target blocks with an explicit finite-observation contract."""

    def write(self, name: str, values: np.ndarray, columns: list[str]):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or not columns:
            raise ValueError("Target blocks must be two-dimensional and named")
        if not np.isfinite(array).any(axis=0).all():
            missing = [column for column, valid in zip(columns, np.isfinite(array).any(axis=0)) if not valid]
            raise ValueError(f"Target columns contain no finite observations: {missing}")
        return super().write(name, array, columns)
