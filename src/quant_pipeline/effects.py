"""Semantic effect helpers shared by Phase 1A/1B scan consumers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureSpec


def feature_scan_kind(spec: FeatureSpec) -> str:
    if spec.dtype == "binary":
        return "binary"
    if spec.dtype == "categorical" or spec.classification == "categorical":
        return "categorical"
    return "continuous"


def effect_value(frame: pd.DataFrame, spec: FeatureSpec, target: str) -> float:
    x = pd.to_numeric(frame[spec.name], errors="coerce")
    y = pd.to_numeric(frame[target], errors="coerce")
    valid = x.notna() & y.notna()
    if feature_scan_kind(spec) == "binary":
        return float(y[valid & x.eq(1)].mean() - y[valid & x.eq(0)].mean())
    try:
        bins = pd.qcut(x[valid].rank(method="first"), 10, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    means = y[valid].groupby(bins, observed=True).mean()
    return float(means.iloc[-1] - means.iloc[0]) if len(means) > 1 else np.nan


def signed_effect_value(frame: pd.DataFrame, spec: FeatureSpec, target: str, direction_sign: int | float = 1) -> float:
    return float(direction_sign) * effect_value(frame, spec, target)


def session_effect_series(frame: pd.DataFrame, spec: FeatureSpec, target: str) -> pd.Series:
    return frame.groupby("session_date", sort=False).apply(lambda part: effect_value(part, spec, target), include_groups=False)


def screen_direction_hint(spec: FeatureSpec, fallback: float = 1.0) -> float:
    return float(spec.expected_direction or fallback)


def validate_binary_semantics(frame: pd.DataFrame, specs: list[FeatureSpec], mode: str = "error") -> list[str]:
    """Reject accidental continuous treatment of a true 0/1 signal."""
    issues: list[str] = []
    if mode == "off":
        return issues
    for spec in specs:
        if spec.dtype in {"binary", "categorical"} or spec.name not in frame:
            continue
        values = pd.to_numeric(frame[spec.name], errors="coerce").dropna()
        if len(values) and values.isin([0, 1]).all():
            issues.append(spec.name)
    if issues and mode == "error":
        raise ValueError(f"Continuous features with binary 0/1 semantics: {issues}")
    return issues
