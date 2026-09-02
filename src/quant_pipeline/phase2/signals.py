from __future__ import annotations

import pandas as pd


def feature_names(family: str, lookback: int) -> tuple[str, ...]:
    supported_lookbacks = {
        "session_range": {5}, "vwap_slope": {5}, "conditional_higher_high": {15},
        "overnight_gap_reversal": {5}, "market_residual_reversal": {60}, "afternoon_residual_reversal": {60},
        "opening_breakout": {5, 15, 30}, "opening_breakdown": {5, 15, 30},
    }
    if lookback not in supported_lookbacks.get(family, set()):
        raise ValueError(f"Phase 1 has no validated {family} feature for lookback {lookback}m")
    mapping = {
        "session_range": ("session_range_position",),
        "opening_breakout": (f"opening_breakout_{lookback}m",),
        "opening_breakdown": (f"opening_breakdown_{lookback}m",),
        "vwap_slope": ("vwap_slope",),
        "market_residual_reversal": (f"market_residual_return_{lookback}",),
        "afternoon_residual_reversal": (f"market_residual_return_{lookback}",),
        "conditional_higher_high": ("higher_high",),
        "overnight_gap_reversal": ("overnight_gap",),
    }
    if family not in mapping:
        raise ValueError(f"No Phase 2 signal adapter for {family}")
    return mapping[family]


def build_signal(frame: pd.DataFrame, family: str, features: tuple[str, ...]) -> pd.DataFrame:
    work = frame.copy()
    signal = work[features[0]].astype(float)
    if family in {"market_residual_reversal", "afternoon_residual_reversal", "overnight_gap_reversal"}:
        signal = -signal
    work["signal"] = signal
    return work
