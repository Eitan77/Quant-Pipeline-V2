from __future__ import annotations

from .config import StrategyConfig


INITIAL_BATCH_FAMILIES = (
    "session_range",
    "opening_breakout",
    "opening_breakdown",
    "vwap_slope",
    "market_residual_reversal",
    "conditional_higher_high",
    "overnight_gap_reversal",
    "afternoon_residual_reversal",
)

# Full Phase 2 library is registered here even when the YAML enables only the
# requested initial batch.  Unsupported raw inputs are recorded explicitly by
# the runner; they are never silently proxied by a different Phase 1 feature.
STRATEGY_LIBRARY = {
    "session_range": ("phase1_supported", "morning_continuation"),
    "opening_breakout": ("phase1_supported", "morning_continuation"),
    "opening_breakdown": ("phase1_supported", "morning_continuation"),
    "vwap_slope": ("phase1_supported", "morning_continuation"),
    "market_residual_reversal": ("phase1_supported", "reversal"),
    "conditional_higher_high": ("phase1_conditional", "morning_continuation"),
    "relative_strength": ("new_diversification_hypothesis", "morning_continuation"),
    "soft_confirmation": ("new_diversification_hypothesis", "morning_continuation"),
    "opening_breakout_pullback": ("new_diversification_hypothesis", "morning_continuation"),
    "opening_breakout_persistence": ("new_diversification_hypothesis", "morning_continuation"),
    "residual_reversal_delayed": ("new_diversification_hypothesis", "reversal"),
    "overnight_gap_reversal": ("new_diversification_hypothesis", "overnight"),
    "overnight_gap_continuation": ("new_diversification_hypothesis", "overnight"),
    "afternoon_residual_reversal": ("new_diversification_hypothesis", "afternoon"),
    "closing_window_continuation": ("new_diversification_hypothesis", "afternoon"),
    "volatility_compression_breakout": ("new_diversification_hypothesis", "volatility_breakout"),
    "benchmark_trend": ("new_diversification_hypothesis", "benchmark_directional"),
    "benchmark_reversal": ("new_diversification_hypothesis", "benchmark_directional"),
}


def initial_batch(configs: tuple[StrategyConfig, ...]) -> tuple[StrategyConfig, ...]:
    by_family = {item.family: item for item in configs}
    missing = set(INITIAL_BATCH_FAMILIES) - set(by_family)
    if missing:
        raise ValueError(f"Initial Phase 2 batch is incomplete: {sorted(missing)}")
    return tuple(by_family[family] for family in INITIAL_BATCH_FAMILIES)
