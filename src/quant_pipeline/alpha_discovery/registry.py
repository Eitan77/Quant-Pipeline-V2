from __future__ import annotations

from dataclasses import dataclass, replace
import json
from itertools import product
from typing import Iterable

import pandas as pd

from .config import AlphaDiscoveryConfig
from .models import CompiledFeatureSpec, CompiledTargetSpec, FeatureConceptSpec, TimeScale, stable_hash
from .timescales import history_sessions, parse_scale, parse_scales


INTRADAY = ("1m", "2m", "5m", "10m", "15m", "30m", "60m", "120m", "240m", "session")
DAILY = ("1d", "2d", "3d", "5d", "10d", "20d", "30d", "40d", "63d", "126d", "252d")
MID = ("30m", "60m", "120m", "240m", "session", "1d", "2d", "3d", "5d", "10d", "20d", "30d", "40d", "63d")
GRIDS = ("intraday_5m", "daily_close", "preclose_1555", "intraday_1m")
REPRESENTATIONS = ("raw", "own_history_zscore", "own_history_percentile")
APPROVED_FAST_SLOW_PAIRS = (
    ("1m", "5m"), ("2m", "10m"), ("5m", "30m"), ("15m", "60m"),
    ("30m", "120m"), ("120m", "session"), ("1d", "5d"), ("2d", "10d"),
    ("5d", "20d"), ("10d", "40d"), ("20d", "63d"), ("40d", "126d"), ("63d", "252d"),
)

MINIMUM_WINDOW = {
    "return_t_stat": 2, "return_rank_std": 2, "vol_acceleration": 4,
    "vol_compression_ratio": 4, "downside_vol_acceleration": 4,
    "upside_vol_acceleration": 4, "gap_vs_prior_vol": 2,
    "return_autocorr_lag1": 3, "return_autocorr_lag2": 4, "return_autocorr_lag3": 5,
    "sign_autocorr_lag1": 3, "abs_return_autocorr_lag1": 3,
    "variance_ratio_2": 4, "variance_ratio_5": 7, "permutation_entropy": 6,
    "hurst_exponent": 32, "market_correlation": 3, "market_lead_response": 3,
    "stock_lead_market_response": 3, "average_pairwise_correlation": 3,
    "breadth_momentum": 5,
}


FAMILIES: dict[str, tuple[str, ...]] = {
    "returns": (
        "simple_return", "log_return", "benchmark_excess_return", "beta_residual_return",
        "return_over_realized_vol", "return_over_downside_vol", "return_t_stat",
        "positive_return_sum", "negative_return_sum_abs", "up_down_return_ratio",
    ),
    "multispeed": (
        "fast_minus_slow_return", "fast_minus_slow_cross_sectional_rank",
        "fast_minus_slow_risk_adjusted_momentum", "momentum_acceleration",
        "momentum_deceleration", "momentum_sign_agreement", "pullback_in_uptrend", "recovery_in_downtrend",
    ),
    "path": (
        "path_efficiency", "trend_slope", "trend_slope_t", "trend_r2", "path_chord_convexity",
        "path_quadratic_curvature", "return_acceleration_halves", "return_acceleration_thirds",
        "positive_bar_fraction", "negative_bar_fraction", "sign_persistence", "sign_switch_rate",
        "average_signed_run_length", "longest_up_run", "longest_down_run", "gross_to_net_path_ratio",
        "window_max_drawdown", "window_max_runup", "close_vs_path_mean", "close_vs_path_median",
    ),
    "consistency": (
        "information_discreteness", "largest_abs_return_share", "top2_abs_return_share",
        "top3_abs_return_share", "top5_abs_return_share", "absolute_return_hhi",
        "signed_return_concentration", "mean_median_return_gap",
    ),
    "ranks": (
        "mean_return_rank", "median_return_rank", "return_rank_std", "return_rank_slope",
        "return_rank_acceleration", "rank_above_80_fraction", "rank_above_90_fraction",
        "rank_below_20_fraction", "rank_range", "rank_change_1d", "rank_change_2d",
        "rank_change_5d", "rank_change_10d", "rank_change_20d", "rank_dwell_top_decile",
        "rank_dwell_bottom_decile",
    ),
    "location": (
        "range_position", "distance_from_high", "distance_from_low", "drawdown_from_high",
        "recovery_from_low", "recovery_fraction", "bars_or_days_since_high", "bars_or_days_since_low",
        "new_high_count", "new_low_count", "breakout_distance", "breakdown_distance",
        "failed_breakout_strength", "failed_breakdown_strength", "prior_high_reclaim_strength",
        "prior_low_reclaim_strength",
    ),
    "candles": (
        "body_to_range", "signed_body_to_range", "upper_wick_fraction", "lower_wick_fraction",
        "close_location_bar", "true_range_pct", "high_low_asymmetry",
    ),
    "moving_average": (
        "price_to_sma", "price_minus_sma_volnorm", "sma_slope", "sma_acceleration",
        "ema_distance", "ema_slope", "bollinger_z", "atr_band_z", "fast_slow_sma_gap",
        "fast_slow_sma_ratio", "fast_slow_ema_gap",
    ),
    "volatility": (
        "close_to_close_vol", "realized_vol", "downside_vol", "upside_vol", "down_up_vol_ratio",
        "atr_pct", "parkinson_vol", "garman_klass_vol", "rogers_satchell_vol", "idiosyncratic_vol",
        "vol_of_vol", "vol_percentile_own", "vol_acceleration", "vol_compression_ratio",
        "downside_vol_acceleration", "upside_vol_acceleration",
    ),
    "distribution": (
        "realized_skewness", "realized_kurtosis", "downside_semivariance", "upside_semivariance",
        "semivariance_ratio", "max_subreturn", "min_subreturn", "p95_subreturn", "p05_subreturn",
        "tail_spread", "left_expected_shortfall", "right_expected_shortfall", "jump_fraction",
        "positive_jump_fraction", "negative_jump_fraction", "jump_variance_share",
    ),
    "serial": (
        "return_autocorr_lag1", "return_autocorr_lag2", "return_autocorr_lag3", "sign_autocorr_lag1",
        "abs_return_autocorr_lag1", "variance_ratio_2", "variance_ratio_5", "return_entropy",
        "sign_entropy", "permutation_entropy", "hurst_exponent",
    ),
    "volume": (
        "raw_volume", "dollar_volume", "relative_volume", "relative_dollar_volume", "volume_zscore",
        "dollar_volume_zscore", "volume_percentile_own", "volume_trend", "volume_acceleration",
        "volume_volatility", "up_bar_volume_share", "down_bar_volume_share", "volume_concentration",
        "largest_bar_volume_share", "trade_count", "relative_trade_count", "average_trade_size",
    ),
    "liquidity": (
        "amihud_illiquidity", "range_per_dollar_volume", "abs_return_per_dollar_volume",
        "roll_spread_proxy", "corwin_schultz_spread_proxy",
    ),
    "price_volume": (
        "return_x_rvol", "abs_return_x_rvol", "return_x_volume_z", "signed_volume_balance", "obv_slope",
        "return_volume_corr", "absreturn_volume_corr", "price_shock_minus_volume_shock",
        "volume_shock_minus_price_shock", "close_vs_vwap", "vwap_drift", "money_flow_proxy",
    ),
    "session": (
        "overnight_return", "regular_session_return", "overnight_minus_regular", "overnight_share_total",
        "positive_overnight_fraction", "positive_regular_fraction", "opening_gap", "gap_vs_atr",
        "gap_vs_prior_vol", "gap_fill_fraction_15m", "gap_fill_fraction_30m", "gap_fill_fraction_60m",
        "gap_fill_fraction_eod", "opening_return_5m", "opening_return_10m", "opening_return_15m",
        "opening_return_30m", "opening_return_60m", "closing_return_5m", "closing_return_10m",
        "closing_return_15m", "closing_return_30m", "closing_return_60m", "open_to_midday_return",
        "midday_to_close_return", "opening_range_pct_15m", "opening_range_pct_30m",
        "opening_range_pct_60m", "close_location_daily_range", "session_vwap_distance",
        "vwap_cross_count", "time_above_vwap", "opening_rvol_15m", "opening_rvol_30m",
        "opening_rvol_60m", "closing_volume_share_15m", "closing_volume_share_30m",
        "closing_volume_share_60m", "largest_1m_volume_share_session", "largest_5m_volume_share_session",
    ),
    "same_time": (
        "same_bucket_return_lag1", "same_bucket_return_mean2", "same_bucket_return_mean5",
        "same_bucket_return_mean10", "same_bucket_return_mean20", "same_bucket_rvol",
        "same_bucket_volatility", "same_bucket_return_rank",
    ),
    "market": (
        "market_beta", "downside_beta", "upside_beta", "market_correlation", "beta_change",
        "correlation_change", "market_lead_response", "stock_lead_market_response", "market_return",
        "market_risk_adjusted_momentum", "market_drawdown", "market_volatility", "market_downside_vol",
        "market_vol_percentile", "market_breadth_positive", "market_breadth_above_sma20",
        "market_breadth_above_sma50", "market_breadth_above_sma100", "market_breadth_above_sma200",
        "breadth_change", "breadth_momentum", "cross_sectional_return_dispersion",
        "cross_sectional_residual_dispersion", "cross_sectional_return_skew",
        "cross_sectional_return_kurtosis", "average_pairwise_correlation", "beta_dispersion",
        "market_gap", "market_intraday_return",
    ),
    "statistical_factors": (
        "pca_residual_return", "pca_residual_momentum", "pca_idiosyncratic_vol", "pc1_loading",
        "pc2_loading", "pc1_loading_change", "stat_peer_basket_return", "stock_minus_stat_peer_return",
        "stat_peer_momentum", "stat_peer_dispersion", "stat_peer_correlation", "stat_peer_lagged_return",
    ),
    "calendar": (
        "minute_of_day_sin", "minute_of_day_cos", "day_of_week", "month_end_distance",
        "quarter_end_distance", "pre_holiday_flag", "post_holiday_flag",
    ),
}


WINDOWS = {
    "returns": INTRADAY + DAILY, "multispeed": (),
    "path": MID, "consistency": MID, "ranks": DAILY, "location": MID,
    "candles": INTRADAY[:-1] + ("1d",), "moving_average": ("15m", "30m", "60m", "120m", "240m") + DAILY,
    "volatility": ("5m", "10m", "15m", "30m", "60m", "120m", "240m", "session") + DAILY[:-1],
    "distribution": MID, "serial": ("60m", "120m", "240m", "session", "5d", "10d", "20d", "40d", "63d"),
    "volume": ("5m", "10m", "15m", "30m", "60m", "120m", "240m", "session") + DAILY[:9],
    "liquidity": MID, "price_volume": ("5m", "10m", "15m", "30m", "60m", "120m", "240m", "session") + DAILY[:8],
    "session": ("1d", "2d", "5d", "10d", "20d"), "same_time": ("5m", "30m"),
    "market": ("5m", "15m", "30m", "60m", "120m", "240m", "session") + ("1d", "2d", "5d", "10d", "20d", "40d", "63d", "126d"),
    "statistical_factors": ("5d", "10d", "20d", "30d", "40d", "63d", "126d"), "calendar": ("context",),
}


def concept_catalog() -> tuple[FeatureConceptSpec, ...]:
    concepts: list[FeatureConceptSpec] = []
    for family, names in FAMILIES.items():
        grids = ("daily_close",) if family in {"ranks", "statistical_factors"} else GRIDS
        if family == "session":
            grids = ("daily_close", "preclose_1555")
        if family == "same_time":
            grids = ("intraday_5m", "intraday_1m")
        represented_families = {"returns", "volatility", "volume"}
        representations = ("raw", "own_history_percentile") if family in represented_families else ("raw",)
        required = ("open", "high", "low", "close", "volume")
        for name in names:
            concept_grids = grids
            if family == "market" and (name.startswith("market_breadth") or name.startswith("breadth_")
                                       or name.startswith("cross_sectional_")
                                       or name in {"average_pairwise_correlation", "beta_dispersion"}):
                concept_grids = ("daily_close",)
            scales = parse_scales(WINDOWS[family])
            if family == "multispeed":
                scales = tuple(
                    TimeScale(parse_scale(slow).kind, parse_scale(slow).value or 390, f"{fast}_{slow}")
                    for fast, slow in APPROVED_FAST_SLOW_PAIRS
                )
            active = family != "statistical_factors"
            unavailable_reason = None if active else "monthly frozen prior-only PCA and peer-model cache is not implemented; alternate proxy formulas are prohibited"
            concepts.append(FeatureConceptSpec(
                concept_id=name, family=family, description=name.replace("_", " "),
                valid_scales=scales, required_columns=required,
                decision_grids=concept_grids, representations=representations, min_observations=5,
                builder_key=family, active=active, unavailable_reason=unavailable_reason,
            ))
    for formation, skip in ((5,1),(10,1),(20,1),(20,5),(30,5),(40,5),(63,5),(63,20),(126,20),(252,20),(252,63)):
        concepts.append(FeatureConceptSpec(
            concept_id=f"return_skip_recent_{formation}d_ex_{skip}d", family="returns",
            description="skip momentum", valid_scales=(parse_scale(f"{formation}d"),),
            required_columns=("close",), decision_grids=("daily_close",), representations=REPRESENTATIONS,
            min_observations=formation + 1, builder_key="returns", parameters={"formation": formation, "skip": skip},
        ))
    return tuple(concepts)


@dataclass(frozen=True)
class RegistryBundle:
    concepts: tuple[FeatureConceptSpec, ...]
    features: tuple[CompiledFeatureSpec, ...]
    targets: tuple[CompiledTargetSpec, ...]
    unavailable: tuple[FeatureConceptSpec, ...]
    required_warmup_sessions: int

    def feature_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            vars(item) | {
                "scale": item.scale.label,
                "parameters": json.dumps(item.parameters, sort_keys=True, separators=(",", ":")),
            }
            for item in self.features
        ])

    def target_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(item) | {"horizon": item.horizon.label} for item in self.targets])


def _compile_features(config: AlphaDiscoveryConfig, concepts: Iterable[FeatureConceptSpec]) -> tuple[CompiledFeatureSpec, ...]:
    enabled_grids = {name for name, enabled in config.decision_grids.items() if enabled}
    configured_scales = {
        str(label)
        for labels in config.feature_windows.values()
        for label in labels
    }
    rows: list[CompiledFeatureSpec] = []
    for concept in concepts:
        if not concept.active:
            continue
        for scale, representation, grid in product(concept.valid_scales, concept.representations, concept.decision_grids):
            if grid not in enabled_grids:
                continue
            if scale.label != "context" and scale.label not in configured_scales:
                if concept.family != "multispeed":
                    continue
                fast_label, slow_label = scale.label.split("_", 1)
                if fast_label not in configured_scales or slow_label not in configured_scales:
                    continue
            if scale.kind == "minutes" and not grid.startswith("intraday"):
                continue
            if scale.kind == "sessions" and grid.startswith("intraday"):
                continue
            if scale.kind == "session_to_date" and grid == "daily_close":
                continue
            minimum_window = MINIMUM_WINDOW.get(concept.concept_id, 1)
            if scale.kind in {"minutes", "sessions"} and int(scale.value or 0) < minimum_window:
                continue
            parameters = dict(concept.parameters)
            if concept.family == "multispeed":
                fast_label, slow_label = scale.label.split("_", 1)
                parameters.update({"fast_label": fast_label, "slow_label": slow_label,
                                   "fast": parse_scale(fast_label).value or 390,
                                   "slow": parse_scale(slow_label).value or 390})
            feature_id = f"{concept.concept_id}__{scale.label}__{representation}__{grid}"
            payload = {
                "feature_id": feature_id, "concept_id": concept.concept_id, "family": concept.family,
                "scale": vars(scale), "representation": representation, "decision_grid": grid,
                "availability_rule": "availability_ts_utc <= decision_ts", "builder_key": concept.builder_key,
                "parameters": parameters, "price_basis": "split_consistent",
            }
            minimum_history = max(concept.min_observations, history_sessions(scale))
            if representation in {"own_history_zscore", "own_history_percentile"}:
                minimum_history += 252
            if concept.concept_id in {"benchmark_excess_return", "beta_residual_return", "idiosyncratic_vol",
                                      "market_beta", "downside_beta", "upside_beta", "beta_change",
                                      "cross_sectional_residual_dispersion", "beta_dispersion"}:
                minimum_history += 20 if grid.startswith("intraday") else 63
            if concept.family == "same_time":
                minimum_history += 20
            rows.append(CompiledFeatureSpec(
                feature_id=feature_id, concept_id=concept.concept_id, family=concept.family, scale=scale,
                representation=representation, decision_grid=grid,
                availability_rule="availability_ts_utc <= decision_ts",
                minimum_history=minimum_history,
                redundancy_group=f"{concept.concept_id}::{scale.label}::{grid}", builder_key=concept.builder_key,
                price_basis="split_consistent", definition_hash=stable_hash(payload), parameters=parameters,
            ))
    ids = [row.feature_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate compiled feature IDs")
    smoke_limit = int(config.universe.get("smoke_feature_limit", 0) or 0)
    if smoke_limit and len(rows) > smoke_limit:
        by_family: dict[str, list[CompiledFeatureSpec]] = {}
        for row in rows: by_family.setdefault(row.family, []).append(row)
        selected: list[CompiledFeatureSpec] = []
        while len(selected) < smoke_limit and any(by_family.values()):
            for family in sorted(by_family):
                if by_family[family] and len(selected) < smoke_limit:
                    selected.append(by_family[family].pop(0))
        rows = selected
    return tuple(rows)


def _target(grid: str, label: str, basis: str) -> CompiledTargetSpec:
    horizon = parse_scale(label)
    if grid.startswith("intraday"):
        entry = "first raw 1m bar with bar_start_ts_utc > decision_ts"
        exit_rule = "H-th executable 1m holding-bar close; missing if beyond RTH" if label != "EOD" else "final RTH close"
        family = "intraday"
    elif grid == "daily_close":
        entry = "raw open at D+1 market_open + configured delay"
        exit_rule = "close of H-th session counting entry session as 1"
        family = "interday"
    else:
        entry, exit_rule, family = "first actionable raw 1m bar after decision", "next session open + 1m", "overnight"
    target_id = f"target_{label.lower()}__{basis}__{grid}"
    payload = {"target_id": target_id, "entry_rule": entry, "exit_rule": exit_rule, "basis": basis}
    return CompiledTargetSpec(target_id, family, grid, horizon, entry, exit_rule, basis, "raw_execution", stable_hash(payload))


def _compile_targets(config: AlphaDiscoveryConfig) -> tuple[CompiledTargetSpec, ...]:
    targets: list[CompiledTargetSpec] = []
    bases = config.targets["bases"]
    for grid, enabled in config.decision_grids.items():
        if not enabled:
            continue
        labels = config.targets["intraday"] if grid.startswith("intraday") else config.targets["interday"] if grid == "daily_close" else ["overnight"]
        targets.extend(_target(grid, label, basis) for label, basis in product(labels, bases))
    ids = [row.target_id for row in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate target IDs")
    return tuple(targets)


def compile_registry(config: AlphaDiscoveryConfig) -> RegistryBundle:
    config.validate()
    concepts = concept_catalog()
    features = _compile_features(config, concepts)
    targets = _compile_targets(config)
    warmup = max((item.minimum_history for item in features), default=0) + int(config.warmup["safety_margin_sessions"])
    available = int(config.warmup.get("available_prior_sessions", 0))
    if config.warmup.get("fail_if_full_coverage_warmup_missing", True) and available < warmup:
        raise ValueError(f"Insufficient transitive warmup: registry requires {warmup} prior sessions, config declares {available}")
    return RegistryBundle(concepts, features, targets, tuple(c for c in concepts if not c.active), warmup)
