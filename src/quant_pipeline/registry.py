from __future__ import annotations

import ast
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    family: str
    lookback: int | None = None
    classification: str = "time_series"  # time_series|cross_sectional|context|categorical
    cutoff: str = "completed_bar_available_at"
    calculation_frequency: str = "5m"
    price_basis: str = "split_adjusted_research_price"
    dtype: str = "continuous"
    missing_rule: str = "exclude_pairwise"
    directional_meaning: str = "none"
    status: str = "requested"
    unavailable_reason: str | None = None
    required_columns: tuple[str, ...] = ()
    minimum_history_bars: int = 0
    session_reset: bool = True
    uses_current_bar: bool = True
    uses_previous_sessions: bool = False
    feature_available_offset_minutes: int = 0
    eligibility_requirement: str = "tradable_at_decision"
    baseline_inclusion: str = "not_applicable"
    # Phase 1B provenance fields are appended so existing positional FeatureSpec
    # construction remains compatible.
    discovery_phase: str = "1A"
    arity: int = 1
    parent_features: tuple[str, ...] = ()
    operator: str | None = None
    operator_parameters_json: str | None = None
    lineage_hash: str | None = None
    redundancy_group: str | None = None
    complexity_units: int = 1
    expected_direction: int = 0
    evidence_class: str | None = None


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    horizon_minutes: int | None
    entry_definition: str
    exit_definition: str
    classification: str = "raw"
    overlaps: bool = True
    tier: str = "exploratory"
    price_basis: str = "raw_execution_price"


def target_registry(horizons: Iterable[int] | None = None, primary_horizons: Iterable[int] | None = None) -> list[TargetSpec]:
    horizons = list(horizons) if horizons is not None else list(range(5, 390, 5))
    primary=set(primary_horizons or [5,15,30,60,120])
    raw = [
        TargetSpec(f"fwd_return_{m}m", f"Raw return over next {m} minutes", m, "first bar open at/after decision availability", f"first close at/after entry plus {m} minutes", tier="primary" if m in primary else "exploratory")
        for m in horizons
    ] + [TargetSpec("fwd_return_eod", "Raw return from next actionable open to scheduled RTH close", None, "first bar open at/after decision availability", "scheduled final complete RTH bar close", tier="primary")]
    adjusted = [TargetSpec(t.name+"_benchmark_adjusted", t.description+" minus benchmark return", t.horizon_minutes, t.entry_definition, t.exit_definition, "benchmark_adjusted", t.overlaps, t.tier) for t in raw]
    beta = [TargetSpec(t.name+"_beta_residual", t.description+" minus prior-only beta times benchmark return", t.horizon_minutes, t.entry_definition, t.exit_definition, "beta_residual", t.overlaps, t.tier) for t in raw]
    return raw + adjusted + beta


def feature_registry(lookbacks: Iterable[int], sector_available: bool = False, opening_windows: Iterable[int] | None = None) -> list[FeatureSpec]:
    """Registry is the scanner authority; construction never guesses by name."""
    out: list[FeatureSpec] = []
    def add(name: str, description: str, family: str, lb: int | None = None, cls: str = "time_series", direction: str = "none", status: str = "requested", reason: str | None = None, **metadata) -> None:
        previous_session_names={"overnight_gap","previous_session_return","continuous_return_1"}
        defaults={"required_columns":("symbol","session_date","decision_ts","open_adjusted","high_adjusted","low_adjusted","close_adjusted","volume"),"minimum_history_bars":int(lb or 0),"session_reset":family not in {"normalization","sector"} and name not in previous_session_names,"uses_previous_sessions":family in {"normalization","sector"} or name in previous_session_names or name.startswith("session_return_"),"feature_available_offset_minutes":0,"eligibility_requirement":"point_in_time_tradable_cross_section" if cls=="cross_sectional" else "valid_completed_bar"}
        defaults.update(metadata)
        out.append(FeatureSpec(name, description, family, lb, cls, directional_meaning=direction, status=status, unavailable_reason=reason, **defaults))
    rolling = {
        "return": ("session-reset intraday simple return", "momentum"), "log_return": ("session-reset intraday log return", "momentum"),
        "realized_vol": ("return standard deviation", "volatility"), "downside_vol": ("downside volatility", "volatility"),
        "upside_vol": ("upside volatility", "volatility"), "return_vol_ratio": ("return divided by volatility", "momentum"),
        "volume_sum": ("volume sum", "volume"), "volume_mean": ("average volume", "volume"),
        "relative_volume": ("current volume divided by rolling mean", "volume"), "dollar_volume_mean": ("average dollar volume", "volume"),
        "range_mean": ("average high-low range", "volatility"), "range_ratio": ("current range divided by rolling range", "volatility"),
        "return_consistency": ("share positive returns", "momentum"), "positive_return_sum": ("sum positive returns", "momentum"),
        "negative_return_sum": ("sum negative returns", "momentum"), "largest_positive_bar": ("largest positive return", "momentum"),
        "largest_negative_bar": ("largest negative return", "momentum"), "distance_rolling_high": ("distance from rolling high", "price_location"),
        "distance_rolling_low": ("distance from rolling low", "price_location"), "range_position": ("position in rolling range", "price_location"),
        "breakout_magnitude": ("close above prior rolling high", "momentum"), "breakdown_magnitude": ("close below prior rolling low", "momentum"),
        "volume_acceleration": ("short versus full-window volume", "volume"), "volatility_acceleration": ("short versus full-window volatility", "volatility"),
        "rolling_vwap_distance": ("distance from rolling VWAP", "vwap"), "volume_range_ratio": ("volume divided by range", "volume"),
        "return_volume_product": ("return multiplied by volume", "volume"), "return_outlier_score": ("return divided by recent volatility", "volatility"),
    }
    for n in lookbacks:
        for prefix, (desc, family) in rolling.items(): add(f"{prefix}_{n}", f"{n}-bar {desc}", family, n)
        add(f"relative_volume_prior_{n}",f"current volume divided by prior {n}-bar intraday mean","volume",n,baseline_inclusion="prior_only")
        add(f"relative_volume_inclusive_{n}",f"current volume divided by inclusive {n}-bar intraday mean","volume",n,baseline_inclusion="inclusive")
        for base in ["return", "relative_volume", "realized_vol", "range_position", "return_vol_ratio"]:
            add(f"{base}_rank_{n}", f"cross-sectional percentile rank of {base}_{n}", "cross_sectional", n, "cross_sectional")
    binary_names={"inside_bar","outside_bar","higher_high","higher_low","lower_high","lower_low","vwap_cross"}
    for name, desc, family in [
        ("intraday_return_1", "one-bar session-reset intraday return", "momentum"),
        ("continuous_return_1", "continuous close-to-close return including overnight movement", "momentum"),
        ("return_since_open", "return from session opening price", "opening"), ("overnight_gap", "open versus prior session close", "opening"),
        ("previous_session_return", "previous session close-to-close return", "momentum"), ("bar_range_pct", "bar high-low range / close", "volatility"),
        ("body_to_range", "absolute body divided by range", "bar_structure"), ("upper_wick_to_range", "upper wick divided by range", "bar_structure"),
        ("lower_wick_to_range", "lower wick divided by range", "bar_structure"), ("close_location", "close location in bar", "bar_structure"),
        ("distance_session_vwap", "close / causal session VWAP - 1", "vwap"), ("distance_session_high", "close / session high - 1", "vwap"),
        ("distance_session_low", "close / session low - 1", "vwap"), ("minutes_since_open", "minutes elapsed since regular-session open", "calendar"),
        ("day_of_week", "exchange weekday", "calendar"),
        ("universe_breadth_positive", "share of universe with positive one-bar return", "breadth"),
        ("universe_return_dispersion", "cross-sectional return dispersion", "breadth"),
        ("bar_return", "current completed-bar return", "bar_structure"), ("bar_log_return", "current completed-bar log return", "bar_structure"),
        ("body_pct", "signed bar body divided by open", "bar_structure"), ("absolute_body_pct", "absolute body divided by open", "bar_structure"),
        ("inside_bar", "inside previous bar", "bar_structure"), ("outside_bar", "outside previous bar", "bar_structure"),
        ("higher_high", "high above prior high", "bar_structure"), ("higher_low", "low above prior low", "bar_structure"),
        ("lower_high", "high below prior high", "bar_structure"), ("lower_low", "low below prior low", "bar_structure"),
        ("session_range_position", "position within causal session range", "price_location"),
        ("cumulative_volume", "causal cumulative session volume", "volume"), ("current_volume_share", "current volume share of session volume", "volume"),
        ("vwap_slope", "one-bar change in causal session VWAP", "vwap"), ("vwap_cross", "crossing of causal session VWAP", "vwap"),
        ("consecutive_positive_bars", "positive-bar run length", "momentum"), ("consecutive_negative_bars", "negative-bar run length", "momentum"),
        ("month", "calendar month", "calendar"), ("quarter", "calendar quarter", "calendar"),
        ("minute_of_session", "regular-session minute", "calendar"), ("minutes_until_close", "minutes to regular close", "calendar"),
        ("decision_time_bucket", "30-minute decision-time bucket", "calendar"),
        ("market_up", "benchmark positive at decision", "market_context"), ("high_market_vol", "benchmark volatility above expanding median", "market_context"),
    ]: add(name, desc, family, cls="categorical" if name in {"day_of_week","month","quarter","market_up","high_market_vol","decision_time_bucket"} else "time_series",dtype="categorical" if name in {"day_of_week","month","quarter","market_up","high_market_vol","decision_time_bucket"} else "binary" if name in binary_names else "continuous")
    for name,desc in [("decision_time_sin","cyclical sine encoding of decision minute"),("decision_time_cos","cyclical cosine encoding of decision minute")]:add(name,desc,"calendar")
    for minutes in (list(opening_windows) if opening_windows is not None else [1,3,5,10,15,20,30,45,60,90]):
        for base in ["opening_return", "opening_range", "opening_volume", "opening_realized_vol", "opening_close_location", "distance_opening_high", "distance_opening_low", "opening_breakout", "opening_breakdown"]:
            add(f"{base}_{minutes}m", f"{base.replace('_',' ')} after completed {minutes}-minute opening window", "opening", minutes, dtype="binary" if base in {"opening_breakout","opening_breakdown"} else "continuous")
    for lag in [1, 2, 3, 5, 10]:
        for base in ["lagged_market_return", "lagged_breadth_change", "lagged_dispersion_change", "unreacted_market_move"]:
            add(f"{base}_{lag}", f"{lag}-bar {base.replace('_',' ')}", "lead_lag", lag)
    for base in ["return_1", "relative_volume_10", "realized_vol_10", "range_position_10", "distance_session_vwap"]:
        for bars in [1560, 4680]:
            add(f"{base}_z_{bars}", f"point-in-time rolling z-score of {base}", "normalization", bars)
            # Ratios to a rolling mean are only defined for naturally positive
            # features. Signed returns and VWAP distances have means that cross
            # zero, which creates enormous, meaningless ratio spikes.
            if base in {"relative_volume_10", "realized_vol_10", "range_position_10"}:
                add(f"{base}_mean_ratio_{bars}", f"value divided by point-in-time rolling mean of {base}", "normalization", bars)
            add(f"{base}_median_diff_{bars}", f"value minus point-in-time rolling median of {base}", "normalization", bars)
    for sessions in [20,60]:
        add(f"tod_relative_volume_{sessions}", f"volume versus same session minute over prior {sessions} sessions", "volume", sessions)
        add(f"tod_cumulative_relative_volume_{sessions}", f"cumulative volume versus same session minute over prior {sessions} sessions", "volume", sessions)
    for sessions in [2,5]: add(f"session_return_{sessions}", f"return over prior {sessions} completed sessions", "momentum", sessions)
    for n in lookbacks:
        add(f"market_return_{n}", f"benchmark {n}-bar return", "market_context", n)
        add(f"stock_minus_market_return_{n}", f"stock minus benchmark {n}-bar return", "market_context", n)
    for n in [20,60]:add(f"market_residual_return_{n}",f"stock return minus prior-only rolling beta times benchmark return over {n} bars","market_context",n,uses_previous_sessions=True,baseline_inclusion="prior_only")
    reason = "No point-in-time sector classification supplied" if not sector_available else None
    for name in ["sector_return", "stock_minus_sector_return", "sector_rank", "sector_breadth_positive"]:
        add(name, "Sector-dependent feature", "sector", status="requested" if sector_available else "unavailable", reason=reason)
    return out


def registry_frame(specs: Iterable[FeatureSpec | TargetSpec]):
    import pandas as pd
    return pd.DataFrame([asdict(x) for x in specs])


def _load_registry(path: str | Path, cls):
    """Strictly deserialize a persisted CSV/JSON registry.

    Older Phase 1A runs persist CSV, so tuple and nullable fields must be
    reconstructed rather than guessed or silently discarded.
    """
    source = Path(path)
    if source.suffix.lower() == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
    else:
        import pandas as pd
        frame = pd.read_csv(source)
        records = frame.where(frame.notna(), None).to_dict("records")
    allowed = {item.name: item for item in fields(cls)}
    required = {
        item.name for item in fields(cls)
        if item.default is MISSING and item.default_factory is MISSING
    }
    result = []
    for row_number, record in enumerate(records, start=2):
        unknown = set(record) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} registry fields at row {row_number}: {sorted(unknown)}")
        missing = required - {key for key, value in record.items() if value is not None}
        if missing:
            raise ValueError(f"Missing {cls.__name__} registry fields at row {row_number}: {sorted(missing)}")
        values = {}
        for key, value in record.items():
            missing_value = value is None or (not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))) if source.suffix.lower() != ".json" else value is None
            if missing_value:
                values[key] = None
            elif key in {"required_columns", "parent_features"}:
                if isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except (SyntaxError, ValueError) as exc:
                        raise ValueError(f"Malformed tuple field {key} at row {row_number}") from exc
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"Registry field {key} must be a list or tuple at row {row_number}")
                values[key] = tuple(str(item) for item in value)
            elif key in {"session_reset", "uses_current_bar", "uses_previous_sessions", "overlaps"}:
                if isinstance(value, str):
                    if value.lower() not in {"true", "false"}:
                        raise ValueError(f"Malformed boolean field {key} at row {row_number}")
                    value = value.lower() == "true"
                values[key] = bool(value)
            elif key in {"lookback", "minimum_history_bars", "feature_available_offset_minutes", "arity", "complexity_units", "expected_direction", "horizon_minutes"}:
                values[key] = None if value is None else int(value)
            else:
                values[key] = value
        result.append(cls(**values))
    names = [item.name for item in result]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate names in {cls.__name__} registry")
    return result


def load_feature_registry(path: str | Path) -> list[FeatureSpec]:
    return _load_registry(path, FeatureSpec)


def load_target_registry(path: str | Path) -> list[TargetSpec]:
    return _load_registry(path, TargetSpec)


def write_registry_json(specs: Iterable[FeatureSpec | TargetSpec], path: str | Path) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps([asdict(item) for item in specs], indent=2), encoding="utf-8")
    temporary.replace(destination)
