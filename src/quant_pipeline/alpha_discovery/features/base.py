from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

import numpy as np
import pandas as pd

from ..models import CompiledFeatureSpec
from .formulas import FORMULAS, absolute_return_hhi, corwin_schultz_pair, information_discreteness, path_chord_convexity, path_efficiency, path_quadratic_curvature, top_abs_return_share


def cross_sectional_rank(values: pd.Series, decisions: pd.Series) -> pd.Series:
    return values.groupby(decisions, sort=False).rank(method="average", pct=True)


def apply_representation(values: pd.Series, frame: pd.DataFrame, representation: str, baseline: int = 252) -> pd.Series:
    if representation == "raw":
        return values
    groups = [frame.security_id]
    shifted = values.groupby(groups, sort=False).shift(1)
    if representation == "own_history_zscore":
        rolling = shifted.groupby(groups, sort=False).rolling(baseline, min_periods=max(5, min(63, baseline))).agg(["mean", "std"]).reset_index(level=0, drop=True)
        return (values - rolling["mean"]) / rolling["std"].replace(0, np.nan)
    if representation == "own_history_percentile":
        def percentile(window: np.ndarray) -> float:
            return float(np.mean(window[:-1] <= window[-1])) if len(window) > 1 else np.nan
        return values.groupby(groups, sort=False).rolling(baseline + 1, min_periods=max(6, min(64, baseline + 1))).apply(percentile, raw=True).reset_index(level=0, drop=True)
    if representation == "same_time_of_day_zscore":
        local = pd.to_datetime(frame.decision_ts, utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        keys = [frame.security_id, local]
        prior = values.groupby(keys, sort=False).shift(1)
        stats = prior.groupby(keys, sort=False).rolling(20, min_periods=10).agg(["mean", "std"]).reset_index(level=[0, 1], drop=True)
        return (values - stats["mean"]) / stats["std"].replace(0, np.nan)
    raise ValueError(f"Unknown representation: {representation}")


def _rolling_array(series: pd.Series, groups: pd.Series, window: int, function: Callable[[np.ndarray], float], minimum: int = 5) -> pd.Series:
    required = min(window, max(minimum, 5, ceil(0.60 * window)))
    return series.groupby(groups, sort=False).rolling(window, min_periods=required).apply(function, raw=True).reset_index(level=0, drop=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.Series(denominator, index=numerator.index, dtype=float)
    finite = np.abs(denominator.to_numpy(dtype=float)); finite = finite[np.isfinite(finite)]
    scale = float(np.median(finite)) if len(finite) else np.nan
    eps = max(np.finfo(float).eps, scale * 1e-12) if np.isfinite(scale) else np.finfo(float).eps
    valid = denominator.abs().gt(eps)
    return pd.Series(np.where(valid, numerator / denominator, np.nan), index=numerator.index, dtype=float)


def _linear_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]
    y = y[valid]
    x -= x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator else np.nan


def _slope_acceleration(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    midpoint = len(y) // 2
    if midpoint < 2 or len(y) - midpoint < 2:
        return np.nan
    left = _linear_slope(y[:midpoint])
    right = _linear_slope(y[midpoint:])
    return right - left if np.isfinite(left) and np.isfinite(right) else np.nan


def _normalized_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    return _linear_slope(y) * (len(y) - 1) / 2


def _trend_stat(values: np.ndarray, output: str) -> float:
    y = np.log(np.asarray(values, dtype=float))
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    y = y[valid]; x = np.linspace(-1.0, 1.0, len(y)); design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients; residual = y - fitted
    sst = float(np.sum((y - y.mean()) ** 2)); sse = float(np.sum(residual ** 2))
    if output == "r2": return 1 - sse / sst if sst > 0 else np.nan
    if output == "slope": return float(coefficients[1])
    variance = sse / (len(y) - 2); inverse = np.linalg.inv(design.T @ design)
    se = np.sqrt(max(variance * inverse[1, 1], 0))
    return float(coefficients[1] / se) if se > 0 else np.nan


def _return_acceleration(values: np.ndarray, thirds: bool = False) -> float:
    prices = np.asarray(values, dtype=float)
    pieces = 3 if thirds else 2
    if len(prices) < pieces + 1 or np.any(prices <= 0): return np.nan
    boundaries = np.linspace(0, len(prices) - 1, pieces + 1).round().astype(int)
    first = prices[boundaries[1]] / prices[boundaries[0]] - 1
    last = prices[boundaries[-1]] / prices[boundaries[-2]] - 1
    return float(last - first)


def _run_stat(values: np.ndarray, mode: str) -> float:
    signs = np.sign(np.asarray(values, dtype=float)); runs: list[tuple[float, int]] = []
    for sign in signs:
        if sign == 0 or not np.isfinite(sign): continue
        if runs and runs[-1][0] == sign: runs[-1] = (sign, runs[-1][1] + 1)
        else: runs.append((sign, 1))
    if not runs or not len(signs): return np.nan
    if mode == "average_signed_run_length": return float(np.mean([length for _, length in runs]) / len(signs))
    wanted = 1 if mode == "longest_up_run" else -1
    return float(max([length for sign, length in runs if sign == wanted] or [0]) / len(signs))


def _corwin_window(high_low_pairs: np.ndarray) -> float:
    values = np.asarray(high_low_pairs, dtype=float)
    if len(values) < 2: return np.nan
    spreads = [corwin_schultz_pair(values[i-1:i+1, 0], values[i-1:i+1, 1]) for i in range(1, len(values))]
    valid = np.asarray(spreads, dtype=float); valid = valid[np.isfinite(valid)]
    return float(np.median(valid)) if len(valid) else np.nan


def _standardized_moment(values: np.ndarray, order: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(5, order + 1):
        return np.nan
    centered = values - values.mean()
    sigma = float(np.sqrt(np.mean(centered ** 2)))
    return float(np.mean((centered / sigma) ** order)) if sigma > 0 else np.nan


def _prior_percentile(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values[-1]):
        return np.nan
    prior = values[:-1]
    prior = prior[np.isfinite(prior)]
    return float(np.mean(prior <= values[-1])) if len(prior) else np.nan


def _regression_slope(pair: np.ndarray) -> float:
    pair = np.asarray(pair, dtype=float)
    valid = np.isfinite(pair).all(axis=1)
    if valid.sum() < 3:
        return np.nan
    y, x = pair[valid, 0], pair[valid, 1]
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else np.nan


def _rolling_corwin(high: pd.Series, low: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    output = pd.Series(np.nan, index=high.index, dtype=float)
    minimum = max(2, min(window, max(5, ceil(0.60 * window))))
    for _, indices in high.groupby(groups, sort=False).groups.items():
        loc = np.asarray(indices)
        h = high.loc[loc].to_numpy(float); l = low.loc[loc].to_numpy(float)
        values = np.full(len(loc), np.nan)
        for end in range(minimum - 1, len(loc)):
            start = max(0, end - window + 1)
            values[end] = _corwin_window(np.column_stack([h[start:end + 1], l[start:end + 1]]))
        output.loc[loc] = values
    return output


@dataclass
class FeatureBuilder:
    """Causal feature builder for ordered single-scale panels.

    Large production runs call this on security/family partitions and persist
    each block, avoiding a global observations-by-features dense matrix.
    """
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"security_id", "decision_ts", "open", "high", "low", "close", "volume"}
        missing = required - set(self.frame)
        if missing:
            raise ValueError(f"Feature panel missing columns: {sorted(missing)}")
        self.frame = self.frame.sort_values(["security_id", "decision_ts"], kind="mergesort").copy()
        if "research_close" not in self.frame:
            self.frame["research_close"] = self.frame.close
        self._cache: dict[tuple[str, int], pd.Series] = {}
        self._direct_cache: dict[tuple, pd.Series] = {}

    def build(self, spec: CompiledFeatureSpec) -> pd.Series:
        intraday = spec.decision_grid.startswith("intraday")
        n = 390 if spec.scale.kind == "session_to_date" and intraday else max(1, int(spec.scale.value or 1))
        if spec.scale.kind in {"minutes", "session_to_date"} and intraday:
            group = self.frame.security_id.astype(str) + "|" + self.frame.session_date.astype(str)
            self._group_kind = "security_session"
        else:
            group = self.frame.security_id
            self._group_kind = "security"
        self._current_group = group
        price = self.frame.research_close.astype(float)
        returns = price.groupby(group, sort=False).pct_change(fill_method=None)
        benchmark_price = self.frame["benchmark_close"].astype(float) if "benchmark_close" in self.frame else pd.Series(np.nan, index=self.frame.index)
        benchmark_returns = benchmark_price.groupby(group, sort=False).pct_change(fill_method=None)
        self._benchmark_simple = benchmark_price / benchmark_price.groupby(group, sort=False).shift(n) - 1
        self._benchmark_returns = benchmark_returns
        self._intraday = intraday
        concept = spec.concept_id

        direct_key = (concept, n, spec.decision_grid, tuple(sorted(spec.parameters.items())))
        if direct_key not in self._direct_cache:
            self._direct_cache[direct_key] = self._direct(concept, n, group, price, returns, spec)
        direct = self._direct_cache[direct_key]
        direct.name = spec.feature_id
        return apply_representation(direct, self.frame, spec.representation)

    def _direct(self, concept: str, n: int, group: pd.Series, price: pd.Series, returns: pd.Series, spec: CompiledFeatureSpec) -> pd.Series:
        shifted = price.groupby(group, sort=False).shift(n)
        simple = price / shifted - 1
        if concept == "simple_return": return simple
        if concept == "log_return": return np.log(price / shifted)
        if concept.startswith("return_skip_recent_"):
            formation, skip = int(spec.parameters["formation"]), int(spec.parameters["skip"])
            return price.groupby(group).shift(skip) / price.groupby(group).shift(formation) - 1
        if concept in {"benchmark_excess_return", "beta_residual_return"}:
            benchmark = self._benchmark_simple
            beta = self._beta(returns)
            return simple - (beta if concept == "beta_residual_return" else 1.0) * benchmark
        rolling_std = self._rolling(returns.rename("returns"), n, "std", 3)
        rolling_mean = self._rolling(returns.rename("returns"), n, "mean", 2)
        realized = np.sqrt(self._rolling(returns.pow(2).rename("squared_returns"), n, "sum", 3))
        downside = np.sqrt(self._rolling(returns.clip(upper=0).pow(2).rename("down_squared"), n, "mean", 3))
        if concept == "return_over_realized_vol": return _safe_divide(simple, realized)
        if concept == "return_over_downside_vol": return _safe_divide(simple, downside)
        if concept == "return_t_stat": return _safe_divide(rolling_mean, rolling_std / np.sqrt(n))
        if concept == "positive_return_sum": return returns.clip(lower=0).groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        if concept == "negative_return_sum_abs": return returns.clip(upper=0).abs().groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        if concept == "up_down_return_ratio":
            up = returns.clip(lower=0).groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
            down = returns.clip(upper=0).abs().groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
            return up / down.replace(0, np.nan)
        if concept in FORMULAS and concept not in {"parkinson_vol", "garman_klass_vol", "rogers_satchell_vol", "corwin_schultz_spread_proxy"}:
            source = returns if concept in {"information_discreteness", "absolute_return_hhi", "return_entropy", "sign_entropy"} else price
            func = FORMULAS[concept]
            if concept in {"return_entropy", "sign_entropy"}:
                func = lambda x: FORMULAS[concept](np.sign(x))
            return _rolling_array(source, group, n, func)
        if concept.startswith("top") and concept.endswith("_abs_return_share"):
            k = int(concept[3])
            return _rolling_array(returns, group, n, lambda x: top_abs_return_share(x, k))
        if concept in {"close_to_close_vol", "realized_vol", "downside_vol", "upside_vol", "down_up_vol_ratio", "downside_semivariance", "upside_semivariance", "semivariance_ratio"}:
            down = returns.clip(upper=0).pow(2).groupby(group).rolling(n, min_periods=min(n, 5)).mean().reset_index(level=0, drop=True)
            up = returns.clip(lower=0).pow(2).groupby(group).rolling(n, min_periods=min(n, 5)).mean().reset_index(level=0, drop=True)
            values = {"close_to_close_vol": rolling_std, "realized_vol": np.sqrt(returns.pow(2).groupby(group).rolling(n, min_periods=min(n, 5)).sum().reset_index(level=0, drop=True)),
                      "downside_vol": np.sqrt(down), "upside_vol": np.sqrt(up), "down_up_vol_ratio": np.sqrt(down) / np.sqrt(up).replace(0, np.nan),
                      "downside_semivariance": down, "upside_semivariance": up, "semivariance_ratio": down / up.replace(0, np.nan)}
            return values[concept]
        if concept in {"realized_skewness", "realized_kurtosis"}:
            order = 3 if concept == "realized_skewness" else 4
            return self._apply(returns, n, lambda x: _standardized_moment(x, order), concept, 5)
        if concept in {"max_subreturn", "min_subreturn", "p95_subreturn", "p05_subreturn"}:
            functions = {"max_subreturn": np.nanmax, "min_subreturn": np.nanmin, "p95_subreturn": lambda x: np.nanquantile(x, .95), "p05_subreturn": lambda x: np.nanquantile(x, .05)}
            return _rolling_array(returns, group, n, functions[concept])
        if concept == "tail_spread":
            return _rolling_array(returns, group, n, lambda x: np.nanquantile(x, .95) - np.nanquantile(x, .05))
        if concept in {"raw_volume", "dollar_volume"}:
            source = self.frame.volume if concept == "raw_volume" else self.frame.volume * ((self.frame.high + self.frame.low + self.frame.close) / 3)
            return source.groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        if concept == "body_to_range": return (self.frame.close - self.frame.open).abs() / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept == "signed_body_to_range": return (self.frame.close - self.frame.open) / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept == "upper_wick_fraction": return (self.frame.high - self.frame[["open", "close"]].max(axis=1)) / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept == "lower_wick_fraction": return (self.frame[["open", "close"]].min(axis=1) - self.frame.low) / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept == "close_location_bar": return (self.frame.close - self.frame.low) / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept == "high_low_asymmetry":
            return (self.frame.high + self.frame.low - self.frame.open - self.frame.close) / (self.frame.high - self.frame.low).replace(0, np.nan)
        if concept in {"range_position", "recovery_fraction"}:
            high = price.groupby(group).rolling(n, min_periods=1).max().reset_index(level=0, drop=True); low = price.groupby(group).rolling(n, min_periods=1).min().reset_index(level=0, drop=True)
            return (price - low) / (high - low).replace(0, np.nan)
        if concept in {"distance_from_high", "drawdown_from_high"}: return price / price.groupby(group).rolling(n, min_periods=1).max().reset_index(level=0, drop=True) - 1
        if concept in {"distance_from_low", "recovery_from_low"}: return price / price.groupby(group).rolling(n, min_periods=1).min().reset_index(level=0, drop=True) - 1
        if concept in {"price_to_sma", "bollinger_z"}:
            mean = price.groupby(group).rolling(n, min_periods=min(n, 2)).mean().reset_index(level=0, drop=True)
            return price / mean - 1 if concept == "price_to_sma" else (price - mean) / price.groupby(group).rolling(n, min_periods=min(n, 2)).std().reset_index(level=0, drop=True)
        if concept == "ema_distance":
            ema = price.groupby(group, group_keys=False).apply(lambda s: s.ewm(span=n, adjust=False, min_periods=min(n, 2)).mean(), include_groups=False)
            return price / ema.reset_index(level=0, drop=True) - 1
        if concept == "amihud_illiquidity":
            dollar = self.frame.volume * ((self.frame.high + self.frame.low + self.frame.close) / 3)
            ratio = returns.abs() / dollar.replace(0, np.nan)
            return ratio.groupby(group).rolling(n, min_periods=min(n, 2)).mean().reset_index(level=0, drop=True)
        if concept == "abs_return_per_dollar_volume":
            dollar = self.frame.volume * ((self.frame.high + self.frame.low + self.frame.close) / 3)
            total_dollar = dollar.groupby(group).rolling(n, min_periods=min(n, 2)).sum().reset_index(level=0, drop=True)
            return _safe_divide(simple.abs(), total_dollar)
        if concept == "signed_volume_balance":
            signed = np.sign(returns) * self.frame.volume
            return signed.groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True) / self.frame.volume.groupby(group).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        if concept == "close_vs_vwap":
            dollar = self.frame.get("vwap", (self.frame.high + self.frame.low + self.frame.close) / 3) * self.frame.volume
            rolling_dollar = dollar.groupby(group).rolling(n, min_periods=min(n, 2)).sum().reset_index(level=0, drop=True)
            rolling_volume = self.frame.volume.groupby(group).rolling(n, min_periods=min(n, 2)).sum().reset_index(level=0, drop=True)
            window_vwap = _safe_divide(rolling_dollar, rolling_volume)
            return _safe_divide(self.frame.close - window_vwap, self.frame.close)
        if concept in {"minute_of_day_sin", "minute_of_day_cos"}:
            local = pd.to_datetime(self.frame.decision_ts, utc=True).dt.tz_convert("America/New_York")
            minute = (local.dt.hour * 60 + local.dt.minute - 570).clip(0, 390)
            angle = 2 * np.pi * minute / 390
            return np.sin(angle) if concept.endswith("sin") else np.cos(angle)
        if concept == "day_of_week": return pd.to_datetime(self.frame.decision_ts, utc=True).dt.dayofweek.astype(float)
        extended = self._extended(concept, n, group, price, returns, simple, spec)
        if extended is not None:
            return extended
        raise KeyError(f"No exact calculation mapping for active feature concept {concept!r}")

    def _beta(self, returns: pd.Series) -> pd.Series:
        if "beta_prior" in self.frame:
            return self.frame.beta_prior.astype(float)
        key = ("beta_prior", 20 * 390 if self._intraday else 63)
        if key not in self._cache:
            observations_per_session = 78 if self._intraday and self.frame.decision_grid.astype(str).eq("intraday_5m").all() else 390
            window = 20 * observations_per_session if self._intraday else 63
            minimum = 10 * observations_per_session if self._intraday else 40
            stock_source = self.frame.bucket_return.astype(float) if self._intraday and "bucket_return" in self.frame else returns
            market_source = self.frame.benchmark_bucket_return.astype(float) if self._intraday and "benchmark_bucket_return" in self.frame else self._benchmark_returns
            stock_prior = stock_source.groupby(self.frame.security_id, sort=False).shift(1)
            market_prior = market_source.groupby(self.frame.security_id, sort=False).shift(1)
            security = self.frame.security_id
            mean_x = stock_prior.groupby(security, sort=False).rolling(window, min_periods=minimum).mean().reset_index(level=0, drop=True)
            mean_y = market_prior.groupby(security, sort=False).rolling(window, min_periods=minimum).mean().reset_index(level=0, drop=True)
            mean_xy = (stock_prior * market_prior).groupby(security, sort=False).rolling(window, min_periods=minimum).mean().reset_index(level=0, drop=True)
            mean_y2 = market_prior.pow(2).groupby(security, sort=False).rolling(window, min_periods=minimum).mean().reset_index(level=0, drop=True)
            covariance = mean_xy - mean_x * mean_y
            variance = mean_y2 - mean_y.pow(2)
            self._cache[key] = _safe_divide(covariance, variance)
        return self._cache[key]

    def _rolling(self, values: pd.Series, n: int, operation: str, minimum: int = 2) -> pd.Series:
        key = (f"{values.name}:{operation}:{self._group_kind}", n)
        if key in self._cache: return self._cache[key]
        required = min(n, max(minimum, 5, ceil(0.60 * n)))
        rolling = values.groupby(self._current_group, sort=False).rolling(n, min_periods=required)
        result = getattr(rolling, operation)().reset_index(level=0, drop=True)
        self._cache[key] = result
        return result

    def _apply(self, values: pd.Series, n: int, function: Callable[[np.ndarray], float], name: str, minimum: int = 3) -> pd.Series:
        key = (f"{name}:{self._group_kind}", n)
        if key not in self._cache:
            self._cache[key] = _rolling_array(values, self._current_group, n, function, minimum)
        return self._cache[key]

    def _rolling_corr(self, left: pd.Series, right: pd.Series, n: int, minimum: int = 3) -> pd.Series:
        output = pd.Series(np.nan, index=self.frame.index, dtype=float)
        for _, indices in self.frame.groupby(self._current_group, sort=False).groups.items():
            index = pd.Index(indices)
            output.loc[index] = left.loc[index].rolling(n, min_periods=min(n, minimum)).corr(right.loc[index]).to_numpy()
        return output

    def _rolling_pair_slope(self, dependent: pd.Series, independent: pd.Series, n: int,
                            minimum: int = 3, condition: str | None = None) -> pd.Series:
        output = pd.Series(np.nan, index=self.frame.index, dtype=float)
        required = min(n, max(minimum, 5, ceil(.60 * n)))
        for _, indices in dependent.groupby(self._current_group, sort=False).groups.items():
            loc = np.asarray(indices); y = dependent.loc[loc].to_numpy(float); x = independent.loc[loc].to_numpy(float)
            for end in range(required - 1, len(loc)):
                start = max(0, end - n + 1); pair = np.column_stack([y[start:end + 1], x[start:end + 1]])
                if condition == "down": pair = pair[pair[:, 1] < 0]
                elif condition == "up": pair = pair[pair[:, 1] > 0]
                output.loc[loc[end]] = _regression_slope(pair)
        return output

    def _extended(self, concept: str, n: int, group: pd.Series, price: pd.Series, returns: pd.Series, simple: pd.Series, spec: CompiledFeatureSpec) -> pd.Series | None:
        f = self.frame; eps = np.finfo(float).eps
        fast = int(spec.parameters.get("fast", max(1, n // 4))); slow = int(spec.parameters.get("slow", max(n, fast + 1)))
        fast_return = price / price.groupby(group).shift(fast) - 1; slow_return = price / price.groupby(group).shift(slow) - 1
        rstd = self._rolling(returns.rename("returns"), n, "std", 3); rmean = self._rolling(returns.rename("returns"), n, "mean", 2)
        high_n = self._rolling(f.high.rename("high"), n, "max", 1); low_n = self._rolling(f.low.rename("low"), n, "min", 1)
        volume_n = self._rolling(f.volume.astype(float).rename("volume"), n, "sum", 1)
        dollar = (((f.high + f.low + f.close) / 3) * f.volume).rename("dollar")
        dollar_n = self._rolling(dollar, n, "sum", 1)

        if concept in {"fast_minus_slow_return", "momentum_acceleration", "momentum_deceleration", "momentum_sign_agreement", "pullback_in_uptrend", "recovery_in_downtrend", "fast_minus_slow_cross_sectional_rank", "fast_minus_slow_risk_adjusted_momentum"}:
            acceleration = fast_return / fast - slow_return / slow
            values = {
                "fast_minus_slow_return": fast_return - slow_return,
                "momentum_acceleration": acceleration, "momentum_deceleration": -acceleration,
                "momentum_sign_agreement": np.sign(fast_return) * np.sign(slow_return),
                "pullback_in_uptrend": ((fast_return < 0) & (slow_return > 0)).astype(float) * fast_return.abs() * slow_return.clip(lower=0),
                "recovery_in_downtrend": ((fast_return > 0) & (slow_return < 0)).astype(float) * fast_return.abs() * slow_return.clip(upper=0).abs(),
                "fast_minus_slow_cross_sectional_rank": cross_sectional_rank(fast_return, f.decision_ts) - cross_sectional_rank(slow_return, f.decision_ts),
                "fast_minus_slow_risk_adjusted_momentum": fast_return / (self._rolling(returns.rename("returns"), fast, "std", 2) + eps) - slow_return / (rstd + eps),
            }
            return values[concept]

        path_functions = {
            "trend_slope": lambda x: _trend_stat(x, "slope"),
            "trend_r2": lambda x: _trend_stat(x, "r2"),
            "return_acceleration_halves": lambda x: _return_acceleration(x, False),
            "return_acceleration_thirds": lambda x: _return_acceleration(x, True),
            "close_vs_path_mean": lambda x: (x[-1] - np.mean(x)) / (x[-1] * np.std(np.diff(np.log(x))) + eps),
            "close_vs_path_median": lambda x: (x[-1] - np.median(x)) / (x[-1] * np.std(np.diff(np.log(x))) + eps),
        }
        if concept in path_functions: return self._apply(price, n + 1, path_functions[concept], concept, 4)
        if concept == "trend_slope_t": return self._apply(price, n + 1, lambda x: _trend_stat(x, "t"), concept, 4)
        if concept in {"positive_bar_fraction", "negative_bar_fraction"}: return self._rolling(((returns > 0) if concept.startswith("positive") else (returns < 0)).astype(float).rename(concept), n, "mean", 1)
        if concept in {"sign_persistence", "sign_switch_rate"}:
            signs = np.sign(returns); prior_sign = signs.groupby(group).shift(1)
            valid = signs.ne(0) | prior_sign.eq(0)
            same = signs.eq(prior_sign).where(valid)
            persistence = same.astype(float).where(valid)
            return self._rolling((persistence if concept == "sign_persistence" else 1.0 - persistence).rename(concept), n, "mean", 2)
        if concept in {"average_signed_run_length", "longest_up_run", "longest_down_run"}:
            return self._apply(returns, n, lambda x: _run_stat(x, concept), concept, 3)
        if concept == "gross_to_net_path_ratio": return self._rolling(returns.abs().rename("abs_returns"), n, "sum", 1) / self._rolling(returns.rename("returns"), n, "sum", 1).abs().replace(0,np.nan)

        if concept == "largest_abs_return_share": return self._apply(returns, n, lambda x: top_abs_return_share(x,1), concept)
        if concept == "signed_return_concentration": return self._apply(returns,n,lambda x: np.sum(x*np.abs(x))/(np.sum(np.abs(x))**2+eps),concept)
        if concept == "mean_median_return_gap": return self._apply(returns,n,lambda x:(np.mean(x)-np.median(x))/(np.std(x)+eps),concept)

        decision_rank = cross_sectional_rank(returns, f.decision_ts).rename("decision_rank")
        rank_ops = {"mean_return_rank":"mean","median_return_rank":"median","return_rank_std":"std","rank_range":None,
                    "rank_above_80_fraction":None,"rank_above_90_fraction":None,"rank_below_20_fraction":None}
        if concept in rank_ops:
            if rank_ops[concept]: return self._rolling(decision_rank,n,rank_ops[concept],2)
            if concept=="rank_range": return self._rolling(decision_rank,n,"max",1)-self._rolling(decision_rank,n,"min",1)
            threshold=.9 if "90" in concept else .8 if "80" in concept else .2
            flag=(decision_rank<=threshold) if "below" in concept else (decision_rank>=threshold)
            return self._rolling(flag.astype(float).rename(concept),n,"mean",1)
        if concept == "return_rank_slope": return self._apply(decision_rank,n,_normalized_slope,concept,3)
        if concept == "return_rank_acceleration": return self._apply(decision_rank,n,_slope_acceleration,concept,4)
        if concept.startswith("rank_change_"): return decision_rank - decision_rank.groupby(group).shift(int(concept.split("_")[-1][:-1]))
        if concept in {"rank_dwell_top_decile","rank_dwell_bottom_decile"}:
            flag=(decision_rank>=.9) if "top" in concept else (decision_rank<=.1)
            return self._apply(flag.astype(float),n,lambda x: next((i for i,v in enumerate(x[::-1]) if not v),len(x))/len(x),concept,1)

        if concept in {"bars_or_days_since_high","bars_or_days_since_low"}: return self._apply(price,n,lambda x:(len(x)-1-(np.where(x== (np.max(x) if "high" in concept else np.min(x)))[0][-1]))/len(x),concept,2)
        if concept in {"new_high_count","new_low_count"}: return self._apply(price,n,lambda x:np.mean([x[i]>(np.max(x[:i]) if i else -np.inf) if "high" in concept else x[i]<(np.min(x[:i]) if i else np.inf) for i in range(len(x))]),concept,2)
        prior_high=self._rolling(price.groupby(group).shift(1).rename("prior_price"),n,"max",1); prior_low=self._rolling(price.groupby(group).shift(1).rename("prior_price"),n,"min",1)
        previous=price.groupby(group).shift(1)
        true_range=pd.concat([(f.high-f.low),(f.high-previous).abs(),(f.low-previous).abs()],axis=1).max(axis=1)
        atr=_safe_divide(self._rolling(true_range.rename("tr"),n,"mean",2), previous)
        if concept=="breakout_distance": return (price/prior_high-1).clip(lower=0)
        if concept=="breakdown_distance": return (price/prior_low-1).clip(upper=0)
        if concept=="failed_breakout_strength": return ((high_n>prior_high)&(price<=prior_high)).astype(float)*(high_n/prior_high-1)/(atr+eps)
        if concept=="failed_breakdown_strength": return ((low_n<prior_low)&(price>=prior_low)).astype(float)*(1-low_n/prior_low)/(atr+eps)
        if concept=="prior_high_reclaim_strength": return ((low_n<prior_high)&(price>prior_high)).astype(float)*(price/prior_high-1)/(atr+eps)
        if concept=="prior_low_reclaim_strength": return ((high_n>prior_low)&(price<prior_low)).astype(float)*(1-price/prior_low)/(atr+eps)

        if concept=="true_range_pct": return true_range/previous
        sma=self._rolling(price.rename("price"),n,"mean",2); pstd=self._rolling(price.rename("price"),n,"std",2)
        if concept=="price_minus_sma_volnorm": return (price-sma)/(price*rstd+eps)
        slope_window = min(n, max(5, round(n / 2)))
        if concept=="sma_slope": return self._apply(sma,slope_window,_normalized_slope,concept,3)
        if concept=="sma_acceleration": return self._apply(sma,n,_slope_acceleration,concept,6)
        if concept=="ema_slope":
            ema = price.groupby(group, group_keys=False).transform(lambda x: x.ewm(span=n, adjust=False, min_periods=min(n, 2)).mean())
            return self._apply(ema.rename("ema"),slope_window,_normalized_slope,concept,3)
        if concept=="atr_band_z": return (price-sma)/(self._rolling(true_range.rename("tr"),n,"mean",2)+eps)
        if concept in {"fast_slow_sma_gap","fast_slow_sma_ratio","fast_slow_ema_gap"}:
            fast_sma=self._rolling(price.rename("price"),fast,"mean",2)
            if concept=="fast_slow_sma_gap":return fast_sma/sma-1
            if concept=="fast_slow_sma_ratio":return fast_sma/(sma+eps)
            ef=price.groupby(group,group_keys=False).transform(lambda x:x.ewm(span=fast,adjust=False).mean()); es=price.groupby(group,group_keys=False).transform(lambda x:x.ewm(span=slow,adjust=False).mean());return ef/es-1

        if concept=="atr_pct": return self._rolling(true_range.rename("tr"),n,"mean",2)/(price+eps)
        if concept in {"parkinson_vol","garman_klass_vol","rogers_satchell_vol"}:
            if concept=="parkinson_vol": value=np.log(f.high/f.low).pow(2)/(4*np.log(2))
            elif concept=="garman_klass_vol": value=.5*np.log(f.high/f.low).pow(2)-(2*np.log(2)-1)*np.log(f.close/f.open).pow(2)
            else:value=np.log(f.high/f.close)*np.log(f.high/f.open)+np.log(f.low/f.close)*np.log(f.low/f.open)
            return np.sqrt(self._rolling(value.rename(concept),n,"mean",2).clip(lower=0))
        if concept=="idiosyncratic_vol":
            residual=returns-self._beta(returns)*self._benchmark_returns
            return self._rolling(residual.rename("idio"),n,"std",3)
        if concept=="vol_of_vol":
            short=max(2,min(5,n//4)); short_vol=self._rolling(returns.rename("returns"),short,"std",2)
            return self._rolling(short_vol.rename("short_vol"),n,"std",3)
        if concept=="vol_percentile_own": return self._apply(rstd,n+1,_prior_percentile,concept,6)
        if concept in {"vol_acceleration","vol_compression_ratio","downside_vol_acceleration","upside_vol_acceleration"}:
            if concept=="downside_vol_acceleration": source=returns.clip(upper=0)
            elif concept=="upside_vol_acceleration": source=returns.clip(lower=0)
            else: source=returns
            recent=self._rolling(source.rename(f"{concept}_fast"),fast,"std",2)
            baseline=self._rolling(source.rename(f"{concept}_slow"),slow,"std",3)
            return _safe_divide(recent,baseline)-1

        if concept in {"left_expected_shortfall","right_expected_shortfall"}: return self._apply(returns,n,lambda x:np.mean(x[x<=np.quantile(x,.05)]) if "left" in concept else np.mean(x[x>=np.quantile(x,.95)]),concept,5)
        if concept in {"jump_fraction","positive_jump_fraction","negative_jump_fraction","jump_variance_share"}:
            prior=returns.groupby(group,sort=False).shift(1)
            center=self._apply(prior.rename("jump_prior"),n,lambda x:float(np.nanmedian(x)),"jump_center",5)
            sigma=self._apply(prior.rename("jump_prior"),n,lambda x:float(1.4826*np.nanmedian(np.abs(x-np.nanmedian(x)))),"jump_sigma",5)
            jump=(returns-center).abs().gt(3*sigma); valid=sigma.gt(0)&returns.notna()
            if concept=="jump_fraction": value=jump.astype(float)
            elif concept=="positive_jump_fraction": value=(jump&(returns>0)).astype(float)
            elif concept=="negative_jump_fraction": value=(jump&(returns<0)).astype(float)
            else: value=returns.pow(2).where(jump,0)
            numerator=self._rolling(value.where(valid).rename(concept),n,"sum",2)
            denominator=self._rolling((returns.pow(2) if concept=="jump_variance_share" else valid.astype(float)).rename("jump_den"),n,"sum",2)
            return _safe_divide(numerator,denominator)
        if concept.startswith("return_autocorr_lag"): lag=int(concept[-1]); return self._apply(returns,n,lambda x:np.corrcoef(x[lag:],x[:-lag])[0,1],concept,lag+3)
        if concept=="sign_autocorr_lag1": return self._apply(returns,n,lambda x:np.corrcoef(np.sign(x[1:]),np.sign(x[:-1]))[0,1],concept,4)
        if concept=="abs_return_autocorr_lag1": return self._apply(returns,n,lambda x:np.corrcoef(np.abs(x[1:]),np.abs(x[:-1]))[0,1],concept,4)
        if concept.startswith("variance_ratio_"):
            k=int(concept[-1]);return self._apply(returns,n,lambda x:np.var(np.convolve(x,np.ones(k),mode="valid"),ddof=1)/(k*np.var(x,ddof=1)+eps),concept,k+3)
        if concept=="permutation_entropy": return self._apply(returns,n,lambda x:-sum(p*np.log(p) for p in np.unique([tuple(np.argsort(x[i:i+3])) for i in range(len(x)-2)],axis=0,return_counts=True)[1]/max(len(x)-2,1))/np.log(6),concept,6)
        if concept=="hurst_exponent": return self._apply(returns,n,lambda x:np.polyfit(np.log([1,2,4]),np.log([np.var(x),np.var(np.add.reduceat(x,np.arange(0,len(x),2))),np.var(np.add.reduceat(x,np.arange(0,len(x),4)))]+np.array([eps]*3)),1)[0]/2,concept,32)

        prior_volume=f.volume.groupby(group).shift(1); expected=self._rolling(prior_volume.rename("prior_volume"),n,"mean",2); vstd=self._rolling(prior_volume.rename("prior_volume"),n,"std",3)
        if concept=="relative_volume":return f.volume/(expected+eps)
        if concept=="relative_dollar_volume":return dollar/(self._rolling(dollar.groupby(group).shift(1).rename("prior_dollar"),n,"mean",2)+eps)
        if concept=="volume_zscore":return (f.volume-expected)/(vstd+eps)
        if concept=="dollar_volume_zscore":
            dmean=self._rolling(dollar.groupby(group).shift(1).rename("prior_dollar"),n,"mean",2);dstd=self._rolling(dollar.groupby(group).shift(1).rename("prior_dollar"),n,"std",3);return (dollar-dmean)/(dstd+eps)
        if concept=="volume_percentile_own":return self._apply(f.volume.astype(float),n,lambda x:float(np.mean(x<=x[-1])),concept,2)
        if concept=="volume_trend":return self._apply(np.log1p(f.volume),n,_normalized_slope,concept,3)
        if concept=="volume_acceleration":return self._apply(np.log1p(f.volume),n,_slope_acceleration,concept,6)
        if concept=="volume_volatility":return self._rolling(np.log1p(f.volume).groupby(group).diff().rename("dv"),n,"std",3)
        if concept in {"up_bar_volume_share","down_bar_volume_share"}:
            selected=f.volume.where(returns>0 if concept.startswith("up") else returns<0,0);return self._rolling(selected.rename(concept),n,"sum",1)/(volume_n+eps)
        if concept=="volume_concentration":return self._apply(f.volume.astype(float),n,lambda x:np.sum((x/(np.sum(x)+eps))**2),concept,2)
        if concept=="largest_bar_volume_share":return self._rolling(f.volume.astype(float).rename("volume"),n,"max",1)/(volume_n+eps)
        if concept=="trade_count":return self._rolling(f.trade_count.astype(float).rename("tc"),n,"sum",1)
        if concept=="relative_trade_count":
            tc=f.trade_count.astype(float);return _safe_divide(tc,self._rolling(tc.groupby(group).shift(1).rename("ptc"),n,"mean",2))
        if concept=="average_trade_size":return _safe_divide(volume_n,self._rolling(f.trade_count.astype(float).rename("tc"),n,"sum",1))

        if concept=="range_per_dollar_volume":return ((high_n-low_n)/(price.groupby(group).shift(n)+eps))/(dollar_n+eps)
        if concept=="roll_spread_proxy":return self._apply(price,n,FORMULAS["roll_spread_proxy"],concept,4)
        if concept=="corwin_schultz_spread_proxy":
            return _rolling_corwin(f.high.astype(float),f.low.astype(float),group,n)
        if concept=="return_x_rvol":return simple*(f.volume/(expected+eps))
        if concept=="abs_return_x_rvol":return simple.abs()*(f.volume/(expected+eps))
        if concept=="return_x_volume_z":return simple*((f.volume-expected)/(vstd+eps))
        if concept=="obv_slope":return self._apply((np.sign(returns)*f.volume).groupby(group).cumsum(),n,_linear_slope,concept,3)
        if concept in {"return_volume_corr","absreturn_volume_corr"}:return self._rolling_corr(returns.abs() if concept.startswith("abs") else returns,np.log1p(f.volume),n,3)
        if concept in {"price_shock_minus_volume_shock","volume_shock_minus_price_shock"}:
            pz=(simple-self._rolling(simple.rename("simple"),n,"mean",2))/(self._rolling(simple.rename("simple"),n,"std",3)+eps);vz=(f.volume-expected)/(vstd+eps);return pz-vz if concept.startswith("price") else vz-pz
        if concept=="vwap_drift":return self._apply(f.get("vwap",price).astype(float),n,_linear_slope,concept,3)
        if concept=="money_flow_proxy":return self._rolling((((f.high+f.low+f.close)/3).groupby(group).diff()*f.volume).rename("mf"),n,"sum",2)/(dollar_n+eps)

        # Session, same-time, market, statistical-factor and calendar concepts
        # are built from their causal panel columns when available; otherwise
        # their definitions below derive only from current/prior rows.
        session_open=f.get("session_open", f.groupby([f.security_id,f.session_date],sort=False,observed=True).open.transform("first") if "session_date" in f else f.open)
        prior_close=f.get("prior_session_close", f.groupby(f.security_id, sort=False).close.shift(1))
        benchmark_gap = f.benchmark_session_open / f.benchmark_prior_session_close - 1
        benchmark_intraday = f.benchmark_close / f.benchmark_session_open - 1
        basic={"overnight_return":f.open/prior_close-1,"regular_session_return":f.close/session_open-1,"opening_gap":f.open/prior_close-1,
               "market_return":self._benchmark_simple,"market_gap":benchmark_gap,
               "market_intraday_return":benchmark_intraday}
        if concept in basic:return basic[concept]
        if concept=="overnight_minus_regular":return basic["overnight_return"]-basic["regular_session_return"]
        if concept=="overnight_share_total":return basic["overnight_return"]/(basic["overnight_return"].abs()+basic["regular_session_return"].abs()+eps)
        if concept in {"positive_overnight_fraction","positive_regular_fraction"}:return self._rolling((basic["overnight_return" if "overnight" in concept else "regular_session_return"]>0).astype(float).rename(concept),n,"mean",1)
        if concept in {"gap_vs_atr","gap_vs_prior_vol"}:return basic["opening_gap"]/(atr if concept=="gap_vs_atr" else rstd+eps)
        if concept.startswith(("opening_return_","closing_return_")):return f[concept].astype(float)
        if concept.startswith("gap_fill_fraction_"):
            suffix=concept.removeprefix("gap_fill_fraction_")
            endpoint=f.session_close if suffix=="eod" else f[f"gap_fill_price_{suffix}"]
            return _safe_divide(session_open-endpoint,session_open-prior_close).clip(-2,2)
        if concept.startswith("opening_range_pct_"):return f[concept].astype(float)
        if concept=="close_location_daily_range":return _safe_divide(f.session_close-f.session_low,f.session_high-f.session_low)
        if concept=="session_vwap_distance":return _safe_divide(price-f.session_vwap,price)
        if concept in {"time_above_vwap","vwap_cross_count","open_to_midday_return","midday_to_close_return"}:return f[concept].astype(float)
        if concept.startswith("opening_rvol_"):
            suffix=concept.removeprefix("opening_rvol_"); current=f[f"opening_volume_{suffix}"]
            prior=current.groupby(f.security_id,sort=False).shift(1)
            baseline=prior.groupby(f.security_id,sort=False).rolling(20,min_periods=10).mean().reset_index(level=0,drop=True)
            return _safe_divide(current,baseline)
        if concept.startswith("closing_volume_share_") or concept.startswith("largest_"):return f[concept].astype(float)
        local_bucket=pd.to_datetime(f.decision_ts,utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        bucket_return=f.get("bucket_return",returns).astype(float)
        bucket_keys=[f.security_id,local_bucket]
        if concept.startswith("same_bucket_return_lag1"):return bucket_return.groupby(bucket_keys,sort=False).shift(1)
        if concept.startswith("same_bucket_return_mean"):
            k=int(concept.removeprefix("same_bucket_return_mean"));return bucket_return.groupby(bucket_keys,sort=False).transform(lambda x:x.shift(1).rolling(k,min_periods=max(1,min(k,2))).mean())
        if concept=="same_bucket_rvol":
            prior=f.volume.groupby(bucket_keys,sort=False).shift(1); baseline=prior.groupby(bucket_keys,sort=False).rolling(20,min_periods=10).mean().reset_index(level=[0,1],drop=True)
            return _safe_divide(f.volume,baseline)
        if concept=="same_bucket_volatility":return bucket_return.groupby(bucket_keys,sort=False).transform(lambda x:x.shift(1).rolling(20,min_periods=10).std())
        if concept=="same_bucket_return_rank":
            ranks=cross_sectional_rank(bucket_return,f.decision_ts)
            return ranks.groupby(bucket_keys,sort=False).transform(lambda x:x.shift(1).rolling(20,min_periods=10).mean())

        market=self._benchmark_returns
        needs_beta = concept in {"market_beta","downside_beta","upside_beta","beta_change","correlation_change",
                                 "idiosyncratic_vol","cross_sectional_residual_dispersion","average_pairwise_correlation",
                                 "beta_dispersion","pca_residual_return","pca_residual_momentum","pca_idiosyncratic_vol",
                                 "pc1_loading","pc2_loading","pc1_loading_change"}
        beta=self._beta(returns) if needs_beta else pd.Series(np.nan,index=f.index)
        if concept=="market_beta":return pd.Series(beta,index=f.index,dtype=float)
        if concept in {"downside_beta","upside_beta"}:
            stock_prior=returns.groupby(f.security_id,sort=False).shift(1); market_prior=market.groupby(f.security_id,sort=False).shift(1)
            return self._rolling_pair_slope(stock_prior,market_prior,n,5,"down" if concept.startswith("down") else "up")
        if concept=="market_correlation":return self._rolling_corr(returns,market,n,3)
        if concept=="beta_change":return pd.Series(beta,index=f.index).groupby(group).diff(fast)
        if concept=="correlation_change":
            corr=self._rolling_corr(returns,market,n,3); return corr.groupby(group).diff(fast)
        if concept in {"market_lead_response","stock_lead_market_response"}:
            if concept=="market_lead_response": dependent,independent=returns,market.groupby(group).shift(1)
            else: dependent,independent=market,returns.groupby(group).shift(1)
            return self._rolling_pair_slope(dependent,independent,n,3)
        if concept=="market_risk_adjusted_momentum":return _safe_divide(self._benchmark_simple,self._rolling(market.rename("market"),n,"std",3))
        benchmark=f.benchmark_close.astype(float)
        if concept=="market_drawdown":return benchmark/self._rolling(benchmark.rename("benchmark"),n,"max",1)-1
        if concept in {"market_volatility","market_downside_vol"}:
            source=market.clip(upper=0) if concept=="market_downside_vol" else market
            return self._rolling(source.rename(concept),n,"std",3)
        if concept=="market_vol_percentile":
            market_vol=self._rolling(market.rename("market"),n,"std",3);return self._apply(market_vol,max(n,20)+1,_prior_percentile,concept,6)
        if concept=="market_breadth_positive":return (simple>0).groupby(f.decision_ts).transform("mean")
        if concept.startswith("market_breadth_above_sma"):
            k=int(concept.removeprefix("market_breadth_above_sma")); stock_sma=self._rolling(price.rename("price"),k,"mean",2)
            return price.gt(stock_sma).groupby(f.decision_ts).transform("mean")
        breadth=(simple>0).groupby(f.decision_ts).transform("mean")
        unique_breadth=breadth.groupby(f.decision_ts,sort=False).first(); lag=int(spec.parameters.get("lag",fast))
        if concept=="breadth_change":return f.decision_ts.map(unique_breadth-unique_breadth.shift(lag))
        if concept=="breadth_momentum":
            slope=unique_breadth.rolling(n,min_periods=max(5,ceil(.6*n))).apply(_normalized_slope,raw=True)
            return f.decision_ts.map(slope)
        if concept.startswith("cross_sectional_return_"):
            operation="std" if "dispersion" in concept else "kurt" if "kurtosis" in concept else "skew"
            return simple.groupby(f.decision_ts).transform(operation)
        if concept=="cross_sectional_residual_dispersion":return (simple-beta*market).groupby(f.decision_ts).transform("std")
        if concept=="average_pairwise_correlation":return self._rolling_corr(market,returns,n,2).groupby(f.decision_ts).transform("mean")
        if concept=="beta_dispersion":return pd.Series(beta,index=f.index).groupby(f.decision_ts).transform("std")
        if concept.startswith(("pca_","pc1_","pc2_","stat_peer_","stock_minus_stat_peer")):
            if concept in f:return f[concept].astype(float)
            residual=returns-beta*market
            if concept=="pca_residual_return":return residual
            if concept=="pca_residual_momentum":return self._rolling(residual.rename("residual"),n,"sum",2)
            if concept=="pca_idiosyncratic_vol":return self._rolling(residual.rename("residual"),n,"std",3)
            if concept in {"pc1_loading","pc2_loading","pc1_loading_change"}:return pd.Series(beta,index=f.index,dtype=float)
            peers=returns.groupby(f.decision_ts).transform("mean")
            if concept=="stock_minus_stat_peer_return":return returns-peers
            if concept in {"stat_peer_basket_return","stat_peer_momentum","stat_peer_lagged_return"}:return peers
            if concept=="stat_peer_dispersion":return returns.groupby(f.decision_ts).transform("std")
            return self._rolling_corr(returns,peers,n,3)
        if concept in {"month_end_distance","quarter_end_distance"}:
            dates=pd.to_datetime(f.get("session_date",f.decision_ts)); end=dates+pd.offsets.MonthEnd(0) if concept.startswith("month") else dates+pd.offsets.QuarterEnd(0);return (end-dates).dt.days.astype(float)
        if concept in {"pre_holiday_flag","post_holiday_flag"}:
            import exchange_calendars as xcals
            dates=pd.to_datetime(f.get("session_date",f.decision_ts)).dt.normalize(); calendar=xcals.get_calendar("XNYS")
            start=dates.min()-pd.Timedelta(days=10); end=dates.max()+pd.Timedelta(days=10)
            sessions=pd.DatetimeIndex(calendar.sessions_in_range(start,end)).tz_localize(None).normalize(); flags={}
            for index,date in enumerate(sessions):
                previous=sessions[index-1] if index else date; following=sessions[index+1] if index+1<len(sessions) else date
                pre=np.busday_count((date+pd.Timedelta(days=1)).date(),following.date())>0
                post=np.busday_count((previous+pd.Timedelta(days=1)).date(),date.date())>0
                flags[date]=(float(pre),float(post))
            position=0 if concept.startswith("pre") else 1
            return dates.map(lambda value:flags.get(value,(0.0,0.0))[position]).astype(float)
        return None
