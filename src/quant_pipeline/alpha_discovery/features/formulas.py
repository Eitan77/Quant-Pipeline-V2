from __future__ import annotations

from math import log, sqrt
from typing import Callable

import numpy as np


EPS = np.finfo(np.float64).eps


def simple_return(prices: np.ndarray) -> float:
    return float(prices[-1] / prices[0] - 1) if len(prices) >= 2 and prices[0] > 0 else np.nan


def log_return(prices: np.ndarray) -> float:
    return float(np.log(prices[-1] / prices[0])) if len(prices) >= 2 and prices[0] > 0 and prices[-1] > 0 else np.nan


def path_efficiency(prices: np.ndarray) -> float:
    p = np.log(np.asarray(prices, dtype=float))
    increments = np.diff(p)
    denominator = np.abs(increments).sum()
    return float(abs(p[-1] - p[0]) / denominator) if len(p) >= 2 and denominator > 0 else np.nan


def path_chord_convexity(prices: np.ndarray) -> float:
    p = np.log(np.asarray(prices, dtype=float))
    if len(p) < 3:
        return np.nan
    chord = np.linspace(p[0], p[-1], len(p))
    scale = np.std(np.diff(p), ddof=1) * sqrt(len(p) - 1)
    return float(np.mean(chord - p) / scale) if scale > 0 else np.nan


def path_quadratic_curvature(prices: np.ndarray) -> float:
    p = np.log(np.asarray(prices, dtype=float))
    if len(p) < 4:
        return np.nan
    scale = np.std(np.diff(p), ddof=1)
    if scale <= 0:
        return np.nan
    t = np.linspace(-1.0, 1.0, len(p))
    coefficient = np.polyfit(t, p, 2)[0]
    return float(coefficient / scale)


def information_discreteness(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    valid = r[np.isfinite(r)]
    if not len(valid):
        return np.nan
    return float(np.sign(valid.sum()) * ((valid < 0).mean() - (valid > 0).mean()))


def absolute_return_hhi(returns: np.ndarray) -> float:
    values = np.abs(np.asarray(returns, dtype=float))
    total = np.nansum(values)
    return float(np.nansum((values / total) ** 2)) if total > 0 else np.nan


def top_abs_return_share(returns: np.ndarray, k: int) -> float:
    values = np.abs(np.asarray(returns, dtype=float))
    total = np.nansum(values)
    return float(np.sort(values[np.isfinite(values)])[-k:].sum() / total) if total > 0 else np.nan


def max_drawdown(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    peaks = np.maximum.accumulate(p)
    return float(np.nanmin(p / peaks - 1)) if len(p) else np.nan


def max_runup(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    troughs = np.minimum.accumulate(p)
    return float(np.nanmax(p / troughs - 1)) if len(p) else np.nan


def parkinson_vol(high: np.ndarray, low: np.ndarray) -> float:
    values = np.log(np.asarray(high) / np.asarray(low)) ** 2
    return float(sqrt(max(np.nanmean(values) / (4 * log(2)), 0)))


def garman_klass_vol(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    value = np.nanmean(0.5 * np.log(np.asarray(high) / np.asarray(low)) ** 2 - (2 * log(2) - 1) * np.log(np.asarray(close) / np.asarray(open_)) ** 2)
    return float(sqrt(max(value, 0)))


def rogers_satchell_vol(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    value = np.nanmean(np.log(np.asarray(high) / np.asarray(close)) * np.log(np.asarray(high) / np.asarray(open_)) + np.log(np.asarray(low) / np.asarray(close)) * np.log(np.asarray(low) / np.asarray(open_)))
    return float(sqrt(max(value, 0)))


def roll_spread_proxy(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    changes = np.diff(p)
    if len(changes) < 3 or p[-1] <= 0:
        return np.nan
    covariance = np.cov(changes[1:], changes[:-1], ddof=1)[0, 1]
    return float(2 * sqrt(max(-covariance, 0)) / p[-1])


def corwin_schultz_pair(high: np.ndarray, low: np.ndarray) -> float:
    if len(high) != 2 or len(low) != 2 or np.any(np.asarray(low) <= 0):
        return np.nan
    beta = np.log(high[0] / low[0]) ** 2 + np.log(high[1] / low[1]) ** 2
    gamma = np.log(max(high) / min(low)) ** 2
    k = 3 - 2 * sqrt(2)
    alpha = max((sqrt(2 * beta) - sqrt(beta)) / k - sqrt(gamma / k), 0)
    return float(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))


def shannon_entropy(states: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(states), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum()) if len(counts) else np.nan


FORMULAS: dict[str, Callable] = {
    "simple_return": simple_return, "log_return": log_return, "path_efficiency": path_efficiency,
    "path_chord_convexity": path_chord_convexity, "path_quadratic_curvature": path_quadratic_curvature,
    "information_discreteness": information_discreteness, "absolute_return_hhi": absolute_return_hhi,
    "window_max_drawdown": max_drawdown, "window_max_runup": max_runup, "parkinson_vol": parkinson_vol,
    "garman_klass_vol": garman_klass_vol, "rogers_satchell_vol": rogers_satchell_vol,
    "roll_spread_proxy": roll_spread_proxy, "corwin_schultz_spread_proxy": corwin_schultz_pair,
    "return_entropy": shannon_entropy, "sign_entropy": shannon_entropy,
}
