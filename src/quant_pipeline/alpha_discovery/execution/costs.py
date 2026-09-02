from __future__ import annotations

import numpy as np


def apply_costs(gross_returns: np.ndarray, one_way_turnover: np.ndarray, cost_bp_per_side: float) -> np.ndarray:
    return np.asarray(gross_returns) - np.asarray(one_way_turnover) * cost_bp_per_side / 10_000


def break_even_cost_bp_per_side(gross_returns: np.ndarray, one_way_turnover: np.ndarray) -> float:
    turnover = np.nansum(one_way_turnover)
    return float(np.nansum(gross_returns) / turnover * 10_000) if turnover > 0 else np.nan
