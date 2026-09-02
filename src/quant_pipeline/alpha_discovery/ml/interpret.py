from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


def stable_permutation_importance(model, x: np.ndarray, y: np.ndarray, repeats: int = 10, seed: int = 1729) -> np.ndarray:
    return permutation_importance(model, x, y, n_repeats=repeats, random_state=seed).importances_mean
