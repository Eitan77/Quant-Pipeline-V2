from __future__ import annotations

import numpy as np


def standardized_specific_return(beta_residual: np.ndarray, decision_codes: np.ndarray) -> np.ndarray:
    output = np.full(len(beta_residual), np.nan)
    for code in np.unique(decision_codes):
        mask = (decision_codes == code) & np.isfinite(beta_residual)
        if mask.sum() < 2: continue
        values = beta_residual[mask]; std = values.std(ddof=1)
        if std > 0: output[mask] = (values - values.mean()) / std
    return output
