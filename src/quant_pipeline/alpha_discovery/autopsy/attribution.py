from __future__ import annotations

import numpy as np
import pandas as pd


def factor_attribution(candidate_returns: np.ndarray, factors: pd.DataFrame) -> pd.DataFrame:
    y = np.asarray(candidate_returns, dtype=float); x = factors.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if valid.sum() <= x.shape[1] + 1: return pd.DataFrame()
    design = np.column_stack([np.ones(valid.sum()), x[valid]]); coefficients, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    fitted = design @ coefficients; residual = y[valid] - fitted
    rows = [{"factor": "intercept", "coefficient": coefficients[0]}]
    rows += [{"factor": name, "coefficient": value} for name, value in zip(factors.columns, coefficients[1:])]
    rows.append({"factor": "residual", "coefficient": np.std(residual, ddof=1)})
    return pd.DataFrame(rows)
