from __future__ import annotations

import numpy as np
import pandas as pd


UNARY = {
    "sign": np.sign, "abs": np.abs,
    "rank": lambda x: pd.Series(x).rank(method="average", pct=True).to_numpy(),
    "zscore": lambda x: (x - np.nanmean(x)) / np.nanstd(x),
}
BINARY = {"min": np.minimum, "max": np.maximum, "add": np.add, "subtract": np.subtract, "multiply": np.multiply}


def delay(values: np.ndarray, length: int) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float); output[length:] = values[:-length]; return output


def delta(values: np.ndarray, length: int) -> np.ndarray:
    return np.asarray(values, dtype=float) - delay(np.asarray(values, dtype=float), length)


def ewma(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).ewm(span=length, adjust=False, min_periods=length).mean().to_numpy()
