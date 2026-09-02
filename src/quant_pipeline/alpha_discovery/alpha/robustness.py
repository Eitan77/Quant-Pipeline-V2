from __future__ import annotations

import numpy as np


def additive_metrics(returns: np.ndarray) -> dict[str, float]:
    values = np.nan_to_num(np.asarray(returns, dtype=float)); equity = 1 + np.cumsum(values); peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1; std = values.std(ddof=1)
    return {"additive_return": float(values.sum()), "sharpe": float(values.mean() / std * np.sqrt(252)) if std > 0 else np.nan,
            "max_drawdown": float(drawdown.min()), "positive_fraction": float(np.mean(values > 0))}
