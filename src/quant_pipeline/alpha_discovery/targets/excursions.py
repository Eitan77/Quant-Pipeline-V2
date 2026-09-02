from __future__ import annotations

import numpy as np
import pandas as pd


def trade_path_diagnostics(path_prices: pd.Series, entry_price: float) -> dict[str, float]:
    returns = path_prices.astype(float) / float(entry_price) - 1.0
    if returns.empty or not np.isfinite(entry_price) or entry_price <= 0:
        return {name: np.nan for name in ("mfe", "mae", "time_to_mfe", "time_to_mae", "terminal_return", "mfe_minus_terminal", "recovery_after_mae")}
    mfe_index, mae_index = int(np.nanargmax(returns)), int(np.nanargmin(returns))
    terminal = float(returns.iloc[-1])
    return {"mfe": float(returns.iloc[mfe_index]), "mae": float(returns.iloc[mae_index]),
            "time_to_mfe": mfe_index + 1, "time_to_mae": mae_index + 1, "terminal_return": terminal,
            "mfe_minus_terminal": float(returns.iloc[mfe_index] - terminal),
            "recovery_after_mae": float(terminal - returns.iloc[mae_index])}
