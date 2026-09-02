from __future__ import annotations

import numpy as np
import pandas as pd


def leave_out_tests(returns: np.ndarray, symbols: np.ndarray, sessions: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"return": returns, "symbol": symbols, "session": sessions}).dropna(subset=["return"])
    contributions = frame.groupby("symbol")["return"].sum().sort_values(ascending=False)
    days = frame.groupby("session")["return"].sum().sort_values(ascending=False)
    rows = [{"test": "full", "mean_return": frame["return"].mean(), "n_obs": len(frame)}]
    for count in (1, 5, 10):
        kept = frame[~frame.symbol.isin(contributions.head(count).index)]
        rows.append({"test": f"leave_top_{count}_symbols", "mean_return": kept["return"].mean(), "n_obs": len(kept)})
    for count in (10, 20):
        kept = frame[~frame.session.isin(days.head(count).index)]
        rows.append({"test": f"leave_top_{count}_days", "mean_return": kept["return"].mean(), "n_obs": len(kept)})
    for fraction in (.01, .05):
        cutoff = frame["return"].abs().quantile(1 - fraction); kept = frame[frame["return"].abs() <= cutoff]
        rows.append({"test": f"leave_top_{fraction:.0%}_abs_returns", "mean_return": kept["return"].mean(), "n_obs": len(kept)})
    return pd.DataFrame(rows)
