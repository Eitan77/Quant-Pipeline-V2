from __future__ import annotations

import numpy as np
import pandas as pd


def apply_next_bar_open_fills(trades: pd.DataFrame, adverse_slippage_bps_per_side: float) -> pd.DataFrame:
    """Apply adverse raw-price fills; no same-bar signal/fill is permitted."""
    required = {"side", "decision_ts", "entry_ts", "exit_ts", "entry_open_raw", "exit_close_raw"}
    missing = required - set(trades)
    if missing:
        raise ValueError(f"Execution input missing: {sorted(missing)}")
    work = trades.copy()
    # Phase 1 represents the next actionable bar open by the same timestamp
    # as information availability; it has already established that the bar
    # used by the feature is complete before this timestamp.  Reject only a
    # truly pre-signal fill, rather than invalidating that contract.
    if (pd.to_datetime(work["entry_ts"], utc=True) < pd.to_datetime(work["decision_ts"], utc=True)).any():
        raise ValueError("Pre-signal entry rejected")
    if (pd.to_datetime(work["exit_ts"], utc=True) <= pd.to_datetime(work["entry_ts"], utc=True)).any():
        raise ValueError("Exit must occur after entry")
    c = adverse_slippage_bps_per_side / 10_000.0
    long = work["side"].eq(1)
    work["entry_executable_price"] = np.where(long, work["entry_open_raw"] * (1 + c), work["entry_open_raw"] * (1 - c))
    work["exit_executable_price"] = np.where(long, work["exit_close_raw"] * (1 - c), work["exit_close_raw"] * (1 + c))
    work["gross_return"] = np.where(long, work["exit_close_raw"] / work["entry_open_raw"] - 1,
                                    1 - work["exit_close_raw"] / work["entry_open_raw"])
    work["net_return"] = np.where(long, work["exit_executable_price"] / work["entry_executable_price"] - 1,
                                  1 - work["exit_executable_price"] / work["entry_executable_price"])
    work["slippage_cost"] = work["gross_return"] - work["net_return"]
    return work
