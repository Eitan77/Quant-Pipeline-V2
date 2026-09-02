from __future__ import annotations

import pandas as pd


def replay_market_orders(orders: pd.DataFrame, raw_bars: pd.DataFrame, entry_delay_minutes: int = 1, exit_delay_minutes: int = 0) -> pd.DataFrame:
    bars = raw_bars.copy(); bars["bar_start_ts_utc"] = pd.to_datetime(bars.bar_start_ts_utc, utc=True)
    rows = []
    for order in orders.itertuples(index=False):
        eligible = bars[(bars.security_id == order.security_id) & bars.bar_start_ts_utc.gt(pd.Timestamp(order.decision_ts))]
        if len(eligible) <= entry_delay_minutes - 1: continue
        entry = eligible.iloc[entry_delay_minutes - 1]
        exits = eligible[eligible.bar_start_ts_utc.ge(pd.Timestamp(order.planned_exit_ts))]
        if len(exits) <= exit_delay_minutes: continue
        exit_bar = exits.iloc[exit_delay_minutes]
        direction = int(order.direction)
        rows.append({"order_id": order.order_id, "security_id": order.security_id, "entry_ts": entry.bar_start_ts_utc,
                     "entry_price": entry.open, "exit_ts": exit_bar.bar_start_ts_utc, "exit_price": exit_bar.open,
                     "gross_return": direction * (exit_bar.open / entry.open - 1), "direction": direction})
    return pd.DataFrame(rows)
