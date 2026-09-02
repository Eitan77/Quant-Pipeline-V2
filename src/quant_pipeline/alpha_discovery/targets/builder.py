from __future__ import annotations

import numpy as np
import pandas as pd


def _raw_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"security_id", "session_date", "bar_start_ts_utc", "bar_end_ts_utc", "open", "close"}
    missing = required - set(bars)
    if missing:
        raise ValueError(f"Raw bars missing target columns: {sorted(missing)}")
    output = bars.copy()
    output["bar_start_ts_utc"] = pd.to_datetime(output.bar_start_ts_utc, utc=True)
    output["bar_end_ts_utc"] = pd.to_datetime(output.bar_end_ts_utc, utc=True)
    return output.sort_values(["security_id", "bar_start_ts_utc"], kind="mergesort")


def build_intraday_targets(decisions: pd.DataFrame, raw_bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60, 120, 240), include_eod: bool = True) -> pd.DataFrame:
    bars = _raw_bars(raw_bars)
    columns = ["observation_id", "security_id", "session_date", "decision_ts"]
    if "decision_grid" in decisions:
        columns.append("decision_grid")
    decision_frame = decisions[columns].copy()
    decision_frame["decision_ts"] = pd.to_datetime(decision_frame.decision_ts, utc=True)
    parts: list[pd.DataFrame] = []
    decision_groups = decision_frame.groupby(["security_id", "session_date"], sort=False)
    for key, group in bars.groupby(["security_id", "session_date"], sort=False):
        try: current = decision_groups.get_group(key).copy()
        except KeyError: continue
        starts = group.bar_start_ts_utc.to_numpy(dtype="datetime64[ns]")
        decisions_ns = current.decision_ts.to_numpy(dtype="datetime64[ns]")
        entry_index = np.searchsorted(starts, decisions_ns, side="right")
        valid_entry = entry_index < len(group)
        if not valid_entry.any(): continue
        current = current.loc[valid_entry].reset_index(drop=True); entry_index = entry_index[valid_entry]
        entry_price = group.open.to_numpy(float)[entry_index]
        entry_ts = group.bar_start_ts_utc.iloc[entry_index].reset_index(drop=True).array
        base = current[["observation_id", "security_id", "decision_ts"]].copy()
        base["entry_ts"] = entry_ts; base["entry_price"] = entry_price
        grid = str(current.decision_grid.iloc[0]) if "decision_grid" in current else "intraday_5m"
        for horizon in horizons:
            exit_index = entry_index + horizon - 1; valid = exit_index < len(group)
            if not valid.any(): continue
            part = base.loc[valid].copy(); chosen = exit_index[valid]
            part["target_id"] = f"target_{horizon}m__raw__{grid}"
            part["exit_ts"] = group.bar_end_ts_utc.iloc[chosen].reset_index(drop=True).array
            part["exit_price"] = group.close.to_numpy(float)[chosen]
            part["target"] = part.exit_price.to_numpy(float) / part.entry_price.to_numpy(float) - 1
            part["target_basis"] = "raw"; parts.append(part)
        if include_eod:
            part = base.copy(); part["target_id"] = f"target_eod__raw__{grid}"
            part["exit_ts"] = group.bar_end_ts_utc.iloc[-1]; part["exit_price"] = float(group.close.iloc[-1])
            part["target"] = part.exit_price / part.entry_price - 1; part["target_basis"] = "raw"; parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_daily_targets(decisions: pd.DataFrame, raw_bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 30, 40, 63, 126), entry_delay_minutes: int = 1) -> pd.DataFrame:
    bars = _raw_bars(raw_bars)
    sessions = sorted(pd.to_datetime(bars.session_date).dt.date.unique())
    session_position = {value: index for index, value in enumerate(sessions)}
    grouped = {(sid, pd.Timestamp(day).date()): group.reset_index(drop=True) for (sid, day), group in bars.groupby(["security_id", "session_date"], sort=False)}
    rows: list[dict] = []
    for decision in decisions.itertuples(index=False):
        date = pd.Timestamp(decision.session_date).date()
        position = session_position.get(date)
        if position is None or position + 1 >= len(sessions):
            continue
        entry_session = sessions[position + 1]
        entry_group = grouped.get((decision.security_id, entry_session))
        if entry_group is None or len(entry_group) <= entry_delay_minutes:
            continue
        entry = entry_group.iloc[entry_delay_minutes]
        for horizon in horizons:
            exit_position = position + horizon
            if exit_position >= len(sessions):
                continue
            exit_group = grouped.get((decision.security_id, sessions[exit_position]))
            if exit_group is None or exit_group.empty:
                continue
            exit_bar = exit_group.iloc[-1]
            rows.append({"observation_id": decision.observation_id, "security_id": decision.security_id,
                         "decision_ts": pd.Timestamp(decision.decision_ts), "entry_ts": entry.bar_start_ts_utc,
                         "entry_price": float(entry.open), "exit_ts": exit_bar.bar_end_ts_utc,
                         "exit_price": float(exit_bar.close), "target": float(exit_bar.close / entry.open - 1),
                         "target_basis": "raw", "target_id": f"target_{horizon}d__raw__daily_close"})
    return pd.DataFrame(rows)


def add_residual_bases(targets: pd.DataFrame, benchmark_targets: pd.DataFrame, beta: pd.Series | float = 1.0) -> pd.DataFrame:
    keys = ["observation_id", "target_id"]
    benchmark = benchmark_targets[keys + ["target"]].rename(columns={"target": "benchmark_target"})
    merged = targets.merge(benchmark, on=keys, how="left", validate="many_to_one")
    merged["benchmark_adjusted_target"] = merged.target - merged.benchmark_target
    merged["beta_residual_target"] = merged.target - beta * merged.benchmark_target
    return merged


def flag_corporate_action_crossings(targets: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on intervals containing an action; never scan a false raw-price jump."""
    output = targets.copy(); output["crosses_split"] = False; output["crosses_cash_dividend"] = False
    if output.empty or actions.empty: return output
    entry_date = pd.to_datetime(output.entry_ts, utc=True).dt.date
    exit_date = pd.to_datetime(output.exit_ts, utc=True).dt.date
    for action in actions.itertuples(index=False):
        action_date = pd.Timestamp(getattr(action, "session_date", getattr(action, "ex_date"))).date()
        mask = output.security_id.eq(action.security_id) & entry_date.lt(action_date) & exit_date.ge(action_date)
        kind = str(action.action_type).lower()
        if "split" in kind: output.loc[mask, "crosses_split"] = True
        if "cash" in kind or "dividend" in kind: output.loc[mask, "crosses_cash_dividend"] = True
    invalid = output.crosses_split | output.crosses_cash_dividend
    output.loc[invalid, "target"] = np.nan
    return output


def build_overnight_targets(decisions: pd.DataFrame, raw_bars: pd.DataFrame) -> pd.DataFrame:
    bars = _raw_bars(raw_bars); sessions = sorted(pd.to_datetime(bars.session_date).dt.date.unique()); positions = {day: index for index, day in enumerate(sessions)}
    grouped = {(sid, pd.Timestamp(day).date()): group.reset_index(drop=True) for (sid, day), group in bars.groupby(["security_id", "session_date"], sort=False)}
    rows = []
    for decision in decisions.itertuples(index=False):
        day = pd.Timestamp(decision.session_date).date(); position = positions.get(day)
        if position is None or position + 1 >= len(sessions): continue
        current = grouped.get((decision.security_id, day)); following = grouped.get((decision.security_id, sessions[position + 1]))
        if current is None or following is None or len(following) <= 1: continue
        actionable = current[current.bar_start_ts_utc.gt(pd.Timestamp(decision.decision_ts))]
        if actionable.empty: continue
        entry, exit_bar = actionable.iloc[0], following.iloc[1]
        rows.append({"observation_id": decision.observation_id, "security_id": decision.security_id,
                     "decision_ts": pd.Timestamp(decision.decision_ts), "entry_ts": entry.bar_start_ts_utc,
                     "entry_price": float(entry.open), "exit_ts": exit_bar.bar_start_ts_utc,
                     "exit_price": float(exit_bar.open), "target": float(exit_bar.open / entry.open - 1),
                     "target_basis": "raw", "target_id": "target_overnight__raw__preclose_1555"})
    return pd.DataFrame(rows)
