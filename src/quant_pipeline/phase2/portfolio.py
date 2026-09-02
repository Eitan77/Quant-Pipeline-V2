from __future__ import annotations

import numpy as np
import pandas as pd


def assign_weights(selected: pd.DataFrame, method: str, form: str, symbol_cap: float = 0.10) -> pd.DataFrame:
    """Assign transparent sleeve weights and enforce dollar/beta neutrality."""
    required = {"decision_ts", "side", "symbol"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"Weighting input missing: {sorted(missing)}")
    if method not in {"equal", "rank", "inverse_volatility"}:
        raise ValueError(f"Unsupported weighting method: {method}")
    if form not in {"long_only", "dollar_neutral", "beta_neutral"}:
        raise ValueError(f"Unsupported portfolio form: {form}")
    work = selected.loc[selected.side.ne(0)].copy()
    if form == "long_only":
        work = work.loc[work.side.eq(1)].copy()
    if method == "equal":
        raw = pd.Series(1.0, index=work.index)
    elif method == "rank":
        if "rank_ascending" not in work or "eligible_count" not in work:
            raise ValueError("Rank weighting requires rank_ascending and eligible_count")
        raw = np.where(work.side.eq(1), work.rank_ascending, work.eligible_count - work.rank_ascending + 1)
        raw = pd.Series(raw, index=work.index, dtype=float)
    else:
        if "prior_volatility" not in work:
            raise ValueError("Inverse volatility weighting requires prior_volatility")
        raw = 1.0 / work.prior_volatility.clip(lower=1e-4)
    work["raw_weight"] = raw
    if form == "long_only":
        denominator = work.groupby("decision_ts", sort=False).raw_weight.transform("sum")
        work["target_weight"] = work.raw_weight / denominator
    else:
        valid = work.groupby("decision_ts", sort=False).side.transform("nunique").eq(2)
        work = work.loc[valid].copy()
        denominator = work.groupby(["decision_ts", "side"], sort=False).raw_weight.transform("sum")
        work["target_weight"] = work.side * 0.5 * work.raw_weight / denominator
        if form == "beta_neutral":
            if "prior_beta" not in work:
                raise ValueError("Beta neutral weighting requires prior_beta")
            beta_contribution = work.target_weight * work.prior_beta
            long_beta = beta_contribution.where(work.side.eq(1), 0.0).groupby(work.decision_ts).transform("sum")
            short_beta = beta_contribution.where(work.side.eq(-1), 0.0).groupby(work.decision_ts).transform("sum")
            factor = (-long_beta / short_beta.replace(0, np.nan)).fillna(1.0)
            work.loc[work.side.eq(-1), "target_weight"] *= factor.loc[work.side.eq(-1)]
            gross = work.target_weight.abs().groupby(work.decision_ts).transform("sum")
            work["target_weight"] /= gross
    work["final_weight"] = work.target_weight.clip(lower=-symbol_cap, upper=symbol_cap)
    # A hard symbol cap is a constraint, not a rescaling suggestion.  When a
    # sparse tail cannot deploy the full sleeve without breaching the cap, the
    # remainder stays in cash.
    return work.reset_index(drop=True)
