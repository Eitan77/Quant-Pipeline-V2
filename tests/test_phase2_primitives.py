from __future__ import annotations

import pandas as pd
import pytest

from quant_pipeline.phase2.config import Phase2Config
from quant_pipeline.phase2.execution import apply_next_bar_open_fills
from quant_pipeline.phase2.portfolio import assign_weights
from quant_pipeline.phase2.selection import select_cross_sectional_tails
from quant_pipeline.phase2.vol_target import causal_leverage


def _signals() -> pd.DataFrame:
    stamp = pd.Timestamp("2024-01-02 10:00", tz="America/New_York")
    return pd.DataFrame({"decision_ts": [stamp] * 10, "symbol": list("ABCDEFGHIJ"), "signal": range(10)})


def test_tail_selection_is_timestamp_local_and_deterministic() -> None:
    selected = select_cross_sectional_tails(_signals().sample(frac=1, random_state=7), 0.20)
    assert set(selected.loc[selected.side.eq(1), "symbol"]) == {"I", "J"}
    assert set(selected.loc[selected.side.eq(-1), "symbol"]) == {"A", "B"}


def test_adverse_fills_and_same_bar_rejection() -> None:
    decision = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    trades = pd.DataFrame({"side": [1, -1], "decision_ts": [decision, decision],
                           "entry_ts": [decision + pd.Timedelta(minutes=5)] * 2,
                           "exit_ts": [decision + pd.Timedelta(minutes=35)] * 2,
                           "entry_open_raw": [100.0, 100.0], "exit_close_raw": [101.0, 99.0]})
    filled = apply_next_bar_open_fills(trades, 1.0)
    assert (filled.net_return < filled.gross_return).all()
    trades.loc[0, "entry_ts"] = decision - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Pre-signal"):
        apply_next_bar_open_fills(trades, 1.0)


def test_beta_neutral_weighting_is_gross_normalized() -> None:
    selected = select_cross_sectional_tails(_signals(), 0.20)
    selected["prior_beta"] = 1.0
    weighted = assign_weights(selected, "equal", "beta_neutral", symbol_cap=0.50)
    assert weighted.final_weight.abs().sum() == pytest.approx(1.0)
    assert (weighted.final_weight * weighted.prior_beta).sum() == pytest.approx(0.0)


def test_symbol_cap_remains_hard_after_weighting() -> None:
    selected = select_cross_sectional_tails(_signals(), 0.20)
    weighted = assign_weights(selected, "equal", "long_only", symbol_cap=0.10)
    assert weighted.final_weight.abs().max() <= 0.10
    assert weighted.final_weight.abs().sum() == pytest.approx(0.20)


def test_volatility_target_uses_only_prior_returns() -> None:
    returns = pd.Series([0.01] * 60 + [-0.20])
    leverage = causal_leverage(returns, 0.08, 2.0, lookback_sessions=60)
    assert leverage.iloc[59] == pytest.approx(1.0)
    assert leverage.iloc[60] > 1.0


def test_phase2_config_refuses_holdout_access(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("""experiment_id: bad\ndiscovery_start: '2024-01-01'\ndiscovery_end: '2026-04-30'\nsealed_holdout_start: '2026-05-01'\nallow_holdout_access: true\nphase1_source_run: a\nphase1b_source_run: b\nstrategies:\n  - family: test\n    classification: phase1_supported\n    cluster: test\n    signal: test\n    decision_times_et: ['10:00']\n    lookbacks_minutes: [5]\n    tails: [0.1]\n    holding_periods_minutes: [30]\n    weighting_methods: [equal]\n    portfolio_forms: [long_only]\n""", encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        Phase2Config.from_yaml(path)
