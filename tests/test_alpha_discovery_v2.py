from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.alpha_discovery.cache.rank_store import build_rank_bins
from quant_pipeline.alpha_discovery.config import AlphaDiscoveryConfig
from quant_pipeline.alpha_discovery.data.aggregation import aggregate_bars
from quant_pipeline.alpha_discovery.data.snapshot import validate_snapshot
from quant_pipeline.alpha_discovery.features.formulas import (
    absolute_return_hhi, information_discreteness, path_chord_convexity,
    path_efficiency, path_quadratic_curvature,
)
from quant_pipeline.alpha_discovery.governance.access import AccessGate
from quant_pipeline.alpha_discovery.governance.state import ResearchStateMachine
from quant_pipeline.alpha_discovery.interactions.formula_factory import FormulaFactory
from quant_pipeline.alpha_discovery.models import ResearchState
from quant_pipeline.alpha_discovery.registry import compile_registry
from quant_pipeline.alpha_discovery.features.base import FeatureBuilder
from quant_pipeline.alpha_discovery.run import AlphaDiscoveryRun, STAGE_ORDER
from quant_pipeline.alpha_discovery.scan.dual_coarse import DualTileScanner
from quant_pipeline.alpha_discovery.scan.fdr import benjamini_hochberg
from quant_pipeline.alpha_discovery.targets.builder import build_daily_targets, build_intraday_targets


CONFIG = Path(__file__).parents[1] / "configs" / "alpha_discovery_v1.yaml"


def minute_bars(days: int = 2, bars_per_day: int = 10) -> pd.DataFrame:
    rows = []
    for day in range(days):
        session = pd.Timestamp("2026-04-27") + pd.Timedelta(days=day)
        start = pd.Timestamp(session.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for index in range(bars_per_day):
            bar_start = (start + pd.Timedelta(minutes=index)).tz_convert("UTC"); price = 100 + day + index
            rows.append({"security_id": "s1", "symbol": "AAA", "session_date": session.date(),
                         "bar_start_ts_utc": bar_start, "bar_end_ts_utc": bar_start + pd.Timedelta(minutes=1),
                         "availability_ts_utc": bar_start + pd.Timedelta(minutes=1), "open": price, "high": price + 1,
                         "low": price - 1, "close": price + .5, "volume": 100 + index, "vwap": price + .25,
                         "trade_count": 10, "feed": "sip", "adjustment": "raw", "ingest_batch_id": "batch"})
    return pd.DataFrame(rows)


def test_config_and_registry_are_standalone_and_deterministic():
    config = AlphaDiscoveryConfig.from_yaml(CONFIG); bundle = compile_registry(config); again = compile_registry(config)
    assert Path(config.project_root).name == "Quant Pipeline V2"
    assert 2_000 <= len(bundle.features) <= 8_000
    assert len(bundle.concepts) >= 250 and len(bundle.targets) == 63
    assert [x.definition_hash for x in bundle.features] == [x.definition_hash for x in again.features]
    assert len({x.feature_id for x in bundle.features}) == len(bundle.features)
    assert bundle.required_warmup_sessions >= 257


def test_snapshot_enforces_left_edge_raw_sip_and_keys():
    bars = minute_bars()
    validation = validate_snapshot(bars)
    assert validation.rows == 20 and validation.feed_values == ("sip",)
    broken = bars.copy(); broken.loc[0, "availability_ts_utc"] = broken.loc[0, "bar_start_ts_utc"]
    with pytest.raises(ValueError, match="timestamp contract"):
        validate_snapshot(broken)


def test_session_safe_aggregation_is_exact():
    bars = minute_bars(days=1, bars_per_day=10); result = aggregate_bars(bars, 5)
    assert len(result) == 2
    assert result.iloc[0].open == 100 and result.iloc[0].close == 104.5
    assert result.iloc[0].high == 105 and result.iloc[0].low == 99
    assert result.iloc[0].volume == sum(range(100, 105))
    assert result.iloc[0].availability_ts_utc == bars.iloc[4].bar_end_ts_utc


def test_intraday_target_0935_enters_0936_and_counts_holding_bars():
    bars = minute_bars(days=1, bars_per_day=12)
    decision_ts = bars.iloc[4].bar_end_ts_utc  # 09:35 ET
    decisions = pd.DataFrame({"observation_id": [1], "security_id": ["s1"], "session_date": [bars.iloc[0].session_date], "decision_ts": [decision_ts]})
    targets = build_intraday_targets(decisions, bars, horizons=(1, 2, 5), include_eod=False)
    assert targets.entry_ts.nunique() == 1
    assert targets.entry_ts.iloc[0] == bars.iloc[6].bar_start_ts_utc  # strictly after 09:35 -> 09:36
    ends = dict(zip(targets.target_id, targets.exit_ts))
    assert ends["target_1m__raw__intraday_5m"] == bars.iloc[6].bar_end_ts_utc
    assert ends["target_2m__raw__intraday_5m"] == bars.iloc[7].bar_end_ts_utc
    assert ends["target_5m__raw__intraday_5m"] == bars.iloc[10].bar_end_ts_utc


def test_daily_one_day_target_exits_entry_session_close():
    bars = minute_bars(days=3, bars_per_day=10); first = bars.iloc[9]
    decisions = pd.DataFrame({"observation_id": [1], "security_id": ["s1"], "session_date": [first.session_date], "decision_ts": [first.bar_end_ts_utc]})
    targets = build_daily_targets(decisions, bars, horizons=(1, 2), entry_delay_minutes=1)
    one = targets[targets.target_id.str.startswith("target_1d")].iloc[0]
    assert one.entry_ts == bars.iloc[11].bar_start_ts_utc
    assert one.exit_ts == bars.iloc[19].bar_end_ts_utc


def test_path_and_consistency_formula_contracts():
    efficient = np.array([100., 101., 102., 103.]); noisy = np.array([100., 102., 99., 103.])
    assert path_efficiency(efficient) == pytest.approx(1.0)
    assert path_efficiency(noisy) < 1
    below_chord = np.array([100., 100.2, 101., 104.])
    assert path_chord_convexity(below_chord) > 0
    convex = np.exp(np.linspace(-1, 1, 9) ** 2)
    assert path_quadratic_curvature(convex) > 0
    returns = np.array([.01, -.005, .002, -.001])
    assert np.isfinite(information_discreteness(returns))
    assert absolute_return_hhi(np.array([1., 0., 0.])) == pytest.approx(1.0)


def test_rank_bins_and_dual_cpu_surface():
    values = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]], dtype=float)
    labels, valid = build_rank_bins(values, np.zeros(4, dtype=int), 3)
    scanner = DualTileScanner(bins=3, prefer_cuda=False)
    result = scanner.scan(labels[:, [0]].astype(int), labels[:, [1]].astype(int), np.array([0., 1., 2., 3.]), valid[:, [0]], valid[:, [1]])
    assert scanner.backend == "torch:cpu"
    assert result.n_obs.iloc[0] == 4 and len(result.surface_means.iloc[0]) == 9


def test_dual_cuda_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(7); rows, pairs = 1000, 7
    a = rng.integers(0, 5, size=(rows, pairs)); b = rng.integers(0, 5, size=(rows, pairs)); y = rng.normal(size=rows)
    cpu = DualTileScanner(bins=5, prefer_cuda=False).scan(a, b, y)
    gpu_scanner = DualTileScanner(bins=5, prefer_cuda=True); gpu = gpu_scanner.scan(a, b, y)
    assert gpu_scanner.backend.startswith("torch:cuda")
    for column in ("best_cell_effect", "worst_cell_effect", "max_abs_incremental_cell", "surface_interaction_energy"):
        assert np.allclose(cpu[column], gpu[column], rtol=1e-12, atol=1e-12)


def test_state_machine_and_source_access_are_fail_closed(tmp_path: Path):
    machine = ResearchStateMachine(tmp_path / "state.json")
    machine.transition(ResearchState.DISCOVERY_OPEN)
    with pytest.raises(ValueError, match="Illegal"):
        machine.transition(ResearchState.REPLICATION_OPEN)
    config = AlphaDiscoveryConfig.from_yaml(CONFIG)
    gate = AccessGate(config, ResearchState.DISCOVERY_OPEN)
    assert "2026-04-30" in gate.sql_predicate()
    assert "2019-06-21" in gate.sql_predicate()
    with pytest.raises(ValueError, match="Later-period"):
        gate.assert_frame(pd.DataFrame({"session_date": ["2026-05-01"]}))


def test_bh_and_formula_factory_are_deterministic():
    adjusted = benjamini_hochberg(np.array([.01, .03, .20]))
    assert np.all(np.diff(np.sort(adjusted)) >= 0)
    factory = FormulaFactory(); formulas = factory.generate(["a", "b"])
    assert len({item.expression_hash for item in formulas}) == len(formulas)
    multiply = next(item for item in formulas if item.expression.get("op") == "multiply")
    assert np.array_equal(factory.evaluate(multiply, {"a": np.array([2.]), "b": np.array([3.])}), np.array([6.]))


def test_every_cli_stage_has_a_concrete_handler():
    missing = [stage for stage in STAGE_ORDER if not callable(getattr(AlphaDiscoveryRun, "_stage_" + stage.replace("-", "_"), None))]
    assert missing == []


def test_every_active_feature_concept_has_an_executable_mapping():
    config = AlphaDiscoveryConfig.from_yaml(CONFIG); bundle = compile_registry(config); rows = []
    for security_index, security_id in enumerate(("A", "B", "C", "D")):
        for index in range(280):
            timestamp = pd.Timestamp("2024-01-02 20:00", tz="UTC") + pd.Timedelta(days=index)
            price = 100 + security_index * 7 + index * .03 + np.sin(index / 8 + security_index)
            rows.append({"security_id": security_id, "symbol": security_id, "session_date": timestamp.date(), "decision_ts": timestamp,
                         "open": price - .2, "high": price + .8, "low": price - .9, "close": price,
                         "research_close": price, "volume": 10_000 + index * 3, "vwap": price - .05,
                         "trade_count": 100 + index % 10, "benchmark_return": .001 * np.sin(index / 7),
                         "benchmark_gap": .0002 * np.cos(index / 5), "benchmark_intraday_return": .0005 * np.sin(index / 4),
                             "benchmark_close": 400 + index * .1, "beta_prior": .8 + security_index * .1,
                             "benchmark_session_open": 399.5 + index * .1,
                             "benchmark_prior_session_close": 399.0 + index * .1,
                             "session_open": price-.2, "session_close": price, "session_high": price+.8,
                             "session_low": price-.9, "session_vwap": price-.05,
                             "prior_session_close": price-.1, "time_above_vwap": .55,
                             "vwap_cross_count": .1, "open_to_midday_return": .002,
                             "midday_to_close_return": .003, "bucket_return": .001})
            for horizon in (5, 10, 15, 30, 60):
                rows[-1][f"opening_return_{horizon}m"] = .001 * horizon
                rows[-1][f"closing_return_{horizon}m"] = .0005 * horizon
                rows[-1][f"opening_range_pct_{horizon}m"] = .002 * horizon
                rows[-1][f"opening_volume_{horizon}m"] = 1000.0 * horizon
                rows[-1][f"gap_fill_price_{horizon}m"] = price - .05
                rows[-1][f"closing_volume_share_{horizon}m"] = horizon / 390
            rows[-1]["largest_1m_volume_share_session"] = .01
            rows[-1]["largest_5m_volume_share_session"] = .03
    builder = FeatureBuilder(pd.DataFrame(rows)); tested = set(); selected = {}
    for spec in bundle.features:
        if spec.representation == "raw" and (spec.concept_id not in selected or int(spec.scale.value or 1) > int(selected[spec.concept_id].scale.value or 1)):
            selected[spec.concept_id] = spec
    for spec in selected.values():
        values = builder.build(spec)
        assert len(values) == len(rows)
        tested.add(spec.concept_id)
    assert tested == {concept.concept_id for concept in bundle.concepts if concept.active}


def test_multispeed_registry_uses_only_approved_explicit_pairs():
    bundle = compile_registry(AlphaDiscoveryConfig.from_yaml(CONFIG))
    pairs = {(item.parameters["fast_label"], item.parameters["slow_label"]) for item in bundle.features if item.family == "multispeed"}
    assert ("1m", "5m") in pairs and ("63d", "252d") in pairs and ("5m", "20d") not in pairs
    assert all(item.parameters["fast"] < item.parameters["slow"] for item in bundle.features if item.family == "multispeed")
