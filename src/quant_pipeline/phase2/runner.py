from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..holdout import assert_pre_holdout_parquet
from .config import Phase2Config
from .registry import initial_batch
from .signals import build_signal, feature_names
from .data import load_phase1_signal_rows
from .selection import select_cross_sectional_tails
from .portfolio import assign_weights
from .execution import apply_next_bar_open_fills
from .evaluation import summarize_returns, drawdown_summary
from .statuses import promote
from .correlation import return_correlations
from .reporting import write_reports


def preflight(config: Phase2Config) -> Path:
    """Create an immutable Phase 2 manifest without reading any holdout rows."""
    config.validate()
    phase1 = Path(config.phase1_source_run)
    phase1b = Path(config.phase1b_source_run)
    if not phase1.exists() or not phase1b.exists():
        raise FileNotFoundError("Configured Phase 1 source run does not exist")
    for root, label in ((phase1, "Phase 1A"), (phase1b, "Phase 1B")):
        for path in root.rglob("*.parquet"):
            assert_pre_holdout_parquet(path, config.sealed_holdout_start, f"Phase 2 {label} source {path.name}", verify_key_rows=False)
    output = Path(config.output_root) / config.experiment_id
    if output.resolve() in {phase1.resolve(), phase1b.resolve()}:
        raise ValueError("Phase 2 output must not overwrite a Phase 1 directory")
    output.mkdir(parents=True, exist_ok=True)
    batch = initial_batch(config.strategies) if config.run_initial_batch_only else config.strategies
    sources = {}
    for root, label in ((phase1, "phase1a"), (phase1b, "phase1b")):
        manifest = next((path for path in (root / "MANIFEST.json", root / "manifest.json", root / "fingerprint.json") if path.exists()), None)
        sources[label] = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest is not None else None
    payload = {
        "config": config.as_dict(), "strategy_families": [item.family for item in batch],
        "source_manifest_hashes": sources, "holdout_access": False,
        "holdout_guard": "all source Parquet files verified pre-2026-05-01 before run",
    }
    (output / "MANIFEST.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (output / "holdout_guard_report.json").write_text(json.dumps({
        "sealed_holdout_start": config.sealed_holdout_start, "holdout_access": False, "status": "PASS"
    }, indent=2), encoding="utf-8")
    return output


def execute_initial_batch(config: Phase2Config) -> Path:
    """Run only supported next-bar-open initial-batch configurations; record every deferral."""
    root = preflight(config)
    batch = initial_batch(config.strategies) if config.run_initial_batch_only else config.strategies
    summaries, trades_out, daily_out, failures = [], [], [], []
    frame_cache: dict[tuple[tuple[str, ...], int, str], pd.DataFrame] = {}
    for family in batch:
        for time in family.decision_times_et:
            for lookback in family.lookbacks_minutes:
                for tail in family.tails:
                    for hold in family.holding_periods_minutes:
                        for weighting in family.weighting_methods:
                            for form in family.portfolio_forms:
                                base_id = family.deterministic_id(time, lookback, tail, hold, weighting, form, "next_bar_open", 0)
                                try:
                                        features = feature_names(family.family, lookback)
                                        cache_key = (features, hold, time)
                                        if cache_key not in frame_cache:
                                            frame_cache[cache_key] = load_phase1_signal_rows(config.phase1_source_run, features, hold, time, config.sealed_holdout_start)
                                        frame = frame_cache[cache_key]
                                        frame = build_signal(frame, family.family, features)
                                        selected = select_cross_sectional_tails(frame, tail)
                                        if form == "long_only": selected = selected.loc[selected.side.eq(1)].copy()
                                        selected["prior_beta"] = selected["beta_at_decision"]
                                        if weighting == "inverse_volatility":
                                            selected = selected.sort_values(["symbol", "session_date"]).copy()
                                            selected["prior_volatility"] = selected.groupby("symbol").raw_return.transform(lambda x: x.shift(1).rolling(60, min_periods=20).std())
                                        weighted = assign_weights(selected, weighting, form)
                                except Exception as exc:
                                        failures.append({"strategy_id": base_id, "family": family.family, "reason": str(exc)})
                                        continue
                                for cost in config.adverse_slippage_bps:
                                    strategy_id = family.deterministic_id(time, lookback, tail, hold, weighting, form, "next_bar_open", cost)
                                    try:
                                        filled = apply_next_bar_open_fills(weighted, cost)
                                        filled["position_return"] = filled.final_weight.abs() * filled.net_return
                                        daily = filled.groupby("session_date").position_return.sum().sort_index()
                                        metrics = summarize_returns(daily)
                                        metrics.update({"strategy_id": strategy_id, "family": family.family, "classification": family.classification,
                                                        "cluster": family.cluster, "decision_time_et": time, "lookback_minutes": lookback,
                                                        "tail": tail, "holding_period_minutes": hold, "weighting": weighting,
                                                        "portfolio_form": form, "execution_model": "next_bar_open", "cost_bps_per_side": cost,
                                                        "trade_count": len(filled), "sessions": len(daily),
                                                        "recent_cagr": summarize_returns(daily[pd.to_datetime(daily.index) >= pd.Timestamp("2025-05-01")])["net_cagr"],
                                                        "positive_year_fraction": float((1 + daily).groupby(pd.to_datetime(daily.index).year).prod().sub(1).gt(0).mean())})
                                        summaries.append(metrics)
                                        trades_out.append(filled.assign(strategy_id=strategy_id))
                                        daily_out.append(pd.DataFrame({"session_date": daily.index, "net_return": daily.values, "strategy_id": strategy_id}))
                                    except Exception as exc:
                                        failures.append({"strategy_id": strategy_id, "family": family.family, "reason": str(exc)})
        # A family is an atomic resume boundary: preserve auditable partial
        # progress if the desktop command times out before the full batch.
        pd.DataFrame(summaries).to_csv(root / "strategy_summary.partial.csv", index=False)
        pd.DataFrame(failures, columns=["strategy_id", "family", "reason"]).to_csv(root / "failed_configurations.partial.csv", index=False)
    summary = promote(pd.DataFrame(summaries)) if summaries else pd.DataFrame()
    failed = pd.DataFrame(failures, columns=["strategy_id", "family", "reason"])
    summary.to_csv(root / "strategy_summary.csv", index=False); failed.to_csv(root / "failed_configurations.csv", index=False)
    trades = pd.concat(trades_out, ignore_index=True) if trades_out else pd.DataFrame()
    daily = pd.concat(daily_out, ignore_index=True) if daily_out else pd.DataFrame()
    trades.to_parquet(root / "trades.parquet", index=False); daily.to_parquet(root / "daily_strategy_returns.parquet", index=False)
    (failed.groupby("reason", dropna=False).size().rename("count").reset_index() if "reason" in failed else pd.DataFrame(columns=["reason", "count"])).to_csv(root / "skip_reason_summary.csv", index=False)
    summary.to_csv(root / "strategy_cost_stress.csv", index=False); summary[[c for c in summary if c in {"strategy_id", "maximum_drawdown", "status", "net_cagr"}]].to_csv(root / "strategy_drawdown_summary.csv", index=False)
    summary[[c for c in summary if c in {"strategy_id", "status", "family", "cluster", "cost_bps_per_side"}]].to_csv(root / "strategy_statuses.csv", index=False)
    if not trades.empty:
        trades.groupby(["strategy_id", "symbol"]).net_return.sum().reset_index().to_csv(root / "strategy_concentration.csv", index=False)
        trades.groupby("strategy_id").agg(trade_count=("symbol", "size"), win_rate=("net_return", lambda x: (x > 0).mean()), average_net_bps=("net_return", lambda x: x.mean() * 10_000)).reset_index().to_csv(root / "strategy_trade_statistics.csv", index=False)
        trades[[c for c in ["strategy_id", "decision_ts", "symbol", "side", "final_weight", "entry_ts", "exit_ts", "entry_executable_price", "exit_executable_price", "slippage_cost"] if c in trades]].to_parquet(root / "execution_diagnostics.parquet", index=False)
        trades[[c for c in ["strategy_id", "session_date", "decision_ts", "symbol", "side", "target_weight", "final_weight"] if c in trades]].to_parquet(root / "positions.parquet", index=False)
    if not daily.empty:
        dated = daily.assign(year=pd.to_datetime(daily.session_date).dt.year, month=pd.to_datetime(daily.session_date).dt.to_period("M").astype(str))
        dated.groupby(["strategy_id", "year"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(root / "strategy_yearly_results.csv", index=False)
        dated.groupby(["strategy_id", "month"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(root / "strategy_monthly_results.csv", index=False)
        summary[[c for c in summary if c in {"strategy_id", "family", "decision_time_et", "lookback_minutes", "tail", "holding_period_minutes", "net_cagr", "sharpe", "maximum_drawdown"}]].to_csv(root / "strategy_parameter_stability.csv", index=False)
        summary.groupby("strategy_id").sessions.max().rename("sessions").reset_index().to_csv(root / "data_coverage.csv", index=False)
    for name in ("strategy_walk_forward_results.csv", "strategy_exposure_statistics.csv", "stress_period_correlations.csv", "strategy_cluster_map.csv", "cluster_correlations.csv", "strategy_overlap.csv", "portfolio_summary.csv", "portfolio_yearly_results.csv", "portfolio_monthly_results.csv", "portfolio_drawdowns.csv", "portfolio_allocations.csv", "portfolio_leverage.csv", "portfolio_sleeve_contributions.csv", "portfolio_leave_one_sleeve_out.csv", "portfolio_cost_stress.csv", "portfolio_statuses.csv"):
        if not (root / name).exists(): pd.DataFrame().to_csv(root / name, index=False)
    if not daily.empty:
        corr_daily, corr_monthly = return_correlations(daily)
        corr_daily.to_csv(root / "daily_return_correlations.csv"); corr_monthly.to_csv(root / "monthly_return_correlations.csv")
    else:
        pd.DataFrame().to_csv(root / "daily_return_correlations.csv")
        pd.DataFrame().to_csv(root / "monthly_return_correlations.csv")
    write_reports(root, summary, failed, config.sealed_holdout_start)
    return root
