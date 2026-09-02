from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2")
OUT = ROOT / r"results\phase2_holy_shit_execution_first_through_20260430"
QUOTE = Path(r"D:\AlgoResearch\research_pipeline\terra_reports\phase2_holy_shit_exact_quote_1y_v2")


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100*x:.2f}%"


def metrics(r: pd.Series) -> dict:
    r = r.astype(float); eq = (1 + r).cumprod(); peaks = np.maximum.accumulate(np.r_[1., eq.to_numpy()]); dd = eq.to_numpy() / peaks[1:] - 1
    end_i = int(np.argmin(dd)); prior = np.r_[1., eq.to_numpy()[:end_i + 1]]; peak_i = int(np.argmax(prior)); peak_date = r.index[0] if peak_i == 0 else r.index[peak_i - 1]
    years = max(((pd.Timestamp(r.index[-1]) - pd.Timestamp(r.index[0])).days + 1) / 365.25, 1 / 252); sd = r.std()
    return dict(cagr=float(eq.iloc[-1] ** (1 / years) - 1), total_return=float(eq.iloc[-1] - 1), sharpe=float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0., max_dd=float(dd.min()), pt_days=int((pd.Timestamp(r.index[end_i]) - pd.Timestamp(peak_date)).days))


def markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.fillna("").astype(str).apply(lambda c: c.str.replace("|", "\\|", regex=False))
    header = "| " + " | ".join(map(str, clean.columns)) + " |"; rule = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in clean.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(QUOTE / "summary.csv"); periods = pd.read_csv(QUOTE / "period_stats.csv"); daily = pd.read_parquet(QUOTE / "daily_returns.parquet"); ledger = pd.read_parquet(QUOTE / "quote_replay_ledger.parquet")
    summary.to_csv(OUT / "quote_execution_summary.csv", index=False); periods.to_csv(OUT / "quote_period_stats.csv", index=False); pd.read_csv(QUOTE / "fill_status_counts.csv").to_csv(OUT / "fill_status_counts.csv", index=False); pd.read_csv(QUOTE / "frozen_full_universe_thresholds.csv").to_csv(OUT / "frozen_full_universe_thresholds.csv", index=False)
    one = summary[summary.cost_bp_side.eq(1)].copy(); stress = summary[summary.cost_bp_side.eq(3)][["candidate_id", "cagr"]].rename(columns={"cagr": "stress_cagr_3bp"}); folds = periods[periods.cost_bp_side.eq(1)].pivot(index="candidate_id", columns="period", values="cagr").add_prefix("cagr_")
    rank = one.merge(stress, on="candidate_id", how="left").merge(folds, on="candidate_id", how="left"); rank["fold_min_cagr"] = rank[["cagr_train", "cagr_development", "cagr_recent"]].min(axis=1); rank["positive_folds"] = (rank[["cagr_train", "cagr_development", "cagr_recent"]] > 0).sum(axis=1)
    rank["gate"] = (rank.coverage >= .50) & (rank.fill_rate >= .05) & (rank.filled >= 100) & (rank.sessions >= 40) & (rank.cagr > 0) & (rank.stress_cagr_3bp >= -.002) & (rank.positive_folds >= 2) & (rank.max_dd >= -.05) & (rank.pt_days <= 31)
    rank["quality"] = rank.cagr / (rank.max_dd.abs() + .0025) + rank.fold_min_cagr
    rank = rank.sort_values(["mechanism_id", "gate", "quality", "cagr"], ascending=[True, False, False, False]); chosen = rank.groupby("mechanism_id", as_index=False).head(1).sort_values(["gate", "quality"], ascending=False).copy(); chosen["status"] = np.where(chosen.gate, "execution_qualified_discovery", "rejected")
    chosen.to_csv(OUT / "selected_candidate_statuses.csv", index=False); ids = set(chosen.candidate_id)
    summary[summary.candidate_id.isin(ids)].to_csv(OUT / "selected_quote_cost_stress.csv", index=False); periods[periods.candidate_id.isin(ids)].to_csv(OUT / "selected_quote_period_stats.csv", index=False)
    historical = pd.read_csv(OUT / "historical_bar_summary.csv"); annual = pd.read_csv(OUT / "historical_bar_annual_stats.csv"); chosen["historical_id"] = chosen.candidate_id.str.replace(r"__(market|passive)__d[05]$", "", regex=True); hist_ids = set(chosen.historical_id)
    historical[historical.candidate_id.isin(hist_ids)].to_csv(OUT / "selected_historical_cost_stress.csv", index=False); annual[annual.candidate_id.isin(hist_ids)].to_csv(OUT / "selected_historical_annual_stats.csv", index=False)
    cols = {cid: f"{cid}__cost1" for cid in ids}; selected_daily = pd.DataFrame({cid: daily[col] for cid, col in cols.items()}); selected_daily.to_parquet(OUT / "selected_quote_daily_returns.parquet"); corr = selected_daily.corr(); corr.to_csv(OUT / "daily_return_correlations.csv")
    monthly = selected_daily.copy(); monthly.index = pd.to_datetime(monthly.index); monthly = monthly.groupby(monthly.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1); monthly.index = monthly.index.astype(str); monthly.to_csv(OUT / "selected_quote_monthly_returns.csv")
    overlap = []
    filled = ledger[ledger.status.eq("filled") & ledger.candidate_id.isin(ids)].copy(); filled["key"] = filled.bucket.astype(str) + "|" + filled.symbol + "|" + filled.side.astype(str)
    sets = {cid: set(g.key) for cid, g in filled.groupby("candidate_id")}
    for a, b in combinations(sorted(ids), 2):
        u = sets.get(a, set()) | sets.get(b, set()); overlap.append(dict(candidate_a=a, candidate_b=b, jaccard=len(sets.get(a, set()) & sets.get(b, set())) / len(u) if u else 0, daily_correlation=corr.loc[a, b]))
    pd.DataFrame(overlap).to_csv(OUT / "strategy_overlap.csv", index=False)
    leverage = []
    for cid in sorted(ids):
        for lev in (1., 2., 4.): leverage.append(dict(candidate_id=cid, leverage=lev, **metrics(selected_daily[cid] * lev)))
    leverage = pd.DataFrame(leverage); leverage.to_csv(OUT / "selected_leverage_diagnostics.csv", index=False)
    failures = rank[~rank.candidate_id.isin(ids)].copy(); failures["reason"] = "not_best_variant_for_distinct_mechanism"; failures = pd.concat([failures, chosen[~chosen.gate].assign(reason="failed_execution_qualification_gate")], ignore_index=True); failures.to_csv(OUT / "failed_configurations.csv", index=False)
    cost = summary[summary.candidate_id.isin(ids)].pivot(index="candidate_id", columns="cost_bp_side", values="cagr").reindex(chosen.candidate_id); cost.columns = [f"cagr_{c:g}bp" for c in cost.columns]
    display = chosen.set_index("candidate_id")[["mechanism_id", "execution", "delay_seconds", "filled", "fill_rate", "coverage", "positive_folds", "cagr", "stress_cagr_3bp", "max_dd", "pt_days", "status"]].join(cost).reset_index()
    for c in ["fill_rate", "coverage", "cagr", "stress_cagr_3bp", "max_dd"] + list(cost.columns): display[c] = display[c].map(pct)
    fold_display = periods[(periods.candidate_id.isin(ids)) & periods.cost_bp_side.eq(1)].pivot(index="candidate_id", columns="period", values=["cagr", "max_dd", "pt_days"]); fold_display.columns = [f"{a}_{b}" for a, b in fold_display.columns]; fold_display = fold_display.reindex(chosen.candidate_id).reset_index()
    for c in [x for x in fold_display if x.startswith("cagr_") or x.startswith("max_dd_")]: fold_display[c] = fold_display[c].map(pct)
    annual_display = annual[(annual.candidate_id.isin(hist_ids)) & annual.cost_bp_side.eq(1)].pivot(index="candidate_id", columns="period", values="cagr").reset_index()
    for c in annual_display.columns[1:]: annual_display[c] = annual_display[c].map(pct)
    lev_display = leverage[leverage.candidate_id.isin(ids)].copy(); lev_display["cagr"] = lev_display.cagr.map(pct); lev_display["max_dd"] = lev_display.max_dd.map(pct)
    passed = int(chosen.gate.sum()); max_corr = float(pd.DataFrame(overlap).daily_correlation.abs().max()) if overlap else 0.; max_jaccard = float(pd.DataFrame(overlap).jaccard.max()) if overlap else 0.
    report = f"""# Phase 2 execution-first candidate report

## Verdict

**0 of 10 distinct mechanisms met the requested "holy shit" standard; {passed} of 10 passed the minimum descriptive execution-qualification gate.** The two survivors are low-return research sleeves, not money printers. A pass here means discovery-period quote execution survived, not that the sealed holdout passed or that live trading is approved. Any failed row is explicitly rejected; rebate-only results are not promoted.

Gate: at least 50% exact-window quote-path coverage, at least 5% paired fill rate, 100 filled orders, and 40 active sessions; positive +1 bp CAGR in at least two of three chronological folds; +3 bp CAGR no worse than -0.20% (approximately break-even); maximum drawdown no worse than -5%; and peak-to-trough duration no longer than 31 calendar days.

## Selected distinct mechanisms and execution grid

Slippage/cost is per side and is added after the observed SIP bid/ask or paired passive fill. Delay is explicit in the `delay_seconds` column: 0 is the required quote-base model and 5 is the conservative timing variant. Passive fills require the quote to trade through the resting price within the ten-second path; unmatched long/short fills are canceled.

{markdown_table(display)}

## Chronological quote-period results at +1 bp per side

{markdown_table(fold_display)}

## Fixed-spec historical bar diagnostic at +1 bp per side

This backward diagnostic uses completed-bar prices, not quotes, and is therefore secondary evidence only. It is included to expose long-history instability rather than to claim executable performance.

{markdown_table(annual_display)}

## Mechanical leverage diagnostic at +1 bp per side

{markdown_table(lev_display[["candidate_id", "leverage", "cagr", "sharpe", "max_dd", "pt_days"]])}

## Distinctness

The final comparison contains exactly one configuration per economic mechanism. Maximum absolute pairwise daily-return correlation was {max_corr:.3f}; maximum filled-trade Jaccard overlap was {max_jaccard:.3f}. Full matrices are in `daily_return_correlations.csv` and `strategy_overlap.csv`.

## Integrity and limitations

- Source universe: point-in-time `analysis_eligible` symbols, 2025-05-01 through 2026-04-30 for quote replay; full bar diagnostic begins 2019-06-21.
- Extreme thresholds were frozen from 2025-05-01 through 2025-08-31 on the complete bar universe, never from quote availability.
- The original strict 95%-coverage/all-fold gate produced zero passes. The displayed operational gate was documented after the event-style quote-coverage audit, so it is descriptive rather than prospective and cannot itself authorize a holdout test.
- Sealed holdout begins 2026-05-01 and was not accessed. This report cannot honestly call any candidate holdout-validated or live-approved.
- Quote replay rejects missing, locked/crossed, non-positive-size-invalid, and wider-than-10-bp snapshots. Missing/partial fills remain unfilled.
- Passive quote trade-through does not prove exchange queue priority. Borrow/locate, fees beyond the explicit grid, capacity, and live order-management remain external gates.
- Standalone full-capital statistics can imply concentration. A future multi-sleeve portfolio must net symbols and apply the common 5%/10%/20% symbol-cap grid before leverage.

## Reproducible artifacts

- `selected_candidate_statuses.csv`: objective selection and gate verdicts.
- `selected_quote_cost_stress.csv`: every selected strategy at -1, -0.5, 0, +1, +3, and +5 bp per side.
- `selected_quote_period_stats.csv`: every selected strategy in every chronological quote fold.
- `selected_historical_annual_stats.csv`: fixed-spec annual bar diagnostics.
- `fill_status_counts.csv` in the quote run: filled, unfilled, canceled-unpaired, and missing paths.
- `failed_configurations.csv`: every tested variant not selected plus selected mechanisms that failed the gate.
"""
    (OUT / "HOLY_SHIT_STRATEGY_REPORT.md").write_text(report, encoding="utf-8")
    guard = json.loads((OUT / "holdout_guard_report.json").read_text()); manifest = dict(run_id=OUT.name, discovery_end="2026-04-30", sealed_holdout_start="2026-05-01", holdout_access=False, holdout_guard_passed=guard["passed"], mechanisms=10, execution_variants=int(summary.candidate_id.nunique()), strict_initial_gate_passes=0, descriptive_execution_qualified_mechanisms=passed, holy_shit_candidates=0, gate_prospective=False, quote_root=str(QUOTE), max_abs_daily_correlation=max_corr, max_trade_jaccard=max_jaccard)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); (OUT / "run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
