from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import ScanConfig
from .holdout import assert_pre_holdout_frame


def write_reports(results: pd.DataFrame, quantiles: dict[tuple[str, str], pd.DataFrame], root: Path, frame: pd.DataFrame | None = None, *, config:ScanConfig|None=None, run_metadata:dict|None=None) -> None:
    """Write diagnostics only; never strategy-level performance metrics."""
    config=config or ScanConfig()
    run_metadata=run_metadata or {}
    assert_pre_holdout_frame(results,config.sealed_holdout_start,"report results")
    if frame is not None:assert_pre_holdout_frame(frame,config.sealed_holdout_start,"report generation")
    candidates = results.head(50) if not results.empty else results
    candidates.to_html(root / "ranked_candidates.html", index=False, float_format=lambda x: f"{x:.6g}")
    charts = root / "charts"; charts.mkdir(exist_ok=True)
    selected={(r.feature,r.target) for r in candidates.itertuples()} if not candidates.empty else set()
    for (feature, target), table in quantiles.items():
        if (feature,target) not in selected:
            continue
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x=table["bin"] if "bin" in table else table["category"]
        if {"ci_low","ci_high"}.issubset(table):
            yerr=[table["mean"]-table.ci_low,table.ci_high-table["mean"]]; ax.errorbar(x,table["mean"],yerr=yerr,fmt="o-")
        else:ax.plot(x,table["mean"],"o-")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{feature} -> {target}", xlabel="Feature quantile", ylabel="Mean forward return")
        fig.tight_layout(); fig.savefig(charts / f"{feature}__{target}.png", dpi=140); plt.close(fig)
    master=pd.read_csv(root/"master_results.csv") if (root/"master_results.csv").exists() else pd.DataFrame()
    feature_registry=pd.read_csv(root/"feature_registry.csv") if (root/"feature_registry.csv").exists() else pd.DataFrame()
    target_registry=pd.read_csv(root/"target_registry.csv") if (root/"target_registry.csv").exists() else pd.DataFrame()
    clusters=results.candidate_cluster.nunique() if "candidate_cluster" in results else 0
    status=results.get("status",pd.Series(dtype=str)); robust=int(status.eq("robust_phase1_anomaly_candidate").sum()); phase2=int(status.eq("requires_phase2_testing").sum())
    primary_tests=int(master.get("primary_test_count",pd.Series([0])).max()) if not master.empty else 0; exploratory_tests=int(master.get("exploratory_test_count",pd.Series([0])).max()) if not master.empty else 0
    text = ["# Final Phase 1 full-pre-holdout discovery", "", f"All permitted data through {config.discovery_end} was used for discovery and candidate ranking. Historical subperiod and recent-period results are robustness diagnostics. {config.sealed_holdout_start} onward was not accessed.", "", "Statistical anomaly evidence only; every promoted finding requires Phase 2 strategy and execution testing.", "", "Regime, scope, exact-time, and Phase 2 recommendation diagnostics are descriptive only and were not used to alter FDR, candidate ranking, candidate selection, promotion, or anomaly status.", "", "## Data and methodology", "", f"- Evidence label: full_pre_holdout_discovery",f"- Discovery start: {config.start}",f"- Discovery end: {config.discovery_end}",f"- Sealed holdout start: {config.sealed_holdout_start}",f"- Holdout access: {str(config.allow_holdout_access).lower()}",f"- Source-data fingerprint: {run_metadata.get('fingerprint','unavailable')}",f"- Git commit: {run_metadata.get('git_commit','unavailable')}",f"- Cache schema: {config.cache_schema_version}",f"- Separate confirmation period: {str(config.use_separate_confirmation_period).lower()}",f"- Primary FDR family tests: {primary_tests}",f"- Exploratory FDR family tests: {exploratory_tests}","- Primary inference: global Benjamini-Hochberg FDR across prespecified primary targets.","- Exploratory inference: separate families; exploratory and recency-weighted evidence cannot independently promote a candidate.","", "## Results funnel", "", f"- Features requested: {len(feature_registry)}",f"- Features successfully built: {len(set(master.feature)) if not master.empty else 0}",f"- Targets requested: {len(target_registry)}",f"- Broad-screen feature-target pairs: {len(master)}",f"- Pairs passing coverage: {int(master.raw_p.notna().sum()) if not master.empty else 0}",f"- Primary globally significant pairs: {int(master.get('primary_global_fdr',pd.Series(dtype=float)).lt(config.primary_fdr_threshold).sum()) if not master.empty else 0}",f"- Exploratory significant pairs: {int(master.get('exploratory_family_fdr',pd.Series(dtype=float)).lt(config.primary_fdr_threshold).sum()) if not master.empty else 0}",f"- Redundancy clusters: {clusters}",f"- Exact diagnostic candidates: {len(results)}",f"- Robust Phase 1 anomaly candidates: {robust}",f"- Candidates requiring Phase 2 testing: {phase2}"]
    if not results.empty:
        columns=[c for c in ["feature","target","top_bottom_spread","bh_fdr_p","recent_5y_effect","recent_3y_effect","recent_2y_effect","recent_12m_effect","jan_apr_2026_effect","recent_classification","symbol_breadth_classification","phase2_recommendation","status"] if c in candidates]
        markdown=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
        markdown += ["| "+" | ".join(str(getattr(row,c)) for c in columns)+" |" for row in candidates[columns].itertuples(index=False)]
        text += ["", "## Top relationships", "", "\n".join(markdown)]
        sections=[("Regime summary",["feature","target","regime_summary_label"]),("Scope summary",["feature","target","sector_scope_status","industry_scope_status","scope_classification"]),("Exact-time summary",["feature","target","strongest_exact_decision_time","weakest_exact_decision_time","time_concentration_label"]),("Phase 2 recommendation",["feature","target","phase2_recommendation","phase2_recommendation_reason","phase2_main_limitation","phase2_suggested_test"])]
        for title,wanted in sections:
            fields=[c for c in wanted if c in candidates]
            if len(fields)<3:continue
            lines=["| "+" | ".join(fields)+" |","| "+" | ".join(["---"]*len(fields))+" |"]+["| "+" | ".join(str(getattr(row,c)) for c in fields)+" |" for row in candidates[fields].itertuples(index=False)]
            text += ["",f"## {title}","","\n".join(lines)]
    (root / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if frame is not None and not candidates.empty:
        stable=root/"stability"; stable.mkdir(exist_ok=True)
        for row in candidates.head(25).itertuples():
            cols=["session_date","symbol","decision_ts",row.feature,row.target]
            z=frame[cols].dropna().copy(); z["year"]=pd.to_datetime(z.session_date).dt.year
            local=pd.to_datetime(z.decision_ts,utc=True).dt.tz_convert("America/New_York"); z["time_bucket"]=(local.dt.hour*60+local.dt.minute).floordiv(60)
            for key in ["year","symbol","time_bucket"]:
                tab=z.groupby(key).agg(observations=(row.target,"size"),mean_target=(row.target,"mean"),spearman=(row.feature,lambda s:s.corr(z.loc[s.index,row.target],method="spearman"))).reset_index()
                tab.to_csv(stable/f"{row.feature}__{row.target}__by_{key}.csv",index=False)
