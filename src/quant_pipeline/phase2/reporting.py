from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


REQUIRED = ("README.md", "READINESS.md", "FINDINGS.md", "STRATEGY_CATALOG.md", "PARAMETER_STABILITY.md", "MULTI_SLEEVE_ANALYSIS.md", "IMPLEMENTATION_NOTES.md", "HOLDOUT_FREEZE_CHECKLIST.md")


def write_reports(root: Path, summary: pd.DataFrame, failed: pd.DataFrame, holdout_start: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    promoted = summary.loc[summary.get("status", pd.Series(index=summary.index, dtype=str)).isin(["standalone_candidate", "diversifying_sleeve_candidate"])]
    (root / "README.md").write_text("# Phase 2 strategy library\n\nDiscovery-only research. The sealed holdout has not been accessed. Next-bar-open fills are reference fills with explicit assumed adverse per-side slippage; they are not quote-base fills.\n", encoding="utf-8")
    (root / "FINDINGS.md").write_text(f"# Findings\n\nConfigurations: {len(summary)}. Promoted discovery candidates: {len(promoted)}.\n", encoding="utf-8")
    (root / "READINESS.md").write_text("# Readiness\n\nNo discovery result is deployment-ready. Quote validation and frozen holdout evaluation remain required.\n", encoding="utf-8")
    (root / "HOLDOUT_FREEZE_CHECKLIST.md").write_text(f"# Holdout freeze\n\nSealed boundary: {holdout_start}. No rows at or after this date may be loaded.\n", encoding="utf-8")
    for name in ("STRATEGY_CATALOG.md", "PARAMETER_STABILITY.md", "MULTI_SLEEVE_ANALYSIS.md", "IMPLEMENTATION_NOTES.md"):
        (root / name).write_text(f"# {name.removesuffix('.md').replace('_', ' ').title()}\n\nSee machine-readable outputs in this directory.\n", encoding="utf-8")
    (root / "run_summary.json").write_text(json.dumps({"configurations": len(summary), "failed_configurations": len(failed), "holdout_access": False}, indent=2), encoding="utf-8")
