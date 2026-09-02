from __future__ import annotations

from pathlib import Path
from typing import Any


REPORTS = (
    "RUN_SUMMARY.md", "DATA_VALIDATION.md", "FEATURE_REGISTRY_SUMMARY.md", "TARGET_REGISTRY_SUMMARY.md",
    "SINGLE_DISCOVERY_REPORT.md", "SINGLE_STABILITY_REPORT.md", "DUAL_DISCOVERY_REPORT.md",
    "DUAL_STABILITY_REPORT.md", "CONTEXT_DISCOVERY_REPORT.md", "FORMULA_FACTORY_REPORT.md",
    "ML_DISCOVERY_REPORT.md", "EDGE_AUTOPSY_SUMMARY.md", "EXHAUSTIVENESS_AUDIT.md", "REPLICATION_REPORT.md",
    "ALPHA_LIBRARY_REPORT.md", "EXECUTION_STRESS_REPORT.md", "FINAL_HOLDOUT_REPORT.md", "HOLDOUT_ACCESS_AUDIT.md",
)


def build_reports(root: str | Path, context: dict[str, Any]) -> list[Path]:
    output = Path(root); output.mkdir(parents=True, exist_ok=True); paths = []
    for name in REPORTS:
        title = name.removesuffix(".md").replace("_", " ").title()
        body = [f"# {title}", ""] + [f"- {key}: {value}" for key, value in sorted(context.items())] + [""]
        path = output / name; path.write_text("\n".join(body), encoding="utf-8"); paths.append(path)
    return paths
