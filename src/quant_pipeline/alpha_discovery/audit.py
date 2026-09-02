from __future__ import annotations

from dataclasses import asdict
import importlib
from pathlib import Path

from .config import AlphaDiscoveryConfig
from .registry import FAMILIES, compile_registry


REQUIRED_MODULES = (
    "data.source", "data.snapshot", "data.universe", "data.calendar", "data.corporate_actions", "data.aggregation", "data.panel",
    "targets.registry", "targets.builder", "targets.excursions", "cache.feature_store", "cache.target_store", "cache.rank_store", "cache.bin_store",
    "scan.singles", "scan.dual_pairs", "scan.dual_coarse", "scan.dual_fine", "scan.dual_exact", "scan.inference", "scan.fdr", "scan.stability",
    "interactions.context_expand", "interactions.formula_factory", "autopsy.run", "ml.models", "ml.neural", "alpha.registry", "alpha.portfolio",
    "execution.costs", "execution.bar_replay", "execution.quote_replay", "report.build", "report.plots", "governance.state", "governance.freeze",
)


def code_readiness(config: AlphaDiscoveryConfig) -> dict:
    missing = []
    for name in REQUIRED_MODULES:
        try: importlib.import_module(f"quant_pipeline.alpha_discovery.{name}")
        except Exception as error: missing.append({"module": name, "error": repr(error)})
    bundle = compile_registry(config)
    mapped_families = set(FAMILIES)
    unexplained = sorted({concept.builder_key for concept in bundle.concepts if concept.active} - mapped_families)
    return {"standalone_project": Path(config.project_root).name == "Quant Pipeline V2", "required_modules_pass": not missing,
            "missing_modules": missing, "compiled_concepts": len(bundle.concepts), "compiled_features": len(bundle.features),
            "compiled_targets": len(bundle.targets), "unexplained_builder_keys": unexplained,
            "registry_mapping_pass": not unexplained, "data_ready": False, "smoke_run_complete": False,
            "full_run_complete": False}
