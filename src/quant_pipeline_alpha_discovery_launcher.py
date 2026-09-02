from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _activate_local_source() -> None:
    """Put this standalone checkout ahead of unrelated quant_pipeline installs."""
    source = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != source]
    sys.path.insert(0, source)


def cli_main() -> int:
    _activate_local_source()
    from quant_pipeline.alpha_discovery.cli import main as command_main
    return command_main()


def main() -> int:
    _activate_local_source()
    from quant_pipeline.alpha_discovery.config import AlphaDiscoveryConfig
    from quant_pipeline.alpha_discovery.run import AlphaDiscoveryRun
    parser = argparse.ArgumentParser(); parser.add_argument("config"); args = parser.parse_args()
    config = AlphaDiscoveryConfig.from_yaml(args.config); run = AlphaDiscoveryRun(config)
    stages: list[tuple[str, str | None]] = [
        ("validate-config", None), ("snapshot", None), ("build-panel", None), ("compile-registry", None),
        ("build-features", None), ("build-targets", None), ("scan-singles", None), ("scan-duals", "coarse"),
    ]
    if config.duals.get("exhaustive_5x5_all_pairs"): stages.append(("scan-duals", "fine"))
    stages += [("exact-duals", None), ("build-stability", None)]
    if config.context_expansion.get("enabled"): stages.append(("expand-context", None))
    if config.formula_factory.get("enabled"): stages.append(("run-formula-factory", None))
    if config.ml.get("enabled"): stages += [("run-ml", None), ("distill-ml", None)]
    stages += [("run-edge-autopsy", None), ("audit-exhaustiveness", None), ("freeze-discovery", None)]
    if config.research_periods.allow_replication_access:
        stages += [("evaluate-replication", None), ("freeze-replication", None), ("build-alphas", None),
                   ("evaluate-alphas", None), ("freeze-portfolio", None)]
    if config.research_periods.allow_final_holdout_access: stages.append(("evaluate-final-holdout", None))
    stages.append(("build-report", None))
    for stage, dual_stage in stages:
        result = run.execute(stage, dual_stage); print(f"{result['stage']}: {result['status']}")
    print(f"Run complete through the authorized research state at {Path(config.output_root) / config.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
