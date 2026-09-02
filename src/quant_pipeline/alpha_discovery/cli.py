from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AlphaDiscoveryConfig
from .run import AlphaDiscoveryRun


STAGES = (
    "validate-config", "snapshot", "build-panel", "compile-registry", "build-features", "build-targets",
    "scan-singles", "scan-duals", "exact-duals", "build-stability", "expand-context",
    "run-formula-factory", "run-ml", "distill-ml", "run-edge-autopsy", "audit-exhaustiveness",
    "freeze-discovery", "evaluate-replication", "freeze-replication", "build-alphas", "evaluate-alphas",
    "freeze-portfolio", "evaluate-final-holdout", "build-report",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="quant-alpha")
    result.add_argument("stage", choices=STAGES); result.add_argument("config"); result.add_argument("--stage", dest="dual_stage", choices=("coarse", "fine"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); config = AlphaDiscoveryConfig.from_yaml(args.config); run = AlphaDiscoveryRun(config)
    result = run.execute(args.stage, args.dual_stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
