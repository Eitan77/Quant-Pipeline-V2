from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from quant_pipeline.alpha_discovery.config import AlphaDiscoveryConfig
    from quant_pipeline.alpha_discovery.run import AlphaDiscoveryRun
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else project / "configs" / "alpha_discovery_v2_full.yaml")
    run = AlphaDiscoveryRun(AlphaDiscoveryConfig.from_yaml(config_path))
    stages = ("validate-config", "snapshot", "build-panel", "compile-registry", "build-features",
              "build-targets", "scan-singles", "scan-duals-coarse", "scan-duals-fine", "exact-duals",
              "build-stability", "audit-exhaustiveness")
    for stage in stages:
        result = run.execute(stage)
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **result}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
