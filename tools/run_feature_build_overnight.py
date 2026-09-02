from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    source = project / "src"
    sys.path.insert(0, str(source))
    workers = str(os.cpu_count() or 1)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from quant_pipeline.alpha_discovery.config import AlphaDiscoveryConfig
    from quant_pipeline.alpha_discovery.run import AlphaDiscoveryRun

    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else project / "configs" / "alpha_discovery_v2_full.yaml")
    config = AlphaDiscoveryConfig.from_yaml(config_path)
    run = AlphaDiscoveryRun(config)
    for stage in ("validate-config", "snapshot", "build-panel", "compile-registry", "build-features"):
        result = run.execute(stage)
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **result}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
