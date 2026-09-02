"""Collision-safe installed launcher for the standalone Phase 1B runner."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    source=str(Path(__file__).resolve().parent)
    sys.path[:]=[entry for entry in sys.path if str(Path(entry or ".").resolve())!=source]
    sys.path.insert(0,source)
    for name in [name for name in sys.modules if name=="quant_pipeline" or name.startswith("quant_pipeline.")]:
        sys.modules.pop(name,None)
    from quant_pipeline.phase1b_launcher import main as phase1b_main
    phase1b_main()


if __name__=="__main__":main()
