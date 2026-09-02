"""Collision-safe launcher for the standalone Quant Pipeline project.

The parent AlgoResearch environment also exposes an older package named
``quant_pipeline``. Put this project's source directory first before importing
the CLI so a corrected run cannot accidentally execute that stale package.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    source=str(Path(__file__).resolve().parent)
    sys.path[:]=[entry for entry in sys.path if str(Path(entry or ".").resolve())!=source]
    sys.path.insert(0,source)
    stale=[name for name in sys.modules if name=="quant_pipeline" or name.startswith("quant_pipeline.")]
    for name in stale:sys.modules.pop(name,None)
    from quant_pipeline.cli import main as pipeline_main
    pipeline_main()


if __name__=="__main__":main()
