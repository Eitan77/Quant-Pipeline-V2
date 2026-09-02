from __future__ import annotations

import argparse
from .config import ScanConfig
from .phase1b_run import run_phase1b


def main() -> None:
    parser=argparse.ArgumentParser(description="Run Phase 1B dual-factor discovery from an immutable Phase 1A source")
    parser.add_argument("--source-run",required=True)
    parser.add_argument("--config",required=True)
    args=parser.parse_args(); config=ScanConfig.from_yaml(args.config)
    root=run_phase1b(args.source_run,config)
    print(f"Phase 1B broad screen complete: {root}")


if __name__ == "__main__":main()
