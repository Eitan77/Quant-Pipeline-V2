from __future__ import annotations

import argparse

from .phase2.config import Phase2Config
from .phase2.runner import execute_initial_batch, preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 holdout-safe strategy research")
    parser.add_argument("config", help="Phase 2 YAML configuration")
    parser.add_argument("--run-initial-batch", action="store_true",
                        help="Execute the configured bar-fill initial batch after preflight")
    args = parser.parse_args()
    config = Phase2Config.from_yaml(args.config)
    print(execute_initial_batch(config) if args.run_initial_batch else preflight(config))


if __name__ == "__main__":
    main()
