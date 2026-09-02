from __future__ import annotations
import argparse
from .config import ScanConfig
from .run import execute

def main() -> None:
    parser=argparse.ArgumentParser(description="Clean-room Phase 1 univariate anomaly scanner")
    parser.add_argument("config",help="YAML scan configuration")
    args=parser.parse_args(); print(execute(ScanConfig.from_yaml(args.config)))

if __name__ == "__main__": main()
