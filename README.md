# Quant Pipeline V2

Quant Pipeline V2 is a standalone causal, CUDA-accelerated alpha-discovery
and validation system. It contains its own code, tests, configs, launchers,
registries, checkpoints, inference, governance, and reporting. It never imports
or reads executable code from the earlier Quant Pipeline project.

## Core contract

- Immutable, fingerprinted Alpaca SIP raw 1-minute snapshots.
- Separate raw execution prices and split-consistent research prices.
- Stable security IDs and point-in-time universe membership.
- Explicit bar start, end, availability, decision, entry, and exit times.
- Registry-driven features and targets with stable definition hashes.
- Exhaustive singles and unordered duals with no parent-quality prefilter.
- Tiled Torch CUDA scans with deterministic CPU parity.
- Purged chronological validation, multiple-testing control, and costs.
- Discovery, replication, portfolio, and final-holdout state gates.
- Mandatory Edge Autopsy and complete trial accounting before replication.

## Install and verify

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
quant-alpha validate-config configs/alpha_discovery_v1.yaml
quant-alpha compile-registry configs/alpha_discovery_v1.yaml
```

## Frozen data contract

The full run uses a project-owned SIP minute snapshot from `2019-06-21`
through `2026-04-30`. The first 258 sessions are warmup only; discovery starts
`2020-07-01`. The frozen snapshot includes raw OHLCV, VWAP, trade count, SPY,
QQQ, point-in-time S&P membership, stable security identifiers where Alpaca
provides them, and a split-consistent research-price view. Replication data from
`2026-05-01` onward is excluded by the catalog and remains sealed.

Data acquisition is separate from research execution. The canonical catalog is
`data/catalog.duckdb`, and `data/catalog.manifest.json` records its source range,
reference hashes, row coverage, and identity-quality counts. Run
`tools/audit_v2_data.py` before any smoke or full run; any unexpected missing
symbol-session, duplicate natural key, null volume/VWAP/trade-count field, or
sealed-period row is a hard failure.

## One-shot resumable run

```powershell
python -m quant_pipeline_alpha_discovery_launcher configs/alpha_discovery_v1.yaml
```

The launcher executes every enabled stage in order, resumes completed blocks,
and stops at sealed replication or final-holdout boundaries unless the matching
access flag is explicitly enabled. Stage commands use the same checkpoint and
configuration-fingerprint contract.

Generated runs, result artifacts, local databases, and caches are ignored.
