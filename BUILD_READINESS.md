# Quant Pipeline V2 readiness

Status: **CLEARED FOR THE FULL DISCOVERY BUILD; NOT STARTED**

## Verified on 2026-09-02

- Standalone V2 project; no runtime dependency on the earlier pipeline.
- Immutable Alpaca SIP/raw catalog: 348,156,379 validated 1m rows, 634 securities, 1,724 sessions.
- PIT membership, stable security IDs, split-consistent research prices, corporate-action ledger, and sealed-period SQL gates are present.
- Full registry: 280 concepts, 6,124 active feature specifications, 63 target specifications, and 572 required warmup sessions.
- The 2019-06-21 warmup supports a full-coverage discovery start of 2021-10-01. Replication begins 2026-05-01 and remains disabled; final holdout begins 2026-09-01 and remains disabled.
- Twelve PCA/statistical-peer concepts are explicitly `UNAVAILABLE`; no beta or peer proxy is silently substituted.
- Large-data panels and targets use bounded DuckDB execution. Feature construction uses 16 one-security worker tasks, 128-column memory-mapped blocks, and direct-formula caching. Dual scans use tiled Torch CUDA and delete transient bin caches after each completed scan.
- Checkpoints bind both configuration and implementation hashes; changed code cannot resume stale artifacts.
- Corporate-action-crossing targets are flagged and set missing rather than turning raw split/dividend discontinuities into signals.
- Full production sizing: 44,666,022 intraday-5m observations, 580,524 daily-close observations, 573,183 preclose observations; raw float32 feature cache estimate 258.2 GiB; D: free space measured at 485.9 GiB.
- Full regression suite: 129 passed.
- Final representative end-to-end smoke: 209,476 source rows, 8 securities, 43 source sessions/20 discovery sessions, 16,838 observations, 120 features, 6 targets, 320/320 singles, and 9,410/9,410 eligible pair-targets at 3-bin, 5-bin, and exact 10-bin scans. Exhaustiveness: PASS. Scanner backend: `torch:cuda:0`.

## Launch

Run only after explicit approval:

```powershell
python tools/run_feature_and_scans.py configs/alpha_discovery_v2_full.yaml
```

The output namespace is `D:/AlgoResearch/Quant Pipeline V2/runs/alpha_discovery_v2_full`. It is currently absent and therefore clean. The stopped 1.6 KB prior attempt was moved to `_aborted_alpha_discovery_v2_full_20260430_warmup2019_20260902` and remains recoverable.
