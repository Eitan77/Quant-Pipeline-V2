# Quant Pipeline design

## Clean-room boundary

The pipeline reads canonical completed bars only. It does not read prior scans,
research matrices, strategies, trades, or backtests. A configured sealed
holdout is rejected unless access is explicitly enabled.

## Point-in-time contract

One observation represents `symbol + session_date + decision_ts`. Features may
use only information available at or before `decision_ts`. Targets enter on the
next actionable bar, remain inside the regular session, and never cross session
boundaries.

## Statistical contract

- CUDA performs dense pair moments, rank approximations, quantile aggregation,
  and batched one-way session-clustered sandwich inference.
- Global Benjamini-Hochberg FDR is the candidate-promotion gate. Family-level
  FDR remains diagnostic.
- CPU and GPU exact diagnostics verify promoted candidates and produce year,
  symbol, time, outlier, and cross-sectional IC stability measures.
- Redundant lookbacks and adjacent horizons remain traceable through explicit
  redundancy groups.
- Signed features are never divided by rolling means that can cross zero.

## Output contract

Each local run records its configuration, validation results, registries,
coverage ledger, master results, exact candidates, progress, and reports.
Generated artifacts and all underlying market data are excluded from version
control.
