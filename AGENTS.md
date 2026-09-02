# Quant Pipeline V2 repository contract

- This repository is standalone. Never import code, configs, run artifacts, or results from `D:\AlgoResearch\Quant Pipeline`.
- External market data may be read only through explicit configured paths and immutable fingerprints.
- Research code must enforce time boundaries in source queries/loaders before rows are materialized.
- Discovery, replication, and final holdout are separate physical access states. Never use a later partition to debug or select an earlier-stage candidate.
- Raw executable prices and split-consistent research prices are separate concepts.
- Preserve complete trial and exclusion ledgers. Never describe a partial scan as exhaustive.
- Prefer vectorized CPU operations and tiled Torch CUDA reductions; retain deterministic CPU parity tests.
- Generated data, caches, runs, reports, and local secrets are not committed.
