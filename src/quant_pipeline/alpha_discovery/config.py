from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .models import stable_hash


@dataclass(frozen=True)
class SourceConfig:
    duckdb_path: str = "D:/AlgoResearch/data/catalog.duckdb"
    bars_1m_raw_table: str = "bars_1m_raw"
    bars_1m_research_table: str = "bars_1m_research"
    alpaca_feed_required: str = "sip"
    alpaca_execution_adjustment_required: str = "raw"
    research_price_basis: str = "split_consistent"
    security_master_table: str = "security_master"
    membership_table: str = "sp500_pit_membership_daily"
    corporate_actions_path: str = "reference/corporate_actions.parquet"
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ")


@dataclass(frozen=True)
class ResearchPeriodsConfig:
    discovery_start: str = "2019-01-02"
    discovery_end: str = "2026-04-30"
    replication_start: str = "2026-05-01"
    replication_end: str = "2026-08-31"
    final_holdout_start: str = "2026-09-01"
    allow_replication_access: bool = False
    allow_final_holdout_access: bool = False


@dataclass(frozen=True)
class ComputeConfig:
    prefer_cuda: bool = True
    gpu_device: str = "cuda:0"
    dynamic_memory_fraction: float = 0.80
    cpu_fallback: bool = True
    deterministic: bool = True
    feature_block_size: int | str = "auto"
    target_block_size: int | str = "auto"
    cpu_workers: int | str = "auto"
    host_memory_fraction: float = 0.90
    duckdb_memory_limit: str = "28GB"
    duckdb_temp_directory: str = "D:/AlgoResearch/Quant Pipeline V2/temp"


@dataclass(frozen=True)
class AlphaDiscoveryConfig:
    run_name: str = "alpha_discovery_v1"
    project_root: str = "."
    output_root: str = "runs"
    source: SourceConfig = field(default_factory=SourceConfig)
    research_periods: ResearchPeriodsConfig = field(default_factory=ResearchPeriodsConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    decision_grids: dict[str, bool] = field(default_factory=lambda: {
        "intraday_5m": True, "daily_close": True, "preclose_1555": True, "intraday_1m": False,
    })
    feature_windows: dict[str, list[str]] = field(default_factory=lambda: {
        "intraday": ["1m", "2m", "5m", "10m", "15m", "30m", "60m", "120m", "240m", "session"],
        "daily": ["1d", "2d", "3d", "5d", "10d", "20d", "30d", "40d", "63d", "126d", "252d"],
    })
    targets: dict[str, Any] = field(default_factory=lambda: {
        "intraday": ["1m", "2m", "5m", "10m", "15m", "30m", "60m", "120m", "240m", "EOD"],
        "interday": ["1d", "2d", "3d", "5d", "10d", "20d", "30d", "40d", "63d", "126d"],
        "overnight": True, "bases": ["raw", "benchmark_adjusted", "beta_residual"],
        "intraday_entry_delay_minutes": 1, "daily_entry_delay_minutes": 1,
    })
    universe: dict[str, Any] = field(default_factory=lambda: {
        "minimum_price": 3.0, "minimum_prior_20d_median_dollar_volume": 10_000_000,
        "require_stable_security_id": True, "require_point_in_time_membership": True,
    })
    duals: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "parent_preselection": False, "coarse_bins": 3,
        "extreme_quantiles": [0.10, 0.20], "exhaustive_5x5_all_pairs": True, "exact_bins": 10,
    })
    stability: dict[str, Any] = field(default_factory=lambda: {
        "chronological_folds": 5, "require_neighbor_analysis": True,
        "require_plateau_detection": True, "minimum_symbol_breadth": 0.50,
    })
    warmup: dict[str, Any] = field(default_factory=lambda: {
        "auto_derive_transitive_history": True, "safety_margin_sessions": 5,
        "fail_if_full_coverage_warmup_missing": True,
    })
    context_expansion: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "only_from_stable_base_structures": True})
    formula_factory: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "max_expression_depth": 3, "max_binary_operators": 2})
    ml: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "purged_chronological_folds": True, "symbol_embeddings": False})
    edge_autopsy: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "mandatory_before_replication": True,
        "threshold_tails": [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01],
        "cost_bps_per_side": [0, 1, 2, 3, 5, 7.5, 10, 15, 20],
        "entry_delay_minutes": [1, 2, 5, 10, 15, 30], "placebo_runs": 1000,
    })
    governance: dict[str, Any] = field(default_factory=lambda: {
        "enforce_state_machine": True, "require_candidate_freeze_manifest": True,
        "require_portfolio_freeze_manifest": True, "require_exhaustiveness_pass_for_freeze": True,
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlphaDiscoveryConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown alpha-discovery config keys: {sorted(unknown)}")
        for name, kind in (("source", SourceConfig), ("research_periods", ResearchPeriodsConfig), ("compute", ComputeConfig)):
            if isinstance(raw.get(name), dict):
                nested_unknown = set(raw[name]) - set(kind.__dataclass_fields__)
                if nested_unknown:
                    raise ValueError(f"Unknown {name} keys: {sorted(nested_unknown)}")
                if name == "source" and isinstance(raw[name].get("benchmark_symbols"), list):
                    raw[name]["benchmark_symbols"] = tuple(raw[name]["benchmark_symbols"])
                raw[name] = kind(**raw[name])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        periods = self.research_periods
        d0, d1 = pd.Timestamp(periods.discovery_start), pd.Timestamp(periods.discovery_end)
        r0, r1, h0 = pd.Timestamp(periods.replication_start), pd.Timestamp(periods.replication_end), pd.Timestamp(periods.final_holdout_start)
        if not d0 <= d1 < r0 <= r1 < h0:
            raise ValueError("Research periods must be strictly chronological and non-overlapping")
        if not 0 < self.compute.dynamic_memory_fraction <= 0.95:
            raise ValueError("dynamic_memory_fraction must be in (0, 0.95]")
        if not 0 < self.compute.host_memory_fraction <= 0.95:
            raise ValueError("host_memory_fraction must be in (0, 0.95]")
        if self.compute.cpu_workers != "auto" and int(self.compute.cpu_workers) <= 0:
            raise ValueError("cpu_workers must be positive or 'auto'")
        if self.duals.get("parent_preselection"):
            raise ValueError("V2 exhaustive dual scans forbid parent preselection")
        root = Path(self.project_root).resolve()
        old = Path("D:/AlgoResearch/Quant Pipeline").resolve()
        if root == old:
            raise ValueError("V2 project_root may not be the legacy Quant Pipeline")

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path(self.project_root).resolve() / path).resolve()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def definition_hash(self) -> str:
        return stable_hash(self.as_dict())
