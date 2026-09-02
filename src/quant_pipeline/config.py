from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SystematicParentSelectionConfig:
    max_parent_features: int = 120
    max_per_feature_family: int = 12
    max_per_redundancy_group: int = 2
    primary_global_fdr_max: float = 0.20
    minimum_absolute_effect_bps: float = 0.50
    minimum_valid_observations: int = 100_000
    minimum_sessions: int = 750
    minimum_symbols: int = 75
    minimum_decision_timestamps: int = 250
    include_top_per_family_when_threshold_not_met: int = 3
    include_top_per_target_family_when_threshold_not_met: int = 5


@dataclass(frozen=True)
class SystematicPairGenerationConfig:
    max_parent_pairs: int = 5_000
    forbid_identical_parent: bool = True
    forbid_same_redundancy_group: bool = True
    maximum_absolute_parent_spearman: float = 0.90
    maximum_pairs_per_family_pair: int = 500
    minimum_joint_observations: int = 250_000
    minimum_joint_sessions: int = 750
    minimum_joint_symbols: int = 75
    minimum_joint_decision_timestamps: int = 250


@dataclass(frozen=True)
class SystematicOperatorConfig:
    aligned_rank_mean: bool = True
    directional_intersection: bool = True
    gated_anchor: bool = True
    persistence_intersection: bool = False


@dataclass(frozen=True)
class SystematicBinaryCoverageConfig:
    minimum_on_observations: int = 10_000
    minimum_off_observations: int = 10_000
    minimum_on_sessions: int = 250
    minimum_off_sessions: int = 250
    minimum_on_symbols: int = 50
    minimum_off_symbols: int = 50
    minimum_activation_rate: float = 0.005
    maximum_activation_rate: float = 0.20


@dataclass(frozen=True)
class SystematicLimitsConfig:
    max_generated_features: int = 15_000
    feature_chunk_size: int = 32


@dataclass(frozen=True)
class SystematicScreeningConfig:
    primary_targets_only: bool = True
    exploratory_family_fdr_max: float = 0.05
    exact_candidate_limit: int = 250


@dataclass(frozen=True)
class SystematicPromotionConfig:
    minimum_absolute_effect_bps: float = 1.0
    minimum_effect_ratio_vs_best_parent: float = 1.25
    minimum_effect_increment_bps_vs_best_parent: float = 0.25
    minimum_positive_historical_fold_fraction: float = 0.60
    minimum_recent_to_full_effect_ratio: float = 0.50
    maximum_recent_to_full_effect_ratio: float = 2.50
    minimum_expected_direction_symbol_fraction: float = 0.55
    maximum_symbol_effect_hhi: float = 0.02
    maximum_top5_symbol_effect_share: float = 0.25
    minimum_remove_top5_effect_retention: float = 0.50


@dataclass(frozen=True)
class SystematicPhase1BConfig:
    enabled: bool = True
    parent_selection: SystematicParentSelectionConfig = field(default_factory=SystematicParentSelectionConfig)
    pair_generation: SystematicPairGenerationConfig = field(default_factory=SystematicPairGenerationConfig)
    operators: SystematicOperatorConfig = field(default_factory=SystematicOperatorConfig)
    binary_state_coverage: SystematicBinaryCoverageConfig = field(default_factory=SystematicBinaryCoverageConfig)
    limits: SystematicLimitsConfig = field(default_factory=SystematicLimitsConfig)
    screening: SystematicScreeningConfig = field(default_factory=SystematicScreeningConfig)
    promotion: SystematicPromotionConfig = field(default_factory=SystematicPromotionConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "SystematicPhase1BConfig":
        values = dict(values or {})
        mapping = {
            "parent_selection": SystematicParentSelectionConfig,
            "pair_generation": SystematicPairGenerationConfig,
            "operators": SystematicOperatorConfig,
            "binary_state_coverage": SystematicBinaryCoverageConfig,
            "limits": SystematicLimitsConfig,
            "screening": SystematicScreeningConfig,
            "promotion": SystematicPromotionConfig,
        }
        unknown = set(values) - ({"enabled"} | set(mapping))
        if unknown:
            raise ValueError(f"Unknown systematic_phase1b keys: {sorted(unknown)}")
        for key, kind in mapping.items():
            raw = values.get(key, {})
            if isinstance(raw, kind):
                continue
            allowed = set(kind.__dataclass_fields__)
            nested_unknown = set(raw or {}) - allowed
            if nested_unknown:
                raise ValueError(f"Unknown systematic_phase1b.{key} keys: {sorted(nested_unknown)}")
            values[key] = kind(**(raw or {}))
        return cls(**values)


@dataclass(frozen=True)
class ScanConfig:
    catalog_path: str = "D:/AlgoResearch/data/catalog.duckdb"
    source_table: str = "derived_bars_5m"
    feed: str = "sip"
    adjustment: str = "raw"
    start: str = "2019-06-21"
    selection_end: str | None = None
    confirmation_start: str | None = None
    discovery_end: str = "2026-04-30"  # sealed holdout begins 2026-05-01
    sealed_holdout_start: str = "2026-05-01"
    allow_holdout_access: bool = False
    use_separate_confirmation_period: bool = False
    run_historical_walk_forward_diagnostics: bool = True
    run_recent_period_diagnostics: bool = True
    run_recency_weighted_diagnostics: bool = True
    recency_half_lives_months: list[int] = field(default_factory=lambda: [6, 12, 24])
    universe: list[str] = field(default_factory=list)
    membership_table: str = "qqq_pit_membership_daily"
    membership_source_quality: str = "effective_date_reconstruction"
    require_membership: bool = True
    benchmark_symbols: list[str] = field(default_factory=lambda: ["QQQ"])
    corporate_actions_path: str = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/reference/corporate_actions_through_20260430.parquet"
    require_corporate_actions: bool = True
    exchange_calendar: str = "XNYS"
    maximum_missing_bars_per_session: int = 0
    decision_times_et: list[str] = field(default_factory=list)
    lookbacks: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 48, 60, 78])
    target_horizons_minutes: list[int] = field(default_factory=lambda: list(range(5, 390, 5)))
    primary_target_horizons_minutes: list[int] = field(default_factory=lambda: [5, 15, 30, 60, 120])
    quantiles: int = 10
    min_observations: int = 500
    min_sessions: int = 100
    min_symbols: int = 20
    min_decision_timestamps: int = 100
    min_years: int = 3
    min_bin_observations: int = 30
    outlier_policy: str = "none"  # none|winsorize_1pct
    output_root: str = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/runs"
    experiment_id: str = "phase1_final_discovery_through_20260430"
    benchmark_symbol: str = "QQQ"
    sector_map_path: str | None = None
    use_cuda: bool = True
    cuda_device: str = "cuda:0"
    cuda_target_batch_group_size: int = 3
    scan_batch_rows: int = 250_000
    feature_chunk_size: int = 24
    feature_build_batch_chunks: int = 2
    feature_workers: int = 16
    exact_workers: int = 6
    target_chunk_size: int = 4
    target_build_batch_chunks: int = 2
    cache_validation_workers: int = 4
    resume: bool = True
    checkpoint_every_pairs: int = 25
    normalization_windows_sessions: list[int] = field(default_factory=lambda: [20, 60])
    opening_windows_minutes: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15, 20, 30, 45, 60, 90])
    cross_sectional_min_symbols: int = 5
    multiple_testing_grouping: str = "feature_family_target_family"
    sensitivity_outlier_policy: str = "winsorize_1pct"
    exact_bootstrap_samples: int = 500
    primary_fdr_threshold: float = 0.05
    minimum_effect_bps: float = 1.0
    confirmation_min_sessions: int = 100
    confirmation_min_symbols: int = 20
    confirmation_min_effect_bps: float = 1.0
    confirmation_min_discovery_ratio: float = 0.25
    confirmation_max_discovery_ratio: float = 4.0
    confirmation_alpha: float = 0.10
    confirmation_min_positive_fold_fraction: float = 0.60
    beta_window_sessions: int = 60
    beta_min_observations: int = 40
    max_candidates_per_feature_family: int = 20
    max_candidates_per_cluster: int = 3
    max_candidates_per_target_family: int = 30
    regime_min_observations: int = 500
    regime_min_sessions: int = 100
    regime_min_symbols: int = 20
    exact_time_min_observations: int = 500
    exact_time_min_sessions: int = 100
    exact_time_min_symbols: int = 20
    scope_min_observations: int = 500
    scope_min_sessions: int = 100
    scope_min_symbols: int = 3
    trend_threshold_bps: float = 20.0
    gap_threshold_bps: float = 20.0
    market_wide_min_direction_pct: float = 0.60
    market_wide_max_group_share: float = 0.40
    specific_scope_min_group_share: float = 0.60
    cache_schema_version: str = "phase1_final"
    # Phase 1B dual-factor discovery.  Disabled by default so Phase 1A runs
    # retain their current registry, cache, and scan behaviour.
    dual_factor_enabled: bool = False
    phase1b_mode: str = "curated_only"
    systematic_phase1b: SystematicPhase1BConfig = field(default_factory=SystematicPhase1BConfig)
    dual_factor_manifest_path: str | None = None
    dual_factor_feature_chunk_size: int = 32
    dual_factor_max_generated_features: int = 500
    dual_factor_emit_parent_conditions: bool = True
    dual_factor_cache_subdir: str = "phase1b_dual_features"
    dual_factor_min_signal_observations: int = 100
    dual_factor_min_signal_sessions: int = 50
    dual_factor_min_signal_symbols: int = 10
    dual_factor_min_activation_rate: float = 0.0005
    dual_factor_max_activation_rate: float = 0.9995
    dual_factor_allowed_operators: list[str] = field(default_factory=lambda: ["intersection", "gated_anchor", "aligned_rank_mean", "persistence_intersection"])
    binary_semantics_validation: str = "error"
    scan_schema_version: str = "phase1ab_v2"
    dual_cache_schema_version: str = "phase1b_v2"
    binary_min_on_observations: int = 100
    binary_min_off_observations: int = 100
    binary_min_on_sessions: int = 50
    binary_min_off_sessions: int = 50
    binary_min_on_symbols: int = 10
    binary_min_off_symbols: int = 10
    binary_primary_screen_inference: str = "two_way_date_symbol"
    bar_interval_minutes: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScanConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if isinstance(values.get("systematic_phase1b"), dict):
            values["systematic_phase1b"] = SystematicPhase1BConfig.from_dict(values["systematic_phase1b"])
        config=cls(**values); config.validate(); return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        import pandas as pd
        if pd.Timestamp(self.discovery_end)>=pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("discovery_end must precede sealed_holdout_start")
        if self.allow_holdout_access:raise ValueError("Phase 1 configuration may not allow holdout access")
        if self.use_separate_confirmation_period and (not self.selection_end or not self.confirmation_start):
            raise ValueError("Separate confirmation mode requires selection_end and confirmation_start")
        if any(value<=0 for value in self.recency_half_lives_months):raise ValueError("Recency half-lives must be positive")
        if self.feature_build_batch_chunks<=0:raise ValueError("feature_build_batch_chunks must be positive")
        if self.target_build_batch_chunks<=0:raise ValueError("target_build_batch_chunks must be positive")
        if self.cuda_target_batch_group_size<=0:raise ValueError("cuda_target_batch_group_size must be positive")
        if self.cache_validation_workers<=0:raise ValueError("cache_validation_workers must be positive")
        if self.dual_factor_feature_chunk_size<=0:raise ValueError("dual_factor_feature_chunk_size must be positive")
        if self.dual_factor_max_generated_features<=0:raise ValueError("dual_factor_max_generated_features must be positive")
        if not 0 <= self.dual_factor_min_activation_rate < self.dual_factor_max_activation_rate <= 1:
            raise ValueError("dual-factor activation-rate bounds must satisfy 0 <= min < max <= 1")
        if self.binary_semantics_validation not in {"error", "warn", "off"}:
            raise ValueError("binary_semantics_validation must be error, warn, or off")
        if self.binary_primary_screen_inference != "two_way_date_symbol":
            raise ValueError("binary_primary_screen_inference must be two_way_date_symbol")
        if self.bar_interval_minutes<=0:raise ValueError("bar_interval_minutes must be positive")
        if self.phase1b_mode not in {"curated_only", "systematic_only", "curated_plus_systematic"}:
            raise ValueError("phase1b_mode must be curated_only, systematic_only, or curated_plus_systematic")
        systematic=self.systematic_phase1b
        if systematic.limits.max_generated_features<=0 or systematic.limits.feature_chunk_size<=0:
            raise ValueError("systematic Phase 1B limits must be positive")
        if not 0 <= systematic.binary_state_coverage.minimum_activation_rate < systematic.binary_state_coverage.maximum_activation_rate <= 1:
            raise ValueError("systematic binary activation bounds must satisfy 0 <= min < max <= 1")
        if systematic.pair_generation.maximum_absolute_parent_spearman < 0 or systematic.pair_generation.maximum_absolute_parent_spearman > 1:
            raise ValueError("systematic maximum parent Spearman must be in [0, 1]")
