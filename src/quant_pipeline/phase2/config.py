from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml


Classification = Literal["phase1_supported", "phase1_conditional", "new_diversification_hypothesis"]


@dataclass(frozen=True)
class StrategyConfig:
    family: str
    classification: Classification
    cluster: str
    signal: str
    decision_times_et: tuple[str, ...]
    lookbacks_minutes: tuple[int, ...]
    tails: tuple[float, ...]
    holding_periods_minutes: tuple[int, ...]
    weighting_methods: tuple[str, ...]
    portfolio_forms: tuple[str, ...]

    def validate(self) -> None:
        if self.classification not in {"phase1_supported", "phase1_conditional", "new_diversification_hypothesis"}:
            raise ValueError(f"Unknown Phase 2 classification: {self.classification}")
        if not self.decision_times_et or not self.holding_periods_minutes:
            raise ValueError(f"{self.family} must declare decision times and holding periods")
        if any(not 0 < tail < 0.5 for tail in self.tails):
            raise ValueError(f"{self.family} tails must be in (0, 0.5)")
        if set(self.weighting_methods) - {"equal", "rank", "inverse_volatility"}:
            raise ValueError(f"{self.family} has unsupported weighting")
        if set(self.portfolio_forms) - {"long_only", "dollar_neutral", "beta_neutral"}:
            raise ValueError(f"{self.family} has unsupported portfolio form")

    def deterministic_id(
        self, decision_time: str, lookback: int, tail: float, holding_period: int,
        weighting: str, portfolio_form: str, execution: str, slippage_bps: float,
        regime: str | None = None,
    ) -> str:
        tail_label = f"top{int(tail * 100):02d}_bottom{int(tail * 100):02d}"
        base = "__".join((self.family, decision_time.replace(":", ""), f"lb{lookback}", tail_label,
                          f"hold{holding_period}", weighting, portfolio_form, execution,
                          f"slip{slippage_bps:g}bps"))
        return f"{base}__{regime}" if regime else base


@dataclass(frozen=True)
class Phase2Config:
    experiment_id: str
    discovery_start: str
    discovery_end: str
    sealed_holdout_start: str
    allow_holdout_access: bool
    phase1_source_run: str
    phase1b_source_run: str
    output_root: str = "C:/Users/decla/Desktop/AlgoResearch/Quant Pipeline V2/results"
    run_initial_batch_only: bool = True
    execution_models: tuple[str, ...] = ("next_bar_open", "quote_base")
    adverse_slippage_bps: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)
    weighting_methods: tuple[str, ...] = ("equal", "inverse_volatility")
    portfolio_forms: tuple[str, ...] = ("long_only", "beta_neutral")
    volatility_targets: tuple[float | None, ...] = (None, 0.06, 0.08, 0.10, 0.12)
    gross_leverage_caps: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    default_target_volatility: float = 0.08
    default_gross_leverage_cap: float = 2.0
    strategies: tuple[StrategyConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase2Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = dict(raw)
        strategies = tuple(StrategyConfig(**item) for item in raw.pop("strategies", []))
        for key in ("execution_models", "adverse_slippage_bps", "weighting_methods", "portfolio_forms", "volatility_targets", "gross_leverage_caps"):
            if key in raw:
                raw[key] = tuple(raw[key])
        config = cls(strategies=strategies, **raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.allow_holdout_access:
            raise ValueError("Phase 2 may not enable sealed holdout access")
        if pd.Timestamp(self.discovery_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("Phase 2 discovery_end must precede sealed_holdout_start")
        if set(self.execution_models) - {"next_bar_open", "quote_base"}:
            raise ValueError("Unsupported Phase 2 execution model")
        if not {0.0, 1.0, 3.0, 5.0}.issubset(set(self.adverse_slippage_bps)):
            raise ValueError("Phase 2 must include 0, 1, 3, and 5 bps adverse cost stress")
        if any(cap <= 0 for cap in self.gross_leverage_caps):
            raise ValueError("Gross leverage caps must be positive")
        if not self.strategies:
            raise ValueError("Phase 2 requires at least one strategy family")
        for strategy in self.strategies:
            strategy.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
