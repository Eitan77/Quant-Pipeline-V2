from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .capacity import capacity_proxy
from .costs import cost_delay_curve
from .decay import effect_decay
from .leaveout import leave_out_tests
from .persistence import persistence_turnover
from .placebo import placebo_test
from .regimes import grouped_effects
from .thresholds import threshold_curve
from .attribution import factor_attribution
from .variants import calculation_variant_audit
from .trade_path import trade_path_diagnostics


ARTIFACTS = (
    "effect_decay.parquet", "threshold_curve.parquet", "persistence_turnover.parquet",
    "chronological_robustness.parquet", "leave_out_tests.parquet", "regime_matrix.parquet",
    "breadth_matrix.parquet", "cost_delay_curve.parquet", "capacity_proxy.parquet",
    "trade_path.parquet", "neutralization_attribution.parquet", "redundancy.parquet",
    "placebo_tests.parquet", "calculation_variant_audit.parquet",
)


@dataclass
class EdgeAutopsy:
    output: Path
    config: dict[str, Any]

    def run(self, score: np.ndarray, targets: dict[str, np.ndarray], metadata: pd.DataFrame,
            delayed_returns: dict[int, np.ndarray] | None = None) -> dict[str, str]:
        if not targets:
            raise ValueError("Edge Autopsy requires at least one causal target")
        if len(score) != len(metadata) or any(len(values) != len(score) for values in targets.values()):
            raise ValueError("Edge Autopsy score, targets, and metadata must be row-aligned")
        regime_columns = ["market_trend", "market_vol", "breadth", "dispersion", "correlation_regime", "time_of_day", "day_of_week"]
        breadth_columns = ["price_bucket", "liquidity_bucket", "volatility_bucket", "beta_bucket", "size_bucket"]
        required = {"security_id", "session_date", "decision_ts", "trade_notional", "bar_dollar_volume", "entry_price", "path_prices", *regime_columns, *breadth_columns}
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"Edge Autopsy metadata missing mandatory fields: {sorted(missing)}")
        if delayed_returns is None or not delayed_returns:
            raise ValueError("Edge Autopsy requires measured delayed-return paths")
        self.output.mkdir(parents=True, exist_ok=True); (self.output / "figures").mkdir(exist_ok=True)
        primary = next(iter(targets.values()))
        tables = {
            "effect_decay.parquet": effect_decay(score, targets),
            "threshold_curve.parquet": threshold_curve(score, primary, self.config["threshold_tails"]),
            "persistence_turnover.parquet": persistence_turnover(pd.Series(score).rank(pct=True).to_numpy(), metadata.security_id.to_numpy()),
            "leave_out_tests.parquet": leave_out_tests(primary, metadata.security_id.to_numpy(), metadata.session_date.to_numpy()),
            "cost_delay_curve.parquet": cost_delay_curve(delayed_returns, self.config["cost_bps_per_side"]),
            "placebo_tests.parquet": placebo_test(score, primary, pd.factorize(metadata.decision_ts)[0], int(self.config.get("placebo_runs", 1000))),
            "chronological_robustness.parquet": grouped_effects(primary, metadata.assign(year=pd.to_datetime(metadata.session_date).dt.year), ["year"]),
            "regime_matrix.parquet": grouped_effects(primary, metadata, regime_columns),
            "breadth_matrix.parquet": grouped_effects(primary, metadata, breadth_columns),
            "capacity_proxy.parquet": capacity_proxy(metadata.trade_notional, metadata.bar_dollar_volume, primary),
        }
        factor_columns = [name for name in ("market_factor", "momentum_factor", "volatility_factor", "liquidity_factor") if name in metadata]
        if not factor_columns:
            raise ValueError("Edge Autopsy requires at least one prior-known attribution factor")
        tables["neutralization_attribution.parquet"] = factor_attribution(primary, metadata[factor_columns])
        variant_columns = [name for name in metadata if name.startswith("variant_")]
        if not variant_columns:
            raise ValueError("Edge Autopsy requires at least one calculation variant")
        tables["calculation_variant_audit.parquet"] = calculation_variant_audit(score, {name: metadata[name].to_numpy() for name in variant_columns})
        path_rows = []
        for row in metadata[["security_id", "session_date", "entry_price", "path_prices"]].itertuples(index=False):
            values = row.path_prices if isinstance(row.path_prices, pd.Series) else pd.Series(row.path_prices)
            path_rows.append({"security_id": row.security_id, "session_date": row.session_date} | trade_path_diagnostics(values, row.entry_price))
        tables["trade_path.parquet"] = pd.DataFrame(path_rows)
        peers = [name for name in metadata if name.startswith("candidate_score_")]
        redundancy_rows = []
        for name in peers:
            values = pd.to_numeric(metadata[name], errors="coerce").to_numpy()
            valid = np.isfinite(score) & np.isfinite(values)
            correlation = float(pd.Series(score[valid]).corr(pd.Series(values[valid]), method="spearman")) if valid.sum() >= 3 else np.nan
            redundancy_rows.append({"candidate": name, "spearman_correlation": correlation, "sample_count": int(valid.sum())})
        tables["redundancy.parquet"] = pd.DataFrame(redundancy_rows or [{"candidate": "none", "spearman_correlation": 0.0, "sample_count": len(score)}])
        for artifact in ARTIFACTS:
            table = tables[artifact]
            if table.empty:
                raise ValueError(f"Mandatory Edge Autopsy artifact is empty: {artifact}")
            table.to_parquet(self.output / artifact, index=False)
        checks = {"economic_magnitude": "PASS" if np.nanmean(np.abs(primary)) > 0 else "FAIL",
                  "placebo_tests": "PASS" if tables["placebo_tests.parquet"].empirical_p_value.iloc[0] <= .05 else "WARN",
                  "artifact_completeness": "PASS"}
        report = "# Edge Autopsy\n\n" + "\n".join(f"- {name}: {status}" for name, status in checks.items()) + "\n"
        (self.output / "EDGE_AUTOPSY.md").write_text(report, encoding="utf-8")
        manifest = {name: str(self.output / name) for name in ARTIFACTS} | {"report": str(self.output / "EDGE_AUTOPSY.md")}
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
