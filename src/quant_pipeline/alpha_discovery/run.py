from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import AlphaDiscoveryConfig
from .eligibility import EligibilityMatrix
from .governance.exhaustiveness import ExhaustivenessAudit
from .models import ResearchState
from .registry import RegistryBundle, compile_registry
from .report.build import build_reports


STAGE_ORDER = (
    "validate-config", "snapshot", "build-panel", "compile-registry", "build-features", "build-targets",
    "scan-singles", "scan-duals-coarse", "scan-duals-fine", "exact-duals", "build-stability",
    "expand-context", "run-formula-factory", "run-ml", "distill-ml", "run-edge-autopsy",
    "audit-exhaustiveness", "freeze-discovery", "evaluate-replication", "freeze-replication",
    "build-alphas", "evaluate-alphas", "freeze-portfolio", "evaluate-final-holdout", "build-report",
)


def _alpha_feature_is_global(spec) -> bool:
    return spec.family in {"ranks", "statistical_factors"} or "cross_sectional" in spec.concept_id or "rank" in spec.concept_id or spec.concept_id.startswith("market_breadth") or spec.concept_id.startswith("breadth_") or spec.concept_id in {"average_pairwise_correlation", "beta_dispersion"}


def _build_alpha_symbol_part(panel_path: str, specs: list, security_ids: list[str], base_path: str) -> tuple[str, str]:
    from .features.base import FeatureBuilder
    frame = pd.read_parquet(panel_path, filters=[("security_id", "in", security_ids)])
    for column in ("security_id", "symbol", "decision_grid", "price_basis"):
        if column in frame:
            frame[column] = frame[column].astype("category")
    for column in ("open", "high", "low", "close", "vwap", "research_open", "research_high", "research_low", "research_close", "split_factor"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], downcast="float")
    builder = FeatureBuilder(frame)
    emitted = builder.frame.emit.to_numpy(dtype=bool) if "emit" in builder.frame else np.ones(len(builder.frame), dtype=bool)
    values = np.column_stack([builder.build(item).to_numpy(dtype=np.float32, na_value=np.nan)[emitted] for item in specs])
    ids_path = base_path + "_ids.npy"; values_path = base_path + "_values.npy"
    np.save(ids_path, builder.frame.loc[emitted, "observation_id"].to_numpy(dtype=np.int64), allow_pickle=False)
    np.save(values_path, values.astype(np.float32, copy=False), allow_pickle=False)
    return ids_path, values_path


class AlphaDiscoveryRun:
    def __init__(self, config: AlphaDiscoveryConfig) -> None:
        self.config = config
        self.root = Path(config.output_root) / config.run_name
        source_root = Path(__file__).resolve().parent
        digest = sha256()
        for path in sorted(source_root.rglob("*.py")):
            digest.update(path.relative_to(source_root).as_posix().encode()); digest.update(path.read_bytes())
        self.implementation_hash = digest.hexdigest()

    def initialize(self) -> None:
        for directory in ("single_results", "dual_trial_ledger", "dual_coarse_results", "dual_fine_results",
                          "dual_exact_results", "stability", "context_expansion", "formula_factory", "ml",
                          "edge_autopsy", "replication", "reports", "cache"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        manifest = self.root / "run_manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config.definition_hash or existing.get("implementation_hash") != self.implementation_hash:
                raise RuntimeError("Run directory belongs to a different configuration or implementation hash")
        else:
            self._atomic_json("run_manifest.json", {"run_name": self.config.run_name, "config_hash": self.config.definition_hash,
                                                    "implementation_hash": self.implementation_hash,
                                                    "created_at": datetime.now(timezone.utc).isoformat(), "standalone": True})
        if not (self.root / "research_state.json").exists():
            self._atomic_json("research_state.json", {"state": ResearchState.BUILD_ONLY.value})

    def execute(self, stage: str, dual_stage: str | None = None) -> dict:
        normalized = f"scan-duals-{dual_stage or 'coarse'}" if stage == "scan-duals" else stage
        if normalized not in STAGE_ORDER:
            raise ValueError(f"Unknown pipeline stage: {normalized}")
        self.initialize()
        marker = self.root / "checkpoints" / f"{normalized}.json"
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") == "complete" and payload.get("config_hash") == self.config.definition_hash and payload.get("implementation_hash") == self.implementation_hash:
                return payload | {"resumed": True}
        started = datetime.now(timezone.utc).isoformat()
        self._atomic_json(f"checkpoints/{normalized}.json", {"stage": normalized, "status": "running", "started_at": started,
                                                             "config_hash": self.config.definition_hash, "implementation_hash": self.implementation_hash})
        handler = getattr(self, "_stage_" + normalized.replace("-", "_"))
        try:
            details = handler() or {}
        except Exception as error:
            self._atomic_json(f"checkpoints/{normalized}.json", {"stage": normalized, "status": "failed", "started_at": started,
                              "failed_at": datetime.now(timezone.utc).isoformat(), "config_hash": self.config.definition_hash,
                              "implementation_hash": self.implementation_hash,
                              "error_type": type(error).__name__, "error": str(error)})
            raise
        payload = {"stage": normalized, "status": "complete", "started_at": started,
                   "completed_at": datetime.now(timezone.utc).isoformat(), "config_hash": self.config.definition_hash,
                   "implementation_hash": self.implementation_hash} | details
        self._atomic_json(f"checkpoints/{normalized}.json", payload)
        return payload

    def _stage_validate_config(self) -> dict:
        self.config.validate()
        return {"project_root": str(Path(self.config.project_root).resolve())}

    def _stage_compile_registry(self) -> dict:
        bundle = self.compile_registry()
        return {"concepts": len(bundle.concepts), "features": len(bundle.features), "targets": len(bundle.targets)}

    def _stage_snapshot(self) -> dict:
        from .data.snapshot import RAW_COLUMNS, RESEARCH_COLUMNS, validate_snapshot
        from .data.source import DuckDBSource
        from .governance.access import AccessGate
        source_path = Path(self.config.source.duckdb_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Configured DuckDB catalog does not exist: {source_path}")
        source = DuckDBSource(source_path, AccessGate(self.config, ResearchState.BUILD_ONLY))
        import duckdb
        start = self.config.warmup.get("snapshot_start", self.config.research_periods.discovery_start)
        end = self.config.research_periods.discovery_end
        smoke_limit = int(self.config.universe.get("smoke_symbol_limit", 0) or 0)
        benchmark_sql = ",".join(f"'{symbol}'" for symbol in self.config.source.benchmark_symbols)
        symbol_filter = ""
        if smoke_limit:
            symbol_filter = (f" AND security_id IN (SELECT security_id FROM {self.config.source.security_master_table} WHERE symbol IN ({benchmark_sql}) "
                             f"UNION SELECT security_id FROM (SELECT security_id FROM {self.config.source.security_master_table} "
                             f"WHERE symbol NOT IN ({benchmark_sql}) ORDER BY symbol LIMIT {smoke_limit}))")
        with duckdb.connect(str(source_path), read_only=True) as connection:
            row_count, symbols, sessions = connection.execute(
                f"SELECT count(*), count(DISTINCT security_id), count(DISTINCT session_date) "
                f"FROM {self.config.source.bars_1m_raw_table} "
                f"WHERE session_date BETWEEN DATE '{start}' AND DATE '{end}'{symbol_filter}"
            ).fetchone()
        if int(row_count) > 5_000_000:
            snapshot = self.root / "snapshot"; snapshot.mkdir(parents=True, exist_ok=True)
            validation_path = Path(self.config.project_root) / "data" / "DATA_VALIDATION.json"
            validation_hash = sha256(validation_path.read_bytes()).hexdigest() if validation_path.exists() else None
            payload = {
                "mode": "immutable_catalog_reference", "catalog": str(source_path.resolve()),
                "start": str(start), "end": str(end), "rows": int(row_count),
                "symbols": int(symbols), "sessions": int(sessions),
                "catalog_validation_sha256": validation_hash,
            }
            self._atomic_json("snapshot/source_reference.json", payload)
            return {"rows": int(row_count), "symbols": int(symbols), "sessions": int(sessions),
                    "snapshot_fingerprint": sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                    "snapshot_mode": "immutable_catalog_reference"}
        raw_available = source.table_columns(self.config.source.bars_1m_raw_table)
        raw_columns = tuple(sorted(RAW_COLUMNS | ({"vwap", "trade_count"} & raw_available)))
        where = symbol_filter.removeprefix(" AND ") or None
        raw = source.read_table(self.config.source.bars_1m_raw_table, raw_columns, where=where)
        research = source.read_table(self.config.source.bars_1m_research_table, tuple(sorted(RESEARCH_COLUMNS)), where=where)
        validation = validate_snapshot(raw, research, self.config.source.alpaca_feed_required,
                                       self.config.source.alpaca_execution_adjustment_required)
        selected_ids = tuple(sorted(raw.security_id.astype(str).unique()))
        membership_where = "security_id IN (" + ",".join("'" + value.replace("'", "''") + "'" for value in selected_ids) + ")"
        membership = source.read_table(self.config.source.membership_table,
                                       ("security_id", "session_date", "in_universe"), where=membership_where)
        security_columns = source.table_columns(self.config.source.security_master_table)
        required_security = {"security_id", "symbol"}
        if not required_security.issubset(security_columns):
            raise ValueError(f"Security master missing columns: {sorted(required_security - security_columns)}")
        security = source.read_dimension(self.config.source.security_master_table, tuple(sorted(required_security)))
        security = security[security.security_id.astype(str).isin(selected_ids)].copy()
        actions_path = Path(self.config.source.corporate_actions_path)
        if not actions_path.exists():
            raise FileNotFoundError(f"Corporate-action ledger does not exist: {actions_path}")
        actions = pd.read_parquet(actions_path)
        if not {"security_id", "session_date", "split_factor"}.issubset(actions):
            raise ValueError("Corporate-action ledger must contain security_id, session_date, and split_factor")
        snapshot = self.root / "snapshot"; snapshot.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(snapshot / "bars_1m_raw.parquet", index=False)
        research.to_parquet(snapshot / "bars_1m_research.parquet", index=False)
        membership.to_parquet(snapshot / "membership.parquet", index=False)
        security.to_parquet(snapshot / "security_master.parquet", index=False)
        actions.to_parquet(snapshot / "corporate_actions.parquet", index=False)
        validation.write(snapshot / "validation.json")
        return {"rows": validation.rows, "symbols": validation.symbols, "sessions": validation.sessions,
                "snapshot_fingerprint": validation.fingerprint}

    def _require(self, stage: str) -> dict:
        marker = self.root / "checkpoints" / f"{stage}.json"
        if not marker.exists():
            raise FileNotFoundError(f"Required stage has not completed: {stage}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("config_hash") != self.config.definition_hash or payload.get("implementation_hash") != self.implementation_hash:
            raise RuntimeError(f"Required stage is incomplete or incompatible: {stage}")
        return payload

    def _stage_build_panel(self) -> dict:
        from .data.panel import build_calculation_panel
        from .data.universe import apply_point_in_time_universe
        self._require("snapshot")
        snapshot = self.root / "snapshot"
        if not (snapshot / "bars_1m_raw.parquet").exists():
            return self._build_panels_out_of_core()
        raw = pd.read_parquet(snapshot / "bars_1m_raw.parquet")
        research = pd.read_parquet(snapshot / "bars_1m_research.parquet")
        membership = pd.read_parquet(snapshot / "membership.parquet")
        security = pd.read_parquet(snapshot / "security_master.parquet")
        research_columns = [column for column in research if column not in {"security_id", "bar_start_ts_utc"}]
        bars = raw.merge(research[["security_id", "bar_start_ts_utc", *research_columns]],
                         on=["security_id", "bar_start_ts_utc"], how="left", validate="one_to_one")
        eligible_bars = apply_point_in_time_universe(bars, membership, security)
        benchmark_ids = set(security.loc[security.symbol.isin(self.config.source.benchmark_symbols), "security_id"])
        benchmark_bars = bars[bars.security_id.isin(benchmark_ids)].copy(); benchmark_bars["in_universe"] = False
        eligible_bars = pd.concat([eligible_bars, benchmark_bars], ignore_index=True).drop_duplicates(["security_id", "bar_start_ts_utc"])
        panel_root = self.root / "cache" / "panels"; panel_root.mkdir(parents=True, exist_ok=True)
        calculation_root = self.root / "cache" / "calculation_panels"; calculation_root.mkdir(parents=True, exist_ok=True)
        rows = 0
        for grid, enabled in self.config.decision_grids.items():
            if not enabled: continue
            calculation, panel = build_calculation_panel(eligible_bars, grid, self.config.source.benchmark_symbols[0])
            calculation.loc[pd.to_datetime(calculation.session_date).lt(pd.Timestamp(self.config.research_periods.discovery_start)), "emit"] = False
            calculation["observation_id"] = -1
            order = calculation.loc[calculation.emit].sort_values(["decision_ts", "security_id"], kind="mergesort").index
            calculation.loc[order, "observation_id"] = np.arange(len(order), dtype=np.int64)
            panel = calculation.loc[calculation.emit].sort_values(["decision_ts", "security_id"], kind="mergesort").copy()
            calculation.to_parquet(calculation_root / f"{grid}.parquet", index=False)
            panel.to_parquet(panel_root / f"{grid}.parquet", index=False); rows += len(panel)
        return {"panel_rows": rows, "enabled_grids": sum(bool(value) for value in self.config.decision_grids.values())}

    def _build_panels_out_of_core(self) -> dict:
        import duckdb
        source = self.config.source
        start = self.config.warmup.get("snapshot_start", self.config.research_periods.discovery_start)
        end = self.config.research_periods.discovery_end
        workers = os.cpu_count() if self.config.compute.cpu_workers == "auto" else int(self.config.compute.cpu_workers)
        temp = Path(self.config.compute.duckdb_temp_directory); temp.mkdir(parents=True, exist_ok=True)
        panel_root = self.root / "cache" / "panels"; panel_root.mkdir(parents=True, exist_ok=True)
        calculation_root = self.root / "cache" / "calculation_panels"; calculation_root.mkdir(parents=True, exist_ok=True)
        benchmark_sql = ",".join(f"'{symbol}'" for symbol in self.config.source.benchmark_symbols)
        total = 0; built = 0
        with duckdb.connect(source.duckdb_path, read_only=True) as connection:
            connection.execute(f"SET threads={workers}")
            connection.execute(f"SET memory_limit='{self.config.compute.duckdb_memory_limit}'")
            connection.execute(f"SET temp_directory='{temp.as_posix().replace(chr(39), chr(39)*2)}'")
            for grid, enabled in self.config.decision_grids.items():
                if not enabled:
                    continue
                calculation_destination = (calculation_root / f"{grid}.parquet").as_posix().replace("'", "''")
                panel_destination = (panel_root / f"{grid}.parquet").as_posix().replace("'", "''")
                cutoff = "" if grid != "preclose_1555" else "WHERE strftime(timezone('America/New_York', availability_ts_utc), '%H:%M') <= '15:55'"
                base = f"""
                    WITH all_joined AS (
                      SELECT r.security_id,r.symbol,r.session_date,r.bar_start_ts_utc,r.bar_end_ts_utc,r.availability_ts_utc,
                             q.research_open AS open,q.research_high AS high,q.research_low AS low,q.research_close AS close,
                             r.open AS execution_open,r.high AS execution_high,r.low AS execution_low,r.close AS execution_close,
                             r.volume,r.vwap/coalesce(nullif(q.split_factor,0),1) AS vwap,r.trade_count,q.split_factor,q.price_basis,
                             coalesce(m.in_universe,false) AS in_universe
                      FROM {source.bars_1m_raw_table} r JOIN {source.bars_1m_research_table} q
                        ON q.security_id=r.security_id AND q.bar_start_ts_utc=r.bar_start_ts_utc
                      LEFT JOIN {source.membership_table} m ON m.security_id=r.security_id AND m.session_date=r.session_date
                      WHERE r.session_date BETWEEN DATE '{start}' AND DATE '{end}'
                        AND (coalesce(m.in_universe,false) OR r.symbol IN ({benchmark_sql}))
                    ), joined AS (
                      SELECT * FROM all_joined {cutoff}
                    ), daily_close AS (
                      SELECT security_id,session_date,last(close ORDER BY availability_ts_utc) AS session_final_close
                      FROM joined GROUP BY security_id,session_date
                    ), close_history AS (
                      SELECT *,lag(session_final_close) OVER(PARTITION BY security_id ORDER BY session_date) AS prior_session_close
                      FROM daily_close
                    )
                """
                if grid.startswith("intraday"):
                    step = 5 if grid == "intraday_5m" else 1
                    calculation_query = base + f"""
                      , staged AS (SELECT j.*,
                         first_value(close) OVER(PARTITION BY j.security_id,j.session_date ORDER BY availability_ts_utc) AS session_open,
                         h.prior_session_close,availability_ts_utc AS decision_ts,'{grid}' AS decision_grid,
                         ((extract(minute FROM timezone('America/New_York',availability_ts_utc)) % {step}=0) AND session_date>=DATE '{self.config.research_periods.discovery_start}') AS emit,
                         close/lag(close,{step}) OVER(PARTITION BY j.security_id,j.session_date ORDER BY availability_ts_utc)-1 AS bucket_return,
                         -1::BIGINT AS observation_id
                       FROM joined j JOIN close_history h USING(security_id,session_date))
                      SELECT staged.*,
                         max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN close END) OVER(PARTITION BY decision_ts) AS benchmark_close,
                         max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN bucket_return END) OVER(PARTITION BY decision_ts) AS benchmark_bucket_return,
                         max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN session_open END) OVER(PARTITION BY decision_ts) AS benchmark_session_open,
                         max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN prior_session_close END) OVER(PARTITION BY decision_ts) AS benchmark_prior_session_close
                      FROM staged
                    """
                else:
                    calculation_query = base + f"""
                      , enriched AS (
                        SELECT *,first_value(open) OVER(PARTITION BY security_id,session_date ORDER BY availability_ts_utc) AS open0,
                         min(availability_ts_utc) OVER(PARTITION BY security_id,session_date) AS first_ts,
                         max(availability_ts_utc) OVER(PARTITION BY security_id,session_date) AS last_ts,
                         lag(sign(close-vwap)) OVER(PARTITION BY security_id,session_date ORDER BY availability_ts_utc) AS prior_vwap_sign,
                         sum(volume) OVER(PARTITION BY security_id,session_date ORDER BY availability_ts_utc ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS volume_5m
                        FROM joined
                      ), aggregated AS (
                        SELECT security_id,symbol,session_date,min(bar_start_ts_utc) AS bar_start_ts_utc,max(bar_end_ts_utc) AS bar_end_ts_utc,
                         max(availability_ts_utc) AS availability_ts_utc,first(open ORDER BY availability_ts_utc) AS open,max(high) AS high,min(low) AS low,
                         last(close ORDER BY availability_ts_utc) AS close,sum(volume) AS volume,
                         sum(vwap*volume)/nullif(sum(volume),0) AS vwap,sum(trade_count) AS trade_count,last(split_factor ORDER BY availability_ts_utc) AS split_factor,
                         last(price_basis ORDER BY availability_ts_utc) AS price_basis,bool_or(in_universe) AS in_universe,
                         first(open ORDER BY availability_ts_utc) AS session_open,last(close ORDER BY availability_ts_utc) AS session_close,
                         max(high) AS session_high,min(low) AS session_low,sum(vwap*volume)/nullif(sum(volume),0) AS session_vwap,
                         max(volume)/nullif(sum(volume),0) AS largest_1m_volume_share_session,
                         max(volume_5m)/nullif(sum(volume),0) AS largest_5m_volume_share_session,
                         sum(CASE WHEN prior_vwap_sign IS NOT NULL AND sign(close-vwap)<>prior_vwap_sign THEN 1 ELSE 0 END)::DOUBLE/nullif(count(*)-1,0) AS vwap_cross_count,
                         avg((close>vwap)::INT) AS time_above_vwap,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 5 MINUTE)/open0-1 AS opening_return_5m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 10 MINUTE)/open0-1 AS opening_return_10m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 15 MINUTE)/open0-1 AS opening_return_15m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 30 MINUTE)/open0-1 AS opening_return_30m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 60 MINUTE)/open0-1 AS opening_return_60m,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=last_ts-INTERVAL 5 MINUTE)-1 AS closing_return_5m,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=last_ts-INTERVAL 10 MINUTE)-1 AS closing_return_10m,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=last_ts-INTERVAL 15 MINUTE)-1 AS closing_return_15m,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=last_ts-INTERVAL 30 MINUTE)-1 AS closing_return_30m,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=last_ts-INTERVAL 60 MINUTE)-1 AS closing_return_60m,
                         (max(high) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 15 MINUTE)-min(low) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 15 MINUTE))/open0 AS opening_range_pct_15m,
                         (max(high) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 30 MINUTE)-min(low) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 30 MINUTE))/open0 AS opening_range_pct_30m,
                         (max(high) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 60 MINUTE)-min(low) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 60 MINUTE))/open0 AS opening_range_pct_60m,
                         sum(volume) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 15 MINUTE) AS opening_volume_15m,
                         sum(volume) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 30 MINUTE) AS opening_volume_30m,
                         sum(volume) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 60 MINUTE) AS opening_volume_60m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 15 MINUTE) AS gap_fill_price_15m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 30 MINUTE) AS gap_fill_price_30m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE availability_ts_utc<=first_ts+INTERVAL 60 MINUTE) AS gap_fill_price_60m,
                         sum(volume) FILTER(WHERE availability_ts_utc>last_ts-INTERVAL 15 MINUTE)/nullif(sum(volume),0) AS closing_volume_share_15m,
                         sum(volume) FILTER(WHERE availability_ts_utc>last_ts-INTERVAL 30 MINUTE)/nullif(sum(volume),0) AS closing_volume_share_30m,
                         sum(volume) FILTER(WHERE availability_ts_utc>last_ts-INTERVAL 60 MINUTE)/nullif(sum(volume),0) AS closing_volume_share_60m,
                         arg_max(close,availability_ts_utc) FILTER(WHERE strftime(timezone('America/New_York',availability_ts_utc),'%H:%M')<='12:00')/open0-1 AS open_to_midday_return,
                         last(close ORDER BY availability_ts_utc)/arg_max(close,availability_ts_utc) FILTER(WHERE strftime(timezone('America/New_York',availability_ts_utc),'%H:%M')<='12:00')-1 AS midday_to_close_return
                        FROM enriched GROUP BY security_id,symbol,session_date,open0
                      )
                      SELECT a.*,h.prior_session_close,availability_ts_utc AS decision_ts,'{grid}' AS decision_grid,(a.session_date>=DATE '{self.config.research_periods.discovery_start}') AS emit,
                        close/lag(close) OVER(PARTITION BY a.security_id ORDER BY session_date)-1 AS bucket_return,
                        max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN close END) OVER(PARTITION BY availability_ts_utc) AS benchmark_close,
                        NULL::DOUBLE AS benchmark_bucket_return,
                        max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN session_open END) OVER(PARTITION BY availability_ts_utc) AS benchmark_session_open,
                        max(CASE WHEN symbol='{self.config.source.benchmark_symbols[0]}' THEN h.prior_session_close END) OVER(PARTITION BY availability_ts_utc) AS benchmark_prior_session_close,
                        -1::BIGINT AS observation_id
                      FROM aggregated a JOIN close_history h USING(security_id,session_date)
                    """
                connection.execute(f"COPY ({calculation_query}) TO '{calculation_destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
                selector = "emit"
                panel_query = f"SELECT row_number() OVER(ORDER BY decision_ts,security_id)-1 AS new_id,* EXCLUDE(observation_id) FROM read_parquet('{calculation_destination}') WHERE {selector}"
                connection.execute(f"COPY (SELECT new_id AS observation_id,* EXCLUDE(new_id) FROM ({panel_query})) TO '{panel_destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
                connection.execute(f"CREATE OR REPLACE TEMP TABLE observation_map AS SELECT security_id,decision_ts,observation_id FROM read_parquet('{panel_destination}')")
                remapped = calculation_destination + ".remapped"
                connection.execute(f"COPY (SELECT c.* EXCLUDE(observation_id),coalesce(m.observation_id,-1) AS observation_id FROM read_parquet('{calculation_destination}') c LEFT JOIN observation_map m USING(security_id,decision_ts)) TO '{remapped}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
                Path(remapped).replace(Path(calculation_destination))
                count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{panel_destination}')").fetchone()[0])
                total += count; built += 1
        return {"panel_rows": total, "enabled_grids": built, "mode": "duckdb_out_of_core",
                "cpu_workers": workers, "memory_limit": self.config.compute.duckdb_memory_limit}

    def _stage_build_features(self) -> dict:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from .cache.feature_store import ArrayStore
        from .features.base import FeatureBuilder
        self._require("build-panel"); bundle = self.compile_registry(); blocks = 0; columns = 0; resumed_blocks = 0
        panel_root = self.root / "cache" / "panels"; feature_root = self.root / "cache" / "features"
        workers = os.cpu_count() if self.config.compute.cpu_workers == "auto" else int(self.config.compute.cpu_workers)
        configured_block = self.config.compute.feature_block_size
        block_size = workers if configured_block == "auto" else int(configured_block)
        for grid, enabled in self.config.decision_grids.items():
            if not enabled: continue
            import duckdb
            import pyarrow.parquet as pq
            calculation_path = self.root / "cache" / "calculation_panels" / f"{grid}.parquet"
            emitted_path = panel_root / f"{grid}.parquet"
            observation_count = int(pq.ParquetFile(emitted_path).metadata.num_rows)
            if observation_count <= 0:
                raise ValueError(f"Emitted feature panel is empty: {emitted_path}")
            with duckdb.connect() as connection:
                minimum, maximum, distinct_count = connection.execute(
                    "SELECT min(observation_id),max(observation_id),count(DISTINCT observation_id) FROM read_parquet(?)",
                    [str(emitted_path)],
                ).fetchone()
                security_ids = [str(row[0]) for row in connection.execute(
                    "SELECT DISTINCT security_id FROM read_parquet(?) ORDER BY security_id", [str(calculation_path)]
                ).fetchall()]
            if int(minimum) != 0 or int(maximum) != observation_count - 1 or int(distinct_count) != observation_count:
                raise ValueError(f"Observation IDs are not dense and unique for {grid}")
            # Partition once by security. The old path made every worker scan the
            # complete multi-gigabyte panel for every feature block.
            local_panel = self._ensure_local_feature_panel(grid, calculation_path, workers)
            groups: list[list[str]] = [[security_id] for security_id in security_ids]
            store = ArrayStore(feature_root / grid)
            all_specs = [item for item in bundle.features if item.decision_grid == grid]
            work: list[tuple[str, list, bool]] = []
            for family in sorted({item.family for item in all_specs}):
                family_specs = [item for item in all_specs if item.family == family]
                for global_flag in (False, True):
                    selected = [item for item in family_specs if _alpha_feature_is_global(item) == global_flag]
                    for part, offset in enumerate(range(0, len(selected), block_size)):
                        batch = selected[offset:offset + block_size]
                        if batch:
                            work.append((f"{family}__{'global' if global_flag else 'local'}__{part:03d}", batch, global_flag))
            local_work = [item for item in work if not item[2]]
            global_work = [item for item in work if item[2]]
            panel = None; builder = None
            with ProcessPoolExecutor(max_workers=workers) as process_pool:
              for name, batch, _ in local_work:
                    expected_columns = [item.feature_id for item in batch]
                    try:
                        existing, existing_columns = store.read(name)
                        if existing.shape[0] == observation_count and existing_columns == expected_columns:
                            blocks += 1; columns += len(batch); resumed_blocks += 1
                            continue
                    except (FileNotFoundError, ValueError, KeyError):
                        pass
                    target = store.root / f"{name}.npy"; temporary = store.root / f"{name}.tmp.npy"
                    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                       shape=(observation_count, len(batch)))
                    part_root = store.root / ".parts" / name; part_root.mkdir(parents=True, exist_ok=True)
                    futures = [process_pool.submit(_build_alpha_symbol_part, str(local_panel), batch, selected_ids,
                                                   str(part_root / f"part_{part:04d}"))
                               for part, selected_ids in enumerate(groups)]
                    written_rows = 0
                    for future in as_completed(futures):
                        ids_path, values_path = future.result()
                        ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
                        part_values = np.load(values_path, mmap_mode="r", allow_pickle=False)
                        # Own the ID buffer before unlinking the Windows-backed
                        # memmap; np.asarray can retain the file mapping.
                        dense_ids = np.array(ids, dtype=np.int64, copy=True)
                        if len(dense_ids) and (dense_ids.min() < 0 or dense_ids.max() >= observation_count):
                            raise ValueError(f"Worker returned out-of-range observation IDs for {grid}")
                        values[dense_ids, :] = part_values
                        written_rows += len(dense_ids)
                        del ids, part_values
                        Path(ids_path).unlink(); Path(values_path).unlink()
                    if written_rows != observation_count:
                        raise ValueError(f"Local feature block {name} wrote {written_rows:,}/{observation_count:,} rows")
                    values.flush(); del values
                    temporary.replace(target)
                    meta = target.with_suffix(".json"); tmp_meta = meta.with_suffix(".tmp.json")
                    tmp_meta.write_text(json.dumps({"shape": [observation_count, len(batch)], "dtype": "float32",
                                                    "columns": expected_columns}, indent=2), encoding="utf-8")
                    tmp_meta.replace(meta)
                    blocks += 1; columns += len(batch)
            # Global/cross-sectional families need the combined panel. Run them
            # only after the process pool exits so idle workers cannot retain RAM.
            for name, batch, _ in global_work:
                expected_columns = [item.feature_id for item in batch]
                try:
                    existing, existing_columns = store.read(name)
                    if existing.shape[0] == observation_count and existing_columns == expected_columns:
                        blocks += 1; columns += len(batch); resumed_blocks += 1
                        continue
                except (FileNotFoundError, ValueError, KeyError):
                    pass
                if panel is None:
                    panel = pd.read_parquet(calculation_path)
                    for column in ("security_id", "symbol", "decision_grid", "price_basis"):
                        if column in panel: panel[column] = panel[column].astype("category")
                    for column in ("open", "high", "low", "close", "vwap", "research_open", "research_high", "research_low", "research_close", "split_factor"):
                        if column in panel: panel[column] = pd.to_numeric(panel[column], downcast="float")
                    builder = FeatureBuilder(panel)
                emitted = builder.frame.emit.to_numpy(dtype=bool) if "emit" in builder.frame else np.ones(len(builder.frame), dtype=bool)
                emitted_ids = builder.frame.loc[emitted, "observation_id"].to_numpy(dtype=np.int64)
                if len(emitted_ids) != observation_count:
                    raise ValueError(f"Global feature panel emitted {len(emitted_ids):,}/{observation_count:,} rows")
                target = store.root / f"{name}.npy"; temporary = store.root / f"{name}.tmp.npy"
                values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                   shape=(observation_count, len(batch)))
                for index, item in enumerate(batch):
                    values[emitted_ids, index] = builder.build(item).to_numpy(dtype=np.float32, na_value=np.nan)[emitted]
                values.flush(); del values
                temporary.replace(target)
                meta = target.with_suffix(".json"); tmp_meta = meta.with_suffix(".tmp.json")
                tmp_meta.write_text(json.dumps({"shape": [observation_count, len(batch)], "dtype": "float32",
                                                "columns": expected_columns}, indent=2), encoding="utf-8")
                tmp_meta.replace(meta)
                builder._cache.clear(); builder._direct_cache.clear()
                blocks += 1; columns += len(batch)
            observation_destination = feature_root / grid / "observations.parquet"
            escaped_source = emitted_path.as_posix().replace("'", "''")
            escaped_destination = observation_destination.as_posix().replace("'", "''")
            with duckdb.connect() as connection:
                connection.execute(
                    f"COPY (SELECT observation_id,security_id,session_date,decision_ts,decision_grid "
                    f"FROM read_parquet('{escaped_source}') ORDER BY observation_id) TO '{escaped_destination}' "
                    "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
                )
        self._atomic_json("cache/feature_build_resources.json", {
            "cpu_workers": workers, "host_memory_fraction": self.config.compute.host_memory_fraction,
            "duckdb_memory_limit": self.config.compute.duckdb_memory_limit,
            "feature_block_size": block_size, "storage": str(feature_root.resolve()),
        })
        return {"feature_blocks": blocks, "feature_columns": columns, "resumed_blocks": resumed_blocks,
                "cpu_workers": workers, "host_memory_fraction": self.config.compute.host_memory_fraction}

    def _ensure_local_feature_panel(self, grid: str, calculation_path: Path, workers: int) -> Path:
        """Create a resume-safe, security-partitioned panel for local workers."""
        import duckdb
        destination = self.root / "cache" / "local_feature_panels" / grid
        manifest = self.root / "cache" / "local_feature_panels" / f"{grid}.json"
        source = {"size": calculation_path.stat().st_size, "mtime_ns": calculation_path.stat().st_mtime_ns}
        if destination.exists() and manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("source") == source and payload.get("config_hash") == self.config.definition_hash:
                return destination
            raise RuntimeError(f"Incompatible local feature panel already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{grid}.{uuid4().hex}.building"
        source_sql = calculation_path.as_posix().replace("'", "''")
        temporary_sql = temporary.as_posix().replace("'", "''")
        temp_directory = Path(self.config.compute.duckdb_temp_directory)
        temp_directory.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as connection:
            connection.execute(f"SET threads={workers}")
            connection.execute(f"SET memory_limit='{self.config.compute.duckdb_memory_limit}'")
            connection.execute(f"SET temp_directory='{temp_directory.as_posix().replace(chr(39), chr(39)*2)}'")
            connection.execute(
                f"COPY (SELECT * FROM read_parquet('{source_sql}')) TO '{temporary_sql}' "
                "(FORMAT PARQUET,PARTITION_BY (security_id),COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)"
            )
        temporary.replace(destination)
        self._atomic_json(f"cache/local_feature_panels/{grid}.json", {
            "grid": grid, "source": source, "config_hash": self.config.definition_hash,
            "partition_key": "security_id", "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return destination

    def _stage_build_targets(self) -> dict:
        from .targets.builder import build_daily_targets, build_intraday_targets, build_overnight_targets, flag_corporate_action_crossings
        self._require("build-panel")
        if not (self.root / "snapshot" / "bars_1m_raw.parquet").exists():
            return self._build_targets_out_of_core()
        raw = pd.read_parquet(self.root / "snapshot" / "bars_1m_raw.parquet")
        actions = pd.read_parquet(self.root / "snapshot" / "corporate_actions.parquet")
        output = self.root / "cache" / "targets"; output.mkdir(parents=True, exist_ok=True)
        rows = 0; target_ids: set[str] = set()
        for grid, enabled in self.config.decision_grids.items():
            if not enabled: continue
            panel = pd.read_parquet(self.root / "cache" / "panels" / f"{grid}.parquet")
            if grid.startswith("intraday"):
                table = build_intraday_targets(panel, raw, tuple(int(x[:-1]) for x in self.config.targets["intraday"] if str(x).endswith("m")), "EOD" in self.config.targets["intraday"])
            elif grid == "daily_close":
                table = build_daily_targets(panel, raw, tuple(int(x[:-1]) for x in self.config.targets["interday"]), int(self.config.targets["daily_entry_delay_minutes"]))
            else:
                table = build_overnight_targets(panel, raw)
            if not table.empty:
                table = flag_corporate_action_crossings(table, actions)
                annotations = panel[[column for column in ("observation_id", "symbol", "beta_prior") if column in panel]].drop_duplicates("observation_id")
                table = table.merge(annotations, on="observation_id", how="left", validate="many_to_one")
                benchmark_symbol = self.config.source.benchmark_symbols[0]
                benchmark = table[table.symbol.eq(benchmark_symbol)][["decision_ts", "target_id", "target"]].rename(columns={"target": "benchmark_target"})
                benchmark = benchmark.drop_duplicates(["decision_ts", "target_id"])
                enriched = table.merge(benchmark, on=["decision_ts", "target_id"], how="left", validate="many_to_one")
                bases = [enriched]
                if "benchmark_adjusted" in self.config.targets["bases"]:
                    adjusted = enriched.copy(); adjusted["target"] = adjusted.target - adjusted.benchmark_target
                    adjusted["target_basis"] = "benchmark_adjusted"; adjusted["target_id"] = adjusted.target_id.str.replace("__raw__", "__benchmark_adjusted__", regex=False); bases.append(adjusted)
                if "beta_residual" in self.config.targets["bases"]:
                    residual = enriched.copy(); beta = residual.get("beta_prior", pd.Series(1.0, index=residual.index)).fillna(1.0)
                    residual["target"] = residual.target - beta * residual.benchmark_target
                    residual["target_basis"] = "beta_residual"; residual["target_id"] = residual.target_id.str.replace("__raw__", "__beta_residual__", regex=False); bases.append(residual)
                table = pd.concat(bases, ignore_index=True).drop(columns=["benchmark_target"], errors="ignore")
            table.to_parquet(output / f"{grid}.parquet", index=False); rows += len(table); target_ids.update(table.get("target_id", []))
        return {"target_rows": rows, "target_ids": len(target_ids)}

    def _build_targets_out_of_core(self) -> dict:
        """Build the large target ledger in DuckDB without materializing source bars."""
        import duckdb
        source = self.config.source
        output = self.root / "cache" / "targets"; output.mkdir(parents=True, exist_ok=True)
        workers = os.cpu_count() if self.config.compute.cpu_workers == "auto" else int(self.config.compute.cpu_workers)
        total_rows = 0; target_ids: set[str] = set(); benchmark_symbol = source.benchmark_symbols[0]
        with duckdb.connect(source.duckdb_path, read_only=True) as connection:
            connection.execute(f"SET threads={workers}")
            connection.execute(f"SET memory_limit='{self.config.compute.duckdb_memory_limit}'")
            for grid, enabled in self.config.decision_grids.items():
                if not enabled: continue
                panel = (self.root / "cache" / "panels" / f"{grid}.parquet").as_posix().replace("'", "''")
                destination = (output / f"{grid}.parquet").as_posix().replace("'", "''")
                beta_window = 1560 if grid == "intraday_5m" else 7800 if grid == "intraday_1m" else 63
                beta_minimum = 780 if grid == "intraday_5m" else 3900 if grid == "intraday_1m" else 40
                snapshot_start = self.config.warmup.get("snapshot_start", self.config.research_periods.discovery_start)
                raw_table = f"(SELECT * FROM {source.bars_1m_raw_table} WHERE session_date BETWEEN DATE '{snapshot_start}' AND DATE '{self.config.research_periods.discovery_end}')"
                decisions = f"""
                  decisions0 AS (SELECT * FROM read_parquet('{panel}')),
                  decisions AS (SELECT *,CASE WHEN count(benchmark_bucket_return) OVER(PARTITION BY security_id ORDER BY decision_ts ROWS BETWEEN {beta_window} PRECEDING AND 1 PRECEDING)>={beta_minimum}
                    THEN covar_samp(bucket_return,benchmark_bucket_return) OVER(PARTITION BY security_id ORDER BY decision_ts ROWS BETWEEN {beta_window} PRECEDING AND 1 PRECEDING)
                    /nullif(var_samp(benchmark_bucket_return) OVER(PARTITION BY security_id ORDER BY decision_ts ROWS BETWEEN {beta_window} PRECEDING AND 1 PRECEDING),0) END AS beta_prior FROM decisions0)
                """
                if grid.startswith("intraday"):
                    horizons = [int(x[:-1]) for x in self.config.targets["intraday"] if str(x).endswith("m")]
                    values = ",".join(f"({h})" for h in horizons)
                    eod_union = "" if "EOD" not in self.config.targets["intraday"] else """
                      UNION ALL SELECT d.observation_id,d.security_id,d.symbol,d.decision_ts,d.beta_prior,e.bar_start_ts_utc,e.open,
                        'eod' AS target_label,z.bar_end_ts_utc,z.exit_close,z.exit_close/e.open-1 AS target
                      FROM decisions d JOIN bars e ON e.security_id=d.security_id AND e.session_date=d.session_date AND e.bar_start_ts_utc=d.decision_ts+INTERVAL 1 MINUTE
                      JOIN session_end z ON z.security_id=d.security_id AND z.session_date=d.session_date
                    """
                    raw_query = f"""
                      WITH {decisions}, bars AS (SELECT *,row_number() OVER(PARTITION BY security_id,session_date ORDER BY bar_start_ts_utc) rn FROM {raw_table}),
                      session_end AS (SELECT security_id,session_date,last(bar_end_ts_utc ORDER BY bar_start_ts_utc) AS bar_end_ts_utc,last(close ORDER BY bar_start_ts_utc) AS exit_close FROM bars GROUP BY security_id,session_date),
                      raw_targets AS (
                        SELECT d.observation_id,d.security_id,d.symbol,d.decision_ts,d.beta_prior,e.bar_start_ts_utc,e.open,
                          cast(h.h AS VARCHAR)||'m' AS target_label,z.bar_end_ts_utc,z.close,z.close/e.open-1 AS target
                        FROM decisions d JOIN bars e ON e.security_id=d.security_id AND e.session_date=d.session_date AND e.bar_start_ts_utc=d.decision_ts+INTERVAL 1 MINUTE
                        CROSS JOIN (VALUES {values}) h(h) JOIN bars z ON z.security_id=e.security_id AND z.session_date=e.session_date AND z.rn=e.rn+h.h-1
                        {eod_union}
                      )
                    """
                elif grid == "daily_close":
                    values = ",".join(f"({int(x[:-1])})" for x in self.config.targets["interday"])
                    delay = int(self.config.targets["daily_entry_delay_minutes"]) + 1
                    raw_query = f"""
                      WITH {decisions}, sessions AS (SELECT session_date,row_number() OVER(ORDER BY session_date) sn FROM (SELECT DISTINCT session_date FROM {raw_table})),
                      bars AS (SELECT *,row_number() OVER(PARTITION BY security_id,session_date ORDER BY bar_start_ts_utc) rn,row_number() OVER(PARTITION BY security_id,session_date ORDER BY bar_start_ts_utc DESC) rev FROM {raw_table}),
                      raw_targets AS (SELECT d.observation_id,d.security_id,d.symbol,d.decision_ts,d.beta_prior,e.bar_start_ts_utc,e.open,
                        cast(h.h AS VARCHAR)||'d' AS target_label,z.bar_end_ts_utc,z.close,z.close/e.open-1 AS target
                        FROM decisions d JOIN sessions s ON s.session_date=d.session_date CROSS JOIN (VALUES {values}) h(h)
                        JOIN sessions se ON se.sn=s.sn+1 JOIN bars e ON e.security_id=d.security_id AND e.session_date=se.session_date AND e.rn={delay}
                        JOIN sessions sx ON sx.sn=s.sn+h.h JOIN bars z ON z.security_id=d.security_id AND z.session_date=sx.session_date AND z.rev=1)
                    """
                else:
                    raw_query = f"""
                      WITH {decisions}, sessions AS (SELECT session_date,row_number() OVER(ORDER BY session_date) sn FROM (SELECT DISTINCT session_date FROM {raw_table})),
                      bars AS (SELECT *,row_number() OVER(PARTITION BY security_id,session_date ORDER BY bar_start_ts_utc) rn FROM {raw_table}),
                      raw_targets AS (SELECT d.observation_id,d.security_id,d.symbol,d.decision_ts,d.beta_prior,e.bar_start_ts_utc,e.open,
                        'overnight' AS target_label,z.bar_start_ts_utc AS bar_end_ts_utc,z.open AS close,z.open/e.open-1 AS target
                        FROM decisions d JOIN sessions s ON s.session_date=d.session_date JOIN bars e ON e.security_id=d.security_id AND e.session_date=d.session_date AND e.bar_start_ts_utc=d.decision_ts+INTERVAL 1 MINUTE
                        JOIN sessions sx ON sx.sn=s.sn+1 JOIN bars z ON z.security_id=d.security_id AND z.session_date=sx.session_date AND z.rn=2)
                    """
                bases = ["raw"] + (["benchmark_adjusted"] if "benchmark_adjusted" in self.config.targets["bases"] else []) + (["beta_residual"] if "beta_residual" in self.config.targets["bases"] else [])
                basis_values = ",".join(f"('{basis}')" for basis in bases)
                actions_path = Path(source.corporate_actions_path).as_posix().replace("'", "''")
                split_cross = f"EXISTS(SELECT 1 FROM read_parquet('{actions_path}') a WHERE a.security_id=enriched.security_id AND a.session_date>CAST(enriched.bar_start_ts_utc AS DATE) AND a.session_date<=CAST(enriched.bar_end_ts_utc AS DATE) AND lower(a.action_type) LIKE '%split%')"
                cash_cross = f"EXISTS(SELECT 1 FROM read_parquet('{actions_path}') a WHERE a.security_id=enriched.security_id AND a.session_date>CAST(enriched.bar_start_ts_utc AS DATE) AND a.session_date<=CAST(enriched.bar_end_ts_utc AS DATE) AND (lower(a.action_type) LIKE '%cash%' OR lower(a.action_type) LIKE '%dividend%'))"
                query = raw_query + f"""
                  , benchmark AS (SELECT decision_ts,target_label,target AS benchmark_target FROM raw_targets WHERE symbol='{benchmark_symbol}'),
                  enriched AS (SELECT r.*,b.benchmark_target FROM raw_targets r LEFT JOIN benchmark b USING(decision_ts,target_label))
                  SELECT observation_id,security_id,decision_ts,bar_start_ts_utc entry_ts,open entry_price,bar_end_ts_utc exit_ts,close exit_price,
                    CASE WHEN {split_cross} OR {cash_cross} THEN NULL ELSE CASE basis WHEN 'raw' THEN target WHEN 'benchmark_adjusted' THEN target-benchmark_target ELSE target-beta_prior*benchmark_target END END AS "target",
                    basis AS target_basis,'target_'||target_label||'__'||basis||'__{grid}' AS target_id,beta_prior,
                    {split_cross} AS crosses_split,{cash_cross} AS crosses_cash_dividend
                  FROM enriched CROSS JOIN (VALUES {basis_values}) q(basis)
                  WHERE basis='raw' OR benchmark_target IS NOT NULL AND (basis<>'beta_residual' OR beta_prior IS NOT NULL)
                """
                connection.execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
                count, ids = connection.execute(f"SELECT count(*),list(DISTINCT target_id) FROM read_parquet('{destination}')").fetchone()
                total_rows += int(count); target_ids.update(ids or [])
        return {"target_rows": total_rows, "target_ids": len(target_ids), "mode": "duckdb_out_of_core", "cpu_workers": workers}

    def _target_ids_and_vector(self, grid: str, observations: pd.DataFrame, target_id: str | None = None):
        """Read one aligned target at a time so production scans stay memory bounded."""
        import duckdb
        path = (self.root / "cache" / "targets" / f"{grid}.parquet").as_posix().replace("'", "''")
        with duckdb.connect() as connection:
            if target_id is None:
                return [row[0] for row in connection.execute(f"SELECT DISTINCT target_id FROM read_parquet('{path}') ORDER BY target_id").fetchall()]
            ids = observations.observation_id.to_numpy(np.int64)
            connection.register("observation_order", pd.DataFrame({"position": np.arange(len(ids)), "observation_id": ids}))
            escaped = target_id.replace("'", "''")
            frame = connection.execute(f"SELECT o.position,t.target FROM observation_order o LEFT JOIN read_parquet('{path}') t ON t.observation_id=o.observation_id AND t.target_id='{escaped}' ORDER BY o.position").fetchdf()
            return frame.target.to_numpy(dtype=np.float32, na_value=np.nan)

    def _stage_scan_singles(self) -> dict:
        from .cache.feature_store import ArrayStore
        from .scan.singles import scan_singles
        self._require("build-features"); self._require("build-targets")
        chunks = 0; tests = 0
        for grid, enabled in self.config.decision_grids.items():
            if not enabled: continue
            observations = pd.read_parquet(self.root / "cache" / "features" / grid / "observations.parquet")
            target_ids = self._target_ids_and_vector(grid, observations)
            if not target_ids: continue
            target_vectors={target_id:self._target_ids_and_vector(grid, observations, target_id) for target_id in target_ids}
            for metadata in sorted((self.root / "cache" / "features" / grid).glob("*.json")):
                values, feature_ids = ArrayStore(metadata.parent).read(metadata.stem)
                results=[]
                for target_id,target_vector in target_vectors.items():
                    results.append(scan_singles(np.asarray(values),target_vector[:,None],feature_ids,[target_id]))
                result=pd.concat(results,ignore_index=True)
                result["fold_id"] = "all"
                sessions = pd.to_datetime(observations.session_date)
                fold_rows = []
                for fold_id, dates in enumerate(np.array_split(np.sort(sessions.unique()), int(self.config.stability["chronological_folds"]))):
                    mask = sessions.isin(dates).to_numpy()
                    if mask.sum() < 3: continue
                    fold=pd.concat([scan_singles(np.asarray(values)[mask],vector[mask,None],feature_ids,[target_id]) for target_id,vector in target_vectors.items()],ignore_index=True)
                    fold["fold_id"] = str(fold_id); fold_rows.append(fold)
                if fold_rows:
                    result = pd.concat([result, *fold_rows], ignore_index=True)
                destination = self.root / "single_results" / grid; destination.mkdir(parents=True, exist_ok=True)
                result.to_parquet(destination / f"{metadata.stem}.parquet", index=False); chunks += 1; tests += int(result.fold_id.eq("all").sum())
        return {"result_chunks": chunks, "attempted_single_tests": tests}

    def _dual_scan(self, bins: int, stage_name: str) -> dict:
        from .cache.bin_store import BinStore
        from .cache.feature_store import ArrayStore
        from .cache.rank_store import build_rank_bins
        from .scan.dual_coarse import DualTileScanner
        from .scan.dual_pairs import pair_id
        self._require("build-features"); self._require("build-targets"); bundle = self.compile_registry()
        eligibility = EligibilityMatrix.build(bundle.features, bundle.targets)
        attempted = 0; excluded = 0; chunks = 0; backend = None
        output_root = self.root / ("dual_coarse_results" if bins == 3 else "dual_fine_results" if bins == 5 else "dual_exact_results")
        ledger_root = self.root / "dual_trial_ledger" / stage_name; ledger_root.mkdir(parents=True, exist_ok=True)
        for grid, enabled in self.config.decision_grids.items():
            if not enabled: continue
            observations = pd.read_parquet(self.root / "cache" / "features" / grid / "observations.parquet")
            decision_codes = pd.factorize(observations.decision_ts, sort=True)[0]
            feature_bins: dict[str, tuple[str, int]] = {}
            for metadata in sorted((self.root / "cache" / "features" / grid).glob("*.json")):
                values, feature_ids = ArrayStore(metadata.parent).read(metadata.stem)
                labels, _ = build_rank_bins(np.asarray(values), decision_codes, bins)
                persisted_labels = labels.view(np.int8)
                bin_path = BinStore(self.root / "cache" / "bins" / str(bins) / grid).write(metadata.stem, persisted_labels, feature_ids, bins)
                feature_bins.update({name: (str(bin_path), index) for index, name in enumerate(feature_ids)})
                del labels, persisted_labels
            target_ids = self._target_ids_and_vector(grid, observations)
            specs = [item for item in bundle.features if item.decision_grid == grid and item.feature_id in feature_bins]
            scanner = DualTileScanner(bins=bins, device_name=self.config.compute.gpu_device,
                                      prefer_cuda=self.config.compute.prefer_cuda, memory_fraction=self.config.compute.dynamic_memory_fraction)
            backend = scanner.backend; block = scanner.recommended_pair_block(len(observations))
            exclusions = []
            for target_id in target_ids:
                target_vector = self._target_ids_and_vector(grid, observations, target_id)
                active = [item for item in specs if target_id in eligibility.active_targets(item.feature_id)]
                batch: list[tuple] = []; part = 0
                for left_index, left in enumerate(active):
                    for right in active[left_index + 1:]:
                        identifier = pair_id(left.feature_id, right.feature_id)
                        if left.redundancy_group == right.redundancy_group:
                            exclusions.append({"pair_id": identifier, "feature_a": left.feature_id, "feature_b": right.feature_id,
                                               "target_id": target_id, "eligible": False, "reason": "exact_alias"}); excluded += 1; continue
                        batch.append((identifier, left.feature_id, right.feature_id))
                        if len(batch) < block: continue
                        chunks += self._write_dual_tile(scanner, batch, feature_bins, target_vector, output_root / grid / target_id, part)
                        attempted += len(batch); part += 1; batch = []
                if batch:
                    chunks += self._write_dual_tile(scanner, batch, feature_bins, target_vector, output_root / grid / target_id, part)
                    attempted += len(batch)
            pd.DataFrame(exclusions, columns=["pair_id", "feature_a", "feature_b", "target_id", "eligible", "reason"]).to_parquet(ledger_root / f"{grid}_exclusions.parquet", index=False)
        import shutil
        bin_cache = (self.root / "cache" / "bins" / str(bins)).resolve()
        if bin_cache.exists() and self.root.resolve() in bin_cache.parents:
            shutil.rmtree(bin_cache)
        return {"bins": bins, "backend": backend, "attempted_pair_target_tests": attempted, "excluded_pair_target_tests": excluded, "result_chunks": chunks}

    @staticmethod
    def _write_dual_tile(scanner, batch: list[tuple], feature_bins: dict[str, tuple[str, int]], target: np.ndarray, output: Path, part: int) -> int:
        destination = output / f"part-{part:08d}.parquet"
        if destination.exists(): return 0
        arrays: dict[str, np.memmap] = {}
        def column(feature_id: str) -> np.ndarray:
            path,index=feature_bins[feature_id]
            if path not in arrays: arrays[path]=np.load(path,mmap_mode="r",allow_pickle=False)
            return arrays[path][:,index]
        a = np.column_stack([column(left) for _, left, _ in batch]); b = np.column_stack([column(right) for _, _, right in batch])
        result = scanner.scan(a, b, target)
        result.insert(0, "pair_id", [item[0] for item in batch]); result.insert(1, "feature_a", [item[1] for item in batch]); result.insert(2, "feature_b", [item[2] for item in batch])
        output.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(".tmp.parquet"); result.to_parquet(temporary, index=False); temporary.replace(destination)
        return 1

    def _stage_scan_duals_coarse(self) -> dict: return self._dual_scan(int(self.config.duals["coarse_bins"]), "coarse")
    def _stage_scan_duals_fine(self) -> dict: return self._dual_scan(5, "fine")
    def _stage_exact_duals(self) -> dict: return self._dual_scan(int(self.config.duals["exact_bins"]), "exact")

    def _stage_build_stability(self) -> dict:
        self._require("scan-singles")
        files = list((self.root / "single_results").glob("*/*.parquet"))
        if not files: raise ValueError("No single-scan results exist")
        table = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        overall = table[table.fold_id.eq("all")].drop(columns="fold_id")
        folds = table[table.fold_id.ne("all")]
        fold_stats = folds.groupby(["feature_id", "target_id"], observed=True).agg(
            fold_median_effect=("top_bottom_spread", "median"),
            fold_sign_consistency=("top_bottom_spread", lambda x: float(abs(np.sign(x.dropna()).mean())) if x.notna().any() else np.nan),
            valid_folds=("rank_ic", "count")).reset_index()
        stability = overall.merge(fold_stats, on=["feature_id", "target_id"], how="left")
        stability["stability_score"] = stability.top_bottom_spread.abs() * stability.fold_sign_consistency.fillna(0) * np.sqrt(stability.valid_folds.fillna(0))
        destination = self.root / "stability" / "single_stability.parquet"; destination.parent.mkdir(parents=True, exist_ok=True); stability.to_parquet(destination, index=False)
        return {"single_structures": len(stability), "valid_fold_results": len(folds)}

    def _stable_rows(self, limit: int = 64) -> pd.DataFrame:
        table = pd.read_parquet(self.root / "stability" / "single_stability.parquet")
        return table.replace([np.inf, -np.inf], np.nan).dropna(subset=["stability_score"]).sort_values("stability_score", ascending=False).head(limit)

    def _stage_expand_context(self) -> dict:
        from .interactions.context_expand import expand_contexts
        self._require("build-stability"); bundle = self.compile_registry(); stable = self._stable_rows()
        structures = [{"source_id": f"single:{row.feature_id}:{row.target_id}", "feature_ids": (row.feature_id,)} for row in stable.itertuples()]
        contexts = [item.feature_id for item in bundle.features if item.family in {"calendar", "market", "session"}]
        rows = expand_contexts(structures, contexts); destination = self.root / "context_expansion" / "registry.parquet"; destination.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(destination, index=False)
        return {"generated_context_interactions": len(rows)}

    def _stage_run_formula_factory(self) -> dict:
        from .interactions.formula_factory import FormulaFactory
        self._require("build-stability"); features = self._stable_rows()["feature_id"].drop_duplicates().head(64).tolist()
        factory = FormulaFactory(int(self.config.formula_factory["max_expression_depth"]), int(self.config.formula_factory["max_binary_operators"]))
        formulas = factory.generate(features); rows = [{"expression": item.expression, "expression_hash": item.expression_hash, "depth": item.depth, "binary_operators": item.binary_operators} for item in formulas]
        destination = self.root / "formula_factory" / "registry.parquet"; destination.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(destination, index=False)
        return {"generated_formulas": len(rows)}

    def _stage_run_ml(self) -> dict:
        from .ml.folds import purged_walk_forward_folds
        from .ml.models import fit_predict_models
        self._require("build-stability"); stable = self._stable_rows(32)
        if stable.empty: raise ValueError("No stable feature-target structures are available for ML")
        target_id = stable.iloc[0].target_id; grid = next(item.decision_grid for item in compile_registry(self.config).targets if item.target_id == target_id)
        feature_ids = stable.feature_id.drop_duplicates().tolist(); observations, x = self._load_features(grid, feature_ids)
        target_long = pd.read_parquet(self.root / "cache" / "targets" / f"{grid}.parquet")
        target = target_long[target_long.target_id.eq(target_id)][["observation_id", "target"]]
        y = observations[["observation_id"]].merge(target, on="observation_id", how="left").target.to_numpy(float)
        valid = np.isfinite(y); x = x[valid]; y = y[valid]; predictions = []
        for fold, (train, validation) in enumerate(purged_walk_forward_folds(len(y), int(self.config.stability["chronological_folds"]), 5)):
            for model, values in fit_predict_models(x, y, train, validation).items():
                predictions.extend({"fold": fold, "model": model, "row": int(row), "prediction": float(value), "actual": float(y[row])} for row, value in zip(validation, values))
        destination = self.root / "ml" / "predictions.parquet"; destination.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(predictions).to_parquet(destination, index=False)
        self._atomic_json("ml/dataset.json", {"grid": grid, "target_id": target_id, "feature_ids": feature_ids})
        return {"ml_predictions": len(predictions), "features": len(feature_ids)}

    def _load_features(self, grid: str, requested: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        from .cache.feature_store import ArrayStore
        observations = pd.read_parquet(self.root / "cache" / "features" / grid / "observations.parquet"); columns = {}
        for metadata in (self.root / "cache" / "features" / grid).glob("*.json"):
            values, names = ArrayStore(metadata.parent).read(metadata.stem)
            for index, name in enumerate(names):
                if name in requested: columns[name] = np.asarray(values[:, index], dtype=float)
        missing = set(requested) - set(columns)
        if missing: raise ValueError(f"Feature cache missing requested columns: {sorted(missing)}")
        return observations, np.column_stack([columns[name] for name in requested])

    def _stage_distill_ml(self) -> dict:
        self._require("run-ml"); predictions = pd.read_parquet(self.root / "ml" / "predictions.parquet")
        if predictions.empty: raise ValueError("ML produced no out-of-fold predictions")
        summary = predictions.groupby("model", observed=True).apply(lambda x: pd.Series({"oof_correlation": x.prediction.corr(x.actual, method="spearman"), "mse": np.mean((x.prediction-x.actual)**2), "rows": len(x)}), include_groups=False).reset_index()
        summary.to_parquet(self.root / "ml" / "distillation.parquet", index=False)
        return {"distilled_models": len(summary)}

    @staticmethod
    def _bucket(values: pd.Series, bins: int = 5) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        try: return pd.qcut(numeric.rank(method="first"), bins, labels=False, duplicates="drop").astype("Int64").astype(str)
        except ValueError: return pd.Series("single", index=values.index)

    def _stage_run_edge_autopsy(self) -> dict:
        from .autopsy.run import EdgeAutopsy
        self._require("build-stability"); stable = self._stable_rows(8)
        if stable.empty: raise ValueError("No stable candidate can enter Edge Autopsy")
        completed = []
        for candidate_index, row in enumerate(stable.itertuples(index=False)):
            feature = next(item for item in compile_registry(self.config).features if item.feature_id == row.feature_id)
            observations, matrix = self._load_features(feature.decision_grid, [row.feature_id])
            panel = pd.read_parquet(self.root / "cache" / "panels" / f"{feature.decision_grid}.parquet")
            target_long = pd.read_parquet(self.root / "cache" / "targets" / f"{feature.decision_grid}.parquet")
            primary_table = target_long[target_long.target_id.eq(row.target_id)]
            aligned = observations.merge(panel.drop(columns=[column for column in ("decision_grid",) if column in panel]), on=["observation_id", "security_id", "session_date", "decision_ts"], how="left", validate="one_to_one")
            aligned = aligned.merge(primary_table[["observation_id", "target", "entry_price", "exit_price"]], on="observation_id", how="left", validate="one_to_one")
            score = matrix[:, 0]; primary = aligned.target.to_numpy(float)
            close = pd.to_numeric(aligned.close, errors="coerce"); volume = pd.to_numeric(aligned.volume, errors="coerce")
            simple = close.groupby(aligned.security_id).pct_change(fill_method=None)
            market = pd.to_numeric(aligned.get("benchmark_return", pd.Series(0.0, index=aligned.index)), errors="coerce").fillna(0)
            aligned["market_trend"] = np.sign(market.rolling(20, min_periods=1).sum()).astype(str)
            aligned["market_vol"] = self._bucket(market.rolling(20, min_periods=2).std())
            aligned["breadth"] = self._bucket(simple.groupby(aligned.decision_ts).transform(lambda x: (x > 0).mean()))
            aligned["dispersion"] = self._bucket(simple.groupby(aligned.decision_ts).transform("std"))
            aligned["correlation_regime"] = self._bucket(market.rolling(20, min_periods=3).std())
            local = pd.to_datetime(aligned.decision_ts, utc=True).dt.tz_convert("America/New_York")
            aligned["time_of_day"] = local.dt.strftime("%H:%M"); aligned["day_of_week"] = local.dt.dayofweek.astype(str)
            aligned["price_bucket"] = self._bucket(close); aligned["liquidity_bucket"] = self._bucket(close * volume)
            aligned["volatility_bucket"] = self._bucket(simple.groupby(aligned.security_id).rolling(20, min_periods=3).std().reset_index(level=0, drop=True))
            aligned["beta_bucket"] = self._bucket(aligned.get("beta_prior", pd.Series(1.0, index=aligned.index)))
            aligned["size_bucket"] = self._bucket(close * volume)
            aligned["trade_notional"] = 10_000.0; aligned["bar_dollar_volume"] = close * volume
            aligned["path_prices"] = [np.array([entry, exit], dtype=float) for entry, exit in zip(aligned.entry_price, aligned.exit_price)]
            aligned["market_factor"] = market; aligned["variant_winsorized"] = pd.Series(score).clip(pd.Series(score).quantile(.01), pd.Series(score).quantile(.99))
            for peer_index, peer in enumerate(stable.feature_id.head(4)):
                if peer == row.feature_id: continue
                try: _, peer_values = self._load_features(feature.decision_grid, [peer]); aligned[f"candidate_score_{peer_index}"] = peer_values[:, 0]
                except (ValueError, StopIteration): continue
            delayed = {}
            for target_id, table in target_long[target_long.target_basis.eq(primary_table.target_basis.iloc[0] if not primary_table.empty else "raw")].groupby("target_id"):
                horizon = ''.join(character for character in target_id.split("__", 1)[0] if character.isdigit())
                if not horizon: continue
                delayed[int(horizon)] = observations[["observation_id"]].merge(table[["observation_id", "target"]], on="observation_id", how="left").target.to_numpy(float)
            metadata = aligned[["security_id", "session_date", "decision_ts", "trade_notional", "bar_dollar_volume", "entry_price", "path_prices",
                                "market_trend", "market_vol", "breadth", "dispersion", "correlation_regime", "time_of_day", "day_of_week",
                                "price_bucket", "liquidity_bucket", "volatility_bucket", "beta_bucket", "size_bucket", "market_factor", "variant_winsorized",
                                *[column for column in aligned if column.startswith("candidate_score_")]]]
            valid = np.isfinite(score) & np.isfinite(primary)
            if valid.sum() < 20: continue
            autopsy_id = f"candidate-{candidate_index:04d}"; EdgeAutopsy(self.root / "edge_autopsy" / autopsy_id, self.config.edge_autopsy).run(score[valid], {row.target_id: primary[valid]}, metadata.loc[valid].reset_index(drop=True), {key: value[valid] for key, value in delayed.items()})
            completed.append({"candidate_id": autopsy_id, "feature_ids": [row.feature_id], "target_id": row.target_id, "autopsy_complete": True, "stability_score": row.stability_score})
        if not completed: raise ValueError("No candidate had enough aligned observations for Edge Autopsy")
        self._atomic_json("edge_autopsy/candidates.json", {"candidates": completed})
        return {"completed_autopsies": len(completed)}

    def _stage_audit_exhaustiveness(self) -> dict:
        bundle = self.compile_registry(); eligibility = EligibilityMatrix.build(bundle.features, bundle.targets)
        expected_singles = int(eligibility.frame.active.sum()); single_files = list((self.root / "single_results").glob("*/*.parquet"))
        attempted_singles = sum(int(pd.read_parquet(path, columns=["fold_id"]).fold_id.eq("all").sum()) for path in single_files)
        raw_pairs = 0; expected_pairs = 0; alias_exclusions = 0
        for grid in {item.decision_grid for item in bundle.features}:
            specs = [item for item in bundle.features if item.decision_grid == grid]; raw_pairs += len(specs) * (len(specs)-1)//2
            for target in [item.target_id for item in bundle.targets if item.decision_grid == grid]:
                active = [item for item in specs if target in eligibility.active_targets(item.feature_id)]
                total = len(active)*(len(active)-1)//2; aliases = sum(1 for i,left in enumerate(active) for right in active[i+1:] if left.redundancy_group == right.redundancy_group)
                expected_pairs += total-aliases; alias_exclusions += aliases
        def count_tree(name: str) -> int:
            return sum(len(pd.read_parquet(path, columns=["pair_id"])) for path in (self.root / name).glob("**/part-*.parquet"))
        coarse, fine = count_tree("dual_coarse_results"), count_tree("dual_fine_results")
        audit = ExhaustivenessAudit(len(bundle.concepts), len(bundle.features), len(bundle.features), len(bundle.unavailable), len(bundle.targets),
            expected_singles, attempted_singles, 0, 0, raw_pairs, {"exact_alias": alias_exclusions}, raw_pairs-alias_exclusions,
            expected_pairs, coarse, coarse, fine, 0, bool(self.config.duals["exhaustive_5x5_all_pairs"]))
        audit.write(self.root / "exhaustiveness_manifest.json")
        return {"exhaustiveness_status": audit.status, "expected_single_tests": expected_singles, "attempted_single_tests": attempted_singles,
                "expected_pair_target_tests": expected_pairs, "coarse_completed": coarse, "fine_completed": fine}

    def _stage_freeze_discovery(self) -> dict:
        from .governance.freeze import freeze_candidates
        from .governance.state import ResearchStateMachine
        self._require("run-edge-autopsy"); self._require("audit-exhaustiveness")
        audit = json.loads((self.root / "exhaustiveness_manifest.json").read_text(encoding="utf-8")); candidates = json.loads((self.root / "edge_autopsy" / "candidates.json").read_text(encoding="utf-8"))["candidates"]
        digest = freeze_candidates(self.root / "candidate_freeze.json", candidates, audit)
        machine = ResearchStateMachine.load(self.root / "research_state.json")
        if machine.state == ResearchState.BUILD_ONLY: machine.transition(ResearchState.DISCOVERY_OPEN)
        machine.transition(ResearchState.DISCOVERY_FROZEN, {"exhaustiveness": audit.get("exhaustiveness_status") == "PASS", "autopsy": bool(candidates)})
        return {"candidate_count": len(candidates), "freeze_sha256": digest}

    def _stage_evaluate_replication(self) -> dict:
        from .governance.freeze import verify_manifest
        from .governance.state import ResearchStateMachine
        self._require("freeze-discovery"); manifest = verify_manifest(self.root / "candidate_freeze.json")
        if not self.config.research_periods.allow_replication_access: raise PermissionError("Replication remains sealed; set allow_replication_access only when the campaign is authorized")
        machine = ResearchStateMachine.load(self.root / "research_state.json")
        machine.transition(ResearchState.REPLICATION_OPEN, {"candidate_freeze": bool(manifest["candidates"])})
        results = self._evaluate_period(ResearchState.REPLICATION_OPEN, manifest["candidates"])
        results.to_parquet(self.root / "replication" / "evaluation.parquet", index=False)
        return {"replication_candidates": len(results)}

    def _stage_freeze_replication(self) -> dict:
        from .governance.freeze import _write_frozen
        from .governance.state import ResearchStateMachine
        self._require("evaluate-replication"); results = pd.read_parquet(self.root / "replication" / "evaluation.parquet")
        digest = _write_frozen(self.root / "replication_freeze.json", {"kind": "replication_freeze", "results": results.to_dict("records")})
        machine = ResearchStateMachine.load(self.root / "research_state.json"); machine.transition(ResearchState.REPLICATION_FROZEN)
        return {"replication_freeze_sha256": digest, "candidates": len(results)}

    def _evaluate_period(self, state: ResearchState, candidates: list[dict]) -> pd.DataFrame:
        from .data.panel import build_decision_panel
        from .data.source import DuckDBSource
        from .data.universe import apply_point_in_time_universe
        from .features.base import FeatureBuilder
        from .governance.access import AccessGate
        from .scan.statistics import pairwise_rank_ic, quantile_spread
        from .targets.builder import build_daily_targets, build_intraday_targets, build_overnight_targets
        source = DuckDBSource(Path(self.config.source.duckdb_path), AccessGate(self.config, state))
        raw_columns = tuple(sorted(source.table_columns(self.config.source.bars_1m_raw_table) & {"security_id","symbol","bar_start_ts_utc","bar_end_ts_utc","availability_ts_utc","session_date","open","high","low","close","volume","vwap","trade_count","feed","adjustment","ingest_batch_id"}))
        raw = source.read_table(self.config.source.bars_1m_raw_table, raw_columns)
        research_columns = tuple(sorted(source.table_columns(self.config.source.bars_1m_research_table) & {"security_id","bar_start_ts_utc","research_open","research_high","research_low","research_close","split_factor","price_basis"}))
        research = source.read_table(self.config.source.bars_1m_research_table, research_columns)
        membership = source.read_table(self.config.source.membership_table, ("security_id","session_date","in_universe"))
        security = source.read_dimension(self.config.source.security_master_table, ("security_id","symbol"))
        bars = raw.merge(research, on=["security_id","bar_start_ts_utc"], how="left", validate="one_to_one")
        bundle = compile_registry(self.config); feature_map = {item.feature_id:item for item in bundle.features}; results=[]
        for grid in sorted({feature_map[candidate["feature_ids"][0]].decision_grid for candidate in candidates}):
            decision_rows = build_decision_panel(bars, grid); panel = apply_point_in_time_universe(decision_rows, membership, security)
            benchmark_ids=set(security.loc[security.symbol.isin(self.config.source.benchmark_symbols),"security_id"]); benchmark=decision_rows[decision_rows.security_id.isin(benchmark_ids)].copy(); benchmark["in_universe"]=False
            panel=pd.concat([panel,benchmark],ignore_index=True).drop_duplicates("observation_id").sort_values(["security_id","decision_ts"],kind="mergesort")
            if grid.startswith("intraday"): targets=build_intraday_targets(panel,raw,tuple(int(x[:-1]) for x in self.config.targets["intraday"] if str(x).endswith("m")),"EOD" in self.config.targets["intraday"])
            elif grid=="daily_close": targets=build_daily_targets(panel,raw,tuple(int(x[:-1]) for x in self.config.targets["interday"]),int(self.config.targets["daily_entry_delay_minutes"]))
            else: targets=build_overnight_targets(panel,raw)
            targets=targets.merge(panel[["observation_id","symbol",*[column for column in ("beta_prior",) if column in panel]]].drop_duplicates("observation_id"),on="observation_id",how="left",validate="many_to_one")
            bench=targets[targets.symbol.eq(self.config.source.benchmark_symbols[0])][["decision_ts","target_id","target"]].rename(columns={"target":"benchmark_target"}).drop_duplicates(["decision_ts","target_id"])
            enriched=targets.merge(bench,on=["decision_ts","target_id"],how="left",validate="many_to_one"); variants=[enriched]
            adjusted=enriched.copy(); adjusted["target"]-=adjusted.benchmark_target; adjusted["target_id"]=adjusted.target_id.str.replace("__raw__","__benchmark_adjusted__",regex=False); variants.append(adjusted)
            residual=enriched.copy(); residual["target"]-=residual.get("beta_prior",pd.Series(1.0,index=residual.index)).fillna(1)*residual.benchmark_target; residual["target_id"]=residual.target_id.str.replace("__raw__","__beta_residual__",regex=False); variants.append(residual)
            target_table=pd.concat(variants,ignore_index=True)
            builder=FeatureBuilder(panel)
            for candidate in [item for item in candidates if feature_map[item["feature_ids"][0]].decision_grid==grid]:
                values=[builder.build(feature_map[feature_id]).to_numpy(float) for feature_id in candidate["feature_ids"]]
                score=np.nanmean(np.column_stack(values),axis=1); selected=target_table[target_table.target_id.eq(candidate["target_id"])][["observation_id","target"]]
                y=panel[["observation_id"]].merge(selected,on="observation_id",how="left").target.to_numpy(float); ic,count=pairwise_rank_ic(score,y)
                results.append({"candidate_id":candidate.get("candidate_id"),"feature_ids":candidate["feature_ids"],"target_id":candidate["target_id"],"n_obs":count,"rank_ic":ic,"top_bottom_spread":quantile_spread(score,y),"period":state.value})
        return pd.DataFrame(results)

    def _stage_build_alphas(self) -> dict:
        from .alpha.registry import AlphaRegistry
        from .governance.freeze import verify_manifest
        from .models import AlphaSpec, stable_hash
        self._require("freeze-replication"); frozen=verify_manifest(self.root/"replication_freeze.json"); registry=AlphaRegistry()
        for index,row in enumerate(frozen["results"]):
            direction=1 if float(row.get("rank_ic") or 0)>=0 else -1; alpha_id=f"alpha-{index:04d}"
            payload={"alpha_id":alpha_id,"features":row["feature_ids"],"target":row["target_id"],"direction":direction}
            registry.add(AlphaSpec(alpha_id,"single" if len(row["feature_ids"])==1 else "dual",tuple(row["feature_ids"]),row["target_id"],"cross_sectional_rank",direction,{"source":"replication_locked"},next(item.decision_grid for item in compile_registry(self.config).targets if item.target_id==row["target_id"]),row["target_id"].split("__")[0],stable_hash(payload)))
        destination=self.root/"alpha_repository.parquet"; registry.write(destination); return {"alphas":len(registry.specs)}

    def _stage_evaluate_alphas(self) -> dict:
        self._require("build-alphas"); alphas=pd.read_parquet(self.root/"alpha_repository.parquet"); replication=pd.read_parquet(self.root/"replication"/"evaluation.parquet")
        evaluated=alphas.merge(replication[["target_id","rank_ic","top_bottom_spread","n_obs"]],on="target_id",how="left")
        evaluated["directional_replication_effect"]=evaluated.direction*evaluated.top_bottom_spread; evaluated.to_parquet(self.root/"replication"/"alpha_evaluation.parquet",index=False)
        return {"evaluated_alphas":len(evaluated)}

    def _stage_freeze_portfolio(self) -> dict:
        from .governance.freeze import freeze_portfolio
        from .governance.state import ResearchStateMachine
        self._require("evaluate-alphas"); evaluated=pd.read_parquet(self.root/"replication"/"alpha_evaluation.parquet").sort_values("directional_replication_effect",ascending=False)
        included=evaluated[evaluated.directional_replication_effect.gt(0)].alpha_id.tolist()
        if not included: raise RuntimeError("No alpha passed the predeclared positive replication-direction gate")
        portfolio={"included_alphas":included,"allocation_rule":"equal_weight","rebalance_rule":"at_each_alpha_decision","risk_limits":{"gross_leverage":1.0,"max_alpha_weight":min(0.35,1/len(included))},"cost_model":{"per_side_bps":[0,1,3,5]},"entry_exit_mappings":evaluated[evaluated.alpha_id.isin(included)][["alpha_id","target_id"]].to_dict("records")}
        digest=freeze_portfolio(self.root/"portfolio_freeze.json",portfolio); machine=ResearchStateMachine.load(self.root/"research_state.json"); machine.transition(ResearchState.PORTFOLIO_FROZEN)
        return {"portfolio_freeze_sha256":digest,"included_alphas":len(included)}

    def _stage_evaluate_final_holdout(self) -> dict:
        from .governance.freeze import verify_manifest
        from .governance.state import ResearchStateMachine
        self._require("freeze-portfolio")
        if not self.config.research_periods.allow_final_holdout_access: raise PermissionError("Final holdout remains sealed; enable it only for the one-way confirmation")
        portfolio=verify_manifest(self.root/"portfolio_freeze.json")["portfolio"]; alphas=pd.read_parquet(self.root/"alpha_repository.parquet"); selected=alphas[alphas.alpha_id.isin(portfolio["included_alphas"])]
        candidates=[{"candidate_id":row.alpha_id,"feature_ids":list(row.feature_ids),"target_id":row.target_id} for row in selected.itertuples()]
        machine=ResearchStateMachine.load(self.root/"research_state.json"); machine.transition(ResearchState.FINAL_HOLDOUT_OPEN)
        results=self._evaluate_period(ResearchState.FINAL_HOLDOUT_OPEN,candidates); results.to_parquet(self.root/"final_holdout_results.parquet",index=False); machine.transition(ResearchState.FINAL_COMPLETE)
        return {"final_candidates":len(results)}

    def _stage_build_report(self) -> dict:
        self.build_report_shells(); markers=[]
        for path in (self.root/"checkpoints").glob("*.json"):
            payload=json.loads(path.read_text(encoding="utf-8")); markers.append({"stage":payload.get("stage"),"status":payload.get("status")})
        pd.DataFrame(markers).to_parquet(self.root/"reports"/"stage_status.parquet",index=False)
        return {"reports":len(list((self.root/"reports").glob("*.md")))}

    def compile_registry(self) -> RegistryBundle:
        self.initialize(); bundle = compile_registry(self.config); eligibility = EligibilityMatrix.build(bundle.features, bundle.targets)
        bundle.feature_frame().to_parquet(self.root / "feature_registry.parquet", index=False)
        bundle.target_frame().to_parquet(self.root / "target_registry.parquet", index=False)
        eligibility.frame.to_parquet(self.root / "eligibility_matrix.parquet", index=False)
        self._atomic_json("registry_summary.json", {"concepts": len(bundle.concepts), "features": len(bundle.features),
                                                    "targets": len(bundle.targets), "unavailable": len(bundle.unavailable),
                                                    "required_warmup_sessions": bundle.required_warmup_sessions})
        return bundle

    def initial_audit(self) -> ExhaustivenessAudit:
        bundle = self.compile_registry(); eligibility = EligibilityMatrix.build(bundle.features, bundle.targets)
        expected_singles = int(eligibility.frame.active.sum())
        features_by_grid: dict[str, int] = {}
        for feature in bundle.features: features_by_grid[feature.decision_grid] = features_by_grid.get(feature.decision_grid, 0) + 1
        raw_pairs = sum(count * (count - 1) // 2 for count in features_by_grid.values())
        audit = ExhaustivenessAudit(len(bundle.concepts), len(bundle.features), len(bundle.features), len(bundle.unavailable),
            len(bundle.targets), expected_singles, 0, 0, 0, raw_pairs, {}, 0, 0, 0, 0, 0, 0,
            bool(self.config.duals["exhaustive_5x5_all_pairs"]))
        audit.write(self.root / "exhaustiveness_manifest.json")
        return audit

    def build_report_shells(self) -> None:
        bundle = self.compile_registry()
        build_reports(self.root / "reports", {"run_name": self.config.run_name, "config_hash": self.config.definition_hash,
                      "compiled_concepts": len(bundle.concepts), "compiled_features": len(bundle.features),
                      "compiled_targets": len(bundle.targets), "required_warmup_sessions": bundle.required_warmup_sessions,
                      "research_state": ResearchState.BUILD_ONLY.value})

    def _atomic_json(self, relative: str, payload: dict) -> None:
        target = self.root / relative; target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"); temporary.replace(target)
