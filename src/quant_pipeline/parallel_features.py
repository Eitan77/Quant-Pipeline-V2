from __future__ import annotations

import gc
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from .config import ScanConfig
from .features import build_features
from .registry import FeatureSpec


GLOBAL_FAMILIES = {"cross_sectional", "breadth", "market_context", "lead_lag", "sector"}


def is_symbol_local(spec: FeatureSpec) -> bool:
    return spec.classification not in {"cross_sectional", "context"} and spec.family not in GLOBAL_FAMILIES


def symbol_groups(canonical_path: Path, workers: int) -> list[list[str]]:
    counts = duckdb.sql(
        "SELECT symbol, count(*) AS n FROM read_parquet(?) GROUP BY symbol ORDER BY symbol",
        params=[str(canonical_path)],
    ).fetchall()
    groups: list[list[str]] = [[] for _ in range(min(workers, len(counts)))]
    loads = [0] * len(groups)
    for symbol, count in sorted(counts, key=lambda row: row[1], reverse=True):
        slot = min(range(len(groups)), key=loads.__getitem__)
        groups[slot].append(symbol)
        loads[slot] += count
    return [sorted(group) for group in groups]


def build_parallel_block(
    canonical_path: Path,
    output_path: Path,
    config: ScanConfig,
    specs: list[FeatureSpec],
    progress_path: Path,
) -> list[FeatureSpec]:
    return build_parallel_blocks(
        canonical_path,
        [(output_path, specs)],
        config,
        progress_path,
    )[0]


def build_parallel_blocks(
    canonical_path: Path,
    outputs: list[tuple[Path, list[FeatureSpec]]],
    config: ScanConfig,
    progress_path: Path,
) -> list[list[FeatureSpec]]:
    """Build several scan blocks in one bounded pass over each symbol shard."""
    if not outputs:
        return []
    temp_roots = [path.parent / f".{path.stem}_parts" for path, _ in outputs]
    for (output_path, _), temp_root in zip(outputs, temp_roots):
        resolved_parent = output_path.parent.resolve()
        if temp_root.resolve().parent != resolved_parent:
            raise RuntimeError(f"Unsafe temporary feature path: {temp_root}")
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True)
    groups = symbol_groups(canonical_path, config.feature_workers)
    futures = {}
    with ProcessPoolExecutor(max_workers=config.feature_workers) as pool:
        for part, symbols in enumerate(groups):
            part_outputs = [
                (temp_root / f"part_{part:02d}.parquet", specs)
                for temp_root, (_, specs) in zip(temp_roots, outputs)
            ]
            futures[pool.submit(_build_symbol_parts, canonical_path, part_outputs, config, symbols)] = part
        for complete, future in enumerate(as_completed(futures), 1):
            future.result()
            progress_path.write_text(json.dumps({
                "stage": "feature_cache_parallel",
                "feature_blocks": [path.name for path, _ in outputs],
                "completed_workers": complete,
                "total_workers": len(groups),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
    built_by_output = []
    for (output_path, specs), temp_root in zip(outputs, temp_roots):
        parts = str(temp_root / "*.parquet").replace("'", "''")
        destination = str(output_path).replace("'", "''")
        connection = duckdb.connect()
        connection.execute(f"PRAGMA threads={config.feature_workers}")
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{parts}') ORDER BY symbol, bar_start_ts) "
            f"TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.close()
        columns = set(pq.ParquetFile(output_path).schema.names)
        built_by_output.append([spec for spec in specs if spec.name in columns])
        shutil.rmtree(temp_root)
    return built_by_output


def _build_symbol_parts(
    canonical_path: Path,
    outputs: list[tuple[Path, list[FeatureSpec]]],
    config: ScanConfig,
    symbols: list[str],
) -> None:
    bars = pd.read_parquet(canonical_path, filters=[("symbol", "in", symbols)])
    for column in ["open", "high", "low", "close", "vwap", "volume"]:
        if column in bars:
            bars[column] = pd.to_numeric(bars[column], downcast="float")
    specs = [spec for _, chunk in outputs for spec in chunk]
    frame, built = build_features(bars, config, specs, symbol_local=True)
    frame["decision_ts"] = frame["available_at_ts"]
    feature_names = {spec.name for spec in built}
    identifiers = [column for column in frame if column not in feature_names]
    for part_path, chunk in outputs:
        columns = identifiers + [spec.name for spec in chunk if spec.name in frame]
        frame.loc[:, columns].to_parquet(part_path, index=False)
    del bars, frame
    gc.collect()
