from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from .evaluation import summarize_returns
from .finalize import finalize_existing
from .statuses import promote


CAP = 0.10
RECENT_START = pd.Timestamp("2025-05-01")


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _replace_with_backup(canonical: Path, corrected: Path) -> Path:
    backup = canonical.with_name(f"{canonical.stem}_pre_symbol_cap_fix{canonical.suffix}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite audit backup: {backup}")
    os.replace(canonical, backup)
    os.replace(corrected, canonical)
    return backup


def _rebuild_summary(root: Path, corrected_daily: Path) -> Path:
    summary = pd.read_csv(root / "strategy_summary.csv")
    daily = pd.read_parquet(corrected_daily)
    daily["session_date"] = pd.to_datetime(daily["session_date"])

    records: dict[str, dict[str, float]] = {}
    for strategy_id, group in daily.groupby("strategy_id", sort=False):
        returns = group.set_index("session_date")["net_return"].sort_index()
        metrics = summarize_returns(returns)
        metrics["recent_cagr"] = summarize_returns(returns.loc[returns.index >= RECENT_START])["net_cagr"]
        metrics["positive_year_fraction"] = float(
            (1 + returns).groupby(returns.index.year).prod().sub(1).gt(0).mean()
        )
        metrics["sessions"] = int(len(returns))
        records[strategy_id] = metrics

    metric_frame = pd.DataFrame.from_dict(records, orient="index")
    for column in metric_frame.columns:
        summary[column] = summary["strategy_id"].map(metric_frame[column])
    summary = promote(summary)
    corrected = root / "strategy_summary.symbol_cap_fixed.tmp.csv"
    summary.to_csv(corrected, index=False)
    return corrected


def repair(root: Path) -> None:
    root = root.resolve()
    trades = root / "trades.parquet"
    daily = root / "daily_strategy_returns.parquet"
    summary = root / "strategy_summary.csv"
    for path in (trades, daily, summary):
        if not path.exists():
            raise FileNotFoundError(path)

    fixed_trades = root / "trades.symbol_cap_fixed.tmp.parquet"
    fixed_daily = root / "daily_strategy_returns.symbol_cap_fixed.tmp.parquet"
    for path in (fixed_trades, fixed_daily, root / "strategy_summary.symbol_cap_fixed.tmp.csv"):
        if path.exists():
            path.unlink()

    con = duckdb.connect(str(root / "symbol_cap_repair.duckdb"))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='24GB'")
    source = _sql_path(trades)
    fixed = _sql_path(fixed_trades)
    clipped = f"greatest(-{CAP}, least({CAP}, target_weight))"
    con.execute(
        f"""
        COPY (
          SELECT * EXCLUDE(final_weight, position_return),
                 {clipped} AS final_weight,
                 abs({clipped}) * net_return AS position_return
          FROM read_parquet('{source}')
        ) TO '{fixed}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    source_check = con.execute(
        f"SELECT count(*) n, max(session_date) max_date FROM read_parquet('{source}')"
    ).fetchone()
    fixed_check = con.execute(
        f"SELECT count(*) n, max(session_date) max_date, max(abs(final_weight)) max_weight "
        f"FROM read_parquet('{fixed}')"
    ).fetchone()
    if source_check[0] != fixed_check[0] or fixed_check[1] >= pd.Timestamp("2026-05-01") or fixed_check[2] > CAP + 1e-12:
        raise RuntimeError(f"Corrected ledger failed validation: source={source_check}, fixed={fixed_check}")

    fixed_daily_sql = _sql_path(fixed_daily)
    con.execute(
        f"""
        COPY (
          SELECT session_date, sum(position_return) AS net_return, strategy_id
          FROM read_parquet('{fixed}')
          GROUP BY session_date, strategy_id
        ) TO '{fixed_daily_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    corrected_summary = _rebuild_summary(root, fixed_daily)

    backups = {
        "trades": _replace_with_backup(trades, fixed_trades),
        "daily": _replace_with_backup(daily, fixed_daily),
        "summary": _replace_with_backup(summary, corrected_summary),
    }

    canonical = _sql_path(trades)
    con.execute(
        f"COPY (SELECT strategy_id, session_date, decision_ts, symbol, side, target_weight, final_weight "
        f"FROM read_parquet('{canonical}')) TO '{_sql_path(root / 'positions.parquet')}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT strategy_id, decision_ts, symbol, side, final_weight, entry_ts, exit_ts, "
        f"entry_executable_price, exit_executable_price, slippage_cost FROM read_parquet('{canonical}')) "
        f"TO '{_sql_path(root / 'execution_diagnostics.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT strategy_id, symbol, sum(position_return) AS profit_contribution "
        f"FROM read_parquet('{canonical}') GROUP BY strategy_id, symbol) "
        f"TO '{_sql_path(root / 'strategy_concentration.csv')}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY (SELECT strategy_id, count(*) AS trade_count, avg(CAST(net_return > 0 AS INTEGER)) AS win_rate, "
        f"avg(net_return) * 10000 AS average_net_bps FROM read_parquet('{canonical}') GROUP BY strategy_id) "
        f"TO '{_sql_path(root / 'strategy_trade_statistics.csv')}' (HEADER, DELIMITER ',')"
    )
    con.close()

    corrected = pd.read_csv(summary)
    corrected.to_csv(root / "strategy_cost_stress.csv", index=False)
    corrected[[c for c in ("strategy_id", "maximum_drawdown", "status", "net_cagr") if c in corrected]].to_csv(
        root / "strategy_drawdown_summary.csv", index=False
    )
    corrected[[c for c in ("strategy_id", "status", "family", "cluster", "cost_bps_per_side") if c in corrected]].to_csv(
        root / "strategy_statuses.csv", index=False
    )
    corrected[[c for c in ("strategy_id", "family", "decision_time_et", "lookback_minutes", "tail", "holding_period_minutes", "net_cagr", "sharpe", "maximum_drawdown") if c in corrected]].to_csv(
        root / "strategy_parameter_stability.csv", index=False
    )
    corrected[["strategy_id", "sessions"]].to_csv(root / "data_coverage.csv", index=False)

    daily_frame = pd.read_parquet(daily)
    daily_frame["session_date"] = pd.to_datetime(daily_frame["session_date"])
    daily_frame["year"] = daily_frame.session_date.dt.year
    daily_frame["month"] = daily_frame.session_date.dt.to_period("M").astype(str)
    daily_frame.groupby(["strategy_id", "year"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(
        root / "strategy_yearly_results.csv", index=False
    )
    daily_frame.groupby(["strategy_id", "month"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(
        root / "strategy_monthly_results.csv", index=False
    )

    finalize_existing(root)
    report = {
        "status": "PASS",
        "symbol_cap": CAP,
        "source_rows": int(source_check[0]),
        "corrected_rows": int(fixed_check[0]),
        "max_session_date": str(fixed_check[1]),
        "max_abs_final_weight": float(fixed_check[2]),
        "audit_backups": {key: str(path) for key, path in backups.items()},
    }
    (root / "symbol_cap_repair_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def finish_reports(root: Path) -> None:
    """Resume after the validated canonical swap without rewriting the trade ledger."""
    root = root.resolve()
    trades = root / "trades.parquet"
    daily = root / "daily_strategy_returns.parquet"
    summary = root / "strategy_summary.csv"
    con = duckdb.connect(str(root / "symbol_cap_repair.duckdb"))
    con.execute("PRAGMA threads=8")
    canonical = _sql_path(trades)
    check = con.execute(
        f"SELECT count(*), max(session_date), max(abs(final_weight)) FROM read_parquet('{canonical}')"
    ).fetchone()
    if check[1] >= pd.Timestamp("2026-05-01") or check[2] > CAP + 1e-12:
        raise RuntimeError(f"Canonical ledger failed validation: {check}")
    con.execute(
        f"COPY (SELECT strategy_id, session_date, decision_ts, symbol, side, target_weight, final_weight "
        f"FROM read_parquet('{canonical}')) TO '{_sql_path(root / 'positions.parquet')}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT strategy_id, decision_ts, symbol, side, final_weight, entry_ts, exit_ts, "
        f"entry_executable_price, exit_executable_price, slippage_cost FROM read_parquet('{canonical}')) "
        f"TO '{_sql_path(root / 'execution_diagnostics.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT strategy_id, symbol, sum(position_return) AS profit_contribution "
        f"FROM read_parquet('{canonical}') GROUP BY strategy_id, symbol) "
        f"TO '{_sql_path(root / 'strategy_concentration.csv')}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY (SELECT strategy_id, count(*) AS trade_count, "
        f"avg(CAST(net_return > 0 AS INTEGER)) AS win_rate, avg(net_return) * 10000 AS average_net_bps "
        f"FROM read_parquet('{canonical}') GROUP BY strategy_id) "
        f"TO '{_sql_path(root / 'strategy_trade_statistics.csv')}' (HEADER, DELIMITER ',')"
    )
    con.close()

    corrected = pd.read_csv(summary)
    corrected.to_csv(root / "strategy_cost_stress.csv", index=False)
    corrected[[c for c in ("strategy_id", "maximum_drawdown", "status", "net_cagr") if c in corrected]].to_csv(root / "strategy_drawdown_summary.csv", index=False)
    corrected[[c for c in ("strategy_id", "status", "family", "cluster", "cost_bps_per_side") if c in corrected]].to_csv(root / "strategy_statuses.csv", index=False)
    corrected[[c for c in ("strategy_id", "family", "decision_time_et", "lookback_minutes", "tail", "holding_period_minutes", "net_cagr", "sharpe", "maximum_drawdown") if c in corrected]].to_csv(root / "strategy_parameter_stability.csv", index=False)
    corrected[["strategy_id", "sessions"]].to_csv(root / "data_coverage.csv", index=False)

    daily_frame = pd.read_parquet(daily)
    daily_frame["session_date"] = pd.to_datetime(daily_frame["session_date"])
    daily_frame["year"] = daily_frame.session_date.dt.year
    daily_frame["month"] = daily_frame.session_date.dt.to_period("M").astype(str)
    daily_frame.groupby(["strategy_id", "year"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(root / "strategy_yearly_results.csv", index=False)
    daily_frame.groupby(["strategy_id", "month"]).net_return.apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(root / "strategy_monthly_results.csv", index=False)
    finalize_existing(root)
    report = {
        "status": "PASS", "symbol_cap": CAP, "source_rows": int(check[0]),
        "corrected_rows": int(check[0]), "max_session_date": str(check[1]),
        "max_abs_final_weight": float(check[2]),
        "audit_backups": {
            "trades": str(root / "trades_pre_symbol_cap_fix.parquet"),
            "daily": str(root / "daily_strategy_returns_pre_symbol_cap_fix.parquet"),
            "summary": str(root / "strategy_summary_pre_symbol_cap_fix.csv"),
        },
    }
    (root / "symbol_cap_repair_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    if args.reports_only:
        finish_reports(args.root)
    else:
        repair(args.root)


if __name__ == "__main__":
    main()
