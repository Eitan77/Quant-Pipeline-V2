from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2")
OUT = ROOT / r"results\phase2_holy_shit_execution_first_through_20260430"
FB = ROOT / r"runs\phase1_final_discovery_through_20260430\blocks\features"
THRESHOLDS = Path(r"D:\AlgoResearch\research_pipeline\terra_reports\phase2_holy_shit_exact_quote_1y_v2\frozen_full_universe_thresholds.csv")
START, END = "2019-06-21", "2026-04-30"

SPECS = [
    dict(id="S01_session_momentum", feature="since_open", schedule="early", h=15, q=None, direction=1, max_side=1),
    dict(id="S02_range_location", feature="range_location", schedule="early", h=30, q=None, direction=1, max_side=1),
    dict(id="S03_opening_continuation", feature="opening_15m", schedule="late", h=15, q=.95, direction=1, max_side=3),
    dict(id="S04_extreme_momentum", feature="move_15m", schedule="early", h=30, q=.95, direction=1, max_side=3),
    dict(id="S05_volume_confirmed", feature="volume_move_15m", schedule="early", h=30, q=.95, direction=1, max_side=3),
    dict(id="S06_vwap_displacement", feature="vwap_distance", schedule="continuous15", h=15, q=.99, direction=1, max_side=3),
    dict(id="S07_market_residual", feature="residual_15m", schedule="early", h=30, q=.95, direction=1, max_side=3),
    dict(id="S08_breadth_reversal", feature="market_lag", schedule="early", h=60, q=.95, direction=-1, max_side=3),
    dict(id="S09_gap_reaction", feature="gap_reaction", schedule="early", h=60, q=.95, direction=1, max_side=3),
    dict(id="S10_bar_structure", feature="bar_close_quality", schedule="early", h=15, q=.95, direction=1, max_side=3),
]


def feature_path(n: int) -> str:
    return next(FB.glob(f"feature_{n:03d}_*.parquet")).as_posix()


def load_year(con: duckdb.DuckDBPyConnection, year: int) -> pd.DataFrame:
    lo = max(pd.Timestamp(START), pd.Timestamp(f"{year}-01-01"))
    hi = min(pd.Timestamp(END), pd.Timestamp(f"{year}-12-31"))
    return con.execute(f"""
      SELECT b.symbol,b.session_date,b.decision_ts AS bucket,b.close_adjusted,b.vwap_adjusted,
        x2.return_3,x2.relative_volume_3,x21.return_since_open,x21.overnight_gap,x21.close_location,
        x22.session_range_position,x22.minute_of_session,x22.minutes_until_close,
        x24.opening_return_15m,x24.opening_close_location_15m,x32.unreacted_market_move_5,x32.stock_minus_market_return_3,
        f15.close_adjusted AS exit15,f30.close_adjusted AS exit30,f60.close_adjusted AS exit60
      FROM read_parquet('{feature_path(0)}') b
      JOIN read_parquet('{feature_path(2)}') x2 USING(symbol,session_date,decision_ts)
      JOIN read_parquet('{feature_path(21)}') x21 USING(symbol,session_date,decision_ts)
      JOIN read_parquet('{feature_path(22)}') x22 USING(symbol,session_date,decision_ts)
      JOIN read_parquet('{feature_path(24)}') x24 USING(symbol,session_date,decision_ts)
      JOIN read_parquet('{feature_path(32)}') x32 USING(symbol,session_date,decision_ts)
      LEFT JOIN read_parquet('{feature_path(0)}') f15 ON f15.symbol=b.symbol AND f15.decision_ts=b.decision_ts+INTERVAL 15 MINUTE
      LEFT JOIN read_parquet('{feature_path(0)}') f30 ON f30.symbol=b.symbol AND f30.decision_ts=b.decision_ts+INTERVAL 30 MINUTE
      LEFT JOIN read_parquet('{feature_path(0)}') f60 ON f60.symbol=b.symbol AND f60.decision_ts=b.decision_ts+INTERVAL 60 MINUTE
      WHERE b.session_date BETWEEN DATE '{lo.date()}' AND DATE '{hi.date()}' AND b.analysis_eligible
    """).fetchdf()


def signal(d: pd.DataFrame, name: str) -> pd.Series:
    values = {
        "since_open": d.return_since_open,
        "range_location": d.session_range_position - .5,
        "opening_15m": d.opening_return_15m,
        "move_15m": d.return_3,
        "volume_move_15m": d.return_3 * d.relative_volume_3.clip(.05, 10).pow(.5),
        "vwap_distance": d.close_adjusted / d.vwap_adjusted - 1,
        "residual_15m": d.stock_minus_market_return_3,
        "market_lag": d.unreacted_market_move_5,
        "gap_reaction": d.overnight_gap * d.opening_close_location_15m,
        "bar_close_quality": d.return_3 * (2 * d.close_location - 1).abs(),
    }
    return values[name]


def select(d: pd.DataFrame, spec: dict, threshold: float | None) -> pd.DataFrame:
    z = d.copy(); z["signal"] = signal(z, spec["feature"])
    if spec["schedule"] == "early": z = z[(z.minute_of_session >= 15) & (z.minute_of_session <= 90)]
    elif spec["schedule"] == "late": z = z[(z.minute_of_session >= 240) & (z.minutes_until_close >= 60)]
    else: z = z[(z.minute_of_session >= 15) & (z.minutes_until_close >= 60) & (z.minute_of_session % 15 == 0)]
    z = z[z.signal.notna() & z[f"exit{spec['h']}"].notna()].sort_values(["bucket", "symbol"], kind="mergesort")
    if threshold is None:
        g = z.groupby("bucket"); hi = g.signal.rank(method="first", ascending=False) <= spec["max_side"]; lo = g.signal.rank(method="first", ascending=True) <= spec["max_side"]
        z = pd.concat([z[hi].assign(side=1), z[lo].assign(side=-1)])
    else:
        z = z[z.signal.abs() >= threshold].copy(); z["side"] = np.sign(z.signal) * spec["direction"]
        z["strength"] = z.signal.abs(); z["side_i"] = z.groupby(["bucket", "side"]).strength.rank(method="first", ascending=False); z = z[z.side_i <= spec["max_side"]]
        counts = z.groupby(["bucket", "side"]).size().unstack(fill_value=0)
        if -1 not in counts: counts[-1] = 0
        if 1 not in counts: counts[1] = 0
        keep = counts[[-1, 1]].min(axis=1).clip(upper=spec["max_side"]).rename("keep"); z = z.join(keep, on="bucket"); z = z[(z.keep > 0) & (z.side_i <= z.keep)]
    cadence = 15 if spec["schedule"] == "continuous15" else 5; cohort = max(1, int(np.ceil(spec["h"] / cadence)))
    z["weight"] = (.5 / z.groupby(["bucket", "side"]).symbol.transform("size")) / cohort
    z["raw_return"] = z.side * (z[f"exit{spec['h']}"] / z.close_adjusted - 1)
    return z[["session_date", "bucket", "symbol", "side", "weight", "raw_return"]]


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod(); peaks = np.maximum.accumulate(np.r_[1., eq.to_numpy()]); dd = eq.to_numpy() / peaks[1:] - 1
    end_i = int(np.argmin(dd)); prior = np.r_[1., eq.to_numpy()[:end_i + 1]]; peak_i = int(np.argmax(prior)); peak_date = r.index[0] if peak_i == 0 else r.index[peak_i - 1]
    years = max(((pd.Timestamp(r.index[-1]) - pd.Timestamp(r.index[0])).days + 1) / 365.25, 1 / 252); sd = r.std()
    return dict(cagr=float(eq.iloc[-1] ** (1 / years) - 1), total_return=float(eq.iloc[-1] - 1), sharpe=float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0., max_dd=float(dd.min()), pt_days=int((pd.Timestamp(r.index[end_i]) - pd.Timestamp(peak_date)).days), positive_day_rate=float((r > 0).mean()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); con = duckdb.connect(); con.execute("PRAGMA threads=16"); con.execute("PRAGMA memory_limit='20GB'")
    tmp = OUT / "historical_duckdb_tmp"; tmp.mkdir(exist_ok=True); con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
    threshold_frame = pd.read_csv(THRESHOLDS); ledgers = []
    for year in range(2019, 2027):
        d = load_year(con, year)
        for spec in SPECS:
            quantiles = [None] if spec["q"] is None else sorted(set([spec["q"]] + ([.99] if spec["q"] < .99 else [])))
            for quantile in quantiles:
                threshold = None if quantile is None else float(threshold_frame[(threshold_frame.mechanism_id.eq(spec["id"])) & np.isclose(threshold_frame["quantile"], quantile)].threshold.iloc[0])
                z = select(d, spec, threshold); z["candidate_id"] = spec["id"] if quantile is None else f"{spec['id']}_q{int(quantile*100)}"; ledgers.append(z)
        print(f"historical {year}: rows={len(d):,}", flush=True)
    ledger = pd.concat(ledgers, ignore_index=True); ledger.to_parquet(OUT / "historical_bar_trade_ledger.parquet", index=False)
    sessions = pd.Index(sorted(pd.to_datetime(ledger.session_date).dt.date.unique()), name="session_date"); summary = []; periods = []; daily_out = {}
    for cid, g in ledger.groupby("candidate_id"):
        for cost in (-1., -.5, 0., 1., 3., 5.):
            z = g.copy(); z["pnl"] = z.weight * (z.raw_return - 2 * cost / 10000); r = z.groupby(pd.to_datetime(z.session_date).dt.date).pnl.sum().reindex(sessions, fill_value=0)
            row = dict(candidate_id=cid, cost_bp_side=cost, trades=len(g), events=g.bucket.nunique(), **metrics(r)); summary.append(row); daily_out[f"{cid}__cost{cost:g}"] = r
            for year in range(2019, 2027):
                rr = r[pd.to_datetime(r.index).year == year]
                if len(rr): periods.append(dict(candidate_id=cid, cost_bp_side=cost, period=str(year), **metrics(rr)))
    pd.DataFrame(summary).to_csv(OUT / "historical_bar_summary.csv", index=False); pd.DataFrame(periods).to_csv(OUT / "historical_bar_annual_stats.csv", index=False); pd.DataFrame(daily_out, index=sessions).to_parquet(OUT / "historical_bar_daily_returns.parquet")
    guard = dict(discovery_cutoff=END, sealed_holdout_start="2026-05-01", max_source_session=str(pd.to_datetime(ledger.session_date).max().date()), passed=bool(pd.to_datetime(ledger.session_date).max() < pd.Timestamp("2026-05-01")))
    (OUT / "holdout_guard_report.json").write_text(json.dumps(guard, indent=2), encoding="utf-8"); print(json.dumps(guard, indent=2), flush=True)


if __name__ == "__main__":
    main()
