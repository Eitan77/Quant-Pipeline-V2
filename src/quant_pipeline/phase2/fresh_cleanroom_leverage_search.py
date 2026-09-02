"""Clean-room conditional strategy search with an untouched internal validation split."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from quant_pipeline.phase2.data import _column_paths, _target_paths


KEYS = "symbol,session_date,bar_start_ts,decision_ts"


def q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("allow_holdout_access") is not False:
        raise RuntimeError("allow_holdout_access must be false")
    if cfg["sealed_holdout_start"] != "2026-05-01":
        raise RuntimeError("sealed holdout boundary changed")
    return cfg


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preflight(cfg: dict[str, Any], config_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    source, out = Path(cfg["source_run"]), Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    features, targets = _column_paths(str(source)), _target_paths(str(source))
    missing = sorted(set(cfg["features"]) - set(features))
    if missing:
        raise RuntimeError(f"missing configured features: {missing}")
    needed_targets = [f"fwd_return_{h}m" for h in cfg["horizons_minutes"]]
    missing_targets = sorted(set(needed_targets) - set(targets))
    if missing_targets:
        raise RuntimeError(f"missing configured targets: {missing_targets}")
    con = duckdb.connect()
    con.execute("SET threads=16")
    try:
        canonical = source / "canonical_bars.parquet"
        min_date, max_date = con.execute(
            f"SELECT min(session_date),max(session_date) FROM read_parquet({q(canonical.as_posix())})"
        ).fetchone()
        if pd.Timestamp(max_date) >= pd.Timestamp(cfg["sealed_holdout_start"]):
            raise RuntimeError(f"holdout contamination in canonical bars: {max_date}")
        unique_paths = {features[n] for n in cfg["features"]}
        unique_paths |= {targets[n] for n in needed_targets}
        for path in sorted(unique_paths, key=str):
            mx = con.execute(f"SELECT max(session_date) FROM read_parquet({q(path.as_posix())})").fetchone()[0]
            if pd.Timestamp(mx) >= pd.Timestamp(cfg["sealed_holdout_start"]):
                raise RuntimeError(f"holdout contamination in {path}: {mx}")
    finally:
        con.close()
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "clean_room": True,
        "prior_findings_used": False,
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "source": fingerprint(source / "canonical_bars.parquet"),
        "source_min_session": str(min_date),
        "source_max_session": str(max_date),
        "sealed_holdout_start": cfg["sealed_holdout_start"],
        "holdout_access": False,
        "timing": "completed 5m feature at decision_ts; target enters at first later eligible bar open",
        "portfolio": "equal-weight dollar-neutral tails; deterministic disjoint ties; 10% symbol cap",
        "execution": "next-bar open/exit-close bar model plus explicit per-side costs",
        "search_controls": "fixed features, horizons, tails, directions and single filters from YAML",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "holdout_guard_report.json").write_text(json.dumps({
        "passed": True, "max_loaded_session": str(max_date),
        "sealed_holdout_start": cfg["sealed_holdout_start"], "holdout_access": False,
    }, indent=2), encoding="utf-8")
    return features, targets


def build_context(cfg: dict[str, Any], features: dict[str, Path]) -> Path:
    out = Path(cfg["output_root"]); dest = out / "context.parquet"
    if dest.exists():
        return dest
    p31 = features["universe_return_dispersion"]
    p32 = features["market_return_6"]
    p27 = features["tod_cumulative_relative_volume_20"]
    con = duckdb.connect(); con.execute("SET threads=16")
    con.execute(f"SET temp_directory={q((out / 'duckdb_tmp').as_posix())}")
    try:
        con.execute(f"""COPY (
          WITH raw AS (
            SELECT CAST(a.session_date AS DATE) session_date,a.decision_ts,
                   CAST(any_value(date_diff('minute',a.scheduled_open,a.decision_ts)) AS INTEGER) minute_of_session,
                   avg(a.universe_return_dispersion) dispersion,
                   avg(a.universe_breadth_positive) breadth,
                   avg(b.market_return_6) market_return_6,
                   median(v.tod_cumulative_relative_volume_20) relative_volume
            FROM read_parquet({q(p31.as_posix())}) a
            JOIN read_parquet({q(p32.as_posix())}) b USING({KEYS})
            JOIN read_parquet({q(p27.as_posix())}) v USING({KEYS})
            WHERE a.analysis_eligible AND CAST(a.session_date AS DATE)<DATE {q(cfg['sealed_holdout_start'])}
            GROUP BY CAST(a.session_date AS DATE),a.decision_ts
          ), med AS (
            SELECT *,
              median(abs(market_return_6)) OVER(PARTITION BY minute_of_session ORDER BY session_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) market_abs_prior_median,
              median(dispersion) OVER(PARTITION BY minute_of_session ORDER BY session_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) dispersion_prior_median,
              median(relative_volume) OVER(PARTITION BY minute_of_session ORDER BY session_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) volume_prior_median
            FROM raw
          ) SELECT * FROM med ORDER BY decision_ts
        ) TO {q(dest.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    finally:
        con.close()
    return dest


def schedule_sql(schedule: str, grid: int) -> str:
    if schedule == "continuous":
        return f"minute_of_session>=30 AND minute_of_session%{grid}=0"
    minute = {"open_5": 5, "open_10": 10, "open_20": 20}[schedule]
    return f"minute_of_session={minute}"


def build_events(cfg: dict[str, Any], features: dict[str, Path], targets: dict[str, Path]) -> Path:
    out = Path(cfg["output_root"]); event_dir = out / "event_blocks"; event_dir.mkdir(exist_ok=True)
    grouped: dict[tuple[Path, str], list[str]] = defaultdict(list)
    for name, meta in cfg["features"].items():
        grouped[(features[name], meta["schedule"])].append(name)
    block_files: list[Path] = []
    for block_no, ((feature_path, schedule), names) in enumerate(sorted(grouped.items(), key=lambda x: (str(x[0][0]), x[0][1]))):
        for horizon in cfg["horizons_minutes"]:
            dest = event_dir / f"block_{block_no:03d}_h{horizon}.parquet"; block_files.append(dest)
            if dest.exists():
                continue
            target = f"fwd_return_{horizon}m"; target_path = targets[target]
            unions = " UNION ALL ".join(
                f"SELECT symbol,session_date,decision_ts,minute_of_session,{q(name)} feature,CAST(\"{name}\" AS DOUBLE) signal,target_return FROM base WHERE \"{name}\" IS NOT NULL"
                for name in names
            )
            tails = ",".join(f"({float(t)})" for t in cfg["tails"])
            time_clause = schedule_sql(schedule, int(cfg["event_grid_minutes"]))
            cols = ",".join(f'f.\"{name}\"' for name in names)
            con = duckdb.connect(); con.execute("SET threads=16")
            con.execute(f"SET temp_directory={q((out / 'duckdb_tmp').as_posix())}")
            try:
                con.execute(f"""COPY (
                  WITH base AS (
                    SELECT f.symbol,CAST(f.session_date AS DATE) session_date,f.decision_ts,
                           CAST(date_diff('minute',f.scheduled_open,f.decision_ts) AS INTEGER) minute_of_session,
                           {cols},CAST(t.\"{target}\" AS DOUBLE) target_return
                    FROM read_parquet({q(feature_path.as_posix())}) f
                    JOIN read_parquet({q(target_path.as_posix())}) t USING({KEYS})
                    WHERE f.analysis_eligible AND t.analysis_eligible
                      AND CAST(f.session_date AS DATE)>=DATE {q(cfg['discovery_start'])}
                      AND CAST(f.session_date AS DATE)<DATE {q(cfg['sealed_holdout_start'])}
                      AND t.\"{target}\" IS NOT NULL AND {time_clause}
                  ), longform AS ({unions}), ranked AS (
                    SELECT *,count(*) OVER(PARTITION BY feature,decision_ts) n,
                      count(DISTINCT signal) OVER(PARTITION BY feature,decision_ts) distinct_n,
                      row_number() OVER(PARTITION BY feature,decision_ts ORDER BY signal DESC,symbol) hi,
                      row_number() OVER(PARTITION BY feature,decision_ts ORDER BY signal,symbol DESC) lo
                    FROM longform
                  ), expanded AS (
                    SELECT *,tail,CAST(floor(n*tail) AS INTEGER) tail_n FROM ranked CROSS JOIN (VALUES {tails}) x(tail)
                  )
                  SELECT feature,{horizon} horizon,{q(schedule)} schedule,session_date,decision_ts,minute_of_session,tail,
                    tail_n,2*tail_n positions,
                    0.5*(avg(target_return) FILTER(WHERE hi<=tail_n)-avg(target_return) FILTER(WHERE lo<=tail_n)) spread_return,
                    avg(signal) FILTER(WHERE hi<=tail_n)-avg(signal) FILTER(WHERE lo<=tail_n) signal_gap
                  FROM expanded WHERE tail_n>=5 AND distinct_n>=10
                  GROUP BY feature,session_date,decision_ts,minute_of_session,tail,tail_n
                  HAVING count(*) FILTER(WHERE hi<=tail_n)=tail_n AND count(*) FILTER(WHERE lo<=tail_n)=tail_n
                  ORDER BY decision_ts,feature,tail
                ) TO {q(dest.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)""")
            finally:
                con.close()
            print(f"event block {block_no+1}/{len(grouped)} h{horizon} features={len(names)}", flush=True)
    combined = out / "events.parquet"
    con = duckdb.connect(); con.execute("SET threads=16")
    files = ",".join(q(x.as_posix()) for x in block_files)
    try:
        con.execute(f"COPY (SELECT * FROM read_parquet([{files}]) WHERE signal_gap>0 ORDER BY decision_ts,feature,horizon,tail) TO {q(combined.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)")
    finally:
        con.close()
    return combined


def filter_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    minute = frame.minute_of_session.to_numpy()
    if name == "all": return np.ones(len(frame), dtype=bool)
    if name == "morning": return minute <= 120
    if name == "afternoon": return minute >= 210
    if name == "market_up": return frame.market_return_6.to_numpy() > 0
    if name == "market_down": return frame.market_return_6.to_numpy() < 0
    if name == "high_market_move": return frame.market_return_6.abs().to_numpy() > frame.market_abs_prior_median.to_numpy()
    if name == "low_market_move": return frame.market_return_6.abs().to_numpy() <= frame.market_abs_prior_median.to_numpy()
    if name == "high_dispersion": return frame.dispersion.to_numpy() > frame.dispersion_prior_median.to_numpy()
    if name == "low_dispersion": return frame.dispersion.to_numpy() <= frame.dispersion_prior_median.to_numpy()
    if name == "high_volume": return frame.relative_volume.to_numpy() > frame.volume_prior_median.to_numpy()
    if name == "low_volume": return frame.relative_volume.to_numpy() <= frame.volume_prior_median.to_numpy()
    raise KeyError(name)


def calendar_for(context: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(context.session_date).drop_duplicates().sort_values())
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def metrics(daily: pd.Series) -> dict[str, float | int]:
    r = daily.fillna(0).astype(float).to_numpy(); dates = pd.DatetimeIndex(daily.index); n = len(r)
    eq = np.r_[1.0, np.cumprod(1+r)]; peaks = np.maximum.accumulate(eq); dd = eq/peaks-1
    tr = int(np.argmin(dd)); pk = int(np.argmax(eq[:tr+1])); peak_date = dates[max(0,pk-1)]; trough_date = dates[max(0,tr-1)]
    monthly = pd.Series(r,index=dates).groupby(dates.to_period("M")).apply(lambda x: np.prod(1+x)-1)
    run=longest=0
    for underwater in dd[1:]<0:
        run=run+1 if underwater else 0;longest=max(longest,run)
    std=np.std(r,ddof=1)
    return {
        "cagr": float(eq[-1]**(252/max(1,n))-1), "annual_volatility": float(std*np.sqrt(252)),
        "sharpe": float(np.mean(r)/std*np.sqrt(252)) if std>0 else np.nan,
        "maximum_drawdown": float(dd.min()), "peak_to_trough_sessions": tr-pk,
        "peak_to_trough_calendar_days": int((trough_date-peak_date).days), "max_underwater_sessions": longest,
        "profitable_day_fraction": float(np.mean(r>0)), "profitable_month_fraction": float(np.mean(monthly>0)),
        "worst_day": float(np.min(r)), "total_return": float(eq[-1]-1),
    }


def candidate_id(feature: str, horizon: int, tail: float, direction: int, filter_name: str) -> str:
    return f"{feature}__h{horizon}__tail{int(tail*100)}__dir{direction:+d}__{filter_name}"


def daily_for(frame: pd.DataFrame, calendar: pd.DatetimeIndex, direction: int, mask: np.ndarray,
              cost: float, cohort: int) -> pd.Series:
    active = frame.loc[mask, ["session_date","spread_return"]].copy()
    active["net"] = (direction*active.spread_return - 2*cost/10000)/cohort
    return active.groupby("session_date").net.sum().reindex(calendar,fill_value=0.0)


def screen(cfg: dict[str, Any]) -> None:
    out=Path(cfg["output_root"]); events=pd.read_parquet(out/"events.parquet"); context=pd.read_parquet(out/"context.parquet")
    for f in (events,context): f["session_date"]=pd.to_datetime(f.session_date)
    events["tail"]=events["tail"].astype(float)
    events=events.merge(context,on=["session_date","decision_ts","minute_of_session"],how="left",validate="many_to_one")
    train_cal=calendar_for(context,cfg["discovery_start"],cfg["train_end"]);dev_cal=calendar_for(context,cfg["development_start"],cfg["development_end"])
    family={n:m["family"] for n,m in cfg["features"].items()};rows=[]; primary=float(cfg["primary_cost_bps_per_side"])
    for (feature,horizon,tail,schedule),g in events.groupby(["feature","horizon","tail","schedule"],sort=False):
        cohort=1 if schedule!="continuous" else max(1,math.ceil(int(horizon)/int(cfg["event_grid_minutes"])))
        for direction in cfg["directions"]:
            for filter_name in cfg["filters"]:
                fm=filter_mask(g,filter_name); cid=candidate_id(feature,int(horizon),float(tail),int(direction),filter_name)
                row={"candidate_id":cid,"feature":feature,"family":family[feature],"horizon":int(horizon),"tail":float(tail),"direction":int(direction),"filter":filter_name,"schedule":schedule,"cohort_divisor":cohort,"events":int(fm.sum()),"positions":int(g.loc[fm,"positions"].sum())}
                for split,cal in (("train",train_cal),("development",dev_cal)):
                    sg=g.session_date.between(cal.min(),cal.max()).to_numpy(); daily=daily_for(g,cal,direction,fm&sg,primary,cohort); row.update({f"{split}_{k}":v for k,v in metrics(daily).items()})
                    daily2=daily_for(g,cal,direction,fm&sg,2.0,cohort); row[f"{split}_cagr_2bps"]=metrics(daily2)["cagr"]
                rows.append(row)
    result=pd.DataFrame(rows)
    result["positive_both_1bp"]=(result.train_cagr>0)&(result.development_cagr>0)
    result["positive_both_2bp"]=(result.train_cagr_2bps>0)&(result.development_cagr_2bps>0)
    result["robust_cagr"]=result[["train_cagr","development_cagr"]].min(axis=1)
    result["worst_dd"]=result[["train_maximum_drawdown","development_maximum_drawdown"]].min(axis=1)
    keys=["feature","filter","schedule","direction"]
    stable=result.positive_both_1bp.groupby([result[k] for k in keys]).transform("sum")
    result["positive_neighbor_count"]=stable
    result["screen_score"]=result.robust_cagr + 0.25*result[["train_cagr_2bps","development_cagr_2bps"]].min(axis=1) + result.worst_dd
    eligible=result[result.positive_both_1bp & (result.positive_neighbor_count>=2) & (result.events>=500) & (result.worst_dd>=-0.075)].copy()
    # A one-event opening schedule is always in the morning; its `morning`
    # wrapper is therefore identical to `all` and is not an independent spec.
    eligible=eligible[~((eligible.schedule!="continuous") & (eligible["filter"]=="morning"))].copy()
    eligible=eligible.sort_values(["screen_score","positive_both_2bp"],ascending=False)
    frozen=eligible.groupby("family",sort=False).head(5).head(20).copy()
    result.to_csv(out/"train_development_screen.csv",index=False);frozen.to_csv(out/"frozen_validation_specs.csv",index=False)
    (out/"VALIDATION_FREEZE.json").write_text(json.dumps({"frozen_before_validation":True,"freeze_revision":2,"spec_count":len(frozen),"selector":"positive train and development at +1bp; >=2 positive neighbors; >=500 events; worst DD <=7.5%; remove redundant opening-morning wrappers; top 5 per surviving family by min-split CAGR plus +2bp and DD score","spec_sha256":sha256((out/"frozen_validation_specs.csv").read_bytes()).hexdigest()},indent=2),encoding="utf-8")
    print(f"screened={len(result)} eligible={len(eligible)} frozen={len(frozen)}",flush=True)
    print(frozen[["candidate_id","train_cagr","development_cagr","train_cagr_2bps","development_cagr_2bps","worst_dd","events"]].head(30).to_string(index=False),flush=True)


def validate(cfg: dict[str, Any]) -> None:
    out=Path(cfg["output_root"]);freeze=json.loads((out/"VALIDATION_FREEZE.json").read_text());frozen_path=out/"frozen_validation_specs.csv"
    if sha256(frozen_path.read_bytes()).hexdigest()!=freeze["spec_sha256"]: raise RuntimeError("frozen spec hash mismatch")
    frozen=pd.read_csv(frozen_path); events=pd.read_parquet(out/"events.parquet");context=pd.read_parquet(out/"context.parquet")
    for f in (events,context):f["session_date"]=pd.to_datetime(f.session_date)
    events["tail"]=events["tail"].astype(float)
    events=events.merge(context,on=["session_date","decision_ts","minute_of_session"],how="left",validate="many_to_one")
    full_cal=calendar_for(context,cfg["discovery_start"],cfg["discovery_end"]); val_cal=calendar_for(context,cfg["validation_start"],cfg["discovery_end"]); recent_cal=calendar_for(context,"2026-01-01",cfg["discovery_end"])
    rows=[];daily_rows=[]
    for spec in frozen.itertuples(index=False):
        g=events[(events.feature==spec.feature)&(events.horizon==spec.horizon)&np.isclose(events["tail"],spec.tail)&(events.schedule==spec.schedule)].copy();fm=filter_mask(g,spec.filter)
        for cost in cfg["costs_bps_per_side"]:
            row={"candidate_id":spec.candidate_id,"feature":spec.feature,"family":spec.family,"horizon":spec.horizon,"tail":spec.tail,"direction":spec.direction,"filter":spec.filter,"cost_bps_per_side":cost,"events":int(fm.sum()),"positions":int(g.loc[fm,"positions"].sum())}
            for split,cal in (("full",full_cal),("validation",val_cal),("jan_apr_2026",recent_cal)):
                sm=g.session_date.between(cal.min(),cal.max()).to_numpy();d=daily_for(g,cal,spec.direction,fm&sm,cost,int(spec.cohort_divisor));row.update({f"{split}_{k}":v for k,v in metrics(d).items()})
                if cost==cfg["primary_cost_bps_per_side"] and split=="full": daily_rows.extend({"candidate_id":spec.candidate_id,"session_date":idx,"net_return":value} for idx,value in d.items())
            rows.append(row)
    result=pd.DataFrame(rows);primary=result[np.isclose(result.cost_bps_per_side,cfg["primary_cost_bps_per_side"])].copy()
    stress=result[np.isclose(result.cost_bps_per_side,2.0)][["candidate_id","validation_cagr"]].rename(columns={"validation_cagr":"validation_cagr_2bps"})
    primary=primary.merge(stress,on="candidate_id",how="left",validate="one_to_one")
    primary["passes_validation"]=(primary.validation_cagr>0)&(primary.validation_cagr_2bps>=0)&(primary.jan_apr_2026_cagr>0)&(primary.full_maximum_drawdown>=-0.05)&(primary.full_peak_to_trough_calendar_days<=45)
    result=result.merge(primary[["candidate_id","passes_validation"]],on="candidate_id",how="left")
    result.to_csv(out/"validation_cost_grid.csv",index=False);pd.DataFrame(daily_rows).to_parquet(out/"frozen_daily_returns.parquet",index=False)
    primary.sort_values(["passes_validation","validation_cagr","full_maximum_drawdown"],ascending=[False,False,False]).to_csv(out/"validation_ranking.csv",index=False)
    print(primary.sort_values(["passes_validation","validation_cagr"],ascending=False)[["candidate_id","full_cagr","validation_cagr","jan_apr_2026_cagr","full_maximum_drawdown","full_peak_to_trough_calendar_days","passes_validation"]].to_string(index=False),flush=True)


def drawdown_overlay(daily: pd.Series) -> tuple[pd.Series,pd.Series]:
    """Predeclared causal 3%/5% drawdown exposure rule, based on prior equity."""
    equity=peak=1.0; returns=[]; exposures=[]
    for value in daily.astype(float):
        drawdown=equity/peak-1
        exposure=0.25 if drawdown<=-0.05 else (0.50 if drawdown<=-0.03 else 1.0)
        returns.append(value*exposure);exposures.append(exposure)
        equity*=1+value*exposure;peak=max(peak,equity)
    return pd.Series(returns,index=daily.index),pd.Series(exposures,index=daily.index)


def causal_vol_target(daily: pd.Series,target: float,cap: float) -> tuple[pd.Series,pd.Series]:
    prior_vol=daily.rolling(60,min_periods=60).std().shift(1)*np.sqrt(252)
    requested=(target/prior_vol.clip(lower=0.04)).clip(upper=cap).fillna(1.0)
    leverage=requested.ewm(span=10,adjust=False).mean().clip(upper=cap)
    return daily*leverage,leverage


def stabilize(cfg: dict[str, Any]) -> None:
    """Evaluate only fixed equal-family ensembles and predeclared overlays."""
    out=Path(cfg["output_root"]);frozen=pd.read_csv(out/"frozen_validation_specs.csv")
    events=pd.read_parquet(out/"events.parquet");context=pd.read_parquet(out/"context.parquet")
    for frame in (events,context):frame["session_date"]=pd.to_datetime(frame.session_date)
    events["tail"]=events["tail"].astype(float)
    events=events.merge(context,on=["session_date","decision_ts","minute_of_session"],how="left",validate="many_to_one")
    full_cal=calendar_for(context,cfg["discovery_start"],cfg["discovery_end"])
    rows=[];daily_rows=[]
    for cost in cfg["costs_bps_per_side"]:
        member_daily={};family_members=defaultdict(list)
        for spec in frozen.itertuples(index=False):
            g=events[(events.feature==spec.feature)&(events.horizon==spec.horizon)&np.isclose(events["tail"],spec.tail)&(events.schedule==spec.schedule)]
            mask=filter_mask(g,spec.filter);daily=daily_for(g,full_cal,spec.direction,mask,cost,int(spec.cohort_divisor))
            member_daily[spec.candidate_id]=daily;family_members[spec.family].append(spec.candidate_id)
        for family,members in family_members.items():
            base=pd.concat([member_daily[m] for m in members],axis=1).mean(axis=1)
            variants={f"{family}_equal_frozen":(base,pd.Series(1.0,index=base.index))}
            controlled,control_exposure=drawdown_overlay(base)
            variants[f"{family}_equal_frozen__dd_control"]=(controlled,control_exposure)
            if family=="opening":
                for target in (0.06,0.08):
                    for cap in (1.5,2.0):
                        levered,leverage=causal_vol_target(base,target,cap)
                        variants[f"{family}_equal_frozen__vol{int(target*100)}__cap{cap:.1f}"]=(levered,leverage)
            for strategy_id,(daily,exposure) in variants.items():
                row={"strategy_id":strategy_id,"family":family,"cost_bps_per_side":cost,"members":len(members),"average_exposure":float(exposure.mean()),"maximum_exposure":float(exposure.max())}
                for split,start,end in (("full",cfg["discovery_start"],cfg["discovery_end"]),("validation",cfg["validation_start"],cfg["discovery_end"]),("jan_apr_2026","2026-01-01",cfg["discovery_end"])):
                    sample=daily.loc[pd.Timestamp(start):pd.Timestamp(end)];row.update({f"{split}_{k}":v for k,v in metrics(sample).items()})
                rows.append(row)
                daily_rows.extend({"strategy_id":strategy_id,"cost_bps_per_side":cost,"session_date":date,"net_return":value,"exposure":exposure.loc[date]} for date,value in daily.items())
    pd.DataFrame(rows).to_csv(out/"stabilization_results.csv",index=False)
    pd.DataFrame(daily_rows).to_parquet(out/"stabilized_daily_returns.parquet",index=False)
    table=pd.DataFrame(rows);base=table[table.strategy_id.eq("opening_equal_frozen")]
    print(base[["cost_bps_per_side","full_cagr","validation_cagr","jan_apr_2026_cagr","full_maximum_drawdown","full_peak_to_trough_calendar_days","full_sharpe"]].to_string(index=False),flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("config",type=Path);p.add_argument("--stage",choices=("all","build","screen","validate","stabilize"),default="all");a=p.parse_args();cfg=load_config(a.config)
    features,targets=preflight(cfg,a.config)
    if a.stage in ("all","build"):
        build_context(cfg,features);build_events(cfg,features,targets)
    if a.stage in ("all","screen"):screen(cfg)
    if a.stage in ("all","validate"):validate(cfg)
    if a.stage in ("all","stabilize"):stabilize(cfg)


if __name__=="__main__":main()
