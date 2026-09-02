from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

CUT = pd.Timestamp("2026-04-30")
FOLDS = (("2019_2021", "2019-06-21", "2021-12-31"), ("2022_2023", "2022-01-01", "2023-12-31"), ("2024_2026", "2024-01-01", "2026-04-30"))


def metrics(r: pd.Series) -> dict[str, float]:
    r = r.fillna(0.0).astype(float)
    eq = (1.0 + r).cumprod()
    dd = eq / pd.concat([pd.Series([1.0]), eq]).cummax().iloc[1:].to_numpy() - 1.0
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 252)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    sharpe = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
    end = int(np.argmin(dd.to_numpy()))
    peak_val = max(1.0, float(eq.iloc[: end + 1].max()))
    peak_date = r.index[0] if peak_val == 1.0 else eq.iloc[: end + 1].idxmax()
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": float(dd.min()), "pt_days": int((r.index[end] - peak_date).days)}


def load(db: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db, read_only=True)
    b = con.execute("""
      select date, symbol, arg_max(open, ingested_at) as px_open, arg_max(close, ingested_at) as px_close
      from bars_1d where adjustment='raw' and date <= DATE '2026-04-30'
      group by date, symbol
    """).fetchdf()
    m = con.execute("""
      select cast(date as date) date,symbol from qqq_pit_membership_daily
      where is_member and cast(date as date) <= DATE '2026-04-30'
    """).fetchdf()
    b.date = pd.to_datetime(b.date); m.date = pd.to_datetime(m.date)
    opens = b.pivot(index="date", columns="symbol", values="px_open").sort_index()
    closes = b.pivot(index="date", columns="symbol", values="px_close").reindex(opens.index)
    mem = m.assign(v=True).pivot(index="date", columns="symbol", values="v").reindex(index=opens.index, columns=opens.columns).fillna(False)
    return opens, closes, mem


def monthly_peers(ret: pd.DataFrame, mem: pd.DataFrame) -> dict[pd.Period, dict[str, tuple[str, float]]]:
    out = {}
    for month, locs in pd.Series(ret.index, index=ret.index).groupby(ret.index.to_period("M")):
        first = locs.iloc[0]; hist = ret.loc[:first].iloc[:-1].tail(252)
        active = mem.loc[first]; cols = [c for c in ret.columns if active.get(c, False) and hist[c].count() >= 120]
        if len(cols) < 2: continue
        raw_corr = hist[cols].corr(min_periods=120)
        arr = raw_corr.to_numpy(copy=True); np.fill_diagonal(arr, np.nan)
        corr = pd.DataFrame(arr, index=raw_corr.index, columns=raw_corr.columns)
        peers = {}
        for s in cols:
            if not corr[s].notna().any(): continue
            p = corr[s].idxmax(); x = hist[p]; y = hist[s]; ok = x.notna() & y.notna()
            if ok.sum() < 120 or corr.at[p, s] < 0.35: continue
            beta = float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / np.var(x[ok], ddof=1))
            if 0.25 <= beta <= 2.5: peers[s] = (p, beta)
        out[month] = peers
    return out


def build_signal(opens: pd.DataFrame, closes: pd.DataFrame, peers: dict, signal_days: int):
    dates = opens.index; close_ret = closes.pct_change().mask(lambda x: x.abs() > .20)
    residual = pd.DataFrame(index=dates, columns=closes.columns, dtype=float)
    peer_name: dict[tuple[pd.Timestamp, str], str] = {}
    for d in dates:
        for s, (p, beta) in peers.get(d.to_period("M"), {}).items():
            residual.at[d, s] = close_ret.at[d, s] - beta * close_ret.at[d, p]
            peer_name[(d, s)] = p
    score_base = residual.rolling(signal_days, min_periods=signal_days).sum()
    z = score_base / residual.rolling(60, min_periods=40).std().mul(np.sqrt(signal_days))
    return z, peer_name


def run(opens: pd.DataFrame, closes: pd.DataFrame, mem: pd.DataFrame, z: pd.DataFrame, peer_name: dict, hold: int, direction: int, allowed: pd.Series | None = None):
    dates = opens.index; pnl = pd.Series(0.0, index=dates); turnover = pd.Series(0.0, index=dates); entries = 0
    for i in range(60, len(dates) - hold):
        sigd = dates[i]; entryd = dates[i + 1]
        if allowed is not None and not bool(allowed.get(sigd, False)): continue
        cand = []
        for s in z.columns:
            zz = z.at[sigd, s]; p = peer_name.get((sigd, s))
            if p and pd.notna(zz) and abs(zz) >= 2.0 and mem.at[sigd, s] and mem.at[sigd, p]: cand.append((abs(zz), s, p, np.sign(zz)))
        used = set(); chosen = []
        for _, s, p, sg in sorted(cand, reverse=True):
            key = tuple(sorted((s, p)))
            if key in used: continue
            used.add(key); chosen.append((s, p, sg))
            if len(chosen) == 10: break
        if not chosen: continue
        entries += len(chosen); sleeve = 1.0 / hold / len(chosen)
        for s, p, sg in chosen:
            side_s = direction * sg; side_p = -side_s
            if any(pd.isna([opens.at[entryd, s], opens.at[entryd, p]])): continue
            for j in range(1, hold + 1):
                d = dates[i + j]; prev = entryd if j == 1 else dates[i + j - 1]
                rs = closes.at[d, s] / (opens.at[entryd, s] if j == 1 else closes.at[prev, s]) - 1
                rp = closes.at[d, p] / (opens.at[entryd, p] if j == 1 else closes.at[prev, p]) - 1
                if pd.isna(rs) or pd.isna(rp): break
                pnl.at[d] += sleeve * .5 * (side_s * rs + side_p * rp)
                if j == 1: turnover.at[d] += sleeve
                if j == hold: turnover.at[d] += sleeve
    return pnl, turnover, entries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=r"D:\AlgoResearch\data\catalog.duckdb"); ap.add_argument("--out", default=r"C:\Users\decla\Desktop\AlgoResearch\Quant Pipeline V2\results\phase2_peer_spread_through_20260430"); ap.add_argument("--filters-only", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    o, c, m = load(a.db); ret = c.pct_change().mask(lambda x: x.abs() > .20); peers = monthly_peers(ret, m)
    signals={sd:build_signal(o,c,peers,sd) for sd in ((1,) if a.filters_only else (1,5))}; rows=[]
    if a.filters_only:
        daily_ret=c.pct_change().mask(lambda x:x.abs()>.20)
        dispersion=daily_ret.std(axis=1); high_disp=dispersion > dispersion.rolling(126,min_periods=60).median().shift(1)
        qqq_up=c["QQQ"] > c["QQQ"].rolling(100,min_periods=60).mean()
        z,pn=signals[1]
        for regime,allowed in (("high_disp",high_disp),("qqq_up",qqq_up)):
            gross,turn,n=run(o,c,m,z,pn,3,1,allowed)
            for cost in (1.,2.,5.):
                p=gross-turn*(cost/10000)
                for fold,lo,hi in (("full",str(o.index.min().date()),str(CUT.date())),)+FOLDS:
                    rows.append({"spec":f"peer_momentum_s1_h3__{regime}","cost_bp_side":cost,"fold":fold,"entries":n,**metrics(p.loc[lo:hi])})
        df=pd.DataFrame(rows); df.to_csv(out/"filter_results.csv",index=False)
        print(df[(df.cost_bp_side==2)].to_string(index=False)); return
    for sd in (1, 5):
      for h in (1, 3, 5):
       for direction, name in ((-1,"reversal"),(1,"momentum")):
        z,pn=signals[sd]; gross,turn,n=run(o,c,m,z,pn,h,direction)
        for cost in (1.,2.,5.):
            p=gross-turn*(cost/10000)
            for fold,lo,hi in (("full",str(o.index.min().date()),str(CUT.date())),)+FOLDS:
                q=p.loc[lo:hi]; rows.append({"spec":f"peer_{name}_s{sd}_h{h}","cost_bp_side":cost,"fold":fold,"entries":n,**metrics(q)})
    df=pd.DataFrame(rows); df.to_csv(out/"results.csv",index=False)
    w=df[df.fold=="full"].pivot(index="spec",columns="cost_bp_side",values="cagr")
    fold=df[(df.fold!="full")&(df.cost_bp_side==2)].pivot(index="spec",columns="fold",values="cagr")
    rank=w.join(fold.add_prefix("fold_")).join(df[(df.fold=="full")&(df.cost_bp_side==2)].set_index("spec")[["entries","max_dd","pt_days","sharpe"]])
    rank["all_folds_positive"]=(rank.filter(like="fold_")>0).all(axis=1); rank=rank.sort_values(["all_folds_positive",2.0],ascending=False)
    rank.to_csv(out/"ranking.csv"); print(rank.to_string())

if __name__ == "__main__": main()
