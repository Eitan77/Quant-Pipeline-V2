from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from .config import ScanConfig


@dataclass(frozen=True)
class BinaryCoverage:
    valid_observations:int
    signal_on_count:int
    signal_off_count:int
    signal_on_sessions:int
    signal_off_sessions:int
    signal_on_symbols:int
    signal_off_symbols:int
    signal_on_decision_timestamps:int
    signal_off_decision_timestamps:int
    activation_rate:float

    def as_dict(self)->dict:return asdict(self)


def binary_coverage(frame:pd.DataFrame,feature:str,target:str|None=None)->BinaryCoverage:
    x=pd.to_numeric(frame[feature],errors="coerce")
    eligible=frame.get("analysis_eligible",pd.Series(True,index=frame.index)).fillna(False).astype(bool)
    valid=eligible&x.notna()&np.isfinite(x)
    if target is not None:
        y=pd.to_numeric(frame[target],errors="coerce"); valid&=y.notna()&np.isfinite(y)
    if not x.loc[valid].isin([0,1]).all():raise ValueError(f"Binary feature contains values outside 0/1: {feature}")
    def state(value:int):
        part=frame.loc[valid&x.eq(value)]
        return len(part),part.session_date.nunique(),part.symbol.nunique(),part.decision_ts.nunique()
    on=state(1);off=state(0);n=int(valid.sum())
    return BinaryCoverage(n,*on[:1],*off[:1],on[1],off[1],on[2],off[2],on[3],off[3],float(on[0]/n) if n else np.nan)


def build_status(metrics:BinaryCoverage,config:ScanConfig)->tuple[str,str|None]:
    checks=[(metrics.signal_on_count,config.dual_factor_min_signal_observations,"insufficient_signal_on_observations"),(metrics.signal_off_count,config.dual_factor_min_signal_observations,"insufficient_signal_off_observations"),(metrics.signal_on_sessions,config.dual_factor_min_signal_sessions,"insufficient_signal_on_sessions"),(metrics.signal_off_sessions,config.dual_factor_min_signal_sessions,"insufficient_signal_off_sessions"),(metrics.signal_on_symbols,config.dual_factor_min_signal_symbols,"insufficient_signal_on_symbols"),(metrics.signal_off_symbols,config.dual_factor_min_signal_symbols,"insufficient_signal_off_symbols")]
    for value,minimum,reason in checks:
        if value<minimum:return "skipped",reason
    if metrics.activation_rate<config.dual_factor_min_activation_rate:return "skipped","activation_rate_too_low"
    if metrics.activation_rate>config.dual_factor_max_activation_rate:return "skipped","activation_rate_too_high"
    return "built",None


def pair_status(metrics:BinaryCoverage,config:ScanConfig)->tuple[str,str|None]:
    checks=[(metrics.signal_on_count,config.binary_min_on_observations,"insufficient_on_observations"),(metrics.signal_off_count,config.binary_min_off_observations,"insufficient_off_observations"),(metrics.signal_on_sessions,config.binary_min_on_sessions,"insufficient_on_sessions"),(metrics.signal_off_sessions,config.binary_min_off_sessions,"insufficient_off_sessions"),(metrics.signal_on_symbols,config.binary_min_on_symbols,"insufficient_on_symbols"),(metrics.signal_off_symbols,config.binary_min_off_symbols,"insufficient_off_symbols")]
    for value,minimum,reason in checks:
        if value<minimum:return "insufficient_data",reason
    return "sufficient",None
