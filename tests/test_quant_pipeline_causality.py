from __future__ import annotations

import pandas as pd
import numpy as np

from quant_pipeline.config import ScanConfig
from quant_pipeline.features import build_features
from quant_pipeline.registry import FeatureSpec
from quant_pipeline.bulk_scan import cuda_screen


def test_previous_session_return_uses_only_completed_prior_sessions():
    rows=[]
    closes=[[10.0,10.0],[20.0,20.0],[30.0,30.0]]
    for day,values in enumerate(closes,1):
        for bar,close in enumerate(values):
            start=pd.Timestamp(f"2024-01-0{day} 14:{30+bar*5}:00",tz="UTC")
            rows.append({"symbol":"A","session_date":pd.Timestamp(f"2024-01-0{day}"),"bar_start_ts":start,"bar_end_ts":start+pd.Timedelta(minutes=5),"available_at_ts":start+pd.Timedelta(minutes=5),"open":close,"high":close,"low":close,"close":close,"vwap":close,"volume":100.0})
    bars=pd.DataFrame(rows)
    spec=FeatureSpec("previous_session_return","prior completed session return","momentum")
    frame,built=build_features(bars,ScanConfig(lookbacks=[]),[spec],symbol_local=True)
    by_day=frame.groupby("session_date").previous_session_return.first()
    assert pd.isna(by_day.iloc[0])
    assert pd.isna(by_day.iloc[1])
    assert by_day.iloc[2]==1.0


def test_sparse_target_constant_subset_is_not_a_correlation(tmp_path):
    n=1000
    feature=pd.DataFrame({"session_date":np.repeat(pd.date_range("2024-01-01",periods=100).date,10),"symbol":["A"]*n,"x":np.r_[np.zeros(500),np.arange(500,dtype=np.float32)]})
    target=pd.DataFrame({"y":np.r_[np.arange(500,dtype=np.float32),np.full(500,np.nan,dtype=np.float32)]})
    result=cuda_screen(feature,target,[FeatureSpec("x","test","test")],["y"],ScanConfig(use_cuda=False,min_observations=10),pd.DataFrame(),tmp_path/"journal.csv")
    assert result.iloc[0].status=="constant_feature"
    assert pd.isna(result.iloc[0].get("raw_p"))
