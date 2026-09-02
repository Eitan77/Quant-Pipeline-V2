import pandas as pd
import pytest

from quant_pipeline.registry import target_registry
from quant_pipeline.table import add_targets, validate_point_in_time


def test_target_begins_after_decision_and_never_crosses_session():
    ts=pd.date_range("2026-04-30 14:30:00+00:00",periods=4,freq="5min")
    frame=pd.DataFrame({"symbol":"ABC","session_date":[pd.Timestamp("2026-04-30").date()]*4,"bar_start_ts":ts,"bar_end_ts":ts+pd.Timedelta(minutes=5),"available_at_ts":ts+pd.Timedelta(minutes=5),"open":[10,11,12,13],"close":[11,12,13,14]})
    out=add_targets(frame,target_registry())
    valid=out.entry_ts.notna()
    assert (out.loc[valid,"entry_ts"] >= out.loc[valid,"decision_ts"]).all()
    assert validate_point_in_time(out,target_registry())["target_timing_violations"] == 0


def test_validator_rejects_future_available_feature():
    frame=pd.DataFrame({"symbol":["A"],"session_date":[pd.Timestamp("2026-05-01").date()],"decision_ts":[pd.Timestamp("2026-05-01",tz="UTC")],"bar_start_ts":[pd.Timestamp("2026-05-01",tz="UTC")],"bar_end_ts":[pd.Timestamp("2026-05-01 00:05",tz="UTC")],"available_at_ts":[pd.Timestamp("2026-05-01",tz="UTC")],"entry_ts":[pd.Timestamp("2026-05-01 00:10",tz="UTC")]})
    with pytest.raises(ValueError,match="availability"):
        validate_point_in_time(frame,target_registry())
