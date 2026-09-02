from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from ..config import AlphaDiscoveryConfig
from ..models import ResearchState


@dataclass(frozen=True)
class AccessGate:
    config: AlphaDiscoveryConfig
    state: ResearchState

    def sql_predicate(self, timestamp_column: str = "session_date") -> str:
        periods = self.config.research_periods
        if self.state in {ResearchState.BUILD_ONLY, ResearchState.DISCOVERY_OPEN, ResearchState.DISCOVERY_FROZEN}:
            snapshot_start = self.config.warmup.get("snapshot_start", periods.discovery_start)
            return f"{timestamp_column} BETWEEN DATE '{snapshot_start}' AND DATE '{periods.discovery_end}'"
        if self.state in {ResearchState.REPLICATION_OPEN, ResearchState.REPLICATION_FROZEN, ResearchState.PORTFOLIO_FROZEN}:
            if not periods.allow_replication_access and self.state == ResearchState.REPLICATION_OPEN:
                raise PermissionError("Replication access is disabled by config")
            return f"{timestamp_column} BETWEEN DATE '{periods.replication_start}' AND DATE '{periods.replication_end}'"
        if not periods.allow_final_holdout_access:
            raise PermissionError("Final holdout access is disabled by config")
        return f"{timestamp_column} >= DATE '{periods.final_holdout_start}'"

    def assert_frame(self, frame: pd.DataFrame, timestamp_column: str = "session_date") -> None:
        values = pd.to_datetime(frame[timestamp_column], errors="raise")
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            values = values.dt.tz_convert("America/New_York").dt.tz_localize(None)
        values = values.dt.normalize()
        p = self.config.research_periods
        if self.state in {ResearchState.BUILD_ONLY, ResearchState.DISCOVERY_OPEN, ResearchState.DISCOVERY_FROZEN}:
            snapshot_start = pd.Timestamp(self.config.warmup.get("snapshot_start", p.discovery_start))
            if values.lt(snapshot_start).any():
                raise ValueError("Pre-snapshot row materialized during discovery")
            if values.gt(pd.Timestamp(p.discovery_end)).any():
                raise ValueError("Later-period row materialized during discovery")
        elif self.state in {ResearchState.REPLICATION_OPEN, ResearchState.REPLICATION_FROZEN, ResearchState.PORTFOLIO_FROZEN}:
            if values.lt(pd.Timestamp(p.replication_start)).any() or values.gt(pd.Timestamp(p.replication_end)).any():
                raise ValueError("Non-replication row materialized during replication")
        elif values.lt(pd.Timestamp(p.final_holdout_start)).any():
            raise ValueError("Pre-holdout row materialized during final confirmation")
