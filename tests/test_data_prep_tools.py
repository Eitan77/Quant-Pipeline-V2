from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "prepare_v2_catalog", Path(__file__).resolve().parents[1] / "tools" / "prepare_v2_catalog.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_symbol_change_links_constituent_transition() -> None:
    membership = pd.DataFrame({
        "symbol": ["OLD", "NEW"],
        "session_date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")],
    })
    changes = pd.DataFrame({
        "old_symbol": ["OLD"], "new_symbol": ["NEW"], "process_date": [pd.Timestamp("2024-01-06")]
    })
    groups = MODULE._identity_groups(membership, changes)
    assert groups["OLD"] == {"OLD", "NEW"}


def test_symbol_change_rejects_overlapping_ticker_reuse() -> None:
    membership = pd.DataFrame({
        "symbol": ["OLD", "OLD", "NEW"],
        "session_date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-08")],
    })
    changes = pd.DataFrame({
        "old_symbol": ["OLD"], "new_symbol": ["NEW"], "process_date": [pd.Timestamp("2024-01-06")]
    })
    groups = MODULE._identity_groups(membership, changes)
    assert "OLD" not in groups and "NEW" not in groups
