from __future__ import annotations


def require_quote_data(columns: set[str]) -> None:
    required = {"bid", "ask", "bid_size", "ask_size", "quote_ts"}
    missing = required - columns
    if missing: raise ValueError(f"Quote replay unavailable; missing columns: {sorted(missing)}")
