from __future__ import annotations

import re
from .models import TimeScale


_PATTERN = re.compile(r"^(?P<n>[1-9][0-9]*)(?P<unit>m|d)$")


def parse_scale(label: str) -> TimeScale:
    normalized = str(label).strip()
    if normalized in {"session", "session_to_date", "completed_session", "EOD"}:
        return TimeScale("session_to_date", None, normalized)
    if normalized in {"context", "overnight"}:
        return TimeScale("context", None, normalized)
    match = _PATTERN.match(normalized)
    if not match:
        raise ValueError(f"Invalid time scale: {label!r}")
    value = int(match.group("n"))
    kind = "minutes" if match.group("unit") == "m" else "sessions"
    return TimeScale(kind, value, normalized)


def parse_scales(labels: list[str] | tuple[str, ...]) -> tuple[TimeScale, ...]:
    return tuple(parse_scale(label) for label in labels)


def history_sessions(scale: TimeScale, intraday_sessions: int = 1) -> int:
    if scale.kind == "sessions":
        return int(scale.value or 0)
    if scale.kind == "minutes":
        return intraday_sessions
    return 1
