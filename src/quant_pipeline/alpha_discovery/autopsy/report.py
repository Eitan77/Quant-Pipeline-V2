from __future__ import annotations


def promotion_status(checks: dict[str, bool | None]) -> dict[str, str]:
    return {name: "PASS" if value is True else "FAIL" if value is False else "WARN" for name, value in checks.items()}
