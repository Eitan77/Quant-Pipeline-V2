from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _write_frozen(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"Frozen manifest already exists: {path}")
    enriched = dict(payload)
    enriched["frozen_at"] = datetime.now(timezone.utc).isoformat()
    digest = sha256(_canonical(enriched)).hexdigest()
    wrapper = {"manifest": enriched, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return digest


def freeze_candidates(path: str | Path, candidates: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    if audit.get("exhaustiveness_status") != "PASS":
        raise RuntimeError("Discovery cannot freeze without exhaustiveness PASS")
    if any(not candidate.get("autopsy_complete", False) for candidate in candidates):
        raise RuntimeError("Every frozen candidate requires a complete Edge Autopsy")
    return _write_frozen(Path(path), {"kind": "candidate_freeze", "candidates": candidates, "audit": audit})


def freeze_portfolio(path: str | Path, portfolio: dict[str, Any]) -> str:
    required = {"included_alphas", "allocation_rule", "rebalance_rule", "risk_limits", "cost_model", "entry_exit_mappings"}
    missing = required - set(portfolio)
    if missing:
        raise ValueError(f"Portfolio freeze missing fields: {sorted(missing)}")
    return _write_frozen(Path(path), {"kind": "portfolio_freeze", "portfolio": portfolio})


def verify_manifest(path: str | Path) -> dict[str, Any]:
    wrapper = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = sha256(_canonical(wrapper["manifest"])).hexdigest()
    if actual != wrapper["sha256"]:
        raise ValueError(f"Frozen manifest hash mismatch: {path}")
    return wrapper["manifest"]
