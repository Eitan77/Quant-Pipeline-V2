"""Holdout-safe Phase 2 strategy research primitives.

Phase 2 is deliberately separate from Phase 1 discovery.  It consumes only
pre-holdout, point-in-time inputs and never mutates a Phase 1 run directory.
"""

from .config import Phase2Config

__all__ = ["Phase2Config"]
