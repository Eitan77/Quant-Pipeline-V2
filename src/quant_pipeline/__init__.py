"""Clean-room univariate market-anomaly research system.

This package deliberately contains no strategy, portfolio, or prior-research
results.  It builds causal feature/target tables from canonical bars.
"""

from .config import ScanConfig

__all__ = ["ScanConfig"]
