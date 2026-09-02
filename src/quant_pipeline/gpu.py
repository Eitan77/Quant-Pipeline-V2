from __future__ import annotations

import numpy as np


class CorrelationBackend:
    """CUDA for dense numeric reductions; CPU remains authoritative fallback."""

    def __init__(self, enabled: bool = True, device: str = "cuda:0") -> None:
        import torch
        self.torch = torch
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.device = torch.device(device if self.enabled else "cpu")

    @property
    def name(self) -> str:
        return f"torch:{self.device}"

    def correlations(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        valid = np.isfinite(x) & np.isfinite(y)
        x=x[valid]; y=y[valid]
        if len(x)<3 or np.std(x)==0 or np.std(y)==0:
            return np.nan, np.nan
        if not self.enabled or len(x)<50_000:
            from scipy.stats import rankdata
            return float(np.corrcoef(x,y)[0,1]), float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
        t=self.torch; tx=t.as_tensor(x,dtype=t.float64,device=self.device); ty=t.as_tensor(y,dtype=t.float64,device=self.device)
        def corr(a,b):
            a=a-a.mean(); b=b-b.mean(); return (a*b).sum()/t.sqrt((a*a).sum()*(b*b).sum())
        # Exact average ranks for ties are unnecessary for continuous features;
        # categorical features stay on CPU through the size gate in scanner.
        rx=t.empty_like(tx); ry=t.empty_like(ty); rx[t.argsort(tx)]=t.arange(len(tx),device=self.device,dtype=t.float64); ry[t.argsort(ty)]=t.arange(len(ty),device=self.device,dtype=t.float64)
        return float(corr(tx,ty).cpu()), float(corr(rx,ry).cpu())
