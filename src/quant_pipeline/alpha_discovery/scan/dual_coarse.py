from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass
class DualTileScanner:
    bins: int = 3
    device_name: str = "cuda:0"
    prefer_cuda: bool = True
    memory_fraction: float = 0.80

    def __post_init__(self) -> None:
        import torch
        self.torch = torch
        use_cuda = self.prefer_cuda and torch.cuda.is_available()
        self.device = torch.device(self.device_name if use_cuda else "cpu")

    @property
    def backend(self) -> str:
        return f"torch:{self.device}"

    def recommended_pair_block(self, observations: int, minimum: int = 16, maximum: int = 512) -> int:
        if self.device.type != "cuda":
            return min(64, maximum)
        free, _ = self.torch.cuda.mem_get_info(self.device)
        usable = free * self.memory_fraction
        bytes_per_pair = max(1, observations) * 2 + self.bins * self.bins * 24
        return max(minimum, min(maximum, int(usable // max(bytes_per_pair, 1))))

    def scan(self, bins_a: np.ndarray, bins_b: np.ndarray, target: np.ndarray, valid_a: np.ndarray | None = None, valid_b: np.ndarray | None = None) -> pd.DataFrame:
        """Scan aligned pair columns; column j represents pair j.

        Accumulates counts, target sums, and sum-squares on device and returns
        compact summaries. Full surfaces are returned only for this tile.
        """
        t = self.torch
        a = t.tensor(np.asarray(bins_a), dtype=t.int64, device=self.device)
        b = t.tensor(np.asarray(bins_b), dtype=t.int64, device=self.device)
        y = t.tensor(np.asarray(target), dtype=t.float64, device=self.device)
        if a.shape != b.shape or a.shape[0] != y.shape[0]:
            raise ValueError("Dual tile dimensions disagree")
        valid = t.isfinite(y)[:, None] & (a >= 0) & (a < self.bins) & (b >= 0) & (b < self.bins)
        if valid_a is not None: valid &= t.as_tensor(valid_a, dtype=t.bool, device=self.device)
        if valid_b is not None: valid &= t.as_tensor(valid_b, dtype=t.bool, device=self.device)
        pairs = a.shape[1]; cells = self.bins * self.bins
        offsets = t.arange(pairs, device=self.device)[None, :] * cells
        flat_index = (offsets + a * self.bins + b)[valid]
        expanded_y = y[:, None].expand(-1, pairs)[valid]
        counts = t.bincount(flat_index, minlength=pairs * cells).reshape(pairs, cells)
        sums = t.zeros(pairs * cells, dtype=t.float64, device=self.device).scatter_add_(0, flat_index, expanded_y).reshape(pairs, cells)
        sumsq = t.zeros_like(sums).reshape(-1).scatter_add_(0, flat_index, expanded_y * expanded_y).reshape(pairs, cells)
        means = sums / counts.clamp_min(1)
        overall = sums.sum(1) / counts.sum(1).clamp_min(1)
        counts_3d = counts.reshape(pairs, self.bins, self.bins)
        sums_3d = sums.reshape(pairs, self.bins, self.bins)
        means_3d = means.reshape(pairs, self.bins, self.bins)
        mean_a = sums_3d.sum(2) / counts_3d.sum(2).clamp_min(1)
        mean_b = sums_3d.sum(1) / counts_3d.sum(1).clamp_min(1)
        incremental = means_3d - mean_a[:, :, None] - mean_b[:, None, :] + overall[:, None, None]
        interaction_energy = (incremental.square() * counts_3d).sum((1, 2)) / counts.sum(1).clamp_min(1)
        best = means.masked_fill(counts == 0, -t.inf).max(1).values
        worst = means.masked_fill(counts == 0, t.inf).min(1).values
        result = pd.DataFrame({"pair_index": np.arange(pairs), "n_obs": counts.sum(1).cpu().numpy(),
                               "best_cell_effect": best.cpu().numpy(), "worst_cell_effect": worst.cpu().numpy(),
                               "max_abs_incremental_cell": incremental.abs().reshape(pairs, cells).max(1).values.cpu().numpy(),
                               "best_positive_incremental_cell": incremental.reshape(pairs, cells).max(1).values.cpu().numpy(),
                               "best_negative_incremental_cell": incremental.reshape(pairs, cells).min(1).values.cpu().numpy(),
                               "surface_interaction_energy": interaction_energy.cpu().numpy(),
                               "cell_min_count": counts.min(1).values.cpu().numpy()})
        result["surface_counts"] = list(counts.cpu().numpy())
        result["surface_means"] = list(means.cpu().numpy())
        result["incremental_surface"] = list(incremental.reshape(pairs, cells).cpu().numpy())
        if self.device.type == "cuda":
            t.cuda.synchronize(self.device)
        return result
