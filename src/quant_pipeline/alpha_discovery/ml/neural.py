from __future__ import annotations

import torch


class SmallAlphaNet(torch.nn.Module):
    def __init__(self, inputs: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)
