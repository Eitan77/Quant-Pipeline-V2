from __future__ import annotations

import numpy as np


def purged_walk_forward_folds(n_samples: int, folds: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    boundaries = np.linspace(0, n_samples, folds + 2, dtype=int)
    output = []
    for index in range(folds):
        validation_start, validation_end = boundaries[index + 1], boundaries[index + 2]
        train_end = max(0, validation_start - embargo)
        train = np.arange(0, train_end); validation = np.arange(validation_start, validation_end)
        if len(train) and len(validation): output.append((train, validation))
    return output
