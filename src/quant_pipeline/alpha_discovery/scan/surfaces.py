from __future__ import annotations

import numpy as np


def neighbor_edges(shape: tuple[int, ...]):
    for index in np.ndindex(shape):
        for axis in range(len(shape)):
            neighbor = list(index); neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                yield index, tuple(neighbor)


def connected_plateaus(effects: np.ndarray, valid: np.ndarray, minimum_effect: float) -> list[list[tuple[int, ...]]]:
    active = valid & np.isfinite(effects) & (np.abs(effects) >= minimum_effect)
    remaining = set(map(tuple, np.argwhere(active)))
    components = []
    adjacency: dict[tuple[int, ...], list[tuple[int, ...]]] = {node: [] for node in remaining}
    for left, right in neighbor_edges(effects.shape):
        if left in remaining and right in remaining and np.sign(effects[left]) == np.sign(effects[right]):
            adjacency[left].append(right); adjacency[right].append(left)
    while remaining:
        start = remaining.pop(); stack = [start]; component = [start]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor); component.append(neighbor)
        components.append(component)
    return components
