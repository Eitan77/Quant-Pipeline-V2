from __future__ import annotations

import numpy as np

from .surfaces import connected_plateaus


def score_plateaus(effects: np.ndarray, coverage: np.ndarray, fold_sign: np.ndarray, economic_floor: float) -> list[dict]:
    components = connected_plateaus(effects, coverage > 0, economic_floor)
    rows = []
    for component in components:
        values = np.array([effects[index] for index in component]); best = float(np.max(np.abs(values)))
        retention = float(np.median(np.abs(values)) / best) if best else 0.0
        sign_consistency = float(abs(np.mean(np.sign(values))))
        fold_overlap = float(np.mean([fold_sign[index] for index in component]))
        roughness = float(np.median(np.abs(np.diff(np.sort(values))))) if len(values) > 1 else 0.0
        score = float(np.median(np.abs(values)) * np.sqrt(len(values)) * retention * sign_consistency * fold_overlap / (1 + roughness))
        rows.append({"cells": component, "area": len(component), "median_effect": float(np.median(values)),
                     "best_effect": best, "neighbor_effect_retention": retention,
                     "neighbor_sign_consistency": sign_consistency, "chronological_fold_overlap": fold_overlap,
                     "surface_roughness": roughness, "plateau_score": score})
    return sorted(rows, key=lambda row: (-row["plateau_score"], -row["area"]))
