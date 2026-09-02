from __future__ import annotations

from hashlib import sha256


def expand_contexts(stable_structures: list[dict], context_feature_ids: list[str]) -> list[dict]:
    rows = []
    for structure in stable_structures:
        for context in context_feature_ids:
            features = tuple(structure["feature_ids"]) + (context,)
            rows.append({"interaction_id": sha256("\0".join(features).encode()).hexdigest()[:24],
                         "feature_ids": features, "source_id": structure["source_id"], "context_id": context})
    return rows
