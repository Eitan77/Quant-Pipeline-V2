from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from typing import Any

import numpy as np

from .operators import BINARY, UNARY


@dataclass(frozen=True)
class Formula:
    expression: dict[str, Any]
    expression_hash: str
    depth: int
    binary_operators: int


class FormulaFactory:
    def __init__(self, max_depth: int = 3, max_binary_operators: int = 2) -> None:
        self.max_depth = max_depth; self.max_binary_operators = max_binary_operators

    def generate(self, feature_ids: list[str]) -> tuple[Formula, ...]:
        expressions: list[dict[str, Any]] = [{"feature": feature} for feature in feature_ids]
        expressions += [{"op": op, "args": [{"feature": feature}]} for feature in feature_ids for op in UNARY]
        for left, right in combinations(feature_ids, 2):
            expressions += [{"op": op, "args": [{"feature": left}, {"feature": right}]} for op in BINARY]
        unique: dict[str, Formula] = {}
        for expression in expressions:
            canonical = json.dumps(expression, sort_keys=True, separators=(",", ":"))
            digest = sha256(canonical.encode()).hexdigest()
            depth = 1 if "feature" in expression else 2
            binary = int(len(expression.get("args", [])) == 2)
            if depth <= self.max_depth and binary <= self.max_binary_operators:
                unique[digest] = Formula(expression, digest, depth, binary)
        return tuple(unique.values())

    def evaluate(self, formula: Formula, features: dict[str, np.ndarray]) -> np.ndarray:
        def visit(node):
            if "feature" in node:
                return features[node["feature"]]
            args = [visit(arg) for arg in node["args"]]
            return (UNARY if len(args) == 1 else BINARY)[node["op"]](*args)
        return np.asarray(visit(formula.expression), dtype=float)
