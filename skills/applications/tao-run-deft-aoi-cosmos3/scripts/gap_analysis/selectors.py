"""Within-group deterministic and seeded selectors."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any


def _embedding(row: dict[str, Any], field: str) -> list[float]:
    value = row.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate {row.get('id')!r} has invalid embedding JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"candidate {row.get('id')!r} has no usable embedding")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"candidate {row.get('id')!r} has an invalid embedding")
        numbers.append(float(item))
    return numbers


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("candidate embedding dimensions are inconsistent")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("candidate embeddings must have non-zero norm")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))


def order_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    selector = config["selector"]
    name = selector["name"]
    stable = sorted(rows, key=lambda row: (-float(row["selection_score"]), str(row["id"])))
    if name == "hardest":
        return stable
    if name == "stratified_random":
        digest = hashlib.sha256(group_key.encode("utf-8")).digest()
        seed = int(config["seed"]) ^ int.from_bytes(digest[:8], "big")
        shuffled = sorted(rows, key=lambda row: str(row["id"]))
        random.Random(seed).shuffle(shuffled)
        return shuffled
    if name not in {"diverse_topk", "hardness_diversity"}:
        raise ValueError(f"unsupported selector {name!r}")
    field = str(selector.get("embedding_field", "embedding"))
    embeddings = {str(row["id"]): _embedding(row, field) for row in rows}
    dimensions = {len(value) for value in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError("candidate embedding dimensions are inconsistent")
    selected: list[dict[str, Any]] = []
    remaining = list(stable)
    while remaining:
        if not selected:
            chosen = remaining[0]
        else:
            hardness_weight = float(selector.get("hardness_weight", 0.0 if name == "diverse_topk" else 0.7))
            if not 0.0 <= hardness_weight <= 1.0:
                raise ValueError("selector.hardness_weight must be in [0, 1]")

            def combined(row: dict[str, Any]) -> tuple[float, str]:
                distance = min(
                    _cosine_distance(
                        embeddings[str(row["id"])], embeddings[str(existing["id"])]
                    )
                    for existing in selected
                )
                score = hardness_weight * float(row["selection_score"]) + (1.0 - hardness_weight) * min(distance / 2.0, 1.0)
                return score, str(row["id"])

            chosen = min(remaining, key=lambda row: (-combined(row)[0], combined(row)[1]))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
