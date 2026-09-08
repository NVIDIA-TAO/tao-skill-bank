#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-first, error-stratified selection over cached image embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable


CANDIDATE_SELECTORS = ("nearest_neighbor", "coverage_stratified_hardness_v1")
SELECTOR_NAME = "coverage_stratified_hardness_v1"
MANIFEST_SCHEMA = "coverage_stratified_hardness_manifest_v1"
INVENTORY_SCHEMA = "coverage_stratified_inventory_v1"
TIER_NAMES = (
    "coverage_positive",
    "hard_positive",
    "coverage_negative",
    "fp_hard_negative",
)
DEFAULT_HARDNESS_SCHEDULE: tuple[dict[str, float], ...] = (
    {
        "coverage_positive": 0.45,
        "hard_positive": 0.15,
        "coverage_negative": 0.25,
        "fp_hard_negative": 0.15,
    },
    {
        "coverage_positive": 0.40,
        "hard_positive": 0.20,
        "coverage_negative": 0.20,
        "fp_hard_negative": 0.20,
    },
    {
        "coverage_positive": 0.35,
        "hard_positive": 0.25,
        "coverage_negative": 0.15,
        "fp_hard_negative": 0.25,
    },
    {
        "coverage_positive": 0.30,
        "hard_positive": 0.30,
        "coverage_negative": 0.15,
        "fp_hard_negative": 0.25,
    },
    {
        "coverage_positive": 0.30,
        "hard_positive": 0.30,
        "coverage_negative": 0.15,
        "fp_hard_negative": 0.25,
    },
)
DETECTION_EMPTY_PRIORS = {
    "Defect Detection": 0.418,
    "Ref_based Defect Detection": 0.424,
}


def validate_hardness_schedule(value: Any) -> list[dict[str, float]]:
    if value is None:
        return [dict(row) for row in DEFAULT_HARDNESS_SCHEDULE]
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError("hardness_schedule must contain exactly five rounds")
    output: list[dict[str, float]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, dict) or set(row) != set(TIER_NAMES):
            raise ValueError(
                f"hardness_schedule round {index} must contain exactly {TIER_NAMES}"
            )
        normalized: dict[str, float] = {}
        for name in TIER_NAMES:
            amount = row[name]
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ValueError(f"hardness_schedule round {index} {name} must be numeric")
            amount = float(amount)
            if not math.isfinite(amount) or amount < 0.0 or amount > 1.0:
                raise ValueError(
                    f"hardness_schedule round {index} {name} must be in [0, 1]"
                )
            normalized[name] = amount
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"hardness_schedule round {index} must sum to 1")
        output.append(normalized)
    return output


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(row: dict[str, Any]) -> str:
    for field in ("source_group_id", "atomic_sample_id", "parent_record_id", "filepath"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("inventory row has no stable parent identity")


def _normalized_embedding(row: dict[str, Any]) -> tuple[float, ...]:
    value = row.get("embedding")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"inventory parent {_stable_id(row)!r} has invalid embedding JSON") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"inventory parent {_stable_id(row)!r} has no embedding")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inventory parent {_stable_id(row)!r} has non-numeric embedding") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"inventory parent {_stable_id(row)!r} has non-finite embedding")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0.0:
        raise ValueError(f"inventory parent {_stable_id(row)!r} has zero-norm embedding")
    return tuple(item / norm for item in vector)


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("inventory embedding dimensions are inconsistent")
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def farthest_first_parent_order(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order unique parents by k-center while cycling visual clusters first."""

    unique: dict[str, dict[str, Any]] = {}
    vectors: dict[str, tuple[float, ...]] = {}
    for original in rows:
        row = dict(original)
        parent = _stable_id(row)
        if parent in unique:
            continue
        unique[parent] = row
        vectors[parent] = _normalized_embedding(row)
    if not unique:
        return []
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("inventory embedding dimensions are inconsistent")

    remaining = set(unique)
    selected: list[str] = []
    used_clusters: set[str] = set()
    centroid = tuple(
        sum(vector[axis] for vector in vectors.values()) / len(vectors)
        for axis in range(next(iter(dimensions)))
    )
    while remaining:
        available_clusters = {
            str(unique[parent].get("visual_cluster", parent)) for parent in remaining
        }
        candidates = [
            parent
            for parent in remaining
            if str(unique[parent].get("visual_cluster", parent)) not in used_clusters
        ]
        if not candidates:
            used_clusters.clear()
            candidates = sorted(remaining)

        def distance(parent: str) -> float:
            if not selected:
                return _squared_distance(vectors[parent], centroid)
            return min(
                _squared_distance(vectors[parent], vectors[chosen])
                for chosen in selected
            )

        chosen = sorted(candidates, key=lambda parent: (-distance(parent), parent))[0]
        selected.append(chosen)
        remaining.remove(chosen)
        used_clusters.add(str(unique[chosen].get("visual_cluster", chosen)))
        if used_clusters.issuperset(available_clusters):
            used_clusters.clear()
    return [unique[parent] for parent in selected]


def _integer_allocation(
    capacities: dict[str, int], weights: dict[str, float], budget: int
) -> dict[str, int]:
    allocation = {key: 0 for key in capacities}
    if budget <= 0 or not capacities:
        return allocation
    budget = min(budget, sum(max(0, value) for value in capacities.values()))
    positive_weights = {
        key: max(0.0, float(weights.get(key, 0.0))) for key in capacities
    }
    if not any(positive_weights.values()):
        positive_weights = {key: 1.0 for key in capacities}
    weight_total = sum(positive_weights.values())
    ideals = {
        key: budget * positive_weights[key] / weight_total for key in capacities
    }
    for _ in range(budget):
        eligible = [key for key in capacities if allocation[key] < capacities[key]]
        if not eligible:
            break
        chosen = sorted(
            eligible,
            key=lambda key: (
                -(ideals[key] - allocation[key]),
                -(positive_weights[key] / max(1, allocation[key] + 1)),
                key,
            ),
        )[0]
        allocation[chosen] += 1
    return allocation


def _bounded_integer_allocation(
    lower: dict[str, int],
    upper: dict[str, int],
    weights: dict[str, float],
    total: int,
) -> dict[str, int]:
    if set(lower) != set(upper) or any(lower[key] > upper[key] for key in lower):
        raise ValueError("invalid bounded integer allocation")
    if total < sum(lower.values()) or total > sum(upper.values()):
        raise ValueError("bounded integer allocation total is infeasible")
    slack = {key: upper[key] - lower[key] for key in lower}
    extra = _integer_allocation(slack, weights, total - sum(lower.values()))
    return {key: lower[key] + extra[key] for key in lower}


def capped_source_waterfill(
    supply: dict[str, int],
    budget: int,
    *,
    source_cap: float = 0.35,
    source_floor: float = 0.10,
) -> tuple[dict[str, int], int, list[str]]:
    """Find the largest integer source allocation satisfying cap and floor."""

    if budget <= 0:
        raise ValueError("selection budget must be positive")
    if not 0.0 < source_cap <= 1.0:
        raise ValueError("source_cap must be in (0, 1]")
    if not 0.0 <= source_floor <= source_cap:
        raise ValueError("source_floor must be in [0, source_cap]")
    normalized = {
        str(source): int(count)
        for source, count in supply.items()
        if int(count) > 0
    }
    requested = min(budget, sum(normalized.values()))
    for accepted in range(requested, 0, -1):
        cap_rows = math.floor(source_cap * accepted + 1e-12)
        if cap_rows <= 0:
            continue
        floor_rows = math.ceil(source_floor * accepted - 1e-12)
        eligible_floor = {
            source
            for source, count in normalized.items()
            if floor_rows and count >= floor_rows
        }
        allocation = {
            source: floor_rows if source in eligible_floor else 0
            for source in normalized
        }
        capacities = {
            source: min(count, cap_rows) for source, count in normalized.items()
        }
        if any(allocation[source] > capacities[source] for source in normalized):
            continue
        if sum(allocation.values()) > accepted or sum(capacities.values()) < accepted:
            continue
        while sum(allocation.values()) < accepted:
            candidates = [
                source
                for source in normalized
                if allocation[source] < capacities[source]
            ]
            if not candidates:
                break
            chosen = sorted(
                candidates,
                key=lambda source: (
                    allocation[source] / capacities[source],
                    allocation[source],
                    source,
                ),
            )[0]
            allocation[chosen] += 1
        if sum(allocation.values()) != accepted:
            continue
        if any(
            allocation[source] / accepted > source_cap + 1e-12
            for source in normalized
        ):
            continue
        if any(allocation[source] < floor_rows for source in eligible_floor):
            continue
        reasons = []
        if accepted < budget:
            reasons.append(
                f"source_capacity_reduced_budget:{budget}->{accepted}"
            )
        return allocation, accepted, reasons
    return (
        {source: 0 for source in normalized},
        0,
        [f"source_caps_infeasible_for_budget:{budget}"],
    )


def _capability_key(row: dict[str, Any]) -> tuple[str, ...]:
    task = str(row.get("task_type", "unknown"))
    phenotype = str(
        row.get("canonical_phenotype", row.get("label", "__unknown__"))
    )
    if "Classification" in task:
        return (task, phenotype)
    return (
        task,
        phenotype,
        str(row.get("log_bbox_area_quartile", "NA")),
        str(row.get("local_contrast_quartile", "NA")),
        str(row.get("gt_count_bin", "0")),
    )


def _cell_key(row: dict[str, Any]) -> str:
    return " | ".join(
        (str(row.get("task_type", "unknown")), str(row.get("source_dataset", "unknown")))
        + _capability_key(row)[1:]
    )


def _proxy_counts(row: dict[str, Any]) -> tuple[float, float]:
    evidence = row.get("defect_detection_evidence")
    evidence_types = set(evidence) if isinstance(evidence, list) else set()
    evidence = evidence if isinstance(evidence, dict) else {}

    def amount(name: str) -> float:
        value = row.get(name, evidence.get(name, 0))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return max(0.0, float(value))

    fn = amount("false_negative_count")
    fp = amount("false_positive_count")
    if fn == 0.0 and evidence_types.intersection(
        {
            "hard_positive_proxy_false_negative",
            "hard_positive_best_overlap_0_lt_iou_lte_0p5",
        }
    ):
        fn = 1.0
    if fp == 0.0 and "hard_negative_proxy_false_positive" in evidence_types:
        fp = 1.0
    if "Classification" in str(row.get("task_type", "")) and fn == fp == 0.0:
        score = row.get("sample_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool) and score < 1.0:
            fn = 1.0
    return fn, fp


def _proxy_error_model(
    proxy_rows: Iterable[dict[str, Any]], *, shrinkage: float = 2.0
) -> tuple[dict[tuple[str, ...], dict[str, float]], dict[str, float], dict[str, set[str]]]:
    raw: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"mass": 0.0, "fp_mass": 0.0, "rows": 0.0}
    )
    task_mass: Counter[str] = Counter()
    task_rows: Counter[str] = Counter()
    fp_clusters: dict[str, set[str]] = defaultdict(set)
    for original in proxy_rows:
        tasks = original.get("task_types")
        if not isinstance(tasks, list) or not tasks:
            tasks = [original.get("task_type", "unknown")]
        phenotypes = original.get("canonical_phenotypes")
        if not isinstance(phenotypes, list) or not phenotypes:
            phenotypes = [
                original.get(
                    "canonical_phenotype", original.get("label", "__unknown__")
                )
            ]
        divisor = max(1, len(tasks) * len(phenotypes))
        for task_value in tasks:
            for phenotype in phenotypes:
                row = {
                    **original,
                    "task_type": str(task_value),
                    "canonical_phenotype": str(phenotype),
                }
                task = str(task_value)
                fn, fp = _proxy_counts(row)
                fn /= divisor
                fp /= divisor
                mass = fn + 0.5 * fp
                key = _capability_key(row)
                raw[key]["mass"] += mass
                raw[key]["fp_mass"] += 0.5 * fp
                raw[key]["rows"] += 1.0 / divisor
                task_mass[task] += mass
                task_rows[task] += 1.0 / divisor
                cluster = row.get("visual_cluster")
                if fp > 0.0 and isinstance(cluster, str) and cluster:
                    fp_clusters[task].add(cluster)
    task_means = {
        task: task_mass[task] / max(1, task_rows[task]) for task in task_rows
    }
    model: dict[tuple[str, ...], dict[str, float]] = {}
    for key, values in raw.items():
        task_mean = task_means.get(key[0], 0.0)
        rows = values["rows"]
        model[key] = {
            **values,
            "error_mass": (values["mass"] + shrinkage * task_mean) / (rows + shrinkage),
        }
    return model, task_means, fp_clusters


def _cell_error(
    row: dict[str, Any],
    model: dict[tuple[str, ...], dict[str, float]],
    task_means: dict[str, float],
) -> tuple[float, float]:
    key = _capability_key(row)
    if key in model:
        return model[key]["error_mass"], model[key]["fp_mass"]
    phenotype_matches = [
        value for candidate, value in model.items() if candidate[:2] == key[:2]
    ]
    if phenotype_matches:
        return (
            sum(value["error_mass"] for value in phenotype_matches)
            / len(phenotype_matches),
            sum(value["fp_mass"] for value in phenotype_matches),
        )
    return task_means.get(key[0], 0.0), 0.0


def _tier_targets(
    task: str,
    budget: int,
    schedule: dict[str, float],
    *,
    iteration_budget: int,
    negative_rows_already_targeted: int,
) -> tuple[dict[str, int], list[str]]:
    reasons: list[str] = []
    if task not in DETECTION_EMPTY_PRIORS:
        positive_weights = {
            "coverage_positive": schedule["coverage_positive"],
            "hard_positive": schedule["hard_positive"],
        }
        targets = _integer_allocation(
            {name: budget for name in positive_weights}, positive_weights, budget
        )
        return {name: targets.get(name, 0) for name in TIER_NAMES}, reasons

    empty_prior = min(0.45, max(0.38, DETECTION_EMPTY_PRIORS[task]))
    negative_target = math.floor(budget * empty_prior + 0.5)
    global_negative_room = max(
        0, math.floor(iteration_budget * 0.25 + 1e-12) - negative_rows_already_targeted
    )
    if negative_target > global_negative_room:
        reasons.append(
            f"global_empty_no_change_cap_reduced:{negative_target}->{global_negative_room}"
        )
        negative_target = global_negative_room
    positive_target = budget - negative_target
    positive = _integer_allocation(
        {"coverage_positive": positive_target, "hard_positive": positive_target},
        {
            "coverage_positive": schedule["coverage_positive"],
            "hard_positive": schedule["hard_positive"],
        },
        positive_target,
    )
    negative = _integer_allocation(
        {"coverage_negative": negative_target, "fp_hard_negative": negative_target},
        {
            "coverage_negative": schedule["coverage_negative"],
            "fp_hard_negative": schedule["fp_hard_negative"],
        },
        negative_target,
    )
    return {
        **{name: positive.get(name, 0) for name in TIER_NAMES[:2]},
        **{name: negative.get(name, 0) for name in TIER_NAMES[2:]},
    }, reasons


def _tier_eligible(
    row: dict[str, Any], tier: str, *, fp_clusters: set[str], fp_mass: float
) -> bool:
    if "Classification" in str(row.get("task_type", "")):
        return tier in {"coverage_positive", "hard_positive"}
    negative = str(row.get("gt_count_bin", "0")) == "0"
    if tier in {"coverage_positive", "hard_positive"}:
        return not negative
    if not negative:
        return False
    if tier == "fp_hard_negative":
        if fp_mass <= 0.0:
            return False
        if fp_clusters:
            return str(row.get("visual_cluster", "")) in fp_clusters
    return True


def _source_tier_matrix(
    source_targets: dict[str, int],
    tier_targets: dict[str, int],
    rows: list[dict[str, Any]],
    *,
    proxy_model: dict[tuple[str, ...], dict[str, float]],
    task_means: dict[str, float],
    fp_clusters: set[str],
) -> dict[tuple[str, str], int]:
    matrix: Counter[tuple[str, str]] = Counter()
    source_remaining = dict(source_targets)
    tier_remaining = dict(tier_targets)
    eligible_supply: Counter[tuple[str, str]] = Counter()
    for row in rows:
        error_mass, fp_mass = _cell_error(row, proxy_model, task_means)
        del error_mass
        source = str(row["source_dataset"])
        for tier in TIER_NAMES:
            if _tier_eligible(row, tier, fp_clusters=fp_clusters, fp_mass=fp_mass):
                eligible_supply[(source, tier)] += 1
    while any(value > 0 for value in tier_remaining.values()):
        pairs = [
            (source, tier)
            for source, source_rows in source_remaining.items()
            for tier, tier_rows in tier_remaining.items()
            if source_rows > 0
            and tier_rows > 0
            and matrix[(source, tier)] < eligible_supply[(source, tier)]
        ]
        if not pairs:
            break
        source, tier = sorted(
            pairs,
            key=lambda pair: (
                -tier_remaining[pair[1]] / max(1, tier_targets[pair[1]]),
                -source_remaining[pair[0]] / max(1, source_targets[pair[0]]),
                pair[0],
                TIER_NAMES.index(pair[1]),
            ),
        )[0]
        matrix[(source, tier)] += 1
        source_remaining[source] -= 1
        tier_remaining[tier] -= 1
    return dict(matrix)


def select_coverage_stratified_candidates(
    inventory_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    *,
    budget: int,
    round_index: int,
    epochs: int,
    iteration_budget: int | None = None,
    hardness_schedule: Any = None,
    source_cap: float = 0.35,
    source_floor: float = 0.10,
    cluster_cap: float = 0.50,
    share_gap_tolerance: float = 0.05,
    inventory_hashes: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select diverse parent rows; Proxy values influence quotas, never ranking."""

    if not inventory_rows:
        raise ValueError("coverage inventory is empty")
    if not proxy_rows:
        raise ValueError("Proxy quota statistics are empty")
    if type(budget) is not int or budget <= 0:
        raise ValueError("selection budget must be a positive integer")
    if type(round_index) is not int or round_index <= 0:
        raise ValueError("round_index must be a positive integer")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not 0.0 < cluster_cap <= 1.0:
        raise ValueError("cluster_cap must be in (0, 1]")
    if not 0.0 <= share_gap_tolerance <= 1.0:
        raise ValueError("share_gap_tolerance must be in [0, 1]")
    iteration_budget = budget if iteration_budget is None else iteration_budget
    if type(iteration_budget) is not int or iteration_budget <= 0:
        raise ValueError("iteration_budget must be a positive integer")
    schedule = validate_hardness_schedule(hardness_schedule)
    round_schedule = schedule[min(round_index, len(schedule)) - 1]

    normalized: list[dict[str, Any]] = []
    seen_memberships: set[tuple[str, str, str]] = set()
    for original in inventory_rows:
        row = dict(original)
        required = ("task_type", "source_dataset", "source_group_id")
        if not all(isinstance(row.get(field), str) and row[field] for field in required):
            raise ValueError(f"inventory row requires non-empty {required}")
        row["embedding"] = list(_normalized_embedding(row))
        row.setdefault("atomic_sample_id", row["source_group_id"])
        row.setdefault("parent_record_id", row["atomic_sample_id"])
        row.setdefault("canonical_phenotype", "__unknown__")
        row.setdefault("log_bbox_area_quartile", "NA")
        row.setdefault("local_contrast_quartile", "NA")
        row.setdefault("gt_count_bin", "0")
        row.setdefault("visual_cluster", row["source_group_id"])
        key = (row["source_group_id"], row["task_type"], _cell_key(row))
        if key in seen_memberships:
            continue
        seen_memberships.add(key)
        normalized.append(row)

    proxy_model, task_means, fp_clusters = _proxy_error_model(proxy_rows)
    tasks = sorted({str(row["task_type"]) for row in normalized})
    task_capacity = {
        task: len({row["source_group_id"] for row in normalized if row["task_type"] == task})
        for task in tasks
    }
    task_weights = {
        task: math.sqrt(max(0.0, task_means.get(task, 0.0)) + 1e-6)
        for task in tasks
    }
    task_targets = _integer_allocation(task_capacity, task_weights, budget)

    selected: list[dict[str, Any]] = []
    selected_parents: set[str] = set()
    tier_target_rows: Counter[str] = Counter()
    tier_realized_rows: Counter[str] = Counter()
    cell_target_rows: Counter[str] = Counter()
    cell_realized_rows: Counter[str] = Counter()
    source_target_rows: dict[str, Counter[str]] = defaultdict(Counter)
    source_realized_rows: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_counts: Counter[tuple[str, str, str]] = Counter()
    shortages: list[str] = []
    negative_targeted = 0

    for task in tasks:
        task_rows = [row for row in normalized if row["task_type"] == task]
        requested = task_targets[task]
        if requested <= 0:
            continue
        source_supply = {
            source: len(
                {
                    row["source_group_id"]
                    for row in task_rows
                    if row["source_dataset"] == source
                }
            )
            for source in sorted({str(row["source_dataset"]) for row in task_rows})
        }
        source_targets, accepted, reasons = capped_source_waterfill(
            source_supply,
            requested,
            source_cap=source_cap,
            source_floor=source_floor,
        )
        shortages.extend(f"{task}:{reason}" for reason in reasons)
        source_target_rows[task].update(source_targets)
        tier_targets, reasons = _tier_targets(
            task,
            accepted,
            round_schedule,
            iteration_budget=iteration_budget,
            negative_rows_already_targeted=negative_targeted,
        )
        shortages.extend(f"{task}:{reason}" for reason in reasons)
        negative_targeted += tier_targets["coverage_negative"] + tier_targets["fp_hard_negative"]
        tier_target_rows.update(tier_targets)
        if accepted == 0:
            continue
        matrix = _source_tier_matrix(
            source_targets,
            tier_targets,
            task_rows,
            proxy_model=proxy_model,
            task_means=task_means,
            fp_clusters=fp_clusters.get(task, set()),
        )
        if sum(matrix.values()) < accepted:
            shortages.append(
                f"{task}:tier_supply_reduced_budget:{accepted}->{sum(matrix.values())}"
            )
        few_targets_by_bucket: dict[tuple[str, str], int] = {}
        if task in DETECTION_EMPTY_PRIORS and any(
            isinstance(row.get("gt_count"), int) for row in task_rows
        ):
            positive_buckets = {
                (source, tier): target
                for (source, tier), target in matrix.items()
                if tier in {"coverage_positive", "hard_positive"} and target > 0
            }
            encoded = {
                f"{source}\0{tier}": (source, tier)
                for source, tier in positive_buckets
            }
            lower: dict[str, int] = {}
            upper: dict[str, int] = {}
            weights: dict[str, float] = {}
            for key, (source, tier) in encoded.items():
                target = positive_buckets[(source, tier)]
                few_supply = len(
                    {
                        row["source_group_id"]
                        for row in task_rows
                        if row["source_dataset"] == source
                        and isinstance(row.get("gt_count"), int)
                        and 0 < row["gt_count"] <= 2
                    }
                )
                other_supply = len(
                    {
                        row["source_group_id"]
                        for row in task_rows
                        if row["source_dataset"] == source
                        and isinstance(row.get("gt_count"), int)
                        and row["gt_count"] > 2
                    }
                )
                lower[key] = max(0, target - other_supply)
                upper[key] = min(target, few_supply)
                weights[key] = float(target)
            positive_total = sum(positive_buckets.values())
            desired_few = math.floor(positive_total * 0.20 + 0.5)
            feasible_few = min(max(desired_few, sum(lower.values())), sum(upper.values()))
            if feasible_few != desired_few:
                shortages.append(
                    f"{task}:few_box_supply_adjusted:{desired_few}->{feasible_few}"
                )
            allocated_few = _bounded_integer_allocation(
                lower, upper, weights, feasible_few
            )
            few_targets_by_bucket = {
                encoded[key]: value for key, value in allocated_few.items()
            }
        for source in sorted(source_targets):
            # Reserve the explicitly FP-conditioned negative supply before the
            # broader coverage-negative tier can consume those same parents.
            selection_tier_order = (
                "coverage_positive",
                "hard_positive",
                "fp_hard_negative",
                "coverage_negative",
            )
            for tier in selection_tier_order:
                target = matrix.get((source, tier), 0)
                if target <= 0:
                    continue
                eligible = []
                for row in task_rows:
                    if row["source_dataset"] != source or row["source_group_id"] in selected_parents:
                        continue
                    error_mass, fp_mass = _cell_error(row, proxy_model, task_means)
                    if _tier_eligible(
                        row,
                        tier,
                        fp_clusters=fp_clusters.get(task, set()),
                        fp_mass=fp_mass,
                    ):
                        eligible.append(row)
                partitions: list[tuple[list[dict[str, Any]], int]]
                if (source, tier) in few_targets_by_bucket:
                    few_target = few_targets_by_bucket[(source, tier)]
                    partitions = [
                        (
                            [
                                row
                                for row in eligible
                                if isinstance(row.get("gt_count"), int)
                                and 0 < row["gt_count"] <= 2
                            ],
                            few_target,
                        ),
                        (
                            [
                                row
                                for row in eligible
                                if isinstance(row.get("gt_count"), int)
                                and row["gt_count"] > 2
                            ],
                            target - few_target,
                        ),
                    ]
                else:
                    partitions = [(eligible, target)]
                selection_allocations: list[
                    tuple[str, list[dict[str, Any]], int]
                ] = []
                for partition_rows, partition_target in partitions:
                    if partition_target <= 0:
                        continue
                    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    weights: dict[str, float] = {}
                    for row in partition_rows:
                        cell = _cell_key(row)
                        by_cell[cell].append(row)
                        error_mass, _ = _cell_error(row, proxy_model, task_means)
                        weights[cell] = math.sqrt(max(0.0, error_mass) + 1e-6)
                    capacities = {
                        cell: len({row["source_group_id"] for row in rows})
                        for cell, rows in by_cell.items()
                    }
                    allocation = _integer_allocation(
                        capacities, weights, partition_target
                    )
                    cell_target_rows.update(allocation)
                    selection_allocations.extend(
                        (cell, by_cell[cell], amount)
                        for cell, amount in allocation.items()
                        if amount > 0
                    )
                bucket_selected = 0
                # The cap applies to the task/tier selection, not independently
                # to each source bucket. Applying a one-row bucket's cap to the
                # global counter would incorrectly reserve a cluster for the
                # first source visited.
                cluster_limit = max(
                    1, math.ceil(cluster_cap * tier_targets[tier])
                )
                for cell, cell_rows, allocated in sorted(
                    selection_allocations, key=lambda item: item[0]
                ):
                    needed = allocated
                    order = farthest_first_parent_order(cell_rows)
                    for row in order:
                        parent = str(row["source_group_id"])
                        cluster = str(row["visual_cluster"])
                        cluster_key = (task, tier, cluster)
                        if parent in selected_parents or cluster_counts[cluster_key] >= cluster_limit:
                            continue
                        item = dict(row)
                        task_is_detection = "Detection" in task
                        evidence: list[str] = []
                        route_tier = "strict"
                        if task_is_detection and tier == "coverage_positive":
                            evidence = ["coverage_stratified_positive"]
                        elif task_is_detection and tier == "hard_positive":
                            evidence = ["hard_positive_proxy_false_negative"]
                        elif task_is_detection and tier == "coverage_negative":
                            evidence = ["calibration_empty_ground_truth"]
                            if task == "Ref_based Defect Detection":
                                evidence.append(
                                    "calibration_reference_no_change_ground_truth"
                                )
                            route_tier = "calibration"
                        elif task_is_detection and tier == "fp_hard_negative":
                            evidence = ["hard_negative_proxy_false_positive"]
                        item.update(
                            {
                                "candidate_selector": SELECTOR_NAME,
                                "selector_tier": tier,
                                "coverage_cell": cell,
                                "route_tier": route_tier,
                                "route_tiers": [route_tier],
                                "route_tier_by_task": {task: route_tier},
                                "query_task_types": [task],
                                "routed_task_types": [task],
                                "defect_detection_evidence": evidence,
                                "matched_target_ids": [],
                                "best_rank": len(selected) + 1,
                            }
                        )
                        selected.append(item)
                        selected_parents.add(parent)
                        cluster_counts[cluster_key] += 1
                        tier_realized_rows[tier] += 1
                        cell_realized_rows[cell] += 1
                        source_realized_rows[task][source] += 1
                        bucket_selected += 1
                        needed -= 1
                        if needed == 0:
                            break
                    if needed:
                        shortages.append(
                            f"{task}:{source}:{tier}:{cell}:short:{needed}"
                        )
                if bucket_selected < target:
                    shortages.append(
                        f"{task}:{source}:{tier}:realized:{bucket_selected}/{target}"
                    )

    denominator = budget
    target_share = {
        tier: tier_target_rows[tier] / denominator for tier in TIER_NAMES
    }
    realized_share = {
        tier: tier_realized_rows[tier] / denominator for tier in TIER_NAMES
    }
    tier_gaps = {
        tier: max(0.0, target_share[tier] - realized_share[tier]) for tier in TIER_NAMES
    }
    cell_gaps = {
        cell: max(0.0, target / denominator - cell_realized_rows[cell] / denominator)
        for cell, target in cell_target_rows.items()
    }
    max_gap = max([0.0, *tier_gaps.values(), *cell_gaps.values()])
    actual_by_task = {
        task: sum(source_realized_rows[task].values()) for task in tasks
    }
    source_share = {
        task: {
            source: count / actual_by_task[task]
            for source, count in sorted(source_realized_rows[task].items())
        }
        for task in tasks
        if actual_by_task[task]
    }
    source_cap_violations = {
        f"{task}:{source}": share
        for task, values in source_share.items()
        for source, share in values.items()
        if share > source_cap + 1e-12
    }
    source_floor_violations: dict[str, float] = {}
    for task, values in source_share.items():
        total = actual_by_task[task]
        floor_rows = math.ceil(source_floor * total - 1e-12)
        supplies = {
            source: len(
                {
                    row["source_group_id"]
                    for row in normalized
                    if row["task_type"] == task and row["source_dataset"] == source
                }
            )
            for source in source_target_rows[task]
        }
        for source, supply in supplies.items():
            if supply >= floor_rows and values.get(source, 0.0) + 1e-12 < source_floor:
                source_floor_violations[f"{task}:{source}"] = values.get(source, 0.0)
    if source_cap_violations:
        shortages.append(f"source_cap_violation:{sorted(source_cap_violations)}")
    if source_floor_violations:
        shortages.append(f"source_floor_violation:{sorted(source_floor_violations)}")
    few_box_positive_share: dict[str, float] = {}
    few_box_share_violations: dict[str, float] = {}
    for task in DETECTION_EMPTY_PRIORS:
        positive = [
            row
            for row in selected
            if row["task_type"] == task
            and isinstance(row.get("gt_count"), int)
            and row["gt_count"] > 0
        ]
        if len(positive) < 4:
            continue
        share = sum(row["gt_count"] <= 2 for row in positive) / len(positive)
        few_box_positive_share[task] = share
        if share < 0.15 - 1e-12 or share > 0.25 + 1e-12:
            few_box_share_violations[task] = share
    if few_box_share_violations:
        shortages.append(
            f"few_box_positive_share_outside_0.15_0.25:{few_box_share_violations}"
        )
    if max_gap > share_gap_tolerance + 1e-12:
        shortages.append(
            f"target_realized_share_gap:{max_gap:.6f}>{share_gap_tolerance:.6f}"
        )
    effective_exposure = float(epochs)
    if effective_exposure > 8.0:
        shortages.append(f"effective_exposure:{effective_exposure:.6f}>8")

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "candidate_selector": SELECTOR_NAME,
        "round": round_index,
        "hardness_schedule": schedule,
        "round_schedule": dict(round_schedule),
        "requested_budget": budget,
        "accepted_budget": sum(
            sum(values.values()) for values in source_target_rows.values()
        ),
        "selected_rows": len(selected),
        "available_unique": len({row["source_group_id"] for row in normalized}),
        "target_share": target_share,
        "realized_share": realized_share,
        "source_share": source_share,
        "cell_share": {
            cell: rows / denominator for cell, rows in sorted(cell_realized_rows.items())
        },
        "task_target_rows": dict(sorted(task_targets.items())),
        "source_target_rows": {
            task: dict(sorted(values.items())) for task, values in source_target_rows.items()
        },
        "tier_target_rows": dict(tier_target_rows),
        "tier_realized_rows": dict(tier_realized_rows),
        "cell_target_rows": dict(sorted(cell_target_rows.items())),
        "cell_realized_rows": dict(sorted(cell_realized_rows.items())),
        "unique_parent_ratio": (
            len(selected_parents) / len(selected) if selected else 0.0
        ),
        "rep": 1.0,
        "effective_exposure": effective_exposure,
        "max_target_realized_gap": max_gap,
        "share_gap_tolerance": share_gap_tolerance,
        "source_cap": source_cap,
        "source_floor": source_floor,
        "cluster_cap": cluster_cap,
        "source_cap_violations": source_cap_violations,
        "source_floor_violations": source_floor_violations,
        "shortage_reason": sorted(set(shortages)),
        "proxy_role": "quota_statistics_only",
        "query_similarity_used_for_selection": False,
        "empty_no_change_selected": sum(
            tier_realized_rows[tier] for tier in TIER_NAMES[2:]
        ),
        "empty_no_change_iteration_share": sum(
            tier_realized_rows[tier] for tier in TIER_NAMES[2:]
        )
        / iteration_budget,
        "few_box_positive_target_range": [0.15, 0.25],
        "few_box_positive_share": few_box_positive_share,
        "few_box_share_violations": few_box_share_violations,
        "inventory_hashes": dict(sorted((inventory_hashes or {}).items())),
    }
    manifest["training_allowed"] = bool(
        selected
        and max_gap <= share_gap_tolerance + 1e-12
        and not source_cap_violations
        and not source_floor_violations
        and not few_box_share_violations
        and effective_exposure <= 8.0
        and manifest["empty_no_change_iteration_share"] <= 0.25 + 1e-12
    )
    return selected, manifest


def require_coverage_training_eligible(manifest: dict[str, Any]) -> None:
    if manifest.get("candidate_selector") != SELECTOR_NAME:
        raise ValueError("coverage training gate received the wrong selector manifest")
    if manifest.get("training_allowed") is not True:
        gap = float(manifest.get("max_target_realized_gap", 1.0))
        reasons = manifest.get("shortage_reason", [])
        raise ValueError(
            f"coverage selector share gap/training gate failed: "
            f"max_gap={gap:.6f}, shortage_reason={reasons}"
        )


def write_inventory_cache(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    hashes: dict[str, str],
) -> dict[str, str]:
    """Atomically cache an inventory parquet with source hashes in metadata."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for the coverage inventory cache") from exc
    if not rows:
        raise ValueError("cannot cache an empty coverage inventory")
    normalized_hashes = {str(key): str(value) for key, value in hashes.items()}
    payload_sha = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        b"schema_version": INVENTORY_SCHEMA.encode("utf-8"),
        b"inventory_payload_sha256": payload_sha.encode("ascii"),
        b"input_hashes": json.dumps(normalized_hashes, sort_keys=True).encode("utf-8"),
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(metadata)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **normalized_hashes,
        "inventory_payload_sha256": payload_sha,
        "inventory_parquet_sha256": file_sha256(path),
    }


def read_inventory_cache(
    path: pathlib.Path, *, expected_hashes: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for the coverage inventory cache") from exc
    path = path.expanduser().resolve()
    if not path.is_file():
        return None
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_version", b"").decode("utf-8") != INVENTORY_SCHEMA:
        return None
    try:
        stored_hashes = json.loads(metadata[b"input_hashes"].decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    normalized_expected = {str(key): str(value) for key, value in expected_hashes.items()}
    if stored_hashes != normalized_expected:
        return None
    rows = table.to_pylist()
    payload_sha = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    stored_payload = metadata.get(b"inventory_payload_sha256", b"").decode("ascii")
    if payload_sha != stored_payload:
        return None
    return rows, {
        **stored_hashes,
        "inventory_payload_sha256": payload_sha,
        "inventory_parquet_sha256": file_sha256(path),
    }


def load_or_build_coverage_inventory(
    path: pathlib.Path,
    *,
    expected_hashes: dict[str, str],
    builder: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    existed = path.expanduser().exists()
    cached = read_inventory_cache(path, expected_hashes=expected_hashes)
    if cached is not None:
        rows, hashes = cached
        return rows, hashes, "reused"
    rows = builder()
    hashes = write_inventory_cache(path, rows, hashes=expected_hashes)
    return rows, hashes, "rebuilt" if existed else "built"
