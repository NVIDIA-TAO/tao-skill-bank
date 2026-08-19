"""Deterministic group quota allocation."""

from __future__ import annotations

from typing import Any


def allocate_quotas(
    groups: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, int]:
    name = config["allocator"]["name"]
    budget = min(int(config["budget"]), sum(len(rows) for rows in groups.values()))
    max_per_group = config.get("max_per_group")
    capacities = {
        key: min(len(rows), max_per_group) if max_per_group is not None else len(rows)
        for key, rows in groups.items()
    }
    budget = min(budget, sum(capacities.values()))
    quotas = {key: 0 for key in sorted(groups)}
    remaining = budget
    minimum = int(config.get("min_per_group") or 0)
    while remaining and any(
        quotas[key] < min(minimum, capacities[key]) for key in quotas
    ):
        progressed = False
        for key in sorted(quotas):
            if remaining == 0:
                break
            if quotas[key] < min(minimum, capacities[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    if name == "worst_group_first":
        ordering = sorted(
            groups,
            key=lambda key: (
                -sum(float(row["selection_score"]) for row in groups[key]) / len(groups[key]),
                key,
            ),
        )
        for key in ordering:
            take = min(remaining, capacities[key] - quotas[key])
            quotas[key] += take
            remaining -= take
        return quotas

    if name in {"global_topk", "equal_task_round_robin", "task_dataset_round_robin"}:
        weights = {key: 1.0 for key in groups}
    elif name == "support_proportional":
        weights = {key: float(len(rows)) for key, rows in groups.items()}
    elif name == "deficit_weighted_round_robin":
        weights = {
            key: max(
                sum(float(row["selection_score"]) for row in rows) / len(rows),
                1e-12,
            )
            for key, rows in groups.items()
        }
    else:
        raise ValueError(f"unsupported allocator {name!r}")

    # Weighted fair allocation: at each step choose the most under-served group
    # according to weight/(allocated+1). Stable group keys break exact ties.
    while remaining:
        eligible = [key for key in sorted(groups) if quotas[key] < capacities[key]]
        if not eligible:
            break
        key = min(
            eligible,
            key=lambda item: (-(weights[item] / (quotas[item] + 1)), item),
        )
        quotas[key] += 1
        remaining -= 1
    return quotas
