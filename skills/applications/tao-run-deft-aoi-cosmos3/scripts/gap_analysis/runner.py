"""Compose scorer, allocator, and selector components."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

from .allocators import allocate_quotas
from .config import validate_config
from .scorers import score_rows
from .selectors import order_rows


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_fields(config: dict[str, Any]) -> list[str]:
    fields = list(config["allocator"].get("group_by", []))
    subgroup = config["allocator"].get("subgroup_by")
    if isinstance(subgroup, str):
        subgroup = [subgroup]
    for field in subgroup or []:
        if field not in fields:
            fields.append(field)
    return fields


def _group_key(row: dict[str, Any], fields: list[str]) -> str:
    if not fields:
        return "__global__"
    missing = [field for field in fields if row.get(field) is None]
    if missing:
        raise ValueError(f"candidate {row.get('id')!r} is missing group fields {missing}")
    return "|".join(f"{field}={row[field]}" for field in fields)


def _eligible_fraction(
    groups: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    fraction = float(config["fraction_per_group"])
    minimum = int(config.get("min_per_group") or 0)
    eligible: dict[str, list[dict[str, Any]]] = {}
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: (-float(row["selection_score"]), str(row["id"])))
        count = math.ceil(len(rows) * fraction)
        count = min(len(rows), max(count, min(minimum, len(rows))))
        eligible[key] = ordered[:count]
    return eligible


def _expected_missing(config: dict[str, Any], present: set[str]) -> list[str]:
    expected = config.get("expected_groups", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_groups must be an object")
    explicit = expected.get("groups")
    if explicit is None:
        return []
    if not isinstance(explicit, list) or not all(isinstance(value, str) for value in explicit):
        raise ValueError("expected_groups.groups must be a string list")
    return sorted(set(explicit) - present)


def run_selection(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = validate_config(config)
    if not candidates:
        raise ValueError("gap analysis requires at least one candidate")
    ids: set[str] = set()
    for row in candidates:
        if row.get("evaluation_role") != "proxy":
            raise ValueError("gap analysis accepts only Proxy candidates for routing")
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("every gap candidate requires a non-empty id")
        if record_id in ids:
            raise ValueError(f"duplicate gap candidate id {record_id!r}")
        ids.add(record_id)

    scored = score_rows(candidates, config)
    fields = _group_fields(config)
    all_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        all_groups[_group_key(row, fields)].append(row)
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if float(row["selection_score"]) > 0.0:
            raw_groups[_group_key(row, fields)].append(row)
    missing_groups = _expected_missing(config, set(all_groups))
    if missing_groups and config["missing_group_policy"] == "error":
        raise ValueError(f"missing expected gap-analysis groups: {missing_groups}")
    groups = _eligible_fraction(raw_groups, config)
    quotas = allocate_quotas(groups, config)

    max_per_dataset = config.get("max_per_dataset")
    dataset_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    ordered_groups = {
        key: order_rows(rows, config, group_key=key) for key, rows in groups.items()
    }
    for key in sorted(ordered_groups):
        taken = 0
        for row in ordered_groups[key]:
            if taken >= quotas[key]:
                break
            dataset = str(row.get("dataset", "unknown"))
            if max_per_dataset is not None and dataset_counts[dataset] >= max_per_dataset:
                continue
            selected.append(row)
            selected_ids.add(str(row["id"]))
            dataset_counts[dataset] += 1
            taken += 1

    target_count = min(int(config["budget"]), sum(len(rows) for rows in groups.values()))
    if len(selected) < target_count:
        remaining = sorted(
            (row for rows in ordered_groups.values() for row in rows if str(row["id"]) not in selected_ids),
            key=lambda row: (-float(row["selection_score"]), str(row["id"])),
        )
        for row in remaining:
            if len(selected) >= target_count:
                break
            dataset = str(row.get("dataset", "unknown"))
            if max_per_dataset is not None and dataset_counts[dataset] >= max_per_dataset:
                continue
            selected.append(row)
            selected_ids.add(str(row["id"]))
            dataset_counts[dataset] += 1

    per_group_selected = Counter(_group_key(row, fields) for row in selected)
    selected_targets = {str(row.get("target_id", row["id"])) for row in selected}
    summary = {
        "schema_version": "gap_analysis_summary_v1",
        "config": config,
        "config_sha256": _hash(config),
        "candidate_sha256": _hash(candidates),
        "selected_ids_sha256": _hash([row["id"] for row in selected]),
        "seed": config["seed"],
        "requested_budget": config["budget"],
        "eligible_candidates": sum(len(rows) for rows in groups.values()),
        "realized_budget": len(selected),
        "budget_shortfall": max(0, int(config["budget"]) - len(selected)),
        "group_fields": fields,
        "per_group_support": {key: len(rows) for key, rows in sorted(all_groups.items())},
        "per_group_mean_weakness": {
            key: sum(float(row["selection_score"]) for row in rows) / len(rows)
            for key, rows in sorted(all_groups.items())
        },
        "per_group_eligible": {
            key: len(groups.get(key, [])) for key in sorted(all_groups)
        },
        "per_group_quota": {
            key: quotas.get(key, 0) for key in sorted(all_groups)
        },
        "per_group_selected": {
            key: per_group_selected.get(key, 0) for key in sorted(all_groups)
        },
        "per_dataset_selected": dict(sorted(dataset_counts.items())),
        "selected_unique_targets": len(selected_targets),
        "duplicate_target_rate": (
            1.0 - len(selected_targets) / len(selected) if selected else 0.0
        ),
        "caps": {
            "max_per_group": config.get("max_per_group"),
            "max_per_dataset": config.get("max_per_dataset"),
        },
        "missing_expected_groups": missing_groups,
    }
    return selected, summary
