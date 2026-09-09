#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically materialize a task-level repetition blend."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import Counter
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 runtime fallback
    import tomli as tomllib


SCHEMA_VERSION = "repetition_blend_manifest_v2"
POLICIES = ("deficit_proportional", "explicit")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "policy": "deficit_proportional",
    "rep_min": 0.5,
    "rep_max": 3.0,
    "budget_multiplier": 1.0,
    "share_gap_tolerance": 0.05,
    "redistribute": True,
    "never_repeat_empty_gt": True,
    "explicit_multipliers": {},
    "row_cap": None,
    "seed": None,
}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_repetition_config(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("repetition blend config must be an object")
    if "repetition_blend" in value:
        value = value["repetition_blend"]
        if not isinstance(value, dict):
            raise ValueError("repetition_blend must be an object")
    value = dict(value)
    if "multipliers" in value:
        if "explicit_multipliers" in value:
            raise ValueError("use only one of multipliers or explicit_multipliers")
        value["explicit_multipliers"] = value.pop("multipliers")
    unknown = set(value) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown repetition blend keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **value}
    if type(config["enabled"]) is not bool:
        raise ValueError("repetition blend enabled must be boolean")
    if config["policy"] not in POLICIES:
        raise ValueError(f"unsupported repetition policy {config['policy']!r}")
    for name in (
        "rep_min",
        "rep_max",
        "budget_multiplier",
        "share_gap_tolerance",
    ):
        number = config[name]
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise ValueError(f"repetition {name} must be numeric")
        config[name] = float(number)
    if not all(
        math.isfinite(config[name])
        for name in (
            "rep_min",
            "rep_max",
            "budget_multiplier",
            "share_gap_tolerance",
        )
    ):
        raise ValueError("repetition numeric controls must be finite")
    if config["rep_min"] <= 0 or config["rep_max"] <= 0:
        raise ValueError("repetition multipliers must be positive")
    if config["rep_min"] > config["rep_max"]:
        raise ValueError("repetition rep_min cannot exceed rep_max")
    if config["budget_multiplier"] <= 0:
        raise ValueError("repetition budget_multiplier must be positive")
    if not 0 <= config["share_gap_tolerance"] <= 1:
        raise ValueError("repetition share_gap_tolerance must be in [0, 1]")
    if type(config["redistribute"]) is not bool:
        raise ValueError("repetition redistribute must be boolean")
    if type(config["never_repeat_empty_gt"]) is not bool:
        raise ValueError("never_repeat_empty_gt must be boolean")
    if config["row_cap"] is not None and (
        type(config["row_cap"]) is not int or config["row_cap"] <= 0
    ):
        raise ValueError("repetition row_cap must be a positive integer")
    if config["seed"] is not None and type(config["seed"]) is not int:
        raise ValueError("repetition seed must be an integer")
    multipliers = config["explicit_multipliers"]
    if not isinstance(multipliers, dict):
        raise ValueError("explicit_multipliers must be an object")
    normalized: dict[str, float] = {}
    for task, multiplier in multipliers.items():
        if not isinstance(task, str) or not task.strip():
            raise ValueError("explicit multiplier task names must be non-empty strings")
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
            raise ValueError(f"explicit multiplier for {task!r} must be numeric")
        if not math.isfinite(float(multiplier)) or float(multiplier) <= 0:
            raise ValueError(f"explicit multiplier for {task!r} must be positive")
        normalized[task] = float(multiplier)
    config["explicit_multipliers"] = dict(sorted(normalized.items()))
    if config["policy"] == "explicit" and not normalized:
        raise ValueError("explicit repetition policy requires explicit_multipliers")
    return config


def load_repetition_config(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.casefold() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    elif resolved.suffix.casefold() in {".toml", ".tml"}:
        with resolved.open("rb") as stream:
            payload = tomllib.load(stream)
    else:
        raise ValueError("repetition config must use .toml or .json")
    return validate_repetition_config(payload)


def merge_repetition_config(
    base: dict[str, Any] | None, overrides: dict[str, Any]
) -> dict[str, Any]:
    config = validate_repetition_config(base)
    merged = {**config, **{key: value for key, value in overrides.items() if value is not None}}
    if "explicit_multipliers" in overrides and overrides["explicit_multipliers"] is not None:
        merged["explicit_multipliers"] = overrides["explicit_multipliers"]
    return validate_repetition_config(merged)


def parse_explicit_multipliers(values: list[str] | None) -> dict[str, float] | None:
    if values is None:
        return None
    result: dict[str, float] = {}
    for value in values:
        task, separator, multiplier_text = value.partition("=")
        task = task.strip()
        if not separator or not task or task in result:
            raise ValueError(
                "--repetition-explicit-multiplier requires unique TASK=MULTIPLIER values"
            )
        try:
            result[task] = float(multiplier_text)
        except ValueError as exc:
            raise ValueError(f"invalid repetition multiplier {value!r}") from exc
    return result


def deficit_weights_from_gap_summary(
    summary: dict[str, Any] | None, tasks: list[str] | tuple[str, ...]
) -> tuple[dict[str, float], str]:
    ordered_tasks = sorted(set(tasks))
    equal = {task: 1.0 for task in ordered_tasks}
    if not isinstance(summary, dict):
        return equal, "equal_fallback"
    direct = summary.get("task_deficit_weights") or summary.get("deficit_weights")
    if isinstance(direct, dict):
        try:
            weights = {
                task: float(direct[task])
                for task in ordered_tasks
                if task in direct and float(direct[task]) >= 0
            }
        except (TypeError, ValueError):
            weights = {}
        if weights and any(weights.values()):
            return {task: weights.get(task, 1.0) for task in ordered_tasks}, "gap_summary"

    means = summary.get("per_group_mean_weakness")
    supports = summary.get("per_group_support", {})
    if not isinstance(means, dict) or not isinstance(supports, dict):
        return equal, "equal_fallback"
    totals: Counter[str] = Counter()
    observations: Counter[str] = Counter()
    for group, raw_mean in means.items():
        if not isinstance(group, str):
            continue
        fields = dict(
            item.split("=", 1)
            for item in group.split("|")
            if "=" in item
        )
        task = fields.get("task_type")
        if task not in equal:
            continue
        try:
            mean = max(0.0, float(raw_mean))
            support = max(0, int(supports.get(group, 1)))
        except (TypeError, ValueError):
            continue
        totals[task] += mean * support
        observations[task] += support
    weights = {
        task: totals[task] / observations[task]
        for task in ordered_tasks
        if observations[task]
    }
    if not weights or not any(weights.values()):
        return equal, "equal_fallback"
    return {task: weights.get(task, 1.0) for task in ordered_tasks}, "gap_summary"


def _allocate_cap(
    requested: dict[str, int],
    *,
    shares: dict[str, float],
    mandatory: dict[str, int],
    row_cap: int,
) -> dict[str, int]:
    if sum(mandatory.values()) > row_cap:
        raise ValueError(
            "row_cap is smaller than the empty-ground-truth rows protected from repetition"
        )
    if sum(requested.values()) <= row_cap:
        return requested
    allocated = dict(mandatory)
    remaining = row_cap - sum(allocated.values())
    while remaining:
        eligible = [task for task in sorted(requested) if allocated[task] < requested[task]]
        if not eligible:
            break
        task = min(
            eligible,
            key=lambda name: (
                -(shares[name] / (allocated[name] + 1)),
                -(requested[name] - allocated[name]),
                name,
            ),
        )
        allocated[task] += 1
        remaining -= 1
    return allocated


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _align_down(value: int, multiple: int) -> int:
    return value - value % multiple


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _deficit_budget(
    *,
    available_total: int,
    budget_multiplier: float,
    row_cap: int,
) -> int:
    requested = _round_half_up(available_total * budget_multiplier)
    budget = min(row_cap, requested)
    if budget <= 0:
        raise ValueError("repetition share budget must be positive")
    return budget


def _count_bounds(
    *,
    available: dict[str, int],
    empty_rows: dict[str, int],
    mandatory: dict[str, int],
    rep_min: float,
    rep_max: float,
    never_repeat_empty_gt: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    for task, count in available.items():
        task_upper = int(math.floor(count * rep_max + 1e-12))
        if never_repeat_empty_gt and empty_rows[task] == count:
            task_upper = min(task_upper, count)
        if mandatory[task] > task_upper:
            raise ValueError(
                f"repetition rep_max cannot retain mandatory rows for {task!r}"
            )
        task_lower = max(mandatory[task], int(math.ceil(count * rep_min - 1e-12)))
        lower[task] = min(task_lower, task_upper)
        upper[task] = task_upper
    return lower, upper


def _allocate_weighted_total(
    *,
    shares: dict[str, float],
    lower: dict[str, int],
    upper: dict[str, int],
    total: int,
) -> dict[str, int]:
    if sum(lower.values()) > total or sum(upper.values()) < total:
        raise ValueError("repetition total is outside the feasible multiplier bounds")
    allocated = dict(lower)
    remaining = total - sum(allocated.values())
    while remaining:
        eligible = [task for task in sorted(allocated) if allocated[task] < upper[task]]
        if not eligible:
            raise ValueError("repetition bounds cannot satisfy the requested budget")
        task = min(
            eligible,
            key=lambda name: (
                (allocated[name] + 1) / shares[name]
                if shares[name] > 0
                else math.inf,
                name,
            ),
        )
        allocated[task] += 1
        remaining -= 1
    return allocated


def _feasible_aligned_total(
    *,
    requested: int,
    lower_total: int,
    upper_total: int,
    row_cap: int,
    row_multiple: int,
) -> int:
    upper_total = min(upper_total, row_cap)
    if lower_total > upper_total:
        raise ValueError("repetition bounds exceed row_cap")
    bounded = min(upper_total, max(lower_total, requested))
    total = _align_down(bounded, row_multiple)
    if total < lower_total:
        total = _align_up(lower_total, row_multiple)
    if not lower_total <= total <= upper_total or total <= 0:
        raise ValueError(
            "repetition bounds cannot form one complete row_multiple group"
        )
    return total


def plan_repetition(
    *,
    available_rows: dict[str, int],
    row_cap: int,
    deficit_weights: dict[str, float] | None = None,
    policy: str = "deficit_proportional",
    rep_min: float = 0.5,
    rep_max: float = 3.0,
    budget_multiplier: float = 1.0,
    redistribute: bool = True,
    explicit_multipliers: dict[str, float] | None = None,
    empty_rows: dict[str, int] | None = None,
    never_repeat_empty_gt: bool = True,
    minimum_rows: dict[str, int] | None = None,
    row_multiple: int = 1,
) -> dict[str, dict[str, Any]]:
    if type(row_cap) is not int or row_cap <= 0:
        raise ValueError("row_cap must be a positive integer")
    if type(row_multiple) is not int or row_multiple <= 0:
        raise ValueError("row_multiple must be a positive integer")
    config = validate_repetition_config(
        {
            "enabled": True,
            "policy": policy,
            "rep_min": rep_min,
            "rep_max": rep_max,
            "budget_multiplier": budget_multiplier,
            "redistribute": redistribute,
            "never_repeat_empty_gt": never_repeat_empty_gt,
            "explicit_multipliers": explicit_multipliers or {},
        }
    )
    available: dict[str, int] = {}
    for task, count in available_rows.items():
        if not isinstance(task, str) or not task:
            raise ValueError("available row task names must be non-empty strings")
        if type(count) is not int or count < 0:
            raise ValueError(f"available rows for {task!r} must be a non-negative integer")
        if count:
            available[task] = count
    if not available:
        raise ValueError("repetition blend requires at least one available row")
    empty_counts = {task: 0 for task in available}
    for task, count in (empty_rows or {}).items():
        if task not in available:
            if count:
                raise ValueError(f"empty rows supplied for unavailable task {task!r}")
            continue
        if type(count) is not int or not 0 <= count <= available[task]:
            raise ValueError(f"invalid empty row count for {task!r}")
        empty_counts[task] = count
    mandatory = {task: 0 for task in available}
    for task, count in (minimum_rows or {}).items():
        if task not in available:
            if count:
                raise ValueError(f"minimum rows supplied for unavailable task {task!r}")
            continue
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid minimum row count for {task!r}")
        mandatory[task] = max(mandatory[task], count)

    lower, upper = _count_bounds(
        available=available,
        empty_rows=empty_counts,
        mandatory=mandatory,
        rep_min=config["rep_min"],
        rep_max=config["rep_max"],
        never_repeat_empty_gt=never_repeat_empty_gt,
    )

    if policy == "deficit_proportional":
        raw_weights: dict[str, float] = {}
        supplied = deficit_weights if isinstance(deficit_weights, dict) else {}
        for task in available:
            raw = supplied.get(task, 1.0)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise ValueError(f"deficit weight for {task!r} must be numeric")
            weight = float(raw)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"deficit weight for {task!r} must be finite and non-negative")
            raw_weights[task] = weight
        if not any(raw_weights.values()):
            raw_weights = {task: 1.0 for task in available}
        total_weight = sum(raw_weights.values())
        shares = {task: raw_weights[task] / total_weight for task in available}
        budget = _deficit_budget(
            available_total=sum(available.values()),
            budget_multiplier=config["budget_multiplier"],
            row_cap=row_cap,
        )
        target_rows = {task: budget * shares[task] for task in available}
        if config["redistribute"]:
            emitted_total = _feasible_aligned_total(
                requested=budget,
                lower_total=sum(lower.values()),
                upper_total=sum(upper.values()),
                row_cap=row_cap,
                row_multiple=row_multiple,
            )
            emitted_counts = _allocate_weighted_total(
                shares=shares,
                lower=lower,
                upper=upper,
                total=emitted_total,
            )
        else:
            requested_counts = {
                task: min(
                    upper[task],
                    max(lower[task], _round_half_up(target_rows[task])),
                )
                for task in available
            }
            emitted_total = _align_down(
                min(row_cap, sum(requested_counts.values())), row_multiple
            )
            if emitted_total < sum(lower.values()) or emitted_total <= 0:
                raise ValueError(
                    "repetition bounds cannot form one complete row_multiple group"
                )
            emitted_counts = _allocate_cap(
                requested_counts,
                shares=shares,
                mandatory=lower,
                row_cap=emitted_total,
            )
        weights = raw_weights
    else:
        requested_rep = {
            task: min(
                config["rep_max"],
                max(
                    config["rep_min"],
                    config["explicit_multipliers"].get(task, 1.0),
                ),
            )
            for task in available
        }
        desired = {
            task: available[task] * requested_rep[task] for task in available
        }
        desired_total = sum(desired.values())
        shares = {task: desired[task] / desired_total for task in available}
        target_rows = desired
        weights = {
            task: config["explicit_multipliers"].get(task, 1.0)
            for task in available
        }
        requested_counts = {
            task: min(
                upper[task],
                max(
                    lower[task],
                    _round_half_up(available[task] * requested_rep[task]),
                ),
            )
            for task in available
        }
        if not any(requested_counts.values()):
            requested_counts[max(available, key=lambda task: (shares[task], task))] = 1
        emitted_total = _align_down(
            min(row_cap, sum(requested_counts.values())), row_multiple
        )
        if emitted_total <= 0 or emitted_total < sum(lower.values()):
            raise ValueError(
                "repetition blend cannot form one complete row_multiple group"
            )
        emitted_counts = _allocate_cap(
            requested_counts,
            shares=shares,
            mandatory=lower,
            row_cap=emitted_total,
        )
    plan: dict[str, dict[str, Any]] = {}
    available_total = sum(available.values())
    for task in sorted(available):
        emitted = emitted_counts[task]
        empty = (empty_rows or {}).get(task, 0)
        unique_emitted = min(available[task], emitted)
        plan[task] = {
            "available": available[task],
            "deficit_weight": weights[task],
            "observed_share_before": available[task] / available_total,
            "target_share": shares[task],
            "target_rows": target_rows[task],
            "rep": emitted / available[task],
            "realized_rep": emitted / available[task],
            "repeated_rows": emitted,
            "additional_repetitions": max(0, emitted - unique_emitted),
            "dropped_rows": max(0, available[task] - unique_emitted),
            "empty_rows": empty,
            "repeated_empty_rows": 0 if never_repeat_empty_gt else None,
        }
    return plan


def _assistant_payload(record: dict[str, Any]) -> Any:
    for message in reversed(record.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def is_empty_ground_truth(record: dict[str, Any]) -> bool:
    return "Detection" in str(record.get("task_type", "")) and _assistant_payload(record) == []


def _seeded_order(
    indices: list[int],
    *,
    rows: list[dict[str, Any]],
    seed: int,
    task: str,
    phase: str,
) -> list[int]:
    return sorted(
        indices,
        key=lambda index: (
            hashlib.sha256(
                f"{seed}\0{task}\0{phase}\0{index}\0{_fingerprint(rows[index])}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            index,
        ),
    )


def apply_repetition_blend(
    rows: list[dict[str, Any]],
    *,
    row_cap: int,
    config: dict[str, Any] | None,
    deficit_weights: dict[str, float] | None = None,
    seed: int | None = None,
    empty_predicate: Callable[[dict[str, Any]], bool] = is_empty_ground_truth,
    deficit_weight_source: str | None = None,
    mandatory_indices: set[int] | None = None,
    row_multiple: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = validate_repetition_config(config)
    if seed is None:
        seed = resolved["seed"] if resolved["seed"] is not None else 17
    elif resolved["seed"] is not None and seed != resolved["seed"]:
        raise ValueError("repetition seed differs between config and materializer")
    if type(seed) is not int:
        raise ValueError("repetition seed must be an integer")
    if type(row_cap) is not int or row_cap <= 0:
        raise ValueError("row_cap must be a positive integer")
    if type(row_multiple) is not int or row_multiple <= 0:
        raise ValueError("row_multiple must be a positive integer")
    if not rows:
        raise ValueError("repetition blend requires at least one row")
    resolved["row_cap"] = row_cap
    resolved["seed"] = seed
    tasks = [str(row.get("task_type") or "unknown") for row in rows]
    row_fingerprints = [_fingerprint(row) for row in rows]
    required_indices = set(mandatory_indices or ())
    if any(type(index) is not int or not 0 <= index < len(rows) for index in required_indices):
        raise ValueError("mandatory repetition indices must refer to available rows")
    available = Counter(tasks)
    empty_flags = [empty_predicate(row) for row in rows]
    empty = Counter(task for task, is_empty in zip(tasks, empty_flags) if is_empty)
    if not resolved["enabled"]:
        total = len(rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "policy": resolved["policy"],
            "seed": seed,
            "configuration": {**resolved, "row_cap": row_cap},
            "deficit_weight_source": "disabled",
            "warnings": [],
            "tasks": {
                task: {
                    "available": available[task],
                    "deficit_weight": None,
                    "observed_share_before": available[task] / total,
                    "target_share": available[task] / total,
                    "target_rows": float(available[task]),
                    "rep": 1.0,
                    "realized_rep": 1.0,
                    "realized_share_after": available[task] / total,
                    "share_gap": 0.0,
                    "repeated_rows": available[task],
                    "unique_rows_emitted": available[task],
                    "additional_repetitions": 0,
                    "dropped_rows": 0,
                    "empty_rows": empty[task],
                    "empty_rows_emitted": empty[task],
                    "repeated_empty_rows": 0,
                }
                for task in sorted(available)
            },
            "totals": {
                "budget": total,
                "budget_multiplier": resolved["budget_multiplier"],
                "rows_before": total,
                "rows_after": total,
                "max_abs_share_gap": 0.0,
                "total_share_gap": 0.0,
                "available": total,
                "row_cap": row_cap,
                "row_multiple": row_multiple,
                "repeated_rows": total,
                "unique_rows_emitted": total,
                "additional_repetitions": 0,
                "dropped_rows": 0,
                "empty_rows": sum(empty.values()),
                "repeated_empty_rows": 0,
            },
            "invariants": {
                "only_available_rows_emitted": True,
                "empty_ground_truth_not_repeated": True,
                "perceptual_hash_filter_applied": False,
            },
            "available_rows_sha256": _fingerprint(rows),
            "materialized_rows_sha256": _fingerprint(rows),
        }
        return list(rows), manifest

    unique_indices: dict[str, list[int]] = {task: [] for task in available}
    seen_fingerprints: set[str] = set()
    for index, (task, fingerprint) in enumerate(zip(tasks, row_fingerprints)):
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        unique_indices[task].append(index)
    unique_available = {
        task: len(indices) for task, indices in unique_indices.items() if indices
    }
    unique_empty = {
        task: sum(empty_flags[index] for index in indices)
        for task, indices in unique_indices.items()
        if indices
    }
    plan = plan_repetition(
        available_rows=unique_available,
        deficit_weights=deficit_weights,
        row_cap=row_cap,
        policy=resolved["policy"],
        rep_min=resolved["rep_min"],
        rep_max=resolved["rep_max"],
        budget_multiplier=resolved["budget_multiplier"],
        redistribute=resolved["redistribute"],
        explicit_multipliers=resolved["explicit_multipliers"],
        empty_rows=unique_empty,
        never_repeat_empty_gt=resolved["never_repeat_empty_gt"],
        minimum_rows=dict(
            Counter(
                tasks[index]
                for index in required_indices
            )
        ),
        row_multiple=row_multiple,
    )
    budget = (
        _deficit_budget(
            available_total=sum(unique_available.values()),
            budget_multiplier=resolved["budget_multiplier"],
            row_cap=row_cap,
        )
        if resolved["policy"] == "deficit_proportional"
        else sum(payload["repeated_rows"] for payload in plan.values())
    )
    occurrence_counts = [0] * len(rows)
    for task in sorted(unique_available):
        indices = [index for index, value in enumerate(tasks) if value == task]
        mandatory = sorted(required_indices.intersection(indices))
        for index in mandatory:
            occurrence_counts[index] = 1
        remaining = plan[task]["repeated_rows"] - len(mandatory)
        if remaining < 0:
            raise ValueError(f"repetition allocation dropped a mandatory row for {task!r}")
        covered_fingerprints = {row_fingerprints[index] for index in mandatory}
        unique_candidates = _seeded_order(
            [
                index
                for index in unique_indices[task]
                if row_fingerprints[index] not in covered_fingerprints
            ],
            rows=rows,
            seed=seed,
            task=task,
            phase="unique",
        )
        for index in unique_candidates[:remaining]:
            occurrence_counts[index] = 1
        remaining -= min(remaining, len(unique_candidates))
        if remaining:
            repeatable_by_fingerprint: dict[str, int] = {}
            for index in indices:
                if occurrence_counts[index] and not (
                    empty_flags[index] and resolved["never_repeat_empty_gt"]
                ):
                    repeatable_by_fingerprint.setdefault(row_fingerprints[index], index)
            repeatable = _seeded_order(
                list(repeatable_by_fingerprint.values()),
                rows=rows,
                seed=seed,
                task=task,
                phase="repeat",
            )
            if not repeatable:
                raise ValueError(
                    f"repetition allocation cannot repeat rows for {task!r}"
                )
            whole, remainder = divmod(remaining, len(repeatable))
            for index in repeatable:
                occurrence_counts[index] += whole
            for index in repeatable[:remainder]:
                occurrence_counts[index] += 1

    output = [
        row
        for cycle in range(max(occurrence_counts, default=0))
        for index, row in enumerate(rows)
        if occurrence_counts[index] > cycle
    ]
    output_fingerprints = Counter(_fingerprint(row) for row in output)
    available_fingerprints = set(row_fingerprints)
    if not set(output_fingerprints).issubset(available_fingerprints):
        raise AssertionError("repetition blend emitted a row outside the accepted pool")
    repeated_empty_rows = sum(
        max(0, occurrence_counts[index] - 1)
        for index, is_empty in enumerate(empty_flags)
        if is_empty
    )
    for task, payload in plan.items():
        task_indices = [index for index, value in enumerate(tasks) if value == task]
        emitted_rows = sum(occurrence_counts[index] for index in task_indices)
        emitted_fingerprints = {
            row_fingerprints[index]
            for index in task_indices
            if occurrence_counts[index]
        }
        payload["repeated_rows"] = emitted_rows
        payload["realized_rep"] = emitted_rows / payload["available"]
        payload["repeated_empty_rows"] = sum(
            max(0, occurrence_counts[index] - 1)
            for index in task_indices
            if empty_flags[index]
        )
        payload["empty_rows_emitted"] = sum(
            occurrence_counts[index]
            for index in task_indices
            if empty_flags[index]
        )
        payload["unique_rows_emitted"] = len(emitted_fingerprints)
        payload["additional_repetitions"] = emitted_rows - len(emitted_fingerprints)
        payload["dropped_rows"] = payload["available"] - len(emitted_fingerprints)
        payload["realized_share_after"] = emitted_rows / len(output)
        payload["share_gap"] = (
            payload["target_share"] - payload["realized_share_after"]
        )
    max_abs_share_gap = max(
        abs(payload["share_gap"]) for payload in plan.values()
    )
    total_share_gap = sum(
        max(0.0, payload["share_gap"]) for payload in plan.values()
    )
    warnings: list[str] = []
    if total_share_gap > resolved["share_gap_tolerance"] + 1e-12:
        warnings.append(
            "repetition task-share shortfall "
            f"{total_share_gap * 100:.2f} percentage points exceeds configured "
            f"tolerance {resolved['share_gap_tolerance'] * 100:.2f} percentage points"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "policy": resolved["policy"],
        "seed": seed,
        "configuration": {**resolved, "row_cap": row_cap},
        "deficit_weight_source": (
            "explicit"
            if resolved["policy"] == "explicit"
            else deficit_weight_source
            or ("provided" if deficit_weights else "equal_fallback")
        ),
        "warnings": warnings,
        "tasks": plan,
        "totals": {
            "budget": budget,
            "budget_multiplier": resolved["budget_multiplier"],
            "rows_before": len(rows),
            "rows_after": len(output),
            "max_abs_share_gap": max_abs_share_gap,
            "total_share_gap": total_share_gap,
            "available": sum(unique_available.values()),
            "row_cap": row_cap,
            "row_multiple": row_multiple,
            "repeated_rows": len(output),
            "unique_rows_emitted": len(output_fingerprints),
            "additional_repetitions": len(output) - len(output_fingerprints),
            "dropped_rows": sum(unique_available.values()) - len(output_fingerprints),
            "empty_rows": sum(unique_empty.values()),
            "repeated_empty_rows": repeated_empty_rows,
        },
        "invariants": {
            "only_available_rows_emitted": True,
            "empty_ground_truth_not_repeated": (
                repeated_empty_rows == 0
                if resolved["never_repeat_empty_gt"]
                else None
            ),
            "perceptual_hash_filter_applied": False,
        },
        "available_rows_sha256": _fingerprint(
            [
                rows[index]
                for task in sorted(unique_indices)
                for index in unique_indices[task]
            ]
        ),
        "materialized_rows_sha256": _fingerprint(output),
    }
    return output, manifest


def bind_repetition_manifest(
    manifest: dict[str, Any], training_jsonl: pathlib.Path
) -> dict[str, Any]:
    path = training_jsonl.expanduser().resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as stream:
        rows = sum(1 for line in stream if line.strip())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported repetition blend manifest schema")
    if rows != manifest.get("totals", {}).get("repeated_rows"):
        raise ValueError("training JSONL row count differs from repetition manifest")
    bound = json.loads(json.dumps(manifest))
    bound["training_jsonl"] = {"path": str(path), "sha256": digest, "rows": rows}
    return bound
