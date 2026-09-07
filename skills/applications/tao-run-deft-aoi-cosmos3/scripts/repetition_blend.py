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


SCHEMA_VERSION = "repetition_blend_manifest_v1"
POLICIES = ("deficit_proportional", "explicit")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "policy": "deficit_proportional",
    "rep_min": 0.5,
    "rep_max": 3.0,
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
    for name in ("rep_min", "rep_max"):
        number = config[name]
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise ValueError(f"repetition {name} must be numeric")
        config[name] = float(number)
    if not all(math.isfinite(config[name]) for name in ("rep_min", "rep_max")):
        raise ValueError("repetition multipliers must be finite")
    if config["rep_min"] <= 0 or config["rep_max"] <= 0:
        raise ValueError("repetition multipliers must be positive")
    if config["rep_min"] > config["rep_max"]:
        raise ValueError("repetition rep_min cannot exceed rep_max")
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


def plan_repetition(
    *,
    available_rows: dict[str, int],
    row_cap: int,
    deficit_weights: dict[str, float] | None = None,
    policy: str = "deficit_proportional",
    rep_min: float = 0.5,
    rep_max: float = 3.0,
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
    protected_empty = {task: 0 for task in available}
    for task, count in (empty_rows or {}).items():
        if task not in available:
            if count:
                raise ValueError(f"empty rows supplied for unavailable task {task!r}")
            continue
        if type(count) is not int or not 0 <= count <= available[task]:
            raise ValueError(f"invalid empty row count for {task!r}")
        protected_empty[task] = count if never_repeat_empty_gt else 0
    mandatory = dict(protected_empty)
    for task, count in (minimum_rows or {}).items():
        if task not in available:
            if count:
                raise ValueError(f"minimum rows supplied for unavailable task {task!r}")
            continue
        if type(count) is not int or not 0 <= count <= available[task]:
            raise ValueError(f"invalid minimum row count for {task!r}")
        mandatory[task] = max(mandatory[task], count)

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
        target_rows = {task: row_cap * shares[task] for task in available}
        requested_rep = {
            task: min(
                config["rep_max"],
                max(config["rep_min"], target_rows[task] / available[task]),
            )
            for task in available
        }
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
        task: (
            available[task]
            if protected_empty[task] == available[task]
            else max(
                mandatory[task],
                int(math.floor(available[task] * requested_rep[task] + 0.5)),
            )
        )
        for task in available
    }
    if not any(requested_counts.values()):
        requested_counts[max(available, key=lambda task: (shares[task], task))] = 1
    emitted_total = min(row_cap, sum(requested_counts.values()))
    emitted_total -= emitted_total % row_multiple
    if emitted_total <= 0:
        raise ValueError(
            "repetition blend cannot form one complete row_multiple group"
        )
    emitted_counts = _allocate_cap(
        requested_counts,
        shares=shares,
        mandatory=mandatory,
        row_cap=emitted_total,
    )
    plan: dict[str, dict[str, Any]] = {}
    for task in sorted(available):
        emitted = emitted_counts[task]
        empty = (empty_rows or {}).get(task, 0)
        unique_emitted = min(available[task], emitted)
        plan[task] = {
            "available": available[task],
            "deficit_weight": weights[task],
            "target_share": shares[task],
            "target_rows": target_rows[task],
            "rep": requested_rep[task],
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
            "tasks": {
                task: {
                    "available": available[task],
                    "deficit_weight": None,
                    "target_share": available[task] / total,
                    "target_rows": float(available[task]),
                    "rep": 1.0,
                    "realized_rep": 1.0,
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

    plan = plan_repetition(
        available_rows=dict(available),
        deficit_weights=deficit_weights,
        row_cap=row_cap,
        policy=resolved["policy"],
        rep_min=resolved["rep_min"],
        rep_max=resolved["rep_max"],
        explicit_multipliers=resolved["explicit_multipliers"],
        empty_rows=dict(empty),
        never_repeat_empty_gt=resolved["never_repeat_empty_gt"],
        minimum_rows=dict(
            Counter(
                tasks[index]
                for index in required_indices
                | {
                    index
                    for index, is_empty in enumerate(empty_flags)
                    if is_empty and resolved["never_repeat_empty_gt"]
                }
            )
        ),
        row_multiple=row_multiple,
    )
    occurrence_counts = [0] * len(rows)
    for task in sorted(available):
        indices = [index for index, value in enumerate(tasks) if value == task]
        fixed = [
            index
            for index in indices
            if empty_flags[index] and resolved["never_repeat_empty_gt"]
        ]
        mandatory = sorted(required_indices.intersection(indices) - set(fixed))
        repeatable = [index for index in indices if index not in fixed]
        for index in fixed:
            occurrence_counts[index] = 1
        for index in mandatory:
            occurrence_counts[index] = 1
        required = plan[task]["repeated_rows"] - len(fixed) - len(mandatory)
        if required < 0:
            raise ValueError(f"repetition allocation dropped a mandatory row for {task!r}")
        if required and not repeatable:
            required = 0
        if repeatable:
            whole, remainder = divmod(required, len(repeatable))
            for index in repeatable:
                occurrence_counts[index] += whole
            ranked = sorted(
                repeatable,
                key=lambda index: (
                    hashlib.sha256(
                        f"{seed}\0{task}\0{index}\0{_fingerprint(rows[index])}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    index,
                ),
            )
            for index in ranked[:remainder]:
                occurrence_counts[index] += 1

    output = [
        row
        for cycle in range(max(occurrence_counts, default=0))
        for index, row in enumerate(rows)
        if occurrence_counts[index] > cycle
    ]
    output_fingerprints = Counter(_fingerprint(row) for row in output)
    available_fingerprints = {_fingerprint(row) for row in rows}
    if not set(output_fingerprints).issubset(available_fingerprints):
        raise AssertionError("repetition blend emitted a row outside the accepted pool")
    repeated_empty_rows = sum(
        max(0, occurrence_counts[index] - 1)
        for index, is_empty in enumerate(empty_flags)
        if is_empty
    )
    unique_emitted = sum(count > 0 for count in occurrence_counts)
    for task, payload in plan.items():
        task_indices = [index for index, value in enumerate(tasks) if value == task]
        emitted_rows = sum(occurrence_counts[index] for index in task_indices)
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
        payload["unique_rows_emitted"] = sum(
            occurrence_counts[index] > 0 for index in task_indices
        )
        payload["additional_repetitions"] = sum(
            max(0, occurrence_counts[index] - 1) for index in task_indices
        )
        payload["dropped_rows"] = sum(
            occurrence_counts[index] == 0 for index in task_indices
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
        "tasks": plan,
        "totals": {
            "available": len(rows),
            "row_cap": row_cap,
            "row_multiple": row_multiple,
            "repeated_rows": len(output),
            "unique_rows_emitted": unique_emitted,
            "additional_repetitions": sum(
                max(0, count - 1) for count in occurrence_counts
            ),
            "dropped_rows": sum(count == 0 for count in occurrence_counts),
            "empty_rows": sum(empty.values()),
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
        "available_rows_sha256": _fingerprint(rows),
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
