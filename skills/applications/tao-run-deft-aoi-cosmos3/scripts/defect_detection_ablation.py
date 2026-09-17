#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and verify the single-image Defect Detection ablation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import Counter
from typing import Any, Iterable

from anchor_rows import ANCHOR_MARK
from answer_profile import is_empty_ground_truth, parse_task_shares, validate_empty_answer_guard_config
from atomic_samples import PAIR_CONTENT_IDENTITY, content_identity_for_paths, sample_from_record
from coverage_rows import COVERAGE_MARK
from repetition_blend import (
    POLICIES as REPETITION_POLICIES,
    apply_repetition_blend,
    bind_repetition_manifest,
    deficit_weights_from_gap_summary,
    load_repetition_config,
    merge_repetition_config,
    parse_explicit_multipliers,
    plan_repetition,
    validate_repetition_config,
)
from select_detection_calibration import derive_proxy_empty_rates
from validate_sharegpt import load_records, resolve_image, target_path


DEFECT_DETECTION_TASK = "Defect Detection"
REFERENCE_DEFECT_DETECTION_TASK = "Ref_based Defect Detection"
MAINTENANCE_TASK_TYPES = (
    "Component Classification",
    "Component Detection",
    "Defect Classification",
    "Ref_based Defect Classification",
    "Ref_based Defect Detection",
)
ALL_TASK_TYPES = (DEFECT_DETECTION_TASK, *MAINTENANCE_TASK_TYPES)
# Mined per-task pool caps (Feature P5-S, 2026-09-17): ``--mined-task-pool-cap TASK=FRACTION``
# bounds the *mined* rows of a task at ``floor(pool_rows * fraction)`` over the whole run, where
# pool_rows is the task's row count in the canonical Mining pool (--source-annotations). Rows the
# cumulative Train JSONL already holds for the task count against the cap unless they carry one of
# these inert markers (calibration, anchor and coverage rows are not mined rows). The fill order
# (``--mined-task-fill-order``) gives the listed maintenance tasks their rows before the others.
# Neither option changes the default selection: no caps and an empty order reproduce the
# round-robin of the parent snapshot byte-for-byte. (``HISTORY_NON_MINED_MARKS`` below.)
MINED_TASK_POOL_CAP_POLICY = "floor_of_task_mining_pool_rows_times_fraction_cumulative_over_iterations_mined_rows_only"
# Cross-task visual de-duplication (Feature P5-S.1, run v12_p5s_pool10_r8 iteration 1, 2026-09-17):
# ``--cross-task-visual-dedup on`` (default) is today's rule: a maintenance-task row is dropped when
# its image path / content (or near-duplicate hash) was already selected for *any* task. In the NVPAW
# pool the single-image Defect Classification (MCQ) and Defect Detection rows are asked on the same
# board images, so Defect Detection consumed the images first and Defect Classification starved
# (858 routed images, 19 available). ``off`` keeps the exact-record, previous-record and benchmark /
# proxy leakage exclusions and the visual de-duplication WITHIN each task type, but no longer drops
# a row because its image was selected for a different task (reference pairs keep their pair
# identity); the manifest records the rows this unlocked per task
# (``maintenance_rows_unlocked_by_cross_task``) and the uniqueness verification keys are evaluated
# within each task type. ``on`` reproduces the parent snapshot byte-for-byte.
CROSS_TASK_VISUAL_DEDUP_MODES = ("on", "off")
DEFAULT_CROSS_TASK_VISUAL_DEDUP = "on"
POSITIVE_EVIDENCE = {
    "coverage_stratified_positive",
    "hard_positive_proxy_false_negative",
    "hard_positive_best_overlap_0_lt_iou_lte_0p5",
}
NEGATIVE_EVIDENCE = {"hard_negative_proxy_false_positive"}
CALIBRATION_EMPTY_EVIDENCE = "calibration_empty_ground_truth"
CALIBRATION_FEW_EVIDENCE = "calibration_few_box_ground_truth"
# Rows selected by select_detection_calibration's profile policy may carry any
# ground-truth box count; they are recognised by this candidate column.
PROFILE_CALIBRATION_POLICY = "kpi_profile_count_bins"
REFERENCE_NO_CHANGE_EVIDENCE = "calibration_reference_no_change_ground_truth"
# Inert top-level marker on emitted calibration rows (like ``deft_anchor``): the
# assembler must never trim them when it reserves anchor / coverage slots under
# the row cap, otherwise the verified calibration contract silently breaks
# (2026-09-14: 382 of 3,000 reference pairs dropped by the task-balanced trim).
CALIBRATION_MARK = "deft_calibration"
# Second inert marker naming the calibration kind: detection calibration rows
# emitted here carry ``detection`` (the assembler's empty-answer guard treats an
# absent or unknown kind as detection as well); ``select_classification_calibration.py``
# writes ``classification``, which the guard never trims.
CALIBRATION_KIND_MARK = "deft_calibration_kind"
DETECTION_CALIBRATION_KIND = "detection"
CLASSIFICATION_CALIBRATION_KIND = "classification"
# rows of the cumulative Train JSONL that are not mined rows (they never count against a mined
# per-task pool cap): detection / classification calibration, anchors, coverage-blend rows
HISTORY_NON_MINED_MARKS = (CALIBRATION_MARK, ANCHOR_MARK, COVERAGE_MARK)
# Zero-new-candidate policy (Phase 4). ``fail_closed`` = every maintenance task
# must be present in the current selection (historical behaviour). With
# ``skip_exhausted`` a maintenance task may be absent only when the routed
# candidate set has zero eligible rows for it after history / identity
# exclusion; the shortage is recorded and the iteration continues. It still
# fails closed when every maintenance task is exhausted or nothing is added
# (2026-09-15: iteration 2 of run 4c-B had zero new candidates for two tasks
# and 4 / 2 / 1024 / 529 for the others; the whole iteration failed).
ZERO_NEW_CANDIDATE_POLICIES = ("fail_closed", "skip_exhausted")
DEFAULT_ZERO_NEW_CANDIDATE_POLICY = "fail_closed"
_PRESENCE_VERIFICATION_KEY = "all_five_maintenance_tasks_present"
# Feature P5-S follow-up (2026-09-17): under ``skip_exhausted`` a maintenance task absent because its
# mined pool cap was consumed before this selection (remainder 0, eligible candidates left) is
# accepted like an exhausted one and recorded in ``capped_absent_tasks`` (disjoint from
# ``exhausted_tasks`` / ``skipped_tasks``). The binding presence verdict under that policy is
# ``maintenance_tasks_present_or_exhausted_or_capped``; the older ``maintenance_tasks_present_or_exhausted``
# stays a raw fact (false when a capped task is absent) and is excluded from ``verified`` there.
# ``fail_closed`` still fails on any absent task; every task absent still fails under both policies.
_PRESENCE_OR_EXHAUSTED_KEY = "maintenance_tasks_present_or_exhausted"
_PRESENCE_OR_EXHAUSTED_OR_CAPPED_KEY = "maintenance_tasks_present_or_exhausted_or_capped"
# Guard-aware calibration selection (Feature B3, run v12_p4b_emptyguard_r4 iteration 4,
# 2026-09-15): with the empty-answer guard on, the fixed calibration slot (512 + 512
# single-image rows, 500 reference pairs) selects at most as many empty rows as the
# caps leave room for (cumulative corpus + this iteration's non-calibration rows) and
# fills the rest of the slot with few-box rows from the same source; the row count of
# the slot is unchanged, only the empty / few-box split moves. Without it the guard
# trimmed almost every calibration negative and the aligned corpus fell back to the
# previous size (growth 0, fail closed).
# Feature B3.1 (run v12_p4b_emptyguard_r5 iteration 1, snapshot 3c4e042b): the substitution is
# best-effort. When the few-box / changed reserve of the feed runs out, the remaining slot rows
# are empty calibration candidates beyond the headroom (never more than the KPI-rate selection
# carried); the overflow is recorded (``calibration_headroom_overflow_rows``,
# ``guard_aware_calibration.status``) and left to the assembler's guard to trim. A substitution
# shortfall alone never fails the materializer; a genuinely short slot still fails closed.
GUARD_AWARE_CALIBRATION_TASKS = (DEFECT_DETECTION_TASK, REFERENCE_DEFECT_DETECTION_TASK)
GUARD_AWARE_CALIBRATION_STATUSES = ("no_substitution_needed", "substituted", "substituted_with_overflow")
# current-selection facts copied into the bound v2 manifest (bind_cumulative_manifest)
CURRENT_SELECTION_COPIED_KEYS = (
    "new_rows_empty",
    "new_rows_empty_by_task",
    "exhausted_tasks",
    "skipped_tasks",
    "calibration_guard_aware",
    "calibration_empty_headroom",
    "calibration_empty_selected",
    "calibration_fewbox_substituted",
    "calibration_headroom_overflow_rows",
    "mined_task_pool_usage",
    "mined_task_fill_realized",
    "capped_tasks",
    "capped_absent_tasks",
    "cross_task_visual_dedup",
    "maintenance_rows_unlocked_by_cross_task",
)
CORRECT_ANCHOR_EVIDENCE = "proxy_correct"
POSITIVE_MARGINS = (
    ("source", "source_strata", None),
    ("phenotype", "phenotype_strata", None),
    ("source_x_phenotype", "source_phenotype_strata", None),
    ("box_area_quartile_1024", "area_strata", ("Q1", "Q2", "Q3", "Q4")),
    ("local_contrast_quartile", "contrast_strata", ("Q1", "Q2", "Q3", "Q4")),
    ("gt_box_count_bin", "count_strata", ("1", "2-3", "4+")),
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verification_policy_exclusions(policy: str) -> list[str]:
    """Verification keys the policy replaces (``maintenance_tasks_present_or_exhausted_or_capped`` covers them)."""
    return [_PRESENCE_VERIFICATION_KEY, _PRESENCE_OR_EXHAUSTED_KEY] if policy == "skip_exhausted" else []


def _manifest_verified(verification: dict[str, Any], policy: str) -> bool:
    excluded = set(_verification_policy_exclusions(policy))
    return all(value for key, value in verification.items() if key not in excluded)


def validate_defect_detection_fraction(value: Any) -> float:
    """The Defect Detection lower bound of an iteration (share of the target rows); (0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Defect Detection fraction must be a number in (0, 1], not {value!r}") from exc
    if not 0.0 < number <= 1.0:
        raise ValueError(f"Defect Detection fraction must be in (0, 1], not {number}")
    return number


def validate_cross_task_visual_dedup(value: Any) -> str:
    """``on`` (today's cross-task exclusion) or ``off`` (within-task only); Feature P5-S.1."""
    if not isinstance(value, str) or value not in CROSS_TASK_VISUAL_DEDUP_MODES:
        raise ValueError(
            f"cross-task visual de-duplication must be one of {list(CROSS_TASK_VISUAL_DEDUP_MODES)}, not {value!r}"
        )
    return value


def validate_mined_task_pool_caps(caps: Any) -> dict[str, float]:
    """``{task: fraction}`` over the six task types, every fraction in (0, 1]; missing tasks are uncapped."""
    if caps is None:
        return {}
    if not isinstance(caps, dict):
        raise ValueError("mined task pool caps must be an object of TASK: FRACTION")
    result: dict[str, float] = {}
    for task, fraction in caps.items():
        if task not in ALL_TASK_TYPES:
            raise ValueError(
                f"mined task pool cap names an unknown task {task!r}; expected one of {list(ALL_TASK_TYPES)}"
            )
        try:
            number = float(fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mined task pool cap for {task!r} must be a fraction in (0, 1], not {fraction!r}") from exc
        if not 0.0 < number <= 1.0:
            raise ValueError(f"mined task pool cap for {task!r} must be in (0, 1], not {number}")
        result[str(task)] = number
    return result


def parse_mined_task_pool_caps(values: Iterable[str] | None) -> dict[str, float]:
    """CLI form: repeatable ``TASK=FRACTION`` (a task may appear once)."""
    caps: dict[str, Any] = {}
    for value in values or []:
        task, separator, fraction = str(value).partition("=")
        if not separator or not task:
            raise ValueError(f"invalid mined task pool cap {value!r}; expected TASK=FRACTION")
        if task in caps:
            raise ValueError(f"mined task pool cap repeats {task!r}")
        caps[task] = fraction
    return validate_mined_task_pool_caps(caps)


def validate_mined_task_fill_order(order: Any) -> list[str]:
    """Maintenance tasks filled first, in this order; Defect Detection is filled by its fraction and
    cannot appear; an empty order is today's round-robin."""
    if order is None:
        return []
    if isinstance(order, (str, bytes)) or not isinstance(order, (list, tuple)):
        raise ValueError("mined task fill order must be a list of maintenance task names")
    result: list[str] = []
    for task in order:
        if task == DEFECT_DETECTION_TASK:
            raise ValueError(
                "Defect Detection is filled by --defect-detection-fraction first and cannot appear in the mined task fill order"
            )
        if task not in MAINTENANCE_TASK_TYPES:
            raise ValueError(
                f"mined task fill order names an unknown maintenance task {task!r}; "
                f"expected a subset of {list(MAINTENANCE_TASK_TYPES)}"
            )
        if task in result:
            raise ValueError(f"mined task fill order repeats {task!r}")
        result.append(str(task))
    return result


def parse_mined_task_fill_order(value: str | None) -> list[str]:
    """CLI form: ``T1,T2,...`` (blank = no priority tasks)."""
    if value is None or not str(value).strip():
        return []
    return validate_mined_task_fill_order([part.strip() for part in str(value).split(",")])


def mined_task_cap_rows(pool_rows: int, fraction: float) -> int:
    """``floor(pool_rows * fraction)`` with a guard against float representation (0.29 * 100)."""
    return max(0, math.floor(pool_rows * fraction + 1e-9))


def empty_headroom(*, cap: float, rows: int, empty_rows: int) -> int:
    """Empty rows that may still be added under ``cap`` when the corpus will hold ``rows`` rows
    (the rows to add are already counted) and ``empty_rows`` of them are empty already.
    A share equal to its cap is within the cap (the assembler's comparison)."""
    return max(0, math.floor(cap * rows - empty_rows + 1e-9))


def allocate_empty_headroom(limits: dict[str, int], total: int) -> dict[str, int]:
    """Split a shared headroom over per-task limits by largest remainder; no task above its
    own limit, ties broken by task name. Not binding when the limits fit."""
    if sum(limits.values()) <= total:
        return dict(limits)
    weight = sum(limits.values())
    raw = {task: total * limit / weight for task, limit in limits.items()}
    allocated = {task: min(limits[task], math.floor(raw[task])) for task in limits}
    remainder = total - sum(allocated.values())
    for task in sorted(limits, key=lambda name: (-(raw[name] - math.floor(raw[name])), name)):
        if remainder <= 0:
            break
        if allocated[task] < limits[task]:
            allocated[task] += 1
            remainder -= 1
    return allocated


def _guard_aware_status(
    enabled: bool, substituted: dict[str, int], overflow: dict[str, int]
) -> str | None:
    """``guard_aware_calibration.status``: None when off; otherwise whether the headroom moved
    any slot rows to few-box / changed rows and whether some of them fell back to empties."""
    if not enabled:
        return None
    if sum(overflow.values()):
        return "substituted_with_overflow"
    if sum(substituted.values()):
        return "substituted"
    return "no_substitution_needed"


def _empty_ledger(records: Iterable[dict[str, Any]], *, context: str) -> tuple[Counter[str], Counter[str]]:
    """(rows, empty rows) per task type."""
    rows: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    for index, record in enumerate(records):
        task = str(record.get("task_type"))
        rows[task] += 1
        empty[task] += is_empty_ground_truth(record, context=f"{context}[{index}]")
    return rows, empty


def _novel_rows(entries: Iterable[dict[str, Any]]) -> int:
    return sum(not item["is_replay"] for item in entries)


def _record_fingerprint(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assistant_text(record: dict[str, Any]) -> str:
    for message in reversed(record.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    raise ValueError(f"record {record.get('id')!r} has no assistant response")


def _ground_truth_objects(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = _assistant_text(record).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"record {record.get('id')!r} has non-JSON detection ground truth"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"record {record.get('id')!r} detection response is not an array")
    objects: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"record {record.get('id')!r} detection item {index} is not an object"
            )
        box = item.get("bbox_2d")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
        ):
            raise ValueError(
                f"record {record.get('id')!r} detection item {index} has invalid bbox_2d"
            )
        if not isinstance(item.get("label"), str) or not item["label"]:
            raise ValueError(
                f"record {record.get('id')!r} detection item {index} has invalid label"
            )
        objects.append(item)
    return objects


def _count_bin(count: int) -> str:
    if count == 1:
        return "1"
    if 2 <= count <= 3:
        return "2-3"
    if count >= 4:
        return "4+"
    raise ValueError("positive Defect Detection rows must contain at least one box")


def _normalized_box_area(objects: list[dict[str, Any]]) -> float:
    # official_v1 coordinates are normalized to [0, 1000]. Convert every box
    # to the reviewed 1024x1024 analysis canvas without touching the row.
    scale = 1024.0 / 1000.0
    areas = []
    for item in objects:
        x1, y1, x2, y2 = (float(value) * scale for value in item["bbox_2d"])
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    return statistics.median(areas)


def _local_contrast(
    record: dict[str, Any], objects: list[dict[str, Any]], media_root: pathlib.Path
) -> float:
    """Measure mean absolute inside-vs-local-ring luminance contrast."""

    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise ValueError("Pillow is required to compute local-contrast strata") from exc
    path = resolve_image(target_path(record, context=str(record.get("id"))), media_root)
    if not path.is_file():
        raise ValueError(f"Defect Detection image is missing: {path}")
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        contrasts: list[float] = []
        for item in objects:
            x1, y1, x2, y2 = (float(value) for value in item["bbox_2d"])
            box = (
                max(0, min(width, round(x1 * width / 1000.0))),
                max(0, min(height, round(y1 * height / 1000.0))),
                max(0, min(width, round(x2 * width / 1000.0))),
                max(0, min(height, round(y2 * height / 1000.0))),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            expand_x = max(1, round((box[2] - box[0]) * 0.25))
            expand_y = max(1, round((box[3] - box[1]) * 0.25))
            outer = (
                max(0, box[0] - expand_x),
                max(0, box[1] - expand_y),
                min(width, box[2] + expand_x),
                min(height, box[3] + expand_y),
            )
            inside_mean = ImageStat.Stat(gray.crop(box)).mean[0]
            outer_image = gray.crop(outer)
            outer_sum = ImageStat.Stat(outer_image).sum[0]
            inside_sum = ImageStat.Stat(gray.crop(box)).sum[0]
            outer_pixels = max(1, outer_image.width * outer_image.height)
            inside_pixels = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            ring_pixels = outer_pixels - inside_pixels
            ring_mean = (
                (outer_sum - inside_sum) / ring_pixels
                if ring_pixels > 0
                else inside_mean
            )
            contrasts.append(abs(inside_mean - ring_mean) / 255.0)
    if not contrasts:
        raise ValueError(f"record {record.get('id')!r} has no measurable GT box")
    return statistics.mean(contrasts)


def _perceptual_hash(path: pathlib.Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise ValueError("Pillow is required for near-duplicate exclusion") from exc
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    bits = [pixels[row * 9 + column] > pixels[row * 9 + column + 1] for row in range(8) for column in range(8)]
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class _HammingIndex:
    """Small BK-tree for exact radius queries over 64-bit perceptual hashes."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        number = int(value, 16)
        if self.root is None:
            self.root = (number, {})
            return
        node = self.root
        while True:
            distance = (number ^ node[0]).bit_count()
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (number, {})
                return
            node = child

    def has_within(self, value: str, radius: int) -> bool:
        if self.root is None:
            return False
        number = int(value, 16)
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (number ^ node[0]).bit_count()
            if distance <= radius:
                return True
            pending.extend(
                child
                for edge, child in node[1].items()
                if distance - radius <= edge <= distance + radius
            )
        return False


def _candidate_visual_identity(
    candidate: dict[str, Any],
    *,
    media_root: pathlib.Path,
    compute_perceptual_hash: bool = True,
) -> tuple[str, str, str]:
    filepath = str(candidate["filepath"])
    path = resolve_image(filepath, media_root)
    content_sha = candidate.get("content_sha256")
    phash = candidate.get("perceptual_hash")
    if candidate.get("sample_kind") == "reference_pair":
        source_paths = candidate.get("source_image_paths")
        if not isinstance(source_paths, (list, tuple)) or len(source_paths) != 2:
            raise ValueError(
                f"reference-pair candidate {filepath!r} requires two source_image_paths"
            )
        if all(pathlib.Path(value).is_file() for value in source_paths):
            # Prefer the ordered constituent bytes over any composite-asset hash.
            content_sha = content_identity_for_paths("reference_pair", source_paths)
        elif content_sha is None:
            raise ValueError(
                f"reference-pair candidate {filepath!r} has missing source images "
                "and no precomputed content_sha256"
            )
    elif content_sha is None:
        content_sha = _sha256(path) if path.is_file() else hashlib.sha256(str(path).encode()).hexdigest()
    if phash is None and compute_perceptual_hash:
        phash = _perceptual_hash(path) if path.is_file() else str(content_sha)[:16]
    elif phash is None:
        # Preserve the fixed-width internal identity without decoding an image
        # when the reviewed launch explicitly disables perceptual filtering.
        phash = str(content_sha)[:16]
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise ValueError(f"candidate {filepath!r} has invalid content_sha256")
    if not isinstance(phash, str) or len(phash) != 16:
        raise ValueError(f"candidate {filepath!r} has invalid perceptual_hash")
    try:
        int(phash, 16)
    except ValueError as exc:
        raise ValueError(f"candidate {filepath!r} has non-hex perceptual_hash") from exc
    atomic_sample_id = candidate.get("atomic_sample_id")
    identity = (
        atomic_sample_id
        if isinstance(atomic_sample_id, str) and atomic_sample_id
        else str(path)
    )
    return identity, content_sha, phash


def _rank_quartiles(entries: list[dict[str, Any]], field: str, output: str) -> None:
    ordered = sorted(entries, key=lambda item: (float(item[field]), item["record_id"]))
    total = len(ordered)
    for rank, item in enumerate(ordered):
        item[output] = f"Q{min(4, (rank * 4) // total + 1)}"


def _quota_payload(
    entries: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    target: int,
    field: str,
    fixed_strata: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    strata = list(fixed_strata) if fixed_strata is not None else sorted(
        {stratum for item in entries for stratum in item[field]}
    )
    requested = {
        stratum: target // len(strata) + (index < target % len(strata))
        for index, stratum in enumerate(strata)
    } if strata else {}
    available = Counter(stratum for item in entries for stratum in item[field])
    actual = Counter(stratum for item in selected for stratum in item[field])
    return {
        "target_rows": target,
        "requested": requested,
        "available": {key: available[key] for key in strata},
        "selected": {key: actual[key] for key in strata},
        "shortages": {
            key: requested[key] - available[key]
            for key in strata
            if available[key] < requested[key]
        },
    }


def _balanced_positive_selection(
    entries: list[dict[str, Any]], target: int, *, max_novel: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not entries or target <= 0:
        return [], _positive_quota_report([], [], target)
    quota_templates = {
        name: _quota_payload(entries, [], target, field, fixed)
        for name, field, fixed in POSITIVE_MARGINS
    }
    counts = {name: Counter() for name, _, _ in POSITIVE_MARGINS}
    remaining = list(entries)
    selected: list[dict[str, Any]] = []
    novel_count = 0
    while remaining and len(selected) < target:
        eligible = [
            item
            for item in remaining
            if item["is_replay"] or novel_count < max_novel
        ]
        if not eligible:
            break

        def priority(item: dict[str, Any]) -> tuple[Any, ...]:
            deficit = 0.0
            for name, field, _ in POSITIVE_MARGINS:
                requested = quota_templates[name]["requested"]
                for stratum in item[field]:
                    quota = max(1, requested.get(stratum, 0))
                    deficit += max(0, quota - counts[name][stratum]) / quota
            if "hard_positive_proxy_false_negative" in item["evidence"]:
                evidence_rank = 0
            elif "hard_positive_best_overlap_0_lt_iou_lte_0p5" in item["evidence"]:
                evidence_rank = 1
            else:
                evidence_rank = 2
            return (
                -deficit,
                item["is_replay"],
                evidence_rank,
                -item["similarity"],
                item["record_id"],
            )

        chosen = min(eligible, key=priority)
        remaining.remove(chosen)
        selected.append(chosen)
        novel_count += not chosen["is_replay"]
        for name, field, _ in POSITIVE_MARGINS:
            counts[name].update(chosen[field])
    return selected, _positive_quota_report(entries, selected, target)


def _positive_quota_report(
    entries: list[dict[str, Any]], selected: list[dict[str, Any]], target: int
) -> dict[str, Any]:
    return {
        name: _quota_payload(entries, selected, target, field, fixed)
        for name, field, fixed in POSITIVE_MARGINS
    }


def _task_balanced(
    entries: list[dict[str, Any]],
    target: int,
    *,
    max_novel: int,
    reference_empty_rate: float | None = None,
    reference_seed: tuple[int, int] | None = None,
    task_limits: dict[str, int] | None = None,
    fill_order: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin over the maintenance tasks. With ``reference_empty_rate`` the
    reference rows track ``floor(total * rate + 0.5)`` empties. ``reference_seed``
    = (rows, empties) already selected outside this call (the reserved reference
    calibration): seeding makes the running total track the *combined* target the
    quota manifest verifies, so two separately rounded slices cannot miss it by one.

    ``task_limits`` (Feature P5-S) bounds the rows a task may contribute (its mined
    pool-cap remainder; absent = unbounded). ``fill_order`` lists maintenance tasks to
    fill first: one row of every task with candidates is placed first (so a later task
    is never starved and the presence policies stay satisfiable), then each listed task
    takes rows in order up to its limit / availability, then the remaining tasks fill
    the rest round-robin as before. Without both options the selection is unchanged."""
    groups = {
        task: sorted(
            (item for item in entries if item["task_type"] == task),
            key=lambda item: (
                item.get("route_tier") != "calibration",
                item["is_replay"],
                -item["similarity"],
                item["record_id"],
            ),
        )
        for task in MAINTENANCE_TASK_TYPES
    }
    selected: list[dict[str, Any]] = []
    novel_count = 0
    positions = Counter()
    reference_groups = {
        "empty": [
            item
            for item in groups[REFERENCE_DEFECT_DETECTION_TASK]
            if not item.get("objects")
        ],
        "positive": [
            item
            for item in groups[REFERENCE_DEFECT_DETECTION_TASK]
            if item.get("objects")
        ],
    }
    reference_positions = Counter()
    reference_selected = Counter()
    if reference_seed is not None:
        reference_selected["total"], reference_selected["empty"] = reference_seed
    limits = dict(task_limits or {})
    selected_by_task: Counter[str] = Counter()

    def take(task: str) -> bool:
        """Append the next eligible row of ``task``; False when it has none left or is at its limit."""
        nonlocal novel_count
        if task in limits and selected_by_task[task] >= limits[task]:
            return False
        if task == REFERENCE_DEFECT_DETECTION_TASK and reference_empty_rate is not None:
            next_total = reference_selected["total"] + 1
            desired_empty = math.floor(next_total * reference_empty_rate + 0.5)
            bucket = (
                "empty"
                if desired_empty > reference_selected["empty"]
                else "positive"
            )
            group = reference_groups[bucket]
            position = reference_positions[bucket]
            while (
                position < len(group)
                and not group[position]["is_replay"]
                and novel_count >= max_novel
            ):
                position += 1
            reference_positions[bucket] = position
            if position >= len(group):
                return False
            chosen = group[position]
            selected.append(chosen)
            novel_count += not chosen["is_replay"]
            reference_positions[bucket] = position + 1
            reference_selected["total"] += 1
            reference_selected["empty"] += bucket == "empty"
            selected_by_task[task] += 1
            return True
        position = positions[task]
        while (
            position < len(groups[task])
            and not groups[task][position]["is_replay"]
            and novel_count >= max_novel
        ):
            position += 1
        positions[task] = position
        if position >= len(groups[task]):
            return False
        chosen = groups[task][position]
        selected.append(chosen)
        novel_count += not chosen["is_replay"]
        positions[task] = position + 1
        selected_by_task[task] += 1
        return True

    priority = list(fill_order or [])
    if priority:
        remaining = [task for task in MAINTENANCE_TASK_TYPES if task not in priority]
        # presence: one row of every task that has one, so the priority fill (and the
        # materializer's later trim of the tail) can never starve a later task
        for task in [*priority, *remaining]:
            if len(selected) < target:
                take(task)
        # priority fill, in order, each task up to its limit / availability
        for task in priority:
            while len(selected) < target and take(task):
                pass
        cycles = [remaining, list(MAINTENANCE_TASK_TYPES)]
    else:
        cycles = [list(MAINTENANCE_TASK_TYPES)]
    for cycle in cycles:
        while len(selected) < target:
            advanced = False
            for task in cycle:
                if take(task):
                    advanced = True
                    if len(selected) == target:
                        break
            if not advanced:
                break
    return selected


def _interleave_groups(
    groups: list[list[dict[str, Any]]],
    *,
    target: int | None = None,
) -> list[dict[str, Any]]:
    limit = sum(len(group) for group in groups) if target is None else target
    selected: list[dict[str, Any]] = []
    positions = [0] * len(groups)
    while len(selected) < limit:
        advanced = False
        for index, group in enumerate(groups):
            if positions[index] >= len(group):
                continue
            selected.append(group[positions[index]])
            positions[index] += 1
            advanced = True
            if len(selected) == limit:
                break
        if not advanced:
            break
    return selected


def _without_visual_duplicates(
    entries: Iterable[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    hamming_distance: int | None,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    content = {item["content_sha256"] for item in selected}
    hashes = _HammingIndex(item["perceptual_hash"] for item in selected)
    paths = {item["resolved_path"] for item in selected}
    output: list[dict[str, Any]] = []
    for item in entries:
        if item["resolved_path"] in paths or item["content_sha256"] in content:
            counters["exact_duplicates_excluded"] += 1
            continue
        if hamming_distance is not None and hashes.has_within(
            item["perceptual_hash"], hamming_distance
        ):
            counters["near_duplicates_excluded"] += 1
            continue
        output.append(item)
        paths.add(item["resolved_path"])
        content.add(item["content_sha256"])
        hashes.add(item["perceptual_hash"])
    return output


def _without_cross_task_visual_duplicates(
    entries: Iterable[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    hamming_distance: int | None,
    counters: Counter[str],
) -> list[dict[str, Any]]:
    """``_without_visual_duplicates`` applied per task type (Feature P5-S.1,
    ``--cross-task-visual-dedup off``): a row is dropped only when a row of the *same* task
    already holds its image identity, and ``selected`` seeds every task with its own
    already-selected rows only. The input order is preserved."""
    items = list(entries)
    kept: set[int] = set()
    for task in dict.fromkeys(item["task_type"] for item in items):
        kept.update(
            id(item)
            for item in _without_visual_duplicates(
                [item for item in items if item["task_type"] == task],
                selected=[item for item in selected if item["task_type"] == task],
                hamming_distance=hamming_distance,
                counters=counters,
            )
        )
    return [item for item in items if id(item) in kept]


def _visual_exclusion_scope(
    kept: list[dict[str, Any]], task: str, *, within_task_only: bool
) -> list[dict[str, Any]]:
    """The rows a re-selection of ``task`` de-duplicates against: every kept row (cross-task
    de-duplication on) or only the kept rows of the same task (Feature P5-S.1, off)."""
    return [item for item in kept if item["task_type"] == task] if within_task_only else kept


def _validation_identities(
    records: list[dict[str, Any]], media_root: pathlib.Path
) -> tuple[set[str], set[str], _HammingIndex, set[str]]:
    paths: set[str] = set()
    content: set[str] = set()
    phashes = _HammingIndex()
    fingerprints: set[str] = set()
    for record in records:
        sample = sample_from_record(
            record, media_root=media_root, context="validation"
        )
        fingerprints.add(_record_fingerprint(record))
        path_text = str(sample["atomic_sample_id"])
        if path_text in paths:
            continue
        paths.add(path_text)
        if sample["sample_kind"] == "reference_pair":
            content.add(
                content_identity_for_paths("reference_pair", sample["image_paths"])
            )
            continue
        path = pathlib.Path(str(sample["target_filepath"]))
        if path.is_file():
            content.add(_sha256(path))
            phashes.add(_perceptual_hash(path))
    return paths, content, phashes, fingerprints


def materialize(
    *,
    candidate_rows: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]] | None = None,
    validation_records: list[dict[str, Any]],
    media_root: pathlib.Path,
    max_rows: int,
    minimum_rows: int | None = None,
    row_multiple: int,
    defect_detection_fraction: float,
    proxy_empty_rate: float,
    epochs: int,
    global_batch: int,
    near_duplicate_hamming_distance: int | None,
    reference_proxy_empty_rate: float | None = None,
    single_image_calibration_max_empty: int | None = None,
    single_image_calibration_max_few: int | None = None,
    reference_calibration_total: int | None = None,
    novel_image_limit: int | None = None,
    repetition_config: dict[str, Any] | None = None,
    deficit_weights: dict[str, float] | None = None,
    repetition_seed: int | None = None,
    deficit_weight_source: str | None = None,
    acquisition_rows: int = 0,
    zero_new_candidate_policy: str = DEFAULT_ZERO_NEW_CANDIDATE_POLICY,
    empty_answer_guard: dict[str, Any] | None = None,
    calibration_guard_aware: bool = False,
    mined_task_pool_caps: dict[str, float] | None = None,
    mined_task_fill_order: Iterable[str] | None = None,
    cross_task_visual_dedup: str = DEFAULT_CROSS_TASK_VISUAL_DEDUP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min(max_rows, row_multiple, epochs, global_batch) <= 0:
        raise ValueError("row, epoch, and global-batch values must be positive")
    # Feature P5-S: launch-recorded mined per-task pool caps and fill order (defaults = no change)
    mined_caps = validate_mined_task_pool_caps(mined_task_pool_caps)
    fill_order = validate_mined_task_fill_order(
        list(mined_task_fill_order) if mined_task_fill_order is not None else None
    )
    # Feature P5-S.1: cross-task visual de-duplication switch (default on = today's exclusion)
    cross_task_visual_dedup = validate_cross_task_visual_dedup(cross_task_visual_dedup)
    within_task_only = cross_task_visual_dedup == "off"
    if zero_new_candidate_policy not in ZERO_NEW_CANDIDATE_POLICIES:
        raise ValueError(
            f"zero_new_candidate_policy must be one of {list(ZERO_NEW_CANDIDATE_POLICIES)}, "
            f"not {zero_new_candidate_policy!r}"
        )
    if minimum_rows is not None and minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive when supplied")
    if global_batch != row_multiple:
        raise ValueError("row_multiple must equal effective global_batch")
    # launch-recorded since Feature P5-S (was a hard-coded 0.5 lower bound; [0.5, 1] before)
    defect_detection_fraction = validate_defect_detection_fraction(defect_detection_fraction)
    if not 0.0 <= proxy_empty_rate <= 1.0:
        raise ValueError("Proxy empty-ground-truth rate must be in [0, 1]")
    if reference_proxy_empty_rate is not None and not (
        0.0 <= reference_proxy_empty_rate <= 1.0
    ):
        raise ValueError("Reference Proxy empty-ground-truth rate must be in [0, 1]")
    calibration_values = (
        single_image_calibration_max_empty,
        single_image_calibration_max_few,
        reference_calibration_total,
    )
    hybrid_calibration = any(value is not None for value in calibration_values)
    if hybrid_calibration and any(value is None for value in calibration_values):
        raise ValueError(
            "hybrid calibration requires both single-image caps and the reference total"
        )
    if hybrid_calibration and any(value < 0 for value in calibration_values):
        raise ValueError("calibration caps and totals must be non-negative")
    if hybrid_calibration and reference_proxy_empty_rate is None:
        raise ValueError("hybrid calibration requires the reference Proxy empty rate")
    guard = empty_answer_guard or validate_empty_answer_guard_config(None, None, None, None)
    if calibration_guard_aware:
        if not guard["enabled"]:
            raise ValueError(
                "guard-aware calibration selection requires an enabled empty-answer guard "
                "(--max-empty-answer-share and/or --max-empty-answer-share-task)"
            )
        if not hybrid_calibration:
            raise ValueError(
                "guard-aware calibration selection requires the fixed calibration slot "
                "(--single-image-calibration-max-empty/-few and --reference-calibration-total)"
            )
    if near_duplicate_hamming_distance is not None and not (
        0 <= near_duplicate_hamming_distance <= 64
    ):
        raise ValueError("near-duplicate Hamming distance must be in [0, 64]")
    resolved_repetition = validate_repetition_config(repetition_config)
    if (
        resolved_repetition["row_cap"] is not None
        and resolved_repetition["row_cap"] != max_rows
    ):
        raise ValueError(
            "repetition row_cap must equal the materialization --max-rows value"
        )
    resolved_repetition["row_cap"] = max_rows
    requested_target_rows = max_rows - max_rows % row_multiple
    if requested_target_rows <= 0:
        raise ValueError("max_rows cannot form one complete effective global batch")
    minimum_rows_requested = (
        requested_target_rows if minimum_rows is None else minimum_rows
    )
    minimum_rows_aligned = (
        math.ceil(minimum_rows_requested / row_multiple) * row_multiple
    )
    if minimum_rows_aligned > requested_target_rows:
        raise ValueError(
            "minimum_rows cannot be satisfied within the batch-aligned materialization cap"
        )
    # Capability-gated acquisition (Phase 3): rows reserved for a task in
    # acquisition mode are excluded from the Defect Detection floor base, so a
    # large two-image acquisition slice does not force an impossible DD quota.
    if type(acquisition_rows) is not int or acquisition_rows < 0:
        raise ValueError("acquisition_rows must be a non-negative integer")
    if acquisition_rows and acquisition_rows > requested_target_rows - row_multiple:
        raise ValueError("acquisition_rows must leave at least one global batch for the other tasks")
    target_rows = requested_target_rows
    dd_target = math.ceil((target_rows - acquisition_rows) * defect_detection_fraction)
    dd_selection_limit = (
        dd_target if resolved_repetition["enabled"] else target_rows
    )
    empty_target = (
        None
        if hybrid_calibration
        else math.floor(dd_target * proxy_empty_rate + 0.5)
    )
    positive_target = None if empty_target is None else dd_target - empty_target
    if novel_image_limit is None:
        novel_image_limit = target_rows
    if not 0 <= novel_image_limit <= target_rows:
        raise ValueError("novel_image_limit must be in [0, target_rows]")

    media_root = media_root.expanduser().resolve()
    by_path: dict[str, list[dict[str, Any]]] = {}
    for record in source_records:
        task = record.get("task_type")
        if task not in {DEFECT_DETECTION_TASK, *MAINTENANCE_TASK_TYPES}:
            continue
        sample = sample_from_record(record, media_root=media_root, context="source")
        by_path.setdefault(str(sample["atomic_sample_id"]), []).append(record)
        if sample["sample_kind"] == "single_image":
            by_path.setdefault(str(sample["target_filepath"]), []).append(record)
    validation_paths, validation_content, validation_phashes, validation_fingerprints = (
        _validation_identities(validation_records, media_root)
    )
    previous_fingerprints = {
        _record_fingerprint(record) for record in (previous_records or [])
    }
    # Mined per-task pool caps (Feature P5-S): the cap is a fraction of the task's rows in the
    # canonical Mining pool, cumulative over the run; the mined rows the cumulative Train JSONL
    # already holds (no calibration / anchor / coverage marker) are used budget.
    pool_rows_by_task: Counter[str] = Counter(
        str(record.get("task_type")) for record in source_records if record.get("task_type") in ALL_TASK_TYPES
    )
    mined_used_before: Counter[str] = Counter(
        str(record.get("task_type"))
        for record in (previous_records or [])
        if record.get("task_type") in ALL_TASK_TYPES
        and not any(record.get(mark) is True for mark in HISTORY_NON_MINED_MARKS)
    )
    cap_rows_by_task = {
        task: mined_task_cap_rows(pool_rows_by_task[task], fraction) for task, fraction in mined_caps.items()
    }
    mined_remaining_before = {
        task: max(0, cap_rows - mined_used_before[task]) for task, cap_rows in cap_rows_by_task.items()
    }
    dd_mined_limit = mined_remaining_before.get(DEFECT_DETECTION_TASK)
    maintenance_limits = {
        task: remaining for task, remaining in mined_remaining_before.items() if task in MAINTENANCE_TASK_TYPES
    }
    # Routed candidates per task before any exclusion: with the skip_exhausted
    # policy a task may only be skipped when nothing was routed / nothing survived.
    routed_by_task: Counter[str] = Counter()
    for candidate in candidate_rows:
        routed_value = candidate.get("routed_task_types")
        if isinstance(routed_value, str):
            try:
                routed_value = json.loads(routed_value)
            except json.JSONDecodeError:
                routed_value = []
        if isinstance(routed_value, (list, tuple)):
            routed_by_task.update(str(task) for task in set(routed_value))
    counters: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    seen_record_fingerprints: set[str] = set()
    for candidate in candidate_rows:
        candidate_tiers = candidate.get("route_tiers") or [candidate.get("route_tier")]
        if isinstance(candidate_tiers, str):
            candidate_tiers = json.loads(candidate_tiers)
        if (
            not isinstance(candidate_tiers, list)
            or not candidate_tiers
            or not set(candidate_tiers).issubset({"strict", "calibration"})
        ):
            counters["non_strict_routes_excluded"] += 1
            continue
        if not isinstance(candidate.get("filepath"), str):
            raise ValueError("every candidate requires filepath")
        resolved_path, content_sha, phash = _candidate_visual_identity(
            candidate,
            media_root=media_root,
            compute_perceptual_hash=near_duplicate_hamming_distance is not None,
        )
        if (
            resolved_path in validation_paths
            or content_sha in validation_content
            or (
                near_duplicate_hamming_distance is not None
                and validation_phashes.has_within(
                    phash, near_duplicate_hamming_distance
                )
            )
        ):
            counters["benchmark_or_proxy_leakage_excluded"] += 1
            continue
        routed = candidate.get("routed_task_types")
        if isinstance(routed, str):
            routed = json.loads(routed)
        if not isinstance(routed, list):
            raise ValueError("every candidate requires routed_task_types")
        route_tier_by_task = candidate.get("route_tier_by_task") or {}
        if isinstance(route_tier_by_task, str):
            route_tier_by_task = json.loads(route_tier_by_task)
        if not isinstance(route_tier_by_task, dict):
            raise ValueError("route_tier_by_task must be an object")
        for record in by_path.get(resolved_path, []):
            task = str(record.get("task_type"))
            if task not in routed:
                continue
            route_tier = route_tier_by_task.get(task)
            if route_tier is None:
                route_tier = (
                    "calibration"
                    if candidate.get("route_tier") == "calibration"
                    and routed == [task]
                    else "strict"
                    if "strict" in candidate_tiers
                    else candidate.get("route_tier")
                )
            if route_tier not in {"strict", "calibration"}:
                counters["non_strict_routes_excluded"] += 1
                continue
            if route_tier == "calibration" and task not in {
                DEFECT_DETECTION_TASK,
                REFERENCE_DEFECT_DETECTION_TASK,
            }:
                counters["invalid_calibration_routes_excluded"] += 1
                continue
            fingerprint = _record_fingerprint(record)
            if fingerprint in previous_fingerprints:
                counters["previous_records_excluded"] += 1
                seen_record_fingerprints.add(fingerprint)
                continue
            if fingerprint in validation_fingerprints or fingerprint in seen_record_fingerprints:
                counters["exact_record_duplicates_excluded"] += 1
                continue
            seen_record_fingerprints.add(fingerprint)
            evidence_value = candidate.get("defect_detection_evidence") or []
            if isinstance(evidence_value, str):
                evidence_value = json.loads(evidence_value)
            if not isinstance(evidence_value, list) or not all(
                isinstance(value, str) for value in evidence_value
            ):
                raise ValueError("defect_detection_evidence must be a string list")
            evidence = sorted(set(evidence_value))
            entry = {
                # calibration rows carry the markers from here on so every
                # fingerprint (candidate, emitted, manifest) agrees
                "record": (
                    {**record, CALIBRATION_MARK: True, CALIBRATION_KIND_MARK: DETECTION_CALIBRATION_KIND}
                    if route_tier == "calibration"
                    else record
                ),
                "record_id": str(record.get("id")),
                "task_type": task,
                "resolved_path": resolved_path,
                "content_sha256": content_sha,
                "perceptual_hash": phash,
                "similarity": float(candidate.get("max_cosine_similarity", 0.0)),
                "evidence": evidence,
                "is_replay": bool(candidate.get("is_replay", False)),
                "route_tier": route_tier,
            }
            # Profile-matched calibration rows (policy kpi_profile_count_bins) carry
            # the KPI set's box-count distribution, so the <= 2-box rule of the
            # legacy few-box bucket does not apply to them; the bin is recorded.
            profile_calibration = (
                route_tier == "calibration"
                and str(candidate.get("calibration_policy") or "") == PROFILE_CALIBRATION_POLICY
            )
            if profile_calibration:
                entry["calibration_count_bin"] = str(candidate.get("calibration_count_bin") or "")
            if task == DEFECT_DETECTION_TASK:
                objects = _ground_truth_objects(record)
                entry["objects"] = objects
                if objects:
                    if route_tier == "calibration" and (
                        CALIBRATION_FEW_EVIDENCE not in evidence
                        or (len(objects) > 2 and not profile_calibration)
                    ):
                        counters["non_empty_calibration_rows_excluded"] += 1
                        continue
                    if route_tier != "calibration" and not (
                        POSITIVE_EVIDENCE.intersection(evidence)
                        or CORRECT_ANCHOR_EVIDENCE in evidence
                    ):
                        counters["off_evidence_positive_excluded"] += 1
                        continue
                    entry["source_strata"] = [str(record.get("dataset", "unknown"))]
                    entry["phenotype_strata"] = sorted({str(item["label"]) for item in objects})
                    entry["source_phenotype_strata"] = [
                        f"{source}::{phenotype}"
                        for source in entry["source_strata"]
                        for phenotype in entry["phenotype_strata"]
                    ]
                    entry["count_strata"] = [_count_bin(len(objects))]
                    entry["box_area_1024"] = _normalized_box_area(objects)
                    entry["local_contrast"] = float(
                        candidate.get("local_contrast")
                        if candidate.get("local_contrast") is not None
                        else _local_contrast(record, objects, media_root)
                    )
                elif not (
                    NEGATIVE_EVIDENCE.intersection(evidence)
                    or CALIBRATION_EMPTY_EVIDENCE in evidence
                    or CORRECT_ANCHOR_EVIDENCE in evidence
                ):
                    counters["off_evidence_empty_excluded"] += 1
                    continue
            elif task == REFERENCE_DEFECT_DETECTION_TASK:
                objects = _ground_truth_objects(record)
                entry["objects"] = objects
                if route_tier == "calibration":
                    required = (
                        CALIBRATION_EMPTY_EVIDENCE
                        if not objects
                        else CALIBRATION_FEW_EVIDENCE
                    )
                    if required not in evidence or (len(objects) > 2 and not profile_calibration):
                        counters["invalid_calibration_routes_excluded"] += 1
                        continue
                    if not objects and REFERENCE_NO_CHANGE_EVIDENCE not in evidence:
                        counters["invalid_reference_no_change_routes_excluded"] += 1
                        continue
            entries.append(entry)
    # eligible rows per task after previous-row / evaluation / duplicate exclusion
    eligible_by_task: Counter[str] = Counter(item["task_type"] for item in entries)
    positive = [
        item
        for item in entries
        if item["task_type"] == DEFECT_DETECTION_TASK and item["objects"]
    ]
    empty = [
        item
        for item in entries
        if item["task_type"] == DEFECT_DETECTION_TASK and not item["objects"]
    ]
    maintenance = [item for item in entries if item["task_type"] in MAINTENANCE_TASK_TYPES]
    if positive:
        _rank_quartiles(positive, "box_area_1024", "area_quartile")
        _rank_quartiles(positive, "local_contrast", "contrast_quartile")
        for item in positive:
            item["area_strata"] = [item["area_quartile"]]
            item["contrast_strata"] = [item["contrast_quartile"]]

    # Prefer newly mined examples within every quota.  Previously selected rows
    # remain eligible as bounded replay so iteration 5 can reach the requested
    # corpus size without violating the cumulative 50% mining-pool cap.
    empty.sort(
        key=lambda item: (
            item["is_replay"],
            (
                0
                if NEGATIVE_EVIDENCE.intersection(item["evidence"])
                else 1
                if CORRECT_ANCHOR_EVIDENCE in item["evidence"]
                else 2
            ),
            -item["similarity"],
            item["record_id"],
        )
    )
    reserved_reference_calibration: list[dict[str, Any]] = []
    reserved_reference_novel = 0
    reference_calibration_empty_target = 0
    if hybrid_calibration:
        assert reference_calibration_total is not None
        assert reference_proxy_empty_rate is not None
        reference_calibration_empty_target = math.floor(
            reference_calibration_total * reference_proxy_empty_rate + 0.5
        )
        reference_calibration_candidates = _without_visual_duplicates(
            (
                item
                for item in maintenance
                if item["task_type"] == REFERENCE_DEFECT_DETECTION_TASK
                and item["route_tier"] == "calibration"
            ),
            selected=[],
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        reserved_reference_calibration = _task_balanced(
            reference_calibration_candidates,
            reference_calibration_total,
            max_novel=novel_image_limit,
            reference_empty_rate=reference_proxy_empty_rate,
        )
        reserved_reference_novel = sum(
            not item["is_replay"] for item in reserved_reference_calibration
        )
    defect_detection_novel_limit = max(
        0, novel_image_limit - reserved_reference_novel
    )
    if hybrid_calibration:
        assert single_image_calibration_max_empty is not None
        assert single_image_calibration_max_few is not None
        calibration_empty = [item for item in empty if item["route_tier"] == "calibration"]
        strict_empty = [item for item in empty if item["route_tier"] == "strict"]
        calibration_positive = [
            item for item in positive if item["route_tier"] == "calibration"
        ]
        strict_positive = [item for item in positive if item["route_tier"] == "strict"]
        calibration_empty_unique = _without_visual_duplicates(
            calibration_empty,
            selected=reserved_reference_calibration,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        selected_calibration_empty: list[dict[str, Any]] = []
        calibration_novel_count = 0
        for item in calibration_empty_unique:
            if (
                not item["is_replay"]
                and calibration_novel_count >= defect_detection_novel_limit
            ):
                continue
            selected_calibration_empty.append(item)
            calibration_novel_count += not item["is_replay"]
            if len(selected_calibration_empty) == single_image_calibration_max_empty:
                break
        calibration_positive_unique = _without_visual_duplicates(
            calibration_positive,
            selected=selected_calibration_empty,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        selected_calibration_positive, _ = _balanced_positive_selection(
            calibration_positive_unique,
            single_image_calibration_max_few,
            max_novel=max(
                0, defect_detection_novel_limit - calibration_novel_count
            ),
        )
        selected_calibration = _interleave_groups(
            [selected_calibration_empty, selected_calibration_positive]
        )
        strict_empty_unique = _without_visual_duplicates(
            strict_empty,
            selected=selected_calibration,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        strict_positive_unique = _without_visual_duplicates(
            strict_positive,
            selected=selected_calibration + strict_empty_unique,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        strict_positive_ordered, _ = _balanced_positive_selection(
            strict_positive_unique,
            len(strict_positive_unique),
            max_novel=defect_detection_novel_limit,
        )
        strict_ordered: list[dict[str, Any]] = []
        for index in range(max(len(strict_empty_unique), len(strict_positive_ordered))):
            if index < len(strict_empty_unique):
                strict_ordered.append(strict_empty_unique[index])
            if index < len(strict_positive_ordered):
                strict_ordered.append(strict_positive_ordered[index])
        selected_strict: list[dict[str, Any]] = []
        selected_novel = sum(not item["is_replay"] for item in selected_calibration)
        for item in strict_ordered:
            # mined pool cap on Defect Detection: only the task-strict (mined) rows count
            if dd_mined_limit is not None and len(selected_strict) >= dd_mined_limit:
                break
            if (
                not item["is_replay"]
                and selected_novel >= defect_detection_novel_limit
            ):
                continue
            selected_strict.append(item)
            selected_novel += not item["is_replay"]
            if len(selected_calibration) + len(selected_strict) == dd_selection_limit:
                break
        selected_dd = _interleave_groups([selected_strict, selected_calibration])
        selected_empty = [item for item in selected_dd if not item["objects"]]
        selected_positive = [item for item in selected_dd if item["objects"]]
        marginal_quotas = _positive_quota_report(
            positive, selected_positive, len(selected_positive)
        )
    else:
        assert empty_target is not None
        assert positive_target is not None
        # mined pool cap on Defect Detection under the proxy-rate policy: no fixed calibration
        # slot exists here, so the cap bounds every selected Defect Detection row (conservative)
        if dd_mined_limit is not None:
            dd_selection_limit = min(dd_selection_limit, dd_mined_limit)
        empty_selection_limit = math.floor(
            dd_selection_limit * proxy_empty_rate + 0.5
        )
        positive_selection_limit = dd_selection_limit - empty_selection_limit
        eligible_empty = _without_visual_duplicates(
            empty, selected=[], hamming_distance=near_duplicate_hamming_distance, counters=counters
        )
        selected_empty = []
        empty_novel_limit = min(empty_selection_limit, novel_image_limit)
        empty_novel_count = 0
        for item in eligible_empty:
            if not item["is_replay"] and empty_novel_count >= empty_novel_limit:
                continue
            selected_empty.append(item)
            empty_novel_count += not item["is_replay"]
            if len(selected_empty) == empty_selection_limit:
                break
        positive_unique = _without_visual_duplicates(
            positive,
            selected=selected_empty,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        selected_empty_novel = empty_novel_count
        selected_positive, marginal_quotas = _balanced_positive_selection(
            positive_unique,
            positive_selection_limit,
            max_novel=min(
                positive_selection_limit,
                novel_image_limit - selected_empty_novel,
            ),
        )
        selected_dd = selected_empty + selected_positive
    maintenance_pool = [
        item
        for item in maintenance
        if not (
            hybrid_calibration
            and item["task_type"] == REFERENCE_DEFECT_DETECTION_TASK
            and item["route_tier"] == "calibration"
        )
    ]
    # Feature P5-S.1: under ``off`` every maintenance task is de-duplicated against its own
    # already-selected rows only (reference pairs keep their pair identity); the rows this unlocks
    # are counted against today's shared exclusion so the effect stays auditable in the manifest.
    maintenance_rows_unlocked: dict[str, int] | None = None
    if within_task_only:
        maintenance_unique = _without_cross_task_visual_duplicates(
            maintenance_pool,
            selected=selected_dd + reserved_reference_calibration,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
        shared_unique = _without_visual_duplicates(
            maintenance_pool,
            selected=selected_dd + reserved_reference_calibration,
            hamming_distance=near_duplicate_hamming_distance,
            counters=Counter(),
        )
        kept_within_task = Counter(item["task_type"] for item in maintenance_unique)
        kept_shared = Counter(item["task_type"] for item in shared_unique)
        maintenance_rows_unlocked = {
            task: int(kept_within_task[task] - kept_shared[task]) for task in MAINTENANCE_TASK_TYPES
        }
    else:
        maintenance_unique = _without_visual_duplicates(
            maintenance_pool,
            selected=selected_dd + reserved_reference_calibration,
            hamming_distance=near_duplicate_hamming_distance,
            counters=counters,
        )
    selected_dd_novel = sum(not item["is_replay"] for item in selected_dd)
    selected_maintenance = reserved_reference_calibration + _task_balanced(
        maintenance_unique,
        max(
            0,
            (
                target_rows - len(selected_dd)
                if resolved_repetition["enabled"]
                else target_rows
            )
            - len(reserved_reference_calibration),
        ),
        max_novel=max(
            0,
            novel_image_limit
            - (
                0
                if not resolved_repetition["enabled"]
                and novel_image_limit >= target_rows
                else selected_dd_novel
            )
            - reserved_reference_novel,
        ),
        reference_empty_rate=reference_proxy_empty_rate,
        # the manifest checks the empty rate over calibration + mined reference rows
        # together; seed the running total so the mined slice completes that target
        # instead of rounding its own slice (3,000 + 80 rows missed it by one, 2026-09-14)
        reference_seed=(
            (
                len(reserved_reference_calibration),
                sum(not item.get("objects") for item in reserved_reference_calibration),
            )
            if hybrid_calibration and reserved_reference_calibration
            else None
        ),
        # Feature P5-S: mined pool-cap remainders and the priority fill order (defaults: unchanged)
        task_limits=maintenance_limits or None,
        fill_order=fill_order or None,
    )
    # Guard-aware calibration (Feature B3): the slot keeps its row count; its empty rows are
    # bounded by the headroom the caps leave once the cumulative corpus and this iteration's
    # non-calibration rows are counted, and few-box rows from the same source fill the rest.
    calibration_empty_headroom: dict[str, int | None] = {
        "overall": None,
        **{f"task:{task}": None for task in GUARD_AWARE_CALIBRATION_TASKS},
    }
    kpi_calibration_empty_targets = {
        DEFECT_DETECTION_TASK: (
            len(selected_calibration_empty) if hybrid_calibration else 0
        ),
        REFERENCE_DEFECT_DETECTION_TASK: sum(
            not item.get("objects") for item in reserved_reference_calibration
        ),
    }
    calibration_fewbox_substituted = {task: 0 for task in GUARD_AWARE_CALIBRATION_TASKS}
    # few-box / changed rows the guard-aware split asked for that the feed did not hold
    calibration_fewbox_shortfall = {task: 0 for task in GUARD_AWARE_CALIBRATION_TASKS}
    # Feature B3.1: empty rows kept beyond the headroom because that reserve ran out
    calibration_headroom_overflow_rows = {task: 0 for task in GUARD_AWARE_CALIBRATION_TASKS}
    headroom_empty_targets: dict[str, int] | None = None
    reference_empty_substituted = 0
    guard_aware_ledger: dict[str, Any] | None = None
    if calibration_guard_aware:
        assert reference_calibration_total is not None
        assert single_image_calibration_max_few is not None
        mined_maintenance = selected_maintenance[len(reserved_reference_calibration):]
        non_calibration = [*selected_strict, *mined_maintenance]
        previous_rows, previous_empty = _empty_ledger(previous_records or [], context="previous record")
        current_rows, current_empty = _empty_ledger(
            (item["record"] for item in non_calibration), context="current non-calibration row"
        )
        slot_rows = {
            DEFECT_DETECTION_TASK: len(selected_calibration),
            REFERENCE_DEFECT_DETECTION_TASK: len(reserved_reference_calibration),
        }
        all_tasks = set(previous_rows) | set(current_rows) | set(slot_rows)

        def headroom_for(cap: float, tasks: Iterable[str]) -> int:
            names = list(tasks)
            rows = sum(previous_rows[task] + current_rows[task] + slot_rows.get(task, 0) for task in names)
            empty = sum(previous_empty[task] + current_empty[task] for task in names)
            return empty_headroom(cap=cap, rows=rows, empty_rows=empty)

        limits = dict(kpi_calibration_empty_targets)
        if guard["max_empty_answer_share"] is not None:
            calibration_empty_headroom["overall"] = headroom_for(guard["max_empty_answer_share"], all_tasks)
        for task in GUARD_AWARE_CALIBRATION_TASKS:
            cap = guard["max_empty_answer_share_task"].get(task)
            if cap is not None:
                calibration_empty_headroom[f"task:{task}"] = headroom_for(cap, [task])
                limits[task] = min(limits[task], calibration_empty_headroom[f"task:{task}"])
        overall_headroom = calibration_empty_headroom["overall"]
        empty_targets = (
            allocate_empty_headroom(limits, overall_headroom) if overall_headroom is not None else limits
        )
        headroom_empty_targets = dict(empty_targets)
        guard_aware_ledger = {
            "previous": {"rows": sum(previous_rows.values()), "empty_rows": sum(previous_empty.values())},
            "previous_by_task": {
                task: {"rows": previous_rows[task], "empty_rows": previous_empty[task]} for task in sorted(previous_rows)
            },
            "current_non_calibration": {"rows": len(non_calibration), "empty_rows": sum(current_empty.values())},
            "current_non_calibration_by_task": {
                task: {"rows": current_rows[task], "empty_rows": current_empty[task]} for task in sorted(current_rows)
            },
            "calibration_slot_rows": slot_rows,
        }
        # Reference pairs first (they were reserved first): the empty bucket is consumed in the
        # same order, so the kept no-change pairs are a prefix of the KPI-rate selection.
        reference_target = empty_targets[REFERENCE_DEFECT_DETECTION_TASK]
        kpi_reference_target = kpi_calibration_empty_targets[REFERENCE_DEFECT_DETECTION_TASK]
        if reference_calibration_total and reference_target < kpi_reference_target:
            kept = [*selected_strict, *mined_maintenance, *selected_calibration]
            reference_pool = _without_visual_duplicates(
                reference_calibration_candidates,
                selected=_visual_exclusion_scope(
                    kept, REFERENCE_DEFECT_DETECTION_TASK, within_task_only=within_task_only
                ),
                hamming_distance=near_duplicate_hamming_distance,
                counters=counters,
            )
            reference_max_novel = max(
                0, novel_image_limit - _novel_rows(selected_strict) - _novel_rows(mined_maintenance) - len(selected_calibration)
            )
            # Best-effort substitution (Feature B3.1): changed pairs take the slot rows the headroom
            # removed from the no-change bucket; when the changed reserve runs out, the rows still
            # missing are no-change pairs beyond the headroom (the next ones of the KPI-rate
            # selection, never more than it carried), so the slot stays full and the overflow is
            # recorded for the assembler's guard instead of failing the count contract here.
            no_change_allowance = reference_target
            while True:
                reserved_reference_calibration = _task_balanced(
                    reference_pool,
                    reference_calibration_total,
                    max_novel=reference_max_novel,
                    reference_empty_rate=no_change_allowance / reference_calibration_total,
                )
                slot_short = reference_calibration_total - len(reserved_reference_calibration)
                if slot_short <= 0 or no_change_allowance >= kpi_reference_target:
                    break
                no_change_allowance = min(kpi_reference_target, no_change_allowance + slot_short)
            reserved_reference_novel = _novel_rows(reserved_reference_calibration)
            selected_maintenance = reserved_reference_calibration + mined_maintenance
            reference_no_change_selected = sum(not item.get("objects") for item in reserved_reference_calibration)
            reference_calibration_empty_target = reference_no_change_selected
            calibration_headroom_overflow_rows[REFERENCE_DEFECT_DETECTION_TASK] = max(
                0, reference_no_change_selected - reference_target
            )
            calibration_fewbox_shortfall[REFERENCE_DEFECT_DETECTION_TASK] = max(
                0,
                (reference_calibration_total - reference_target)
                - (len(reserved_reference_calibration) - reference_no_change_selected),
            )
        reference_empty_substituted = kpi_reference_target - sum(
            not item.get("objects") for item in reserved_reference_calibration
        )
        calibration_fewbox_substituted[REFERENCE_DEFECT_DETECTION_TASK] = reference_empty_substituted
        # Single-image slot: keep the first ``target`` empties (same order), re-run the balanced
        # few-box selection with the slot's row count as its target. Best-effort (Feature B3.1):
        # when the few-box reserve runs out, the next empties of the same ordering fill the slot
        # (never more than the KPI selection carried) and the overflow is recorded.
        single_target = empty_targets[DEFECT_DETECTION_TASK]
        kpi_single_target = kpi_calibration_empty_targets[DEFECT_DETECTION_TASK]
        if single_target < kpi_single_target:
            fewbox_target = single_image_calibration_max_few + (kpi_single_target - single_target)
            empty_allowance = single_target
            while True:
                new_empty = selected_calibration_empty[:empty_allowance]
                kept = [*selected_strict, *selected_maintenance, *new_empty]
                substituted = kpi_single_target - len(new_empty)
                positive_pool = _without_visual_duplicates(
                    calibration_positive,
                    selected=_visual_exclusion_scope(
                        kept, DEFECT_DETECTION_TASK, within_task_only=within_task_only
                    ),
                    hamming_distance=near_duplicate_hamming_distance,
                    counters=counters,
                )
                new_positive, _ = _balanced_positive_selection(
                    positive_pool,
                    single_image_calibration_max_few + substituted,
                    max_novel=max(0, novel_image_limit - _novel_rows(kept)),
                )
                slot_short = single_image_calibration_max_few + substituted - len(new_positive)
                if slot_short <= 0 or empty_allowance >= kpi_single_target:
                    break
                empty_allowance = min(kpi_single_target, empty_allowance + slot_short)
            calibration_fewbox_shortfall[DEFECT_DETECTION_TASK] = max(0, fewbox_target - len(new_positive))
            calibration_fewbox_substituted[DEFECT_DETECTION_TASK] = substituted
            calibration_headroom_overflow_rows[DEFECT_DETECTION_TASK] = len(new_empty) - single_target
            selected_calibration_empty = new_empty
            selected_calibration_positive = new_positive
            selected_calibration = _interleave_groups([new_empty, new_positive])
            selected_dd = _interleave_groups([selected_strict, selected_calibration])
            selected_empty = [item for item in selected_dd if not item["objects"]]
            selected_positive = [item for item in selected_dd if item["objects"]]
            marginal_quotas = _positive_quota_report(positive, selected_positive, len(selected_positive))
    accepted_target_rows: int | None = None
    if not resolved_repetition["enabled"]:
        for candidate_target in range(
            requested_target_rows,
            minimum_rows_aligned - 1,
            -row_multiple,
        ):
            # The configured fraction is a lower bound. Use additional DD
            # capacity when maintenance cannot fill the candidate batch.
            candidate_dd_min = math.ceil(
                max(candidate_target - acquisition_rows, 0) * defect_detection_fraction
            )
            candidate_dd = max(
                candidate_dd_min,
                candidate_target - len(selected_maintenance),
            )
            candidate_maintenance = candidate_target - candidate_dd
            if hybrid_calibration:
                reference_calibration_complete = (
                    len(reserved_reference_calibration)
                    == reference_calibration_total
                    and sum(
                        not item.get("objects")
                        for item in reserved_reference_calibration
                    )
                    == reference_calibration_empty_target
                )
                feasible = (
                    candidate_dd <= len(selected_dd)
                    and candidate_maintenance <= len(selected_maintenance)
                    and candidate_maintenance >= reference_calibration_total
                    and reference_calibration_complete
                )
            else:
                candidate_empty = math.floor(candidate_dd * proxy_empty_rate + 0.5)
                candidate_positive = candidate_dd - candidate_empty
                feasible = (
                    candidate_empty <= len(selected_empty)
                    and candidate_positive <= len(selected_positive)
                    and candidate_maintenance <= len(selected_maintenance)
                )
            if feasible:
                accepted_target_rows = candidate_target
                target_rows = candidate_target
                dd_target = candidate_dd
                if hybrid_calibration:
                    if candidate_target < requested_target_rows:
                        strict_target = min(len(selected_strict), candidate_dd)
                        calibration_target = min(
                            len(selected_calibration), candidate_dd - strict_target
                        )
                        selected_dd = _interleave_groups(
                            [
                                selected_strict[:strict_target],
                                selected_calibration[:calibration_target],
                            ]
                        )
                    else:
                        selected_dd = selected_dd[:candidate_dd]
                    selected_empty = [item for item in selected_dd if not item["objects"]]
                    selected_positive = [item for item in selected_dd if item["objects"]]
                    marginal_quotas = _positive_quota_report(
                        positive, selected_positive, len(selected_positive)
                    )
                else:
                    empty_target = candidate_empty
                    positive_target = candidate_positive
                    selected_empty = selected_empty[:candidate_empty]
                    selected_positive = selected_positive[:candidate_positive]
                    selected_dd = selected_empty + selected_positive
                    marginal_quotas = _positive_quota_report(
                        positive_unique, selected_positive, candidate_positive
                    )
                selected_maintenance = selected_maintenance[:candidate_maintenance]
                break
    base_selected_entries = selected_dd + selected_maintenance
    base_selected_records = [item["record"] for item in base_selected_entries]
    if not base_selected_records:
        raise ValueError(
            "materialization would add zero new rows: every routed candidate was excluded "
            "(previous rows, evaluation targets, duplicates or history); no policy accepts an empty iteration"
        )
    selected_records, repetition_manifest = apply_repetition_blend(
        base_selected_records,
        row_cap=target_rows,
        config=resolved_repetition,
        deficit_weights=deficit_weights,
        seed=repetition_seed,
        deficit_weight_source=deficit_weight_source,
        row_multiple=row_multiple,
    )
    tasks = Counter(str(row.get("task_type")) for row in selected_records)
    output_fingerprints = Counter(_record_fingerprint(row) for row in selected_records)
    emitted_base_entries = [
        item
        for item in base_selected_entries
        if output_fingerprints[_record_fingerprint(item["record"])]
    ]
    selected_paths = {item["resolved_path"] for item in emitted_base_entries}
    selected_content = {item["content_sha256"] for item in emitted_base_entries}
    # Feature P5-S.1: visual identity is verified over all emitted rows (cross-task
    # de-duplication on) or within each task type (off)
    visual_scopes = (
        [[item for item in emitted_base_entries if item["task_type"] == task] for task in ALL_TASK_TYPES]
        if within_task_only
        else [emitted_base_entries]
    )
    near_duplicate_pairs = 0
    for scope in visual_scopes:
        verification_index = _HammingIndex()
        for phash in (item["perceptual_hash"] for item in scope):
            if (
                near_duplicate_hamming_distance is not None
                and verification_index.has_within(phash, near_duplicate_hamming_distance)
            ):
                near_duplicate_pairs += 1
            verification_index.add(phash)
    unique_target_images = all(
        len({item["resolved_path"] for item in scope}) == len(scope) for scope in visual_scopes
    )
    unique_image_content = all(
        len({item["content_sha256"] for item in scope}) == len(scope) for scope in visual_scopes
    )
    selected_empty_count = len(selected_empty)
    selected_positive_count = len(selected_positive)
    materialized_empty_count = (
        repetition_manifest["tasks"]
        .get(DEFECT_DETECTION_TASK, {})
        .get("empty_rows_emitted", 0)
    )
    selected_replay_count = sum(item["is_replay"] for item in emitted_base_entries)
    selected_novel_count = len(emitted_base_entries) - selected_replay_count
    row_count = len(selected_records)
    expected_steps = row_count // global_batch * epochs if row_count % global_batch == 0 else None
    maintenance_target = target_rows - dd_target
    maintenance_requested = {
        task: maintenance_target // len(MAINTENANCE_TASK_TYPES)
        + (index < maintenance_target % len(MAINTENANCE_TASK_TYPES))
        for index, task in enumerate(MAINTENANCE_TASK_TYPES)
    }
    maintenance_available = Counter(
        item["task_type"]
        for item in maintenance_unique + reserved_reference_calibration
    )
    maintenance_selected = Counter(
        item["task_type"] for item in selected_maintenance
    )
    materialized_maintenance = Counter(
        {task: tasks[task] for task in MAINTENANCE_TASK_TYPES}
    )
    selected_reference = [
        item
        for item in selected_maintenance
        if item["task_type"] == REFERENCE_DEFECT_DETECTION_TASK
    ]
    selected_reference_empty = sum(not item.get("objects") for item in selected_reference)
    # the combined (calibration + mined) reference target; guard-aware calibration lowers it by
    # the no-change pairs it replaced with changed pairs (the mined slice keeps the KPI rate)
    reference_empty_target = (
        max(0, math.floor(len(selected_reference) * reference_proxy_empty_rate + 0.5) - reference_empty_substituted)
        if reference_proxy_empty_rate is not None
        else None
    )
    materialized_reference = [
        row
        for row in selected_records
        if row.get("task_type") == REFERENCE_DEFECT_DETECTION_TASK
    ]
    materialized_reference_empty = sum(
        not _ground_truth_objects(row) for row in materialized_reference
    )
    selected_single_calibration = [
        item for item in selected_dd if item["route_tier"] == "calibration"
    ]
    selected_single_calibration_empty = sum(
        not item["objects"] for item in selected_single_calibration
    )
    selected_single_calibration_few = (
        len(selected_single_calibration) - selected_single_calibration_empty
    )
    selected_strict_dd_count = sum(
        item["route_tier"] == "strict" for item in selected_dd
    )
    selected_reference_calibration = [
        item for item in emitted_base_entries
        if item["task_type"] == REFERENCE_DEFECT_DETECTION_TASK
        and item["route_tier"] == "calibration"
    ]
    reference_content_by_record = {
        item["record_id"]: item["content_sha256"]
        for item in selected_reference_calibration
    }
    selected_reference_calibration_empty = sum(
        not item.get("objects") for item in selected_reference_calibration
    )
    kpi_reference_calibration_empty_target = (
        math.floor(reference_calibration_total * reference_proxy_empty_rate + 0.5)
        if hybrid_calibration
        else None
    )
    reference_calibration_empty_target = (
        kpi_reference_calibration_empty_target - reference_empty_substituted
        if kpi_reference_calibration_empty_target is not None
        else None
    )
    # Zero-new-candidate policy: which maintenance tasks are absent, and whether
    # each absent task is exhausted (no eligible routed row this iteration).
    missing_maintenance = [task for task in MAINTENANCE_TASK_TYPES if materialized_maintenance[task] == 0]
    exhausted_tasks = {
        task: {
            "routed_candidates": int(routed_by_task[task]),
            "eligible_after_exclusion": int(eligible_by_task[task]),
            "selected": int(maintenance_selected[task]),
            "materialized": 0,
        }
        for task in missing_maintenance
        if eligible_by_task[task] == 0
    }
    # Feature P5-S follow-up: a maintenance task absent because its mined pool cap was consumed
    # before this selection (remainder 0) while eligible candidates remain; disjoint from
    # exhausted_tasks (a task with no eligible row is exhausted, whatever its cap says).
    capped_absent_tasks = {
        task: {
            "routed_candidates": int(routed_by_task[task]),
            "eligible_after_exclusion": int(eligible_by_task[task]),
            "cap_rows": cap_rows_by_task[task],
            "used_before": int(mined_used_before[task]),
            "selected": int(maintenance_selected[task]),
            "materialized": 0,
        }
        for task in missing_maintenance
        if task not in exhausted_tasks and mined_remaining_before.get(task) == 0
    }
    skip_exhausted = zero_new_candidate_policy == "skip_exhausted"
    all_maintenance_exhausted = len(exhausted_tasks) == len(MAINTENANCE_TASK_TYPES)
    all_maintenance_absent = len(exhausted_tasks) + len(capped_absent_tasks) == len(MAINTENANCE_TASK_TYPES)
    # verdicts: ``present_or_exhausted`` is the pre-P5-S fact (every absent task exhausted);
    # ``present_or_exhausted_or_capped`` is the binding verdict under skip_exhausted
    if not missing_maintenance:
        maintenance_present_or_exhausted = True
        maintenance_present_or_exhausted_or_capped = True
        zero_new_candidate_block_reason = None
    elif not skip_exhausted:
        maintenance_present_or_exhausted = False
        maintenance_present_or_exhausted_or_capped = False
        zero_new_candidate_block_reason = "policy_fail_closed"
    elif row_count == 0:
        maintenance_present_or_exhausted = False
        maintenance_present_or_exhausted_or_capped = False
        zero_new_candidate_block_reason = "zero_new_rows"
    elif all_maintenance_exhausted:
        maintenance_present_or_exhausted = False
        maintenance_present_or_exhausted_or_capped = False
        zero_new_candidate_block_reason = "all_maintenance_tasks_exhausted"
    elif all_maintenance_absent:
        maintenance_present_or_exhausted = False
        maintenance_present_or_exhausted_or_capped = False
        zero_new_candidate_block_reason = "all_maintenance_tasks_exhausted_or_capped"
    elif set(missing_maintenance) != set(exhausted_tasks) | set(capped_absent_tasks):
        maintenance_present_or_exhausted = False
        maintenance_present_or_exhausted_or_capped = False
        zero_new_candidate_block_reason = "maintenance_task_absent_with_eligible_candidates"
    else:
        maintenance_present_or_exhausted = not capped_absent_tasks
        maintenance_present_or_exhausted_or_capped = True
        zero_new_candidate_block_reason = None
    # the exhausted tasks the policy accepted (accepted capped ones are the capped_absent_tasks keys)
    skipped_tasks = (
        sorted(exhausted_tasks) if (skip_exhausted and maintenance_present_or_exhausted_or_capped) else []
    )
    # Mined per-task pool caps (Feature P5-S): usage ledger over the mined rows this selection emits
    # (calibration rows never count). A task whose remainder is gone is ``capped``; when it is absent
    # for that reason it is in ``capped_absent_tasks`` above, never in ``exhausted_tasks``.
    mined_selected_by_task: Counter[str] = Counter(
        item["task_type"] for item in emitted_base_entries if item["route_tier"] != "calibration"
    )
    mined_task_pool_usage: dict[str, dict[str, Any]] = {}
    for task in ALL_TASK_TYPES:
        cap_rows = cap_rows_by_task.get(task)
        used_before = int(mined_used_before[task])
        selected_now = int(mined_selected_by_task[task])
        remaining_after = max(0, cap_rows - used_before - selected_now) if cap_rows is not None else None
        mined_task_pool_usage[task] = {
            "pool_rows": int(pool_rows_by_task[task]),
            "cap_fraction": mined_caps.get(task),
            "cap_rows": cap_rows,
            "used_before": used_before,
            "selected_now": selected_now,
            "remaining_after": remaining_after,
            "capped_this_iteration": cap_rows is not None and remaining_after == 0,
        }
    capped_tasks = [task for task in ALL_TASK_TYPES if mined_task_pool_usage[task]["capped_this_iteration"]]
    mined_task_fill_realized = {task: int(mined_selected_by_task[task]) for task in ALL_TASK_TYPES}
    verification = {
        "target_rows_reached": row_count == target_rows,
        "minimum_rows_reached": row_count >= minimum_rows_aligned,
        "defect_detection_quota_reached": tasks[DEFECT_DETECTION_TASK] >= dd_target,
        "empty_rate_matched": (
            True
            if hybrid_calibration
            else selected_empty_count == empty_target
            if resolved_repetition["never_repeat_empty_gt"]
            else materialized_empty_count == empty_target
        ),
        "empty_rate_matched_before_repetition": (
            True if hybrid_calibration else selected_empty_count == empty_target
        ),
        # guard-aware substitution moves slots from the empty bucket to the few-box bucket;
        # the effective few-box cap grows by exactly the substituted count (slot count unchanged)
        "single_image_calibration_caps_respected": (
            True
            if not hybrid_calibration
            else selected_single_calibration_empty
            <= single_image_calibration_max_empty
            and selected_single_calibration_few
            <= single_image_calibration_max_few + calibration_fewbox_substituted[DEFECT_DETECTION_TASK]
        ),
        "single_image_proxy_rate_policy_respected": (
            not hybrid_calibration or empty_target is None
        ),
        "task_strict_defect_detection_rows_remain_trainable": (
            not hybrid_calibration
            or selected_strict_dd_count > 0
            or not any(item["route_tier"] == "strict" for item in positive + empty)
        ),
        "reference_calibration_contract_reached": (
            True
            if not hybrid_calibration
            else len(selected_reference_calibration) == reference_calibration_total
            and selected_reference_calibration_empty
            == reference_calibration_empty_target
        ),
        "reference_calibration_content_unique": (
            len(reference_content_by_record)
            == len(set(reference_content_by_record.values()))
            == len(selected_reference_calibration)
        ),
        "reference_empty_rate_matched": (
            True
            if reference_proxy_empty_rate is None
            else materialized_reference_empty == reference_empty_target
        ),
        "reference_empty_rate_matched_before_repetition": (
            True
            if reference_proxy_empty_rate is None
            else selected_reference_empty == reference_empty_target
        ),
        "unique_target_images": unique_target_images,
        "unique_image_content": unique_image_content,
        "near_duplicate_free": near_duplicate_pairs == 0,
        "near_duplicate_filter_policy_respected": (
            near_duplicate_hamming_distance is None or near_duplicate_pairs == 0
        ),
        "optimizer_boundary_aligned": expected_steps is not None,
        "novel_image_limit_respected": (
            selected_novel_count <= novel_image_limit
        ),
        "task_strict_with_authorized_empty_calibration_only": True,
        "repetition_only_accepted_rows": repetition_manifest["invariants"][
            "only_available_rows_emitted"
        ],
        "repetition_empty_policy_respected": repetition_manifest["invariants"][
            "empty_ground_truth_not_repeated"
        ] is not False,
        "repetition_did_not_apply_perceptual_hash_filter": repetition_manifest[
            "invariants"
        ]["perceptual_hash_filter_applied"] is False,
        "all_five_maintenance_tasks_present": all(
            materialized_maintenance[task] > 0 for task in MAINTENANCE_TASK_TYPES
        ),
        # pre-P5-S presence fact: every absent task is exhausted (policy-aware as before; false
        # when a capped task is absent, and excluded from ``verified`` under skip_exhausted)
        _PRESENCE_OR_EXHAUSTED_KEY: maintenance_present_or_exhausted,
        # binding verdict under skip_exhausted: an absent task is acceptable only when it is
        # exhausted or capped (and not every task is absent, and rows are added)
        _PRESENCE_OR_EXHAUSTED_OR_CAPPED_KEY: maintenance_present_or_exhausted_or_capped,
        # Feature P5-S: no task emitted more mined rows than its cap remainder allowed
        "mined_task_pool_caps_respected": all(
            usage["selected_now"] <= max(0, usage["cap_rows"] - usage["used_before"])
            for usage in mined_task_pool_usage.values()
            if usage["cap_rows"] is not None
        ),
    }
    # Empty-answer guard evidence: how many of the rows this selection adds carry an
    # empty ground truth ([] / {} / blank), all tasks; the selection itself is unchanged.
    new_rows_empty_by_task: Counter[str] = Counter(
        str(row.get("task_type"))
        for index, row in enumerate(selected_records)
        if is_empty_ground_truth(row, context=f"selected row[{index}]")
    )
    manifest = {
        "schema_version": "defect_detection_quota_manifest_v1",
        "calibration_scope": "current_new_rows_only",
        "new_rows_empty": sum(new_rows_empty_by_task.values()),
        "new_rows_empty_by_task": dict(sorted(new_rows_empty_by_task.items())),
        # Feature B3: guard-aware calibration (empty vs few-box split of the fixed slot)
        "calibration_guard_aware": bool(calibration_guard_aware),
        "calibration_empty_headroom": calibration_empty_headroom,
        "calibration_empty_selected": {
            DEFECT_DETECTION_TASK: selected_single_calibration_empty,
            REFERENCE_DEFECT_DETECTION_TASK: selected_reference_calibration_empty,
            "total": selected_single_calibration_empty + selected_reference_calibration_empty,
        },
        "calibration_fewbox_substituted": {
            **calibration_fewbox_substituted,
            "total": sum(calibration_fewbox_substituted.values()),
        },
        # Feature B3.1: empty calibration rows kept beyond the headroom (reserve ran out)
        "calibration_headroom_overflow_rows": {
            **calibration_headroom_overflow_rows,
            "total": sum(calibration_headroom_overflow_rows.values()),
        },
        "guard_aware_calibration": {
            "enabled": bool(calibration_guard_aware),
            "caps": {
                "overall": guard["max_empty_answer_share"],
                "per_task": {
                    task: guard["max_empty_answer_share_task"].get(task) for task in GUARD_AWARE_CALIBRATION_TASKS
                },
            },
            "policy": (
                "empty_rows_bounded_by_cap_headroom_fewbox_fill_same_source_slot_count_unchanged"
                if calibration_guard_aware
                else None
            ),
            "substitution": (
                "best_effort_reserve_then_empty_overflow_recorded_for_assembler_guard"
                if calibration_guard_aware
                else None
            ),
            "status": _guard_aware_status(
                calibration_guard_aware, calibration_fewbox_substituted, calibration_headroom_overflow_rows
            ),
            "kpi_empty_targets": kpi_calibration_empty_targets,
            "headroom_empty_targets": headroom_empty_targets,
            "ledger": guard_aware_ledger,
            "fewbox_shortfall": calibration_fewbox_shortfall,
        },
        "zero_new_candidate_policy": zero_new_candidate_policy,
        "verification_policy_exclusions": _verification_policy_exclusions(zero_new_candidate_policy),
        # Feature P5-S: launch-recorded Defect Detection fraction, mined per-task pool caps and fill order
        "defect_detection_fraction": defect_detection_fraction,
        "mined_task_pool_caps": dict(mined_caps),
        "mined_task_pool_cap_policy": MINED_TASK_POOL_CAP_POLICY,
        "mined_task_pool_usage": mined_task_pool_usage,
        "mined_task_fill_order": list(fill_order),
        "mined_task_fill_realized": mined_task_fill_realized,
        "capped_tasks": capped_tasks,
        # absent because the cap was consumed (accepted under skip_exhausted, fatal under fail_closed)
        "capped_absent_tasks": capped_absent_tasks,
        # Feature P5-S.1: cross-task visual de-duplication switch; when off, the maintenance rows per
        # task that today's shared exclusion would have dropped (None when on)
        "cross_task_visual_dedup": cross_task_visual_dedup,
        "maintenance_rows_unlocked_by_cross_task": maintenance_rows_unlocked,
        "maintenance_tasks": {
            "present": [task for task in MAINTENANCE_TASK_TYPES if materialized_maintenance[task] > 0],
            "missing": missing_maintenance,
            "routed_candidates": {task: int(routed_by_task[task]) for task in MAINTENANCE_TASK_TYPES},
            "eligible_after_exclusion": {task: int(eligible_by_task[task]) for task in MAINTENANCE_TASK_TYPES},
        },
        "exhausted_tasks": exhausted_tasks,
        "skipped_tasks": skipped_tasks,
        "zero_new_candidate_block_reason": zero_new_candidate_block_reason,
        "previous_records_excluded": counters["previous_records_excluded"],
        "selection_policy": (
            "defect_detection_hybrid_calibration_task_strict_v2"
            if hybrid_calibration
            else "defect_detection_primary_task_strict_v1"
        ),
        "verified": _manifest_verified(verification, zero_new_candidate_policy),
        "verification": verification,
        "configuration": {
            "media_root": str(media_root),
            "materialization_cap": max_rows,
            "requested_target_rows_after_global_batch_alignment": requested_target_rows,
            "target_rows_after_global_batch_alignment": target_rows,
            "minimum_rows_requested": minimum_rows_requested,
            "minimum_rows_batch_aligned": minimum_rows_aligned,
            "defect_detection_minimum_fraction": defect_detection_fraction,
            "acquisition_rows_excluded_from_floor_base": acquisition_rows,
            "near_duplicate_hamming_distance": near_duplicate_hamming_distance,
            "near_duplicate_filter": (
                "disabled"
                if near_duplicate_hamming_distance is None
                else "perceptual_hamming"
            ),
            "novel_mining_pool_image_limit": novel_image_limit,
            "calibration_policy": (
                "fixed_single_image_caps_plus_task_strict_mining"
                if hybrid_calibration
                else "direct_empty_ground_truth_only_when_proxy_fp_hard_negatives_do_not_fill_proxy_matched_empty_quota"
            ),
            "calibration_cohort_policy": (
                "fixed_single_image_proxy_rate_reference"
                if hybrid_calibration
                else "proxy_empty_rate_by_reference_cohort"
            ),
            "annotation_profile": "nvpaw_multitask_v1",
            "prompt_variant": "official_v1",
            "box_serialization_policy": "corpus_native_unmodified",
            "coordinate_policy": "corpus_native_unmodified",
            "image_preprocessing_policy": "corpus_native_unmodified",
            "box_ordering_policy": "corpus_native_deterministic_unmodified",
        },
        "row_counts": {
            "total": row_count,
            "target": target_rows,
            "requested_target": requested_target_rows,
            "defect_detection": tasks[DEFECT_DETECTION_TASK],
            "defect_detection_target": dd_target,
            "defect_detection_minimum_target": math.ceil(
                (target_rows - acquisition_rows) * defect_detection_fraction
            ),
            "task_strict_defect_detection": selected_strict_dd_count,
            "maintenance": sum(tasks[task] for task in MAINTENANCE_TASK_TYPES),
            "maintenance_target": maintenance_target,
            "by_task": dict(sorted(tasks.items())),
            "novel_mining_pool_images": selected_novel_count,
            "replayed_mining_pool_images": selected_replay_count,
            "repetition_rows": repetition_manifest["totals"]["additional_repetitions"],
        },
        "pre_repetition": {
            "total": len(base_selected_entries),
            "defect_detection": len(selected_dd),
            "maintenance": len(selected_maintenance),
            "empty_ground_truth": selected_empty_count,
        },
        "shortfall": {
            "requested_rows": requested_target_rows,
            "accepted_rows": row_count,
            "rows_below_request": max(0, requested_target_rows - row_count),
            "accepted": bool(
                accepted_target_rows is not None
                and row_count < requested_target_rows
                and row_count >= minimum_rows_aligned
            ),
        },
        "maintenance_task_types": list(MAINTENANCE_TASK_TYPES),
        "maintenance_marginal_quota": {
            "requested": maintenance_requested,
            "available": {
                task: maintenance_available[task] for task in MAINTENANCE_TASK_TYPES
            },
            "selected": {
                task: maintenance_selected[task] for task in MAINTENANCE_TASK_TYPES
            },
            "materialized": {
                task: materialized_maintenance[task]
                for task in MAINTENANCE_TASK_TYPES
            },
            "shortages": {
                task: maintenance_requested[task] - maintenance_available[task]
                for task in MAINTENANCE_TASK_TYPES
                if maintenance_available[task] < maintenance_requested[task]
            },
        },
        "defect_detection_evidence": {
            "positive_rows": selected_positive_count,
            "empty_hard_negative_rows": selected_empty_count,
            "by_type": dict(
                sorted(
                    Counter(
                        evidence
                        for item in selected_dd
                        for evidence in item["evidence"]
                    ).items()
                )
            ),
        },
        "single_image_calibration": {
            "max_empty": single_image_calibration_max_empty,
            "max_few_box": single_image_calibration_max_few,
            # the few-box cap after guard-aware substitution (equal to max_few_box unless it substituted)
            "max_few_box_effective": (
                single_image_calibration_max_few + calibration_fewbox_substituted[DEFECT_DETECTION_TASK]
                if hybrid_calibration
                else None
            ),
            "empty_substituted_by_few_box": calibration_fewbox_substituted[DEFECT_DETECTION_TASK],
            # Feature B3.1: empties kept beyond the headroom because the few-box reserve ran out
            "empty_beyond_headroom": calibration_headroom_overflow_rows[DEFECT_DETECTION_TASK],
            "selected_empty": selected_single_calibration_empty,
            "selected_few_box": selected_single_calibration_few,
            "selected_total": len(selected_single_calibration),
            "proxy_empty_rate_binding": not hybrid_calibration,
            "profile_count_bins": dict(
                sorted(
                    Counter(
                        item["calibration_count_bin"]
                        for item in selected_single_calibration
                        if item.get("calibration_count_bin")
                    ).items()
                )
            ),
        },
        "reference_calibration": {
            "content_identity": PAIR_CONTENT_IDENTITY,
            "content_sha256_by_record_id": reference_content_by_record,
            "requested_total": reference_calibration_total,
            "target_no_change": reference_calibration_empty_target,
            # the KPI-rate target before guard-aware substitution (equal unless it substituted)
            "kpi_target_no_change": kpi_reference_calibration_empty_target,
            "no_change_substituted_by_changed": reference_empty_substituted,
            # Feature B3.1: no-change pairs kept beyond the headroom because the changed reserve ran out
            "no_change_beyond_headroom": calibration_headroom_overflow_rows[REFERENCE_DEFECT_DETECTION_TASK],
            "selected_no_change": selected_reference_calibration_empty,
            "selected_changed": (
                len(selected_reference_calibration)
                - selected_reference_calibration_empty
            ),
            "selected_total": len(selected_reference_calibration),
            "proxy_empty_rate_binding": hybrid_calibration,
            "profile_count_bins": dict(
                sorted(
                    Counter(
                        item["calibration_count_bin"]
                        for item in selected_reference_calibration
                        if item.get("calibration_count_bin")
                    ).items()
                )
            ),
        },
        "empty_ground_truth": {
            "proxy_rate": proxy_empty_rate,
            "target_empty": empty_target,
            "selected_empty": selected_empty_count,
            "selected_non_empty": selected_positive_count,
            "selected_rate": selected_empty_count / len(selected_dd) if selected_dd else None,
            "materialized_empty": materialized_empty_count,
            "materialized_non_empty": (
                tasks[DEFECT_DETECTION_TASK] - materialized_empty_count
            ),
            "materialized_rate": (
                materialized_empty_count / tasks[DEFECT_DETECTION_TASK]
                if tasks[DEFECT_DETECTION_TASK]
                else None
            ),
        },
        "empty_ground_truth_by_cohort": {
            "non_reference_based": {
                "task_type": DEFECT_DETECTION_TASK,
                "proxy_rate": proxy_empty_rate,
                "proxy_empty_rate_binding": not hybrid_calibration,
                "target_empty": empty_target,
                "selected_empty": selected_empty_count,
                "selected_non_empty": selected_positive_count,
                "selected_total": len(selected_dd),
                "selected_rate": (
                    selected_empty_count / len(selected_dd) if selected_dd else None
                ),
                "materialized_empty": materialized_empty_count,
                "materialized_total": tasks[DEFECT_DETECTION_TASK],
            },
            "reference_based": {
                "task_type": REFERENCE_DEFECT_DETECTION_TASK,
                "proxy_rate": reference_proxy_empty_rate,
                "proxy_empty_rate_binding": True,
                "target_empty": reference_empty_target,
                "selected_empty": selected_reference_empty,
                "selected_non_empty": (
                    len(selected_reference) - selected_reference_empty
                ),
                "selected_total": len(selected_reference),
                "selected_rate": (
                    selected_reference_empty / len(selected_reference)
                    if selected_reference
                    else None
                ),
                "materialized_empty": materialized_reference_empty,
                "materialized_total": len(materialized_reference),
            },
        },
        "positive_marginal_quotas": marginal_quotas,
        "uniqueness": {
            "scope": "unique_rows_emitted_before_intentional_repetition",
            # the scope of unique_target_images / unique_image_content / near_duplicate_free
            "visual_identity_scope": "within_task" if within_task_only else "all_tasks",
            "unique_rows": len({_record_fingerprint(item["record"]) for item in emitted_base_entries}),
            "unique_images": len(selected_paths),
            "unique_image_content": len(selected_content),
            "selected_near_duplicate_pairs": near_duplicate_pairs,
            **dict(sorted(counters.items())),
        },
        "optimizer_schedule": {
            "epochs": epochs,
            "global_batch": global_batch,
            "steps_per_epoch": row_count // global_batch if expected_steps is not None else None,
            "expected_optimizer_steps": expected_steps,
        },
        "warnings": list(repetition_manifest["warnings"]),
        "repetition_blend": repetition_manifest,
    }
    return selected_records, manifest


def _verify_reference_calibration_content(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Verify current-only quotas against the materializer's ordered-byte identities."""

    reference = manifest.get("reference_calibration") or {}
    required = reference.get("requested_total")
    if required is None or required == 0:
        return
    identities = reference.get("content_sha256_by_record_id")
    media_root = manifest.get("configuration", {}).get("media_root")
    if (
        reference.get("content_identity") != PAIR_CONTENT_IDENTITY
        or not isinstance(identities, dict)
        or not isinstance(media_root, str)
        or not media_root
    ):
        raise ValueError("reference calibration content identity evidence is missing or incompatible")
    if not all(isinstance(value, str) and len(value) == 64 for value in identities.values()):
        raise ValueError("reference calibration content SHA-256 evidence is invalid")
    unique_contents = len(set(identities.values()))
    if (
        unique_contents != len(identities)
        or unique_contents != required
        or unique_contents != reference.get("selected_total")
    ):
        raise ValueError(
            "reference calibration content-unique shortfall or count mismatch: "
            f"required={required} available={unique_contents} "
            f"shortfall={max(0, required - unique_contents)}"
        )
    by_id = {str(row.get("id")): row for row in records}
    no_change = 0
    for record_id, expected_content in identities.items():
        record = by_id.get(record_id)
        if record is None or record.get("task_type") != REFERENCE_DEFECT_DETECTION_TASK:
            raise ValueError(f"reference calibration content record is missing: {record_id}")
        sample = sample_from_record(
            record, media_root=pathlib.Path(media_root), context="reference calibration gate"
        )
        # This is exactly the materializer's identity policy: rehash ordered
        # original bytes when present, otherwise use its precomputed identity
        # for metadata-only/cache-backed verification. Never hash a canvas.
        _, actual_content, _ = _candidate_visual_identity(
            {
                **sample,
                "filepath": sample["target_filepath"],
                "source_image_paths": sample["image_paths"],
                "content_sha256": expected_content,
            },
            media_root=pathlib.Path(media_root),
            compute_perceptual_hash=False,
        )
        if actual_content != expected_content:
            raise ValueError(f"reference calibration ordered-pair content changed: {record_id}")
        no_change += not _ground_truth_objects(record)
    if (
        no_change != reference.get("selected_no_change")
        or no_change != reference.get("target_no_change")
        or unique_contents - no_change != reference.get("selected_changed")
    ):
        raise ValueError("reference calibration content-unique no-change/changed counts disagree")


def bind_manifest(manifest: dict[str, Any], training_jsonl: pathlib.Path) -> dict[str, Any]:
    path = training_jsonl.expanduser().resolve(strict=True)
    bound = json.loads(json.dumps(manifest))
    with path.open("rb") as stream:
        rows = sum(1 for line in stream if line.strip())
    bound["training_jsonl"] = {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": rows,
    }
    if bound["training_jsonl"]["rows"] != bound["row_counts"]["total"]:
        raise ValueError("training JSONL row count differs from the quota manifest")
    return bound


def bind_cumulative_manifest(
    current_manifest: dict[str, Any],
    *,
    current_jsonl: pathlib.Path,
    training_jsonl: pathlib.Path,
    assembly_summary: dict[str, Any],
    epochs: int,
    global_batch: int,
) -> dict[str, Any]:
    """Bind current-selection quotas to the final cumulative train JSONL."""

    if current_manifest.get("schema_version") != "defect_detection_quota_manifest_v1":
        raise ValueError("current quota manifest must use schema version 1")
    if current_manifest.get("verified") is not True:
        raise ValueError("current Defect Detection quota manifest is not verified")
    current_path = current_jsonl.expanduser().resolve(strict=True)
    current_rows = load_records(current_path)
    current_binding = current_manifest.get("training_jsonl", {})
    if (
        current_binding.get("path") != str(current_path)
        or current_binding.get("sha256") != _sha256(current_path)
        or current_binding.get("rows") != len(current_rows)
        or current_manifest.get("row_counts", {}).get("total")
        != len(current_rows)
    ):
        raise ValueError("current quota manifest does not bind the selector output")
    _verify_reference_calibration_content(current_manifest, current_rows)
    final_path = training_jsonl.expanduser().resolve(strict=True)
    if current_path == final_path:
        raise ValueError("selector output cannot also be the cumulative training JSONL")
    final_rows = load_records(final_path)
    assembly_binding = assembly_summary.get("training_jsonl", {})
    if (
        assembly_summary.get("mined_input") != str(current_path)
        or assembly_binding.get("path") != str(final_path)
        or assembly_binding.get("sha256") != _sha256(final_path)
        or assembly_binding.get("rows") != len(final_rows)
    ):
        raise ValueError("assembly summary does not bind the cumulative training JSONL")
    if assembly_summary.get("retained_previous_records") != assembly_summary.get(
        "previous_records"
    ) or assembly_summary.get("previous_fingerprints_subset") is not True:
        raise ValueError("assembly summary does not prove cumulative lineage")
    if len(final_rows) % global_batch:
        raise ValueError("cumulative training rows must align to the global batch")

    bound = json.loads(json.dumps(current_manifest))
    bound["schema_version"] = "defect_detection_quota_manifest_v2"
    bound["current_selection"] = {
        "scope": "current_new_rows_only",
        "training_jsonl": bound.pop("training_jsonl"),
        "row_counts": bound["row_counts"],
        "optimizer_schedule": bound["optimizer_schedule"],
        "single_image_calibration": bound.get("single_image_calibration"),
        "reference_calibration": bound.get("reference_calibration"),
        "repetition_blend": bound.get("repetition_blend"),
        **{key: bound.get(key) for key in CURRENT_SELECTION_COPIED_KEYS},
    }
    bound["repetition_blend"] = assembly_summary["repetition_blend"]
    tasks = Counter(str(row.get("task_type")) for row in final_rows)
    final_count = len(final_rows)
    bound["row_counts"] = {
        "total": final_count,
        "defect_detection": tasks[DEFECT_DETECTION_TASK],
        "maintenance": sum(tasks[task] for task in MAINTENANCE_TASK_TYPES),
        "by_task": dict(sorted(tasks.items())),
    }
    bound["training_jsonl"] = {
        "path": str(final_path),
        "sha256": _sha256(final_path),
        "rows": final_count,
    }
    bound["optimizer_schedule"] = {
        "epochs": epochs,
        "global_batch": global_batch,
        "steps_per_epoch": final_count // global_batch,
        "expected_optimizer_steps": final_count // global_batch * epochs,
    }
    bound["lineage"] = {
        key: assembly_summary[key]
        for key in (
            "previous_iteration",
            "previous_sha256",
            "previous_records",
            "retained_previous_records",
            "previous_fingerprints_subset",
            "output_fingerprints_sha256",
        )
    }
    bound["verification"]["cumulative_lineage_verified"] = True
    # the only presence the policy may leave unmet is an exhausted maintenance task
    bound["verified"] = _manifest_verified(
        bound["verification"], str(bound.get("zero_new_candidate_policy") or DEFAULT_ZERO_NEW_CANDIDATE_POLICY)
    )
    return bound


def verify_bound_manifest(
    manifest_path: pathlib.Path,
    *,
    training_jsonl: pathlib.Path,
    expected_rows: int,
    epochs: int,
    global_batch: int,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.expanduser().resolve(strict=True).read_text())
    if payload.get("schema_version") not in {
        "defect_detection_quota_manifest_v1",
        "defect_detection_quota_manifest_v2",
    }:
        raise ValueError("unsupported Defect Detection quota manifest schema")
    if payload.get("verified") is not True:
        raise ValueError("Defect Detection quota manifest is not verified")
    path = training_jsonl.expanduser().resolve(strict=True)
    binding = payload.get("training_jsonl", {})
    if binding.get("sha256") != _sha256(path):
        raise ValueError("training JSONL SHA-256 differs from the quota manifest")
    if binding.get("path") != str(path):
        raise ValueError("training JSONL path differs from the quota manifest")
    if binding.get("rows") != expected_rows or payload["row_counts"].get("total") != expected_rows:
        raise ValueError("training JSONL row count differs from the launch schedule")
    if payload.get("schema_version") == "defect_detection_quota_manifest_v2":
        lineage = payload.get("lineage", {})
        if (
            lineage.get("retained_previous_records")
            != lineage.get("previous_records")
            or lineage.get("previous_fingerprints_subset") is not True
        ):
            raise ValueError("Defect Detection quota manifest lineage is not monotonic")
        current = payload.get("current_selection", {})
        current_binding = current.get("training_jsonl", {})
        current_path = pathlib.Path(str(current_binding.get("path", "")))
        if (
            not current_path.is_file()
            or current_path.resolve() == path
            or current_binding.get("sha256") != _sha256(current_path)
            or current_binding.get("rows")
            != current.get("row_counts", {}).get("total")
        ):
            raise ValueError("current-selection quota binding is invalid")
        if current.get("reference_calibration") != payload.get("reference_calibration"):
            raise ValueError("reference calibration current-selection content evidence disagrees")
        _verify_reference_calibration_content(payload, load_records(current_path))
    else:
        _verify_reference_calibration_content(payload, load_records(path))
    schedule = payload.get("optimizer_schedule", {})
    expected_steps = expected_rows // global_batch * epochs if expected_rows % global_batch == 0 else None
    if (
        schedule.get("epochs") != epochs
        or schedule.get("global_batch") != global_batch
        or schedule.get("expected_optimizer_steps") != expected_steps
    ):
        raise ValueError("quota manifest optimizer schedule differs from the launch schedule")
    return payload


def _read_parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read routed candidates") from exc
    return pq.read_table(path).to_pylist()


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _proxy_empty_rate(records: list[dict[str, Any]]) -> tuple[float, int, int]:
    detection = [row for row in records if row.get("task_type") == DEFECT_DETECTION_TASK]
    if not detection:
        raise ValueError("Proxy has no single-image Defect Detection rows")
    empty = sum(not _ground_truth_objects(row) for row in detection)
    return empty / len(detection), empty, len(detection)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument(
        "--previous-jsonl",
        type=pathlib.Path,
        help=(
            "Prior cumulative train JSONL; exact prior records are excluded so "
            "calibration caps apply only to current additions."
        ),
    )
    parser.add_argument("--proxy-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--validation-jsonl", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--max-rows", required=True, type=int)
    parser.add_argument(
        "--minimum-rows",
        type=int,
        help=(
            "Accept the largest feasible batch-aligned shortfall at or above "
            "this raw row minimum; omitted preserves exact-target behavior."
        ),
    )
    parser.add_argument("--row-multiple", required=True, type=int)
    parser.add_argument(
        "--defect-detection-fraction",
        default=0.5,
        type=float,
        help=(
            "Lower bound on the single-image Defect Detection share of the target rows, in (0, 1] "
            "(default 0.5; launch-recorded by init_deft_state.py --defect-detection-fraction and "
            "rendered by the runner). Recorded in the quota manifest as defect_detection_fraction."
        ),
    )
    parser.add_argument(
        "--mined-task-pool-cap",
        action="append",
        metavar="TASK=FRACTION",
        help=(
            "Cap the mined rows of a task at floor(rows of the task in --source-annotations * FRACTION) "
            "over the whole run (repeatable; FRACTION in (0, 1]; missing tasks are uncapped). The mined rows "
            "the --previous-jsonl corpus already holds for the task (no calibration / anchor / coverage "
            "marker) are used budget; calibration rows never count. Recorded as mined_task_pool_caps / "
            "mined_task_pool_usage."
        ),
    )
    parser.add_argument(
        "--mined-task-fill-order",
        metavar="T1,T2,...",
        help=(
            "Maintenance tasks whose mined rows are filled first, in this order, each up to its cap "
            "remainder / availability, before the remaining tasks fill round-robin (default: today's "
            "round-robin). Defect Detection keeps its fraction lower bound first. Recorded as "
            "mined_task_fill_order / mined_task_fill_realized."
        ),
    )
    parser.add_argument(
        "--cross-task-visual-dedup",
        choices=CROSS_TASK_VISUAL_DEDUP_MODES,
        default=DEFAULT_CROSS_TASK_VISUAL_DEDUP,
        help=(
            "on (default, today's rule): a maintenance-task row is dropped when its image was already selected "
            "for any task. off: visual de-duplication stays within each task type and every record-level / "
            "leakage exclusion is unchanged, but a row is no longer dropped because its image was selected for a "
            "different task (the NVPAW pool asks Defect Classification and Defect Detection on the same boards). "
            "Launch-recorded by init_deft_state.py and rendered by the runner; recorded as cross_task_visual_dedup / "
            "maintenance_rows_unlocked_by_cross_task."
        ),
    )
    parser.add_argument(
        "--acquisition-rows",
        type=int,
        default=0,
        help="Rows reserved for tasks in acquisition mode (Phase 3 capability gate); excluded from the Defect Detection floor base.",
    )
    parser.add_argument("--single-image-calibration-max-empty", type=int)
    parser.add_argument("--single-image-calibration-max-few", type=int)
    parser.add_argument("--reference-calibration-total", type=int)
    parser.add_argument(
        "--calibration-guard-aware",
        choices=("on", "off"),
        default="off",
        help=(
            "on: bound the empty rows of the fixed calibration slot by the empty-answer guard's headroom "
            "(cumulative corpus + this iteration's non-calibration rows) and fill the rest with few-box rows "
            "from the same source; the slot's row count is unchanged. Requires the caps below (the runner "
            "mirrors them from the assembler options)."
        ),
    )
    parser.add_argument(
        "--max-empty-answer-share",
        type=float,
        help="Empty-answer guard overall cap (read-only copy for --calibration-guard-aware on).",
    )
    parser.add_argument(
        "--max-empty-answer-share-task",
        action="append",
        metavar="TASK=SHARE",
        help="Empty-answer guard per-task cap (read-only copy for --calibration-guard-aware on; repeatable).",
    )
    parser.add_argument(
        "--zero-new-candidate-policy",
        choices=ZERO_NEW_CANDIDATE_POLICIES,
        default=DEFAULT_ZERO_NEW_CANDIDATE_POLICY,
        help=(
            "fail_closed (default): every maintenance task must be present in the current selection. "
            "skip_exhausted: a task with zero eligible routed candidates this iteration is skipped and "
            "recorded (exhausted_tasks / skipped_tasks); still fails when all are exhausted or no rows are added."
        ),
    )
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--global-batch", required=True, type=int)
    near_duplicate = parser.add_mutually_exclusive_group()
    near_duplicate.add_argument(
        "--near-duplicate-hamming-distance",
        dest="near_duplicate_hamming_distance",
        default=3,
        type=int,
    )
    near_duplicate.add_argument(
        "--no-near-duplicate-filter",
        dest="near_duplicate_hamming_distance",
        action="store_const",
        const=None,
    )
    parser.add_argument("--novel-image-limit", type=int)
    parser.add_argument(
        "--gap-analysis-summary",
        type=pathlib.Path,
        help="Current iteration gaps_summary.json used to derive per-task deficit weights.",
    )
    parser.add_argument(
        "--repetition-config",
        type=pathlib.Path,
        help="Repetition-blend launch config in TOML or JSON format.",
    )
    repetition_toggle = parser.add_mutually_exclusive_group()
    repetition_toggle.add_argument(
        "--repetition-blend",
        dest="repetition_blend",
        action="store_true",
        default=None,
        help="Enable the post-exclusion repetition blend.",
    )
    repetition_toggle.add_argument(
        "--no-repetition-blend",
        dest="repetition_blend",
        action="store_false",
        help="Disable repetition even when a config file enables it.",
    )
    parser.add_argument("--repetition-policy", choices=REPETITION_POLICIES)
    parser.add_argument("--repetition-rep-min", type=float)
    parser.add_argument("--repetition-rep-max", type=float)
    parser.add_argument("--repetition-budget-multiplier", type=float)
    parser.add_argument("--repetition-share-gap-tolerance", type=float)
    redistribution_toggle = parser.add_mutually_exclusive_group()
    redistribution_toggle.add_argument(
        "--repetition-redistribute",
        dest="repetition_redistribute",
        action="store_true",
        default=None,
    )
    redistribution_toggle.add_argument(
        "--no-repetition-redistribute",
        dest="repetition_redistribute",
        action="store_false",
    )
    empty_toggle = parser.add_mutually_exclusive_group()
    empty_toggle.add_argument(
        "--repetition-never-repeat-empty-gt",
        dest="repetition_never_repeat_empty_gt",
        action="store_true",
        default=None,
    )
    empty_toggle.add_argument(
        "--repetition-allow-empty-gt",
        dest="repetition_never_repeat_empty_gt",
        action="store_false",
    )
    parser.add_argument(
        "--repetition-explicit-multiplier",
        action="append",
        metavar="TASK=MULTIPLIER",
    )
    parser.add_argument("--repetition-seed", type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument(
        "--repetition-manifest",
        type=pathlib.Path,
        help="Defaults to repetition_blend_manifest.json beside --output.",
    )
    args = parser.parse_args(argv)
    try:
        repetition_config = merge_repetition_config(
            load_repetition_config(args.repetition_config)
            if args.repetition_config is not None
            else None,
            {
                "enabled": args.repetition_blend,
                "policy": args.repetition_policy,
                "rep_min": args.repetition_rep_min,
                "rep_max": args.repetition_rep_max,
                "budget_multiplier": args.repetition_budget_multiplier,
                "share_gap_tolerance": args.repetition_share_gap_tolerance,
                "redistribute": args.repetition_redistribute,
                "never_repeat_empty_gt": args.repetition_never_repeat_empty_gt,
                "explicit_multipliers": parse_explicit_multipliers(
                    args.repetition_explicit_multiplier
                ),
                "seed": args.repetition_seed,
            },
        )
        gap_summary = (
            json.loads(args.gap_analysis_summary.read_text(encoding="utf-8"))
            if args.gap_analysis_summary is not None
            else None
        )
        deficit_weights, deficit_weight_source = deficit_weights_from_gap_summary(
            gap_summary,
            [DEFECT_DETECTION_TASK, *MAINTENANCE_TASK_TYPES],
        )
        repetition_seed = args.repetition_seed
        if repetition_seed is None:
            repetition_seed = repetition_config["seed"]
        if repetition_seed is None and isinstance(gap_summary, dict):
            repetition_seed = gap_summary.get("seed")
        if repetition_seed is None:
            repetition_seed = 17
        proxy = load_records(args.proxy_annotations)
        cohort_rates = derive_proxy_empty_rates(proxy)
        single_contract = cohort_rates["non_reference_based"]
        reference_contract = cohort_rates["reference_based"]
        empty_rate = float(single_contract["empty_rate"])
        proxy_empty = int(single_contract["empty_rows"])
        proxy_rows = int(single_contract["total_rows"])
        validations = proxy[:]
        for path in args.validation_jsonl:
            validations.extend(load_records(path))
        guard_config = validate_empty_answer_guard_config(
            args.max_empty_answer_share, parse_task_shares(args.max_empty_answer_share_task), None, None
        )
        mined_task_pool_caps = parse_mined_task_pool_caps(args.mined_task_pool_cap)
        mined_task_fill_order = parse_mined_task_fill_order(args.mined_task_fill_order)
        rows, manifest = materialize(
            candidate_rows=_read_parquet(args.candidate_parquet),
            source_records=load_records(args.source_annotations),
            previous_records=(
                load_records(args.previous_jsonl)
                if args.previous_jsonl is not None
                else None
            ),
            validation_records=validations,
            media_root=args.media_root,
            max_rows=args.max_rows,
            minimum_rows=args.minimum_rows,
            row_multiple=args.row_multiple,
            defect_detection_fraction=args.defect_detection_fraction,
            acquisition_rows=args.acquisition_rows,
            zero_new_candidate_policy=args.zero_new_candidate_policy,
            proxy_empty_rate=empty_rate,
            reference_proxy_empty_rate=float(reference_contract["empty_rate"]),
            single_image_calibration_max_empty=(
                args.single_image_calibration_max_empty
            ),
            single_image_calibration_max_few=args.single_image_calibration_max_few,
            reference_calibration_total=args.reference_calibration_total,
            epochs=args.epochs,
            global_batch=args.global_batch,
            near_duplicate_hamming_distance=args.near_duplicate_hamming_distance,
            novel_image_limit=args.novel_image_limit,
            repetition_config=repetition_config,
            deficit_weights=deficit_weights,
            repetition_seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            empty_answer_guard=guard_config,
            calibration_guard_aware=args.calibration_guard_aware == "on",
            mined_task_pool_caps=mined_task_pool_caps,
            mined_task_fill_order=mined_task_fill_order,
            cross_task_visual_dedup=args.cross_task_visual_dedup,
        )
        manifest["empty_ground_truth"]["proxy_empty_rows"] = proxy_empty
        manifest["empty_ground_truth"]["proxy_defect_detection_rows"] = proxy_rows
        manifest["empty_ground_truth_by_cohort"]["non_reference_based"].update(
            {
                "proxy_empty_rows": proxy_empty,
                "proxy_detection_rows": proxy_rows,
            }
        )
        manifest["empty_ground_truth_by_cohort"]["reference_based"].update(
            {
                "proxy_empty_rows": int(reference_contract["empty_rows"]),
                "proxy_detection_rows": int(reference_contract["total_rows"]),
            }
        )
        _write_jsonl(args.output, rows)
        repetition_manifest = bind_repetition_manifest(
            manifest["repetition_blend"], args.output
        )
        manifest["repetition_blend"] = repetition_manifest
        manifest = bind_manifest(manifest, args.output)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        repetition_manifest_path = (
            args.repetition_manifest
            or args.output.with_name("repetition_blend_manifest.json")
        )
        repetition_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        repetition_manifest_path.write_text(
            json.dumps(repetition_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not manifest["verified"]:
            raise ValueError(
                "Defect Detection materialization quota is not verified; inspect the manifest"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"defect_detection_ablation: {exc}", file=sys.stderr)
        return 2
    skipped = manifest.get("skipped_tasks") or []
    capped_absent = sorted(manifest.get("capped_absent_tasks") or {})
    notes = (
        ([f" skipped_tasks={skipped}"] if skipped else [])
        + ([f" capped_tasks={capped_absent}"] if capped_absent else [])
    )
    print(
        f"defect_detection_ablation: wrote {len(rows)} rows; "
        f"Defect Detection={manifest['row_counts']['defect_detection']} verified=true"
        + "".join(notes)
        + (" (zero_new_candidate_policy=skip_exhausted)" if notes else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
