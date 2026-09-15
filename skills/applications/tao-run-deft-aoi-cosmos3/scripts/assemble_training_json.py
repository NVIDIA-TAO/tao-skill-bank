#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Assemble monotonic, real-mining-only NVPAW training JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Callable, Iterable

from anchor_rows import (
    ANCHOR_MARK,
    ANCHOR_SOURCE_KIND,
    select_anchors,
    task_shares_from_jsonl,
    validate_anchor_config,
    write_manifest as write_anchor_manifest,
)
from answer_profile import (
    EMPTY_DEFINITION,
    GUARD_MODES,
    is_empty_ground_truth,
    parse_task_shares,
    profile_rows,
    row_profile,
    validate_empty_answer_guard_config,
)
from atomic_samples import logical_record_identity, sample_from_record
from coverage_rows import (
    COVERAGE_MARK,
    COVERAGE_SOURCE_KIND,
    POOL_STATUS_KEY,
    cell_of,
    check_floor_budget,
    joint_targets,
    select_coverage,
    validate_coverage_config,
    write_manifest as write_coverage_manifest,
)
from defect_detection_ablation import (
    CALIBRATION_KIND_MARK,
    CALIBRATION_MARK,
    CLASSIFICATION_CALIBRATION_KIND,
    bind_cumulative_manifest,
)
from nvpaw_annotations import TASK_SPECS
from repetition_blend import (
    POLICIES as REPETITION_POLICIES,
    apply_repetition_blend,
    bind_repetition_manifest,
    deficit_weights_from_gap_summary,
    load_repetition_config,
    merge_repetition_config,
    parse_explicit_multipliers,
    validate_repetition_config,
)
from validate_sharegpt import load_records, target_path


def _fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# Inert top-level slice markers. They must not make a re-mined copy of a
# retained anchor/coverage row look like new content: a previous anchor row
# (marker present) and the same pool row mined again (marker absent) are one
# training record, and the canonical validator requires unique ids.
SLICE_MARKS = (ANCHOR_MARK, COVERAGE_MARK, CALIBRATION_MARK, CALIBRATION_KIND_MARK)
_SLICE_MARKS = SLICE_MARKS

# Classification calibration rows (select_classification_calibration.py) enter
# as a second current-row source; they are protected from the cap trim like
# every ``deft_calibration`` row and counted separately in the summary.
CLASSIFICATION_CALIBRATION_SOURCE_KIND = "classification_calibration"
_CURRENT_KINDS = frozenset({"current_mining", CLASSIFICATION_CALIBRATION_SOURCE_KIND})


def marker_free_key(record: dict[str, Any]) -> str:
    """Content fingerprint with the inert slice markers removed (the dedup identity)."""
    return _fingerprint({key: value for key, value in record.items() if key not in SLICE_MARKS})


_dedup_key = marker_free_key


def record_identity(
    record: dict[str, Any], *, media_root: pathlib.Path | None, context: str = "record"
) -> str:
    """The atomic sample identity the assembler dedups and leak-checks by.

    With a media root, relative image paths are resolved first (``atomic_sample_id``);
    without one the logical path identity is used. Selectors that build exclusion
    sets for the assembler must call this with the same media root.
    """
    if media_root is not None:
        return str(
            sample_from_record(
                record, media_root=media_root.expanduser().resolve(), context=context
            )["atomic_sample_id"]
        )
    return logical_record_identity(record, context=context)


def _load_optional_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Like ``load_records`` but an empty file is an empty list (a zero-row selector output)."""
    with path.open(encoding="utf-8") as stream:
        if not any(line.strip() for line in stream):
            return []
    return load_records(path)


# Empty-answer guard (Phase 4 step 4c-B). The cumulative corpus of the anchors
# run was 43% empty-ground-truth rows (full pool: 22%) and the model answered
# "[]" on 58% of the single-image MCQ; the guard reports the answer profile
# every iteration and, when caps are launch-recorded, trims empty rows added
# this iteration in a fixed order so the share cannot grow again.
# (GUARD_MODES, parse_task_shares and validate_empty_answer_guard_config live in
# answer_profile.py so the materializer can read the same caps; re-exported here.)
GUARD_MAX_PASSES = 25
# Feature B3: with the guard on, anchors may not fill leftover global-batch slots
# beyond their share (r4 reached 20.4% of the corpus against a 0.10 share); the
# aligned size shrinks instead. ``allow`` keeps the parent behaviour byte-for-byte.
ANCHOR_OVERFILL_POLICIES = ("allow", "forbid")
# operator_attention: anchor_share_exceeded when the cumulative anchor share is above
# the configured share by more than this
ANCHOR_SHARE_TOLERANCE = 0.02
# exceeded_untrimmable = a cap is still exceeded but every remaining empty row that
# counts toward it is never-trimmed (previous rows, anchors, coverage, classification
# calibration); the residual is recorded and the iteration continues. exceeded = a
# trimmable remainder is left (enforce mode fails closed).
GUARD_STATUSES = ("within_caps", "trimmed_to_caps", "exceeded_untrimmable", "exceeded")
GUARD_TRIM_ORDER = ("detection_calibration_negative", "mined_empty")
GUARD_NEVER_TRIMMED = (
    "previous_iteration",
    ANCHOR_SOURCE_KIND,
    COVERAGE_SOURCE_KIND,
    CLASSIFICATION_CALIBRATION_SOURCE_KIND,
)
_CLASSIFICATION_FORMATS = ("BCQ", "MCQ")


def check_calibration_guard_aware_manifest(manifest: dict[str, Any], *, expected: bool, source: str) -> None:
    """Fail closed when the materializer's quota manifest disagrees with the launch on guard-aware
    calibration (the materializer runs only with ``--calibration-guard-aware on`` on its command)."""
    actual = manifest.get("calibration_guard_aware") is True
    if actual != expected:
        raise ValueError(
            "guard-aware calibration mismatch: the launch expects --calibration-guard-aware "
            f"{'on' if expected else 'off'} but the current quota manifest {source} was materialized with it "
            f"{'on' if actual else 'off'}; put the same --calibration-guard-aware value on the selector command "
            "(render_iteration_mining_runner.py mirrors the caps to the materializer) or pass the assembler "
            "--calibration-guard-aware explicitly"
        )


def _share(empty: int, rows: int) -> float:
    return empty / rows if rows else 0.0


def _over_cap(empty: int, rows: int, cap: float) -> bool:
    # a share equal to its cap is within the cap
    return rows > 0 and empty > cap * rows + 1e-9


class _GuardLedger:
    """Running row / empty-row counters (overall, per task, classification) for the guard."""

    def __init__(self, profiles: Iterable[dict[str, Any]]) -> None:
        self.rows = 0
        self.empty = 0
        self.task_rows: Counter[str] = Counter()
        self.task_empty: Counter[str] = Counter()
        self.cls_rows = 0
        self.cls_empty = 0
        self.cls_task_rows: Counter[str] = Counter()
        self.cls_task_empty: Counter[str] = Counter()
        for profile in profiles:
            self.add(profile)

    def add(self, profile: dict[str, Any], sign: int = 1) -> None:
        empty = int(bool(profile["empty"])) * sign
        task = profile["task_type"]
        self.rows += sign
        self.empty += empty
        self.task_rows[task] += sign
        self.task_empty[task] += empty
        if profile["format"] in _CLASSIFICATION_FORMATS:
            self.cls_rows += sign
            self.cls_empty += empty
            self.cls_task_rows[task] += sign
            self.cls_task_empty[task] += empty

    def remove(self, profile: dict[str, Any]) -> None:
        self.add(profile, -1)

    def shares(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "empty_rows": self.empty,
            "overall_share": _share(self.empty, self.rows),
            "per_task": {
                task: _share(self.task_empty[task], rows) for task, rows in sorted(self.task_rows.items()) if rows
            },
            "classification_share": _share(self.cls_empty, self.cls_rows),
            "classification_per_task": {
                task: _share(self.cls_task_empty[task], rows)
                for task, rows in sorted(self.cls_task_rows.items())
                if rows
            },
        }

    def counts_for(self, cap_name: str, config: dict[str, Any]) -> tuple[int, int, float]:
        """(empty rows, rows, cap) behind one cap name returned by ``exceeded``."""
        if cap_name == "overall":
            return self.empty, self.rows, float(config["max_empty_answer_share"])
        if cap_name.startswith("classification_task:"):
            task = cap_name.partition(":")[2]
            return self.cls_task_empty[task], self.cls_task_rows[task], float(config["max_classification_empty_share"])
        if cap_name == "classification":
            return self.cls_empty, self.cls_rows, float(config["max_classification_empty_share"])
        if cap_name.startswith("task:"):
            task = cap_name.partition(":")[2]
            return self.task_empty[task], self.task_rows[task], float(config["max_empty_answer_share_task"][task])
        raise ValueError(f"unknown empty-answer cap {cap_name!r}")

    def exceeded(self, config: dict[str, Any]) -> list[str]:
        if not config["enabled"]:
            return []
        names: list[str] = []
        cap = config["max_empty_answer_share"]
        if cap is not None and _over_cap(self.empty, self.rows, cap):
            names.append("overall")
        for task, task_cap in sorted(config["max_empty_answer_share_task"].items()):
            if _over_cap(self.task_empty[task], self.task_rows[task], task_cap):
                names.append(f"task:{task}")
        cap = config["max_classification_empty_share"]
        if cap is not None:
            if _over_cap(self.cls_empty, self.cls_rows, cap):
                names.append("classification")
            for task in sorted(self.cls_task_rows):
                if _over_cap(self.cls_task_empty[task], self.cls_task_rows[task], cap):
                    names.append(f"classification_task:{task}")
        return names


def _rows_over_cap(empty: int, rows: int, cap: float) -> int:
    """Smallest number of empty rows to remove (without replacement) to reach the cap."""
    if rows <= 0 or cap >= 1.0 or not _over_cap(empty, rows, cap):
        return 0
    return max(0, math.ceil((empty - cap * rows) / (1.0 - cap) - 1e-9))


def _guard_row_helps(profile: dict[str, Any], exceeded: list[str]) -> bool:
    task = profile["task_type"]
    if "overall" in exceeded or f"task:{task}" in exceeded:
        return True
    return profile["format"] in _CLASSIFICATION_FORMATS and (
        "classification" in exceeded or f"classification_task:{task}" in exceeded
    )


def _guard_trim_source(record: dict[str, Any], item: dict[str, Any], profile: dict[str, Any]) -> str | None:
    """Which trim group an aligned-corpus row belongs to, or None when it is never trimmed."""
    if item["source_kind"] != "current_mining" or not profile["empty"]:
        return None
    if record.get(CALIBRATION_MARK) is True:
        if record.get(CALIBRATION_KIND_MARK) == CLASSIFICATION_CALIBRATION_KIND:
            return None
        return GUARD_TRIM_ORDER[0]
    return GUARD_TRIM_ORDER[1]


def _guard_trim_plan(
    merged: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    ledger: "_GuardLedger",
    config: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    """Greedy plan over one aligned corpus: detection calibration negatives first, then mined
    empties, tail first, a row only when it lowers a cap that is still exceeded. Mutates ``ledger``."""

    candidates: dict[str, list[int]] = {source: [] for source in GUARD_TRIM_ORDER}
    for index in range(len(merged) - 1, -1, -1):
        source = _guard_trim_source(merged[index], provenance[index], profiles[index])
        if source is not None:
            candidates[source].append(index)
    plan: list[tuple[dict[str, Any], str]] = []
    for source in GUARD_TRIM_ORDER:
        exceeded = ledger.exceeded(config)
        if not exceeded:
            break
        for index in candidates[source]:
            if not _guard_row_helps(profiles[index], exceeded):
                continue
            ledger.remove(profiles[index])
            plan.append((merged[index], source))
            exceeded = ledger.exceeded(config)
            if not exceeded:
                break
    return plan


def _run_empty_answer_guard(
    candidate_merged: list[dict[str, Any]],
    candidate_provenance: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    materialize: Callable[
        [list[dict[str, Any]], list[dict[str, Any]]], tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Report the empty-answer shares of the aligned corpus and, in enforce mode, trim
    empty candidates *before* alignment until the aligned corpus is within the caps.

    ``materialize`` is the cap / global-batch step as a function of the candidate
    set. Each pass materializes, measures the aligned corpus, plans which of its
    trimmable empty rows to drop, removes those rows from the candidate set and
    re-materializes, so the standard selection back-fills from the remaining
    (non-empty) candidates and the aligned size stays unchanged whenever enough
    candidates remain; it shrinks only when candidates run out (reported).
    """

    if materialize is None:
        def materialize(merged: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            return list(merged), list(provenance)

    cand_merged = list(candidate_merged)
    cand_provenance = list(candidate_provenance)
    enforce = config["enabled"] and config["mode"] == "enforce"
    trimmed_by_source: Counter[str] = Counter()
    trimmed_by_task: Counter[str] = Counter()
    first_ids: list[str] | None = None
    before: dict[str, Any] = {}
    exceeded_before: list[str] = []
    passes = 0
    while True:
        passes += 1
        merged, provenance = materialize(cand_merged, cand_provenance)
        profiles = [
            row_profile(record, context=f"empty-answer guard row[{index}]") for index, record in enumerate(merged)
        ]
        ledger = _GuardLedger(profiles)
        if first_ids is None:
            first_ids = [str(record.get("id")) for record in merged]
            before = ledger.shares()
            exceeded_before = ledger.exceeded(config)
        exceeded_after = ledger.exceeded(config)
        if not enforce or not exceeded_after or passes >= GUARD_MAX_PASSES:
            break
        plan = _guard_trim_plan(merged, provenance, profiles, ledger, config)
        if not plan:
            break
        drop = {str(record.get("id")) for record, _ in plan}
        for record, source in plan:
            trimmed_by_source[source] += 1
            trimmed_by_task[str(record.get("task_type"))] += 1
        kept = [
            (record, item)
            for record, item in zip(cand_merged, cand_provenance)
            if str(record.get("id")) not in drop
        ]
        cand_merged = [record for record, _ in kept]
        cand_provenance = [item for _, item in kept]
    final_ids = [str(record.get("id")) for record in merged]
    assert first_ids is not None
    trimmed_total = sum(trimmed_by_source.values())
    # Residual excess per still-exceeded cap: which rows carry it and whether any
    # could still be trimmed. Excess that sits only in never-trimmed rows (previous
    # rows, anchors, coverage, classification calibration) is recorded and the
    # iteration continues (2026-09-15, 4c-B r3 iteration 2: Component Classification
    # had no new candidates and its only new rows were empty-answer anchors, so the
    # classification cap could not be met by trimming). A trimmable remainder fails closed.
    untrimmable_excess: dict[str, Any] = {}
    trimmable_left_any = False
    for cap_name in exceeded_after:
        by_source: Counter[str] = Counter()
        trimmable_left = 0
        for index, profile in enumerate(profiles):
            if not profile["empty"] or not _guard_row_helps(profile, [cap_name]):
                continue
            source = _guard_trim_source(merged[index], provenance[index], profile)
            if source is None:
                by_source[str(provenance[index]["source_kind"])] += 1
            else:
                by_source[source] += 1
                trimmable_left += 1
        empty, rows, cap = ledger.counts_for(cap_name, config)
        untrimmable_excess[cap_name] = {
            "cap": cap,
            "share_after": _share(empty, rows),
            "rows_over_cap": _rows_over_cap(empty, rows, cap),
            "rows_by_source": dict(sorted(by_source.items())),
            "trimmable_rows_left": trimmable_left,
        }
        trimmable_left_any = trimmable_left_any or trimmable_left > 0
    if not config["enabled"]:
        status = "within_caps"
    elif exceeded_after:
        status = "exceeded" if trimmable_left_any else "exceeded_untrimmable"
    elif trimmed_total:
        status = "trimmed_to_caps"
    else:
        status = "within_caps"
    report = {
        "enabled": config["enabled"],
        "mode": config["mode"],
        "caps": {
            "overall": config["max_empty_answer_share"],
            "per_task": dict(config["max_empty_answer_share_task"]),
            "classification": config["max_classification_empty_share"],
        },
        "empty_definition": EMPTY_DEFINITION,
        "policy": "trim_empty_candidates_before_alignment_backfill_from_remaining_candidates",
        "trim_order": list(GUARD_TRIM_ORDER),
        "never_trimmed": list(GUARD_NEVER_TRIMMED),
        "before": before,
        "after": ledger.shares(),
        "exceeded_before": exceeded_before,
        "exceeded_after": exceeded_after,
        "rows_trimmed_total": trimmed_total,
        "rows_trimmed_by_source": {
            source: trimmed_by_source[source] for source in GUARD_TRIM_ORDER if trimmed_by_source[source]
        },
        "rows_trimmed_by_task": dict(sorted(trimmed_by_task.items())),
        "aligned_rows_before": len(first_ids),
        "aligned_rows_after": len(final_ids),
        "aligned_rows_shrunk": max(0, len(first_ids) - len(final_ids)),
        # rows (remaining mined candidates or anchors) that entered the aligned corpus
        # because the trimmed rows freed their slots
        "backfilled_rows": len(set(final_ids) - set(first_ids)),
        "passes": passes,
        "untrimmable_excess": untrimmable_excess,
        "status": status,
    }
    return merged, provenance, report


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_set_sha256(fingerprints: set[str]) -> str:
    digest = hashlib.sha256()
    for fingerprint in sorted(fingerprints):
        digest.update(hashlib.sha256(fingerprint.encode("utf-8")).digest())
    return digest.hexdigest()


def _task_balanced_indices(
    indices: list[int], records: list[dict[str, Any]], limit: int
) -> list[int]:
    """Select deterministically across tasks while preserving order within each task."""

    if limit >= len(indices):
        return indices
    groups: dict[str, list[int]] = {}
    for index in indices:
        task = str(records[index].get("task_type") or "unknown")
        groups.setdefault(task, []).append(index)
    positions = {task: 0 for task in groups}
    selected: list[int] = []
    while len(selected) < limit:
        advanced = False
        for task in sorted(groups):
            position = positions[task]
            if position >= len(groups[task]):
                continue
            selected.append(groups[task][position])
            positions[task] = position + 1
            advanced = True
            if len(selected) == limit:
                break
        if not advanced:
            break
    if len(selected) != limit:
        raise ValueError(
            f"task-balanced materialization selected {len(selected)} rows, expected {limit}"
        )
    return selected


def _exposure_rows(
    records: list[dict[str, Any]], provenance: list[dict[str, Any]], *, prior_anchor_rows: int, prior_coverage_rows: int,
    prior_classification_rows: int = 0,
) -> dict[str, Any]:
    """Per-purpose row ledger of the materialized corpus (rows; tokens are not measured here)."""
    kinds = Counter(item["source_kind"] for item in provenance)
    previous_total = kinds.get("previous_iteration", 0)
    anchor_total = kinds.get(ANCHOR_SOURCE_KIND, 0) + prior_anchor_rows
    coverage_total = kinds.get(COVERAGE_SOURCE_KIND, 0) + prior_coverage_rows
    total = len(records)
    return {
        "unit": "rows",
        "current_mining": kinds.get("current_mining", 0),
        "previous_mined": max(0, previous_total - prior_anchor_rows - prior_coverage_rows - prior_classification_rows),
        "previous_anchor": prior_anchor_rows,
        "previous_coverage": prior_coverage_rows,
        "previous_classification_calibration": prior_classification_rows,
        "anchor_new": kinds.get(ANCHOR_SOURCE_KIND, 0),
        "coverage_new": kinds.get(COVERAGE_SOURCE_KIND, 0),
        "classification_calibration_new": kinds.get(CLASSIFICATION_CALIBRATION_SOURCE_KIND, 0),
        "repetition_copies": kinds.get("repetition", 0),
        "total": total,
        "shares": {
            "anchor": anchor_total / total if total else 0.0,
            "coverage": coverage_total / total if total else 0.0,
        },
    }


def assemble(
    previous_path: pathlib.Path | None,
    mined_path: pathlib.Path,
    *,
    previous_sha256: str | None = None,
    validation_paths: list[pathlib.Path],
    max_rows: int | None = None,
    row_multiple: int | None = None,
    repetition_config: dict[str, Any] | None = None,
    deficit_weights: dict[str, float] | None = None,
    repetition_seed: int | None = None,
    deficit_weight_source: str | None = None,
    media_root: pathlib.Path | None = None,
    anchor_config: dict[str, Any] | None = None,
    anchor_seed: int | None = None,
    coverage_config: dict[str, Any] | None = None,
    coverage_seed: int | None = None,
    classification_calibration_path: pathlib.Path | None = None,
    empty_answer_guard: dict[str, Any] | None = None,
    calibration_guard_aware: bool | None = None,
    anchor_overfill: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor = anchor_config or validate_anchor_config(None, None, None, None)
    coverage = coverage_config or validate_coverage_config(None, None, None, None)
    guard = empty_answer_guard or validate_empty_answer_guard_config(None, None, None, None)
    # Feature B3: anchors may not over-fill leftover slots when the guard is on (default
    # forbid under the guard, allow otherwise so parent-equivalent runs reproduce byte-for-byte)
    anchor_overfill_policy = anchor_overfill or ("forbid" if guard["enabled"] else "allow")
    if anchor_overfill_policy not in ANCHOR_OVERFILL_POLICIES:
        raise ValueError(f"anchor_overfill must be one of {list(ANCHOR_OVERFILL_POLICIES)}, not {anchor_overfill!r}")
    if anchor["enabled"] and (repetition_config or {}).get("enabled"):
        raise ValueError("correct-row anchors and the repetition blend cannot be combined")
    if coverage["enabled"] and (repetition_config or {}).get("enabled"):
        raise ValueError("the coverage blend and the repetition blend cannot be combined")
    if guard["enabled"] and guard["mode"] == "enforce" and (repetition_config or {}).get("enabled"):
        raise ValueError("empty-answer guard enforcement cannot be combined with the repetition blend")
    if anchor["share"] + coverage["share"] >= 1.0:
        raise ValueError("anchor share plus coverage blend share must be below 1")
    for name, value in (("max_rows", max_rows), ("row_multiple", row_multiple)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    resolved_repetition = validate_repetition_config(repetition_config)
    configured_row_cap = resolved_repetition["row_cap"]
    if max_rows is None:
        max_rows = configured_row_cap
    elif configured_row_cap is not None and configured_row_cap != max_rows:
        raise ValueError("repetition row_cap must equal the materialization max_rows")
    if resolved_repetition["enabled"] and max_rows is None:
        raise ValueError("enabled repetition blend requires a materialization row_cap")
    if max_rows is not None:
        resolved_repetition["row_cap"] = max_rows
    if previous_path is None and previous_sha256 is not None:
        raise ValueError("previous_sha256 requires previous_path")
    if previous_path is not None:
        resolved_previous_path = previous_path.expanduser().resolve(strict=True)
        actual_previous_sha256 = sha256_file(resolved_previous_path)
        if (
            previous_sha256 is not None
            and previous_sha256 != actual_previous_sha256
        ):
            raise ValueError("previous training JSONL SHA-256 changed")
    else:
        resolved_previous_path = None
        actual_previous_sha256 = None
    resolved_mined_path = mined_path.expanduser().resolve(strict=True)
    mined = load_records(resolved_mined_path)
    if not mined:
        raise ValueError("the current iteration must contribute at least one mined record")
    previous = (
        load_records(resolved_previous_path)
        if resolved_previous_path is not None
        else []
    )
    resolved_classification_path: pathlib.Path | None = None
    classification_rows: list[dict[str, Any]] = []
    if classification_calibration_path is not None:
        resolved_classification_path = classification_calibration_path.expanduser().resolve(strict=True)
        classification_rows = _load_optional_records(resolved_classification_path)
        for index, record in enumerate(classification_rows):
            if (
                record.get(CALIBRATION_MARK) is not True
                or record.get(CALIBRATION_KIND_MARK) != CLASSIFICATION_CALIBRATION_KIND
            ):
                raise ValueError(
                    f"classification calibration rows must carry {CALIBRATION_MARK}=true and "
                    f"{CALIBRATION_KIND_MARK}={CLASSIFICATION_CALIBRATION_KIND!r} "
                    f"(select_classification_calibration.py output): {resolved_classification_path}:{index}"
                )
    resolved_media_root = media_root.expanduser().resolve() if media_root else None

    def identity(record: dict[str, Any], context: str) -> str:
        return record_identity(record, media_root=resolved_media_root, context=context)

    evaluation_targets: dict[str, str] = {}
    for path in validation_paths:
        for index, record in enumerate(load_records(path)):
            evaluation_targets[identity(record, f"{path}:{index}")] = str(path)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    corpus_ids: set[str] = set()
    duplicate_count = 0
    classification_duplicates = 0
    tasks: Counter[str] = Counter()
    provenance: list[dict[str, Any]] = []
    for source_kind, source_path, records in (
        ("previous_iteration", resolved_previous_path, previous),
        ("current_mining", resolved_mined_path, mined),
        (CLASSIFICATION_CALIBRATION_SOURCE_KIND, resolved_classification_path, classification_rows),
    ):
        if source_path is None:
            continue
        for index, record in enumerate(records):
            target = target_path(record, context=f"{source_path}:{index}")
            sample_identity = identity(record, f"{source_path}:{index}")
            if sample_identity in evaluation_targets:
                raise ValueError(
                    f"train/evaluation leakage: atomic sample containing target "
                    f"{target!r} also occurs in {evaluation_targets[sample_identity]}"
                )
            key = _dedup_key(record)
            record_id = str(record.get("id"))
            # A current row is a duplicate when its marker-free content or its id is
            # already in the corpus (previous rows are retained as they are).
            if source_kind in _CURRENT_KINDS and (key in seen or record_id in corpus_ids):
                if source_kind == CLASSIFICATION_CALIBRATION_SOURCE_KIND:
                    classification_duplicates += 1
                else:
                    duplicate_count += 1
                continue
            seen.add(key)
            corpus_ids.add(record_id)
            merged.append(record)
            tasks[str(record.get("task_type", "unknown"))] += 1
            provenance.append(
                {
                    "source_kind": source_kind,
                    "source": str(source_path),
                    "source_index": index,
                    "id": record.get("id"),
                }
            )
    if not merged:
        raise ValueError("real-mining assembly produced no training records")
    anchor_report: dict[str, Any] = {"enabled": False}
    coverage_report: dict[str, Any] = {"enabled": False}
    # Slices carried over from previous iterations (inert top-level markers).
    prior_anchor_rows = sum(
        1 for record, item in zip(merged, provenance)
        if item["source_kind"] == "previous_iteration" and record.get(ANCHOR_MARK) is True
    )
    prior_coverage_rows = sum(
        1 for record, item in zip(merged, provenance)
        if item["source_kind"] == "previous_iteration" and record.get(COVERAGE_MARK) is True
    )
    prior_classification_rows = sum(
        1 for record, item in zip(merged, provenance)
        if item["source_kind"] == "previous_iteration"
        and record.get(CALIBRATION_KIND_MARK) == CLASSIFICATION_CALIBRATION_KIND
    )
    # rows a launch-recorded acquisition slice adds on top of the parent corpus do
    # not count toward the anchor share base (anchor volume stays parent-like)
    share_excluded_rows = min(int(anchor.get("share_exclude_rows", 0) or 0), len(merged))
    base_rows = len(merged) - prior_anchor_rows - prior_coverage_rows - share_excluded_rows
    slice_targets = joint_targets(
        base_rows,
        {"anchor": anchor["share"] if anchor["enabled"] else 0.0,
         "coverage": coverage["share"] if coverage["enabled"] else 0.0},
        {"anchor": prior_anchor_rows, "coverage": prior_coverage_rows},
    )
    def excluded(record: dict[str, Any], source: pathlib.Path) -> str | None:
        if str(record.get("id")) in corpus_ids or _dedup_key(record) in seen:
            return "already_in_corpus"
        sample_identity = identity(record, f"{source}:{record.get('id')}")
        if sample_identity in evaluation_targets:
            return "evaluation_target"
        return None

    if anchor["enabled"]:
        anchor_source = pathlib.Path(anchor["source"]).expanduser().resolve(strict=True)
        anchor_candidates = load_records(anchor_source)
        non_anchor_rows = len(merged) - prior_anchor_rows
        anchor_total, anchor_new = slice_targets["anchor"]
        shares = task_shares_from_jsonl(pathlib.Path(anchor["task_shares"]))
        def anchor_excluded(record: dict[str, Any]) -> str | None:
            reason = excluded(record, anchor_source)
            if reason is None and guard["enabled"] and is_empty_ground_truth(record, context="anchor candidate"):
                # With the empty-answer guard on, anchors rehearse answers, not "[]":
                # an empty-answer anchor is never trimmed and could only leave an
                # untrimmable excess behind (4c-B r3 iteration 2, 2026-09-15).
                return "empty_ground_truth"
            return reason

        anchors, selection = select_anchors(
            anchor_candidates,
            new_rows=anchor_new,
            task_shares=shares,
            source_cap=anchor["source_cap"],
            seed=17 if anchor_seed is None else anchor_seed,
            is_excluded=anchor_excluded,
        )
        for index, picked in enumerate(anchors):
            record = {**picked, ANCHOR_MARK: True}
            seen.add(_dedup_key(record))
            corpus_ids.add(str(record.get("id")))
            merged.append(record)
            tasks[str(record.get("task_type", "unknown"))] += 1
            provenance.append(
                {
                    "source_kind": ANCHOR_SOURCE_KIND,
                    "source": str(anchor_source),
                    "source_index": index,
                    "id": record.get("id"),
                    "purpose_tags": ["anchor"],
                }
            )
        anchor_report = {
            "enabled": True,
            "unit": anchor["unit"],
            "requested_share": anchor["share"],
            "source": str(anchor_source),
            "source_sha256": sha256_file(anchor_source),
            "task_shares_source": anchor["task_shares"],
            "task_shares": shares,
            "prior_anchor_rows": prior_anchor_rows,
            "non_anchor_rows": non_anchor_rows,
            "base_rows": base_rows,
            "share_base_excluded_rows": share_excluded_rows,
            "target_anchor_rows_total": anchor_total,
            "selection": selection,
            # with the empty-answer guard enabled, empty-ground-truth candidates are skipped
            "prefer_non_empty_rows": bool(guard["enabled"]),
            "anchor_empty_rows_excluded": int(selection["skipped"].get("empty_ground_truth", 0)),
        }
    if coverage["enabled"]:
        coverage_source = pathlib.Path(coverage["source"]).expanduser().resolve(strict=True)
        coverage_candidates = load_records(coverage_source)
        coverage_total, coverage_new = slice_targets["coverage"]
        prior_cells: Counter[tuple[str, str]] = Counter(
            cell_of(record) for record, item in zip(merged, provenance)
            if item["source_kind"] == "previous_iteration" and record.get(COVERAGE_MARK) is True
        )
        eventual_budget = None
        if max_rows is not None:
            cap_rows = max_rows - (max_rows % row_multiple if row_multiple else 0)
            eventual_budget = int(round(cap_rows * coverage["share"]))
        picked, coverage_selection = select_coverage(
            coverage_candidates,
            new_rows=coverage_new,
            mode=coverage["mode"],
            min_rows_per_cell=coverage["min_rows_per_cell"],
            seed=17 if coverage_seed is None else coverage_seed,
            is_excluded=lambda record: excluded(record, coverage_source),
            prior_cell_counts=dict(prior_cells),
        )
        # Eligible cells (after corpus/evaluation exclusion) must fit the eventual budget under the cap.
        floor_report = check_floor_budget(
            int(coverage_selection["cells"]), coverage["min_rows_per_cell"], eventual_budget
        )
        for index, (candidate, is_fallback) in enumerate(picked):
            record = {key: value for key, value in candidate.items() if key != POOL_STATUS_KEY}
            record[COVERAGE_MARK] = True
            seen.add(_dedup_key(record))
            corpus_ids.add(str(record.get("id")))
            merged.append(record)
            tasks[str(record.get("task_type", "unknown"))] += 1
            provenance.append(
                {
                    "source_kind": COVERAGE_SOURCE_KIND,
                    "source": str(coverage_source),
                    "source_index": index,
                    "id": record.get("id"),
                    "purpose_tags": ["coverage", "anchor"] if is_fallback else ["coverage"],
                    "pool_status": str(candidate.get(POOL_STATUS_KEY) or "unscored"),
                }
            )
        coverage_report = {
            "enabled": True,
            "unit": coverage["unit"],
            "mode": coverage["mode"],
            "requested_share": coverage["share"],
            "source": str(coverage_source),
            "source_sha256": sha256_file(coverage_source),
            "min_rows_per_cell": coverage["min_rows_per_cell"],
            "prior_coverage_rows": prior_coverage_rows,
            "base_rows": base_rows,
            "target_coverage_rows_total": coverage_total,
            "floor_check": floor_report,
            "selection": coverage_selection,
        }
    uncapped_records = len(merged)
    repetition_manifest: dict[str, Any] | None = None
    selection_policy = "monotonic_current_fill_task_balanced_v1"
    if resolved_repetition["enabled"]:
        assert max_rows is not None
        materialized_rows = max_rows
        if row_multiple is not None:
            materialized_rows -= materialized_rows % row_multiple
            if materialized_rows < row_multiple:
                raise ValueError(
                    "training materialization cannot form one complete global batch: "
                    f"row_cap={max_rows}, row_multiple={row_multiple}"
                )
        current = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] in _CURRENT_KINDS
        ]
        prior = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "previous_iteration"
        ]
        if not current:
            raise ValueError(
                "repetition blend requires at least one unique current Mining record"
            )
        if len(prior) + 1 > materialized_rows:
            raise ValueError(
                "training materialization cannot retain all previous iteration records "
                "and include current Mining data under the configured cap"
            )
        # Keep fresh corrective examples at the front of the first repetition
        # pass, matching the non-repetition materializer's ordering contract.
        available_indices = current + prior
        available_records = [merged[index] for index in available_indices]
        available_provenance = [provenance[index] for index in available_indices]
        first_prior = len(current)
        merged, repetition_manifest = apply_repetition_blend(
            available_records,
            row_cap=materialized_rows,
            config=resolved_repetition,
            deficit_weights=deficit_weights,
            seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            mandatory_indices=set(range(first_prior, len(available_records))) | {0},
            row_multiple=row_multiple or 1,
        )
        provenance_by_fingerprint: dict[str, list[dict[str, Any]]] = {}
        for record, item in zip(available_records, available_provenance):
            provenance_by_fingerprint.setdefault(_fingerprint(record), []).append(item)
        provenance_positions = {key: 0 for key in provenance_by_fingerprint}
        materialized_provenance: list[dict[str, Any]] = []
        for record in merged:
            key = _fingerprint(record)
            choices = provenance_by_fingerprint[key]
            position = provenance_positions[key]
            materialized_provenance.append(
                choices[position] if position < len(choices) else choices[0]
            )
            provenance_positions[key] = position + 1
        provenance = materialized_provenance
        tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)
        selection_policy = "monotonic_deficit_repetition_blend_v1"
    # The cap / global-batch step is a function of the candidate set so the
    # empty-answer guard can trim empty candidates *before* alignment and re-run
    # it: the standard selection then back-fills from the remaining (non-empty)
    # mined candidates and the aligned size stays unchanged whenever enough
    # candidates remain (it shrinks only when they run out).
    def materialize_capped(
        cand_merged: list[dict[str, Any]], cand_provenance: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        merged = list(cand_merged)
        provenance = list(cand_provenance)
        if max_rows is None and row_multiple is None:
            return merged, provenance
        candidate_records = len(merged)
        materialized_rows = min(candidate_records, max_rows or candidate_records)
        if row_multiple is not None:
            materialized_rows -= materialized_rows % row_multiple
            if materialized_rows < row_multiple:
                raise ValueError(
                    "training materialization cannot form one complete global batch: "
                    f"available={candidate_records}, row_multiple={row_multiple}"
                )
        # Retain the full prior iteration to make the training set monotonic.
        # Current Mining owns the remaining slots under the cap, selected in
        # task-balanced order and emitted first so fresh corrective examples
        # are not pushed to the tail of a deterministic epoch.
        current = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] in _CURRENT_KINDS
        ]
        prior = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "previous_iteration"
        ]
        if len(prior) > materialized_rows:
            raise ValueError(
                "training materialization cannot retain all previous iteration records: "
                f"previous={len(prior)}, materialized={materialized_rows}"
            )
        anchors_idx = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == ANCHOR_SOURCE_KIND
        ]
        # Anchors are a share of the *materialized* corpus. Reserve their slots
        # before filling current Mining so the max_rows / row_multiple trim is
        # absorbed by the mined slice instead of silently deleting the anchors
        # (2,304 mined rows + 256 anchors rounded to 2,304 used to keep zero).
        coverage_idx = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == COVERAGE_SOURCE_KIND
        ]
        forbid_overfill = anchor_overfill_policy == "forbid"

        def slots_for(rows: int) -> tuple[int, int, int]:
            """(share base, anchor slots, coverage slots) reserved at an aligned size of ``rows``."""
            base = max(0, rows - min(share_excluded_rows, rows))
            anchor_wanted = coverage_wanted = 0
            if anchor["enabled"]:
                anchor_wanted = max(0, min(len(anchors_idx), int(round(base * anchor["share"])) - prior_anchor_rows))
            if coverage["enabled"]:
                coverage_wanted = max(
                    0, min(len(coverage_idx), int(round(rows * coverage["share"])) - prior_coverage_rows)
                )
            return base, anchor_wanted, coverage_wanted

        shrunk_for_share = 0
        if forbid_overfill:
            # Anchors may not fill leftover slots beyond their share (r4: 1,094 of 5,376 rows =
            # 20.4% against 0.10). When the retained rows, the share-bound anchors, the coverage
            # rows and this iteration's candidates cannot fill the aligned size, shrink it to the
            # largest global-batch multiple they do fill instead of drawing extra anchors.
            def fills(rows: int) -> bool:
                return len(prior) + slots_for(rows)[1] + len(coverage_idx) + len(current) >= rows

            step = row_multiple or 1
            fitted = materialized_rows
            while fitted - step >= len(prior) and not fills(fitted):
                fitted -= step
            if not fills(fitted):
                fitted = len(prior)  # nothing aligned above the previous corpus is fillable
            shrunk_for_share = materialized_rows - fitted
            materialized_rows = fitted
        share_base_rows, anchor_slots, coverage_slots = slots_for(materialized_rows)
        current_capacity = max(0, materialized_rows - len(prior))
        current_limit = min(len(current), current_capacity - anchor_slots - coverage_slots)
        if current and current_limit <= 0:
            if forbid_overfill:
                raise ValueError(
                    "training materialization would add zero rows of this iteration under the empty-answer "
                    "guard (anchors may not over-fill the global batch): "
                    f"current_rows_after_guard={len(current)}, anchor_slots={anchor_slots}, "
                    f"coverage_slots={coverage_slots}, row_multiple={row_multiple}, previous_rows={len(prior)}, "
                    f"aligned_rows={materialized_rows}"
                )
            raise ValueError(
                "training materialization cannot retain all previous iteration records "
                "and include current Mining data under the configured cap"
            )
        current_limit = max(0, current_limit)
        leftover = materialized_rows - len(prior) - anchor_slots - coverage_slots - current_limit
        leftover_fill_anchors = 0
        if leftover > 0:
            if not forbid_overfill:
                leftover_fill_anchors = min(leftover, len(anchors_idx) - anchor_slots)
                anchor_slots += leftover_fill_anchors
                leftover -= leftover_fill_anchors
            extra_coverage = min(leftover, len(coverage_idx) - coverage_slots)
            coverage_slots += extra_coverage
            leftover -= extra_coverage
            current_limit += min(leftover, len(current) - current_limit)
        # Calibration rows (marked by the materializer) carry a verified quota
        # contract; the cap / anchor / coverage trim may only displace the other
        # current rows. Mined rows first, then the protected calibration rows.
        protected = [index for index in current if merged[index].get(CALIBRATION_MARK) is True]
        trimmable = [index for index in current if merged[index].get(CALIBRATION_MARK) is not True]
        alignment_fill_anchors = 0
        if len(protected) > current_limit:
            # The global-batch rounding cannot be absorbed by trimmable rows (a
            # calibration-dominated iteration: 3,475 calibration rows + 5 mined rows,
            # 2026-09-14). Round the corpus UP to the next multiple instead and fill
            # the gap with extra correct-row anchors, which are plentiful and inert,
            # rather than deleting verified calibration rows. Fail closed only when
            # the cap or the anchor supply makes that impossible.
            keep_rows = len(prior) + len(current) + coverage_slots
            # the round-up fill draws anchors beyond their share: not available under forbid
            fillable = anchor["enabled"] and row_multiple is not None and not forbid_overfill
            rounded = (
                -(-(keep_rows + anchor_slots) // row_multiple) * row_multiple if fillable else keep_rows + anchor_slots
            )
            if not fillable or (max_rows is not None and rounded > max_rows):
                raise ValueError(
                    "training materialization cannot retain the calibration rows under the "
                    f"configured cap: calibration={len(protected)}, current_limit={current_limit}"
                    + (" (anchor over-fill forbidden under the empty-answer guard)" if forbid_overfill else "")
                )
            gap = rounded - keep_rows - anchor_slots
            if gap > 0:
                extra_pool = anchors_idx[anchor_slots:]
                take = min(gap, len(extra_pool))
                anchor_slots += take
                gap -= take
            if gap > 0:
                extra, extra_selection = select_anchors(
                    anchor_candidates,
                    new_rows=gap,
                    task_shares=shares,
                    source_cap=anchor["source_cap"],
                    seed=(17 if anchor_seed is None else anchor_seed) + 1,
                    is_excluded=anchor_excluded,
                )
                if len(extra) < gap:
                    raise ValueError(
                        "training materialization cannot align the corpus without deleting calibration rows: "
                        f"needs {gap} more anchors, only {len(extra)} available"
                    )
                for offset, picked in enumerate(extra):
                    record = {**picked, ANCHOR_MARK: True}
                    seen.add(_dedup_key(record))
                    corpus_ids.add(str(record.get("id")))
                    merged.append(record)
                    provenance.append(
                        {
                            "source_kind": ANCHOR_SOURCE_KIND,
                            "source": str(anchor_source),
                            "source_index": len(anchors_idx) + offset,
                            "id": record.get("id"),
                            "purpose_tags": ["anchor", "alignment_fill"],
                        }
                    )
                    anchors_idx.append(len(merged) - 1)
                anchor_slots += gap
                alignment_fill_anchors = gap
            materialized_rows = rounded
            current_limit = len(current)
        selected = _task_balanced_indices(trimmable, merged, current_limit - len(protected))
        selected.extend(protected)
        selected.extend(coverage_idx[:coverage_slots])
        selected.extend(anchors_idx[:anchor_slots])
        selected.extend(prior)
        displaced_total = max(0, min(len(current), current_capacity) - current_limit)
        anchor_displaced = min(anchor_slots, displaced_total) if coverage["enabled"] else displaced_total
        if anchor["enabled"]:
            anchor_report["cap_reservation"] = {
                "policy": "anchor_share_of_materialized_rows_v1",
                "materialized_rows": materialized_rows,
                "share_base_rows": share_base_rows,
                "share_base_excluded_rows": materialized_rows - share_base_rows,
                "wanted_anchor_rows_total": int(round(share_base_rows * anchor["share"])),
                "new_anchor_slots": anchor_slots,
                "new_anchors_available": len(anchors_idx),
                "current_rows_displaced_by_anchors": anchor_displaced,
                "calibration_rows_protected": len(protected),
                # rows added to round the corpus up to the global batch when the
                # calibration rows alone exceeded the aligned-down capacity
                "alignment_fill_anchors": alignment_fill_anchors,
                "alignment_policy": (
                    "round_up_fill_with_anchors" if alignment_fill_anchors else "round_down_trim_mined_rows"
                ),
                # Feature B3: anchors drawn to fill leftover slots beyond their share (allow only)
                # and the rows the aligned size shrank by instead (forbid only)
                "anchor_overfill": anchor_overfill_policy,
                "leftover_fill_anchors": leftover_fill_anchors,
                "aligned_rows_shrunk_for_share": shrunk_for_share,
            }
        if coverage["enabled"]:
            coverage_report["cap_reservation"] = {
                "policy": "coverage_share_of_materialized_rows_v1",
                "materialized_rows": materialized_rows,
                "wanted_coverage_rows_total": int(round(materialized_rows * coverage["share"])),
                "new_coverage_slots": coverage_slots,
                "new_coverage_available": len(coverage_idx),
                "current_rows_displaced_by_coverage": max(0, displaced_total - anchor_displaced),
            }
        return [merged[index] for index in selected], [provenance[index] for index in selected]

    if resolved_repetition["enabled"]:
        # report only: enforcement together with the repetition blend is rejected above
        guard_report = _run_empty_answer_guard(merged, provenance, config=guard, materialize=None)[2]
    else:
        merged, provenance, guard_report = _run_empty_answer_guard(
            merged, provenance, config=guard, materialize=materialize_capped
        )
        tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)
    # Feature B3 launch-recorded option group (None = the assembler was not told)
    guard_report["calibration_guard_aware"] = calibration_guard_aware
    guard_report["anchor_overfill"] = anchor_overfill_policy
    if repetition_manifest is None:
        merged, repetition_manifest = apply_repetition_blend(
            merged,
            row_cap=max_rows or len(merged),
            config=resolved_repetition,
            deficit_weights=deficit_weights,
            seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            row_multiple=row_multiple or 1,
        )
    records_truncated = (
        repetition_manifest["totals"]["dropped_rows"]
        if resolved_repetition["enabled"]
        else uncapped_records - len(merged)
    )
    mined_fingerprints = {_fingerprint(record) for record in mined}
    if not any(_fingerprint(record) in mined_fingerprints for record in merged):
        raise ValueError("current mined records were all lost during assembly")
    previous_fingerprints = {_fingerprint(record) for record in previous}
    output_fingerprints = {_fingerprint(record) for record in merged}
    previous_fingerprints_subset = previous_fingerprints.issubset(output_fingerprints)
    retained_previous_records = len(
        {
            (item["source"], item["source_index"])
            for item in provenance
            if item["source_kind"] == "previous_iteration"
        }
    )
    if retained_previous_records != len(previous):
        raise ValueError(
            "training materialization must retain every previous record: "
            f"retained={retained_previous_records}, previous={len(previous)}"
        )
    if not previous_fingerprints_subset:
        raise ValueError("previous training fingerprints are not a subset of output")
    row_profiles = [row_profile(record, context=f"answer profile row[{index}]") for index, record in enumerate(merged)]
    # cumulative anchor share of all materialized rows (marker-based, so it is reported every
    # iteration, also when anchors are off this iteration but retained from earlier ones)
    anchor_rows_total = sum(record.get(ANCHOR_MARK) is True for record in merged)
    cumulative_anchor_share = anchor_rows_total / len(merged) if merged else 0.0
    operator_attention: list[str] = []
    if anchor["enabled"] and cumulative_anchor_share > anchor["share"] + ANCHOR_SHARE_TOLERANCE + 1e-9:
        operator_attention.append("anchor_share_exceeded")
    return merged, {
        "schema_version": 2,
        "format": "jsonl",
        "annotation_profile": "nvpaw_multitask_v1",
        "training_source": "mined_real_samples_only",
        "previous_iteration": (
            str(resolved_previous_path)
            if resolved_previous_path is not None
            else None
        ),
        "previous_sha256": actual_previous_sha256,
        "mined_input": str(resolved_mined_path),
        "validation_inputs": [str(path) for path in validation_paths],
        "previous_records": len(previous),
        "previous_fingerprint_count": len(previous_fingerprints),
        "previous_fingerprints_sha256": _fingerprint_set_sha256(
            previous_fingerprints
        ),
        "previous_fingerprints_subset": previous_fingerprints_subset,
        "output_fingerprint_count": len(output_fingerprints),
        "output_fingerprints_sha256": _fingerprint_set_sha256(output_fingerprints),
        "mined_records": len(mined),
        "output_records": len(merged),
        "growth_rows": len(merged) - len(previous),
        "uncapped_records": uncapped_records,
        "materialization_cap": max_rows,
        "row_multiple": row_multiple,
        "selection_policy": selection_policy,
        "records_truncated": records_truncated,
        "duplicates_skipped": duplicate_count,
        "retained_previous_records": retained_previous_records,
        "selected_current_records": len(
            {
                (item["source"], item["source_index"])
                for item in provenance
                if item["source_kind"] == "current_mining"
            }
        ),
        "materialized_previous_records": sum(
            item["source_kind"] == "previous_iteration" for item in provenance
        ),
        "materialized_current_records": sum(
            item["source_kind"] == "current_mining" for item in provenance
        ),
        "materialized_anchor_records": sum(
            item["source_kind"] == ANCHOR_SOURCE_KIND for item in provenance
        ),
        "anchor": {
            **anchor_report,
            "materialized_anchor_records_total": sum(
                item["source_kind"] == ANCHOR_SOURCE_KIND for item in provenance
            ) + (anchor_report.get("prior_anchor_rows", 0) if anchor_report.get("enabled") else 0),
            # share relative to the share base (all rows minus the launch-recorded
            # excluded rows); ``realized_share_of_all_rows`` keeps the plain ratio
            "realized_share_rows": (
                (sum(item["source_kind"] == ANCHOR_SOURCE_KIND for item in provenance)
                 + anchor_report.get("prior_anchor_rows", 0))
                / max(1, len(merged) - min(share_excluded_rows, len(merged)))
                if anchor_report.get("enabled") and merged else 0.0
            ),
            "realized_share_of_all_rows": (
                (sum(item["source_kind"] == ANCHOR_SOURCE_KIND for item in provenance)
                 + anchor_report.get("prior_anchor_rows", 0)) / len(merged)
                if anchor_report.get("enabled") and merged else 0.0
            ),
            "share_base_excluded_rows": share_excluded_rows if anchor_report.get("enabled") else 0,
            # anchor rows (retained + new) over all rows, every iteration; operator_attention
            # flags anchor_share_exceeded above requested_share + share_tolerance
            "cumulative_share_rows": cumulative_anchor_share,
            "cumulative_anchor_rows": anchor_rows_total,
            "share_tolerance": ANCHOR_SHARE_TOLERANCE,
        },
        "operator_attention": operator_attention,
        "materialized_calibration_records": sum(
            record.get(CALIBRATION_MARK) is True for record in merged
        ),
        "materialized_classification_calibration_records": sum(
            record.get(CALIBRATION_KIND_MARK) == CLASSIFICATION_CALIBRATION_KIND for record in merged
        ),
        "classification_calibration": {
            "enabled": resolved_classification_path is not None,
            "source": str(resolved_classification_path) if resolved_classification_path is not None else None,
            "source_sha256": (
                sha256_file(resolved_classification_path) if resolved_classification_path is not None else None
            ),
            "input_records": len(classification_rows),
            "duplicates_skipped": classification_duplicates,
            "prior_rows": prior_classification_rows,
            "materialized_new": sum(
                item["source_kind"] == CLASSIFICATION_CALIBRATION_SOURCE_KIND for item in provenance
            ),
            "materialized_total": sum(
                record.get(CALIBRATION_KIND_MARK) == CLASSIFICATION_CALIBRATION_KIND for record in merged
            ),
            "markers": {CALIBRATION_MARK: True, CALIBRATION_KIND_MARK: CLASSIFICATION_CALIBRATION_KIND},
            "protected_from_cap_trim": True,
        },
        "materialized_coverage_records": sum(
            item["source_kind"] == COVERAGE_SOURCE_KIND for item in provenance
        ),
        "coverage_blend": {
            **coverage_report,
            "materialized_coverage_records_total": sum(
                item["source_kind"] == COVERAGE_SOURCE_KIND for item in provenance
            ) + (coverage_report.get("prior_coverage_rows", 0) if coverage_report.get("enabled") else 0),
            "materialized_fallback_correct_records": sum(
                item["source_kind"] == COVERAGE_SOURCE_KIND and "anchor" in item.get("purpose_tags", [])
                for item in provenance
            ),
            "realized_share_rows": (
                (sum(item["source_kind"] == COVERAGE_SOURCE_KIND for item in provenance)
                 + coverage_report.get("prior_coverage_rows", 0)) / len(merged)
                if coverage_report.get("enabled") and merged else 0.0
            ),
            "materialized_per_cell_cumulative": (
                {
                    f"{cell[0]}|{cell[1]}": count
                    for cell, count in sorted(
                        Counter(cell_of(record) for record in merged if record.get(COVERAGE_MARK) is True).items()
                    )
                }
                if coverage_report.get("enabled") else {}
            ),
        },
        "exposure_rows": _exposure_rows(
            merged, provenance, prior_anchor_rows=prior_anchor_rows, prior_coverage_rows=prior_coverage_rows,
            prior_classification_rows=prior_classification_rows,
        ),
        "answer_profile": profile_rows(row_profiles),
        "answer_profile_new_rows": profile_rows(
            profile
            for profile, item in zip(row_profiles, provenance)
            if item["source_kind"] != "previous_iteration"
        ),
        "empty_answer_guard": guard_report,
        "tasks": dict(sorted(tasks.items())),
        "warnings": list(repetition_manifest["warnings"]),
        "provenance": provenance,
        "repetition_blend": repetition_manifest,
    }


def bind_summary(
    summary: dict[str, Any], training_jsonl: pathlib.Path
) -> dict[str, Any]:
    path = training_jsonl.expanduser().resolve(strict=True)
    rows = load_records(path)
    bound = json.loads(json.dumps(summary))
    if bound.get("output_records") != len(rows):
        raise ValueError("training JSONL row count differs from assembly summary")
    fingerprints = {_fingerprint(record) for record in rows}
    if bound.get("output_fingerprints_sha256") != _fingerprint_set_sha256(
        fingerprints
    ):
        raise ValueError("training JSONL fingerprints differ from assembly summary")
    bound["training_jsonl"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
    }
    return bound


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-jsonl", type=pathlib.Path)
    parser.add_argument("--previous-sha256")
    parser.add_argument("--mined-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--current-quota-manifest", type=pathlib.Path)
    parser.add_argument("--quota-manifest", type=pathlib.Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--validation-jsonl", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--row-multiple", type=int)
    parser.add_argument("--gap-analysis-summary", type=pathlib.Path)
    parser.add_argument("--repetition-config", type=pathlib.Path)
    repetition_toggle = parser.add_mutually_exclusive_group()
    repetition_toggle.add_argument(
        "--repetition-blend",
        dest="repetition_blend",
        action="store_true",
        default=None,
    )
    repetition_toggle.add_argument(
        "--no-repetition-blend",
        dest="repetition_blend",
        action="store_false",
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
    parser.add_argument("--anchor-share", type=float, help="Correct-row anchor share of the cumulative corpus (rows); 0/absent = off.")
    parser.add_argument("--anchor-source", type=pathlib.Path, help="anchor_candidates.jsonl from build_anchor_candidates.py")
    parser.add_argument("--anchor-task-shares", type=pathlib.Path, help="Evaluation JSONL whose task row shares set the anchor task quotas (the KPI set).")
    parser.add_argument("--anchor-source-cap", type=float, help="Max share of one dataset inside a task's anchors (default 0.35).")
    parser.add_argument("--anchor-seed", type=int)
    parser.add_argument("--anchor-share-exclude-rows", type=int, help="Cumulative corpus rows excluded from the anchor share base (a launch-recorded acquisition slice); default 0.")
    parser.add_argument("--anchor-manifest", type=pathlib.Path, help="Defaults to anchor_manifest.json beside --output.")
    parser.add_argument("--coverage-blend-share", type=float, help="Cross-dataset coverage share of the cumulative corpus (rows); 0/absent = off.")
    parser.add_argument("--coverage-blend-mode", choices=("plain", "residual"), help="plain = uniform pool rows; residual = scored-wrong rows with correct-row fallback.")
    parser.add_argument("--coverage-blend-source", type=pathlib.Path, help="coverage_candidates_<mode>.jsonl from build_coverage_candidates.py")
    parser.add_argument("--coverage-blend-min-rows-per-dataset", type=int, help="Per (task, dataset) floor K checked against the eventual budget under the row cap (default 8).")
    parser.add_argument("--coverage-blend-seed", type=int)
    parser.add_argument("--coverage-blend-manifest", type=pathlib.Path, help="Defaults to coverage_blend_manifest.json beside --output.")
    parser.add_argument(
        "--classification-calibration-jsonl",
        type=pathlib.Path,
        help=(
            "select_classification_calibration.py output (rows marked deft_calibration + "
            "deft_calibration_kind=classification) added as current rows and protected from the cap trim."
        ),
    )
    parser.add_argument("--max-empty-answer-share", type=float, help="Empty-answer guard: cap on the empty-ground-truth share of the whole cumulative corpus, e.g. 0.30.")
    parser.add_argument("--max-empty-answer-share-task", action="append", metavar="TASK=SHARE", help='Per-task cap, e.g. "Defect Detection=0.45" (repeatable).')
    parser.add_argument("--max-classification-empty-share", type=float, help="Cap on empty BCQ+MCQ rows together and per classification task, e.g. 0.10.")
    parser.add_argument("--empty-answer-guard-mode", choices=GUARD_MODES, help="enforce (default when any cap is given: trim this iteration's empty rows, fail closed if still exceeded) or report.")
    parser.add_argument(
        "--calibration-guard-aware",
        choices=("on", "off"),
        help=(
            "Whether the materializer bounded its calibration empties by the guard headroom (default on when a cap "
            "is given); recorded in the summary and cross-checked against the current quota manifest."
        ),
    )
    parser.add_argument(
        "--anchor-overfill",
        choices=ANCHOR_OVERFILL_POLICIES,
        help=(
            "forbid (default when a cap is given): anchors may not fill leftover global-batch slots beyond their "
            "share, the aligned size shrinks instead; allow (default otherwise): parent behaviour."
        ),
    )
    parser.add_argument(
        "--repetition-manifest",
        type=pathlib.Path,
        help="Defaults to repetition_blend_manifest.json beside --output.",
    )
    args = parser.parse_args(argv)
    try:
        if args.previous_jsonl is not None and args.previous_sha256 is None:
            raise ValueError("--previous-sha256 is required with --previous-jsonl")
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
            gap_summary, list(TASK_SPECS)
        )
        repetition_seed = args.repetition_seed
        if repetition_seed is None:
            repetition_seed = repetition_config["seed"]
        if repetition_seed is None and isinstance(gap_summary, dict):
            repetition_seed = gap_summary.get("seed")
        if repetition_seed is None:
            repetition_seed = 17
        anchor_config = validate_anchor_config(
            args.anchor_share, args.anchor_source, args.anchor_task_shares, args.anchor_source_cap,
            args.anchor_share_exclude_rows,
        )
        coverage_config = validate_coverage_config(
            args.coverage_blend_share,
            args.coverage_blend_mode,
            args.coverage_blend_source,
            args.coverage_blend_min_rows_per_dataset,
        )
        guard_config = validate_empty_answer_guard_config(
            args.max_empty_answer_share,
            parse_task_shares(args.max_empty_answer_share_task),
            args.max_classification_empty_share,
            args.empty_answer_guard_mode,
        )
        calibration_guard_aware = (
            args.calibration_guard_aware == "on" if args.calibration_guard_aware else guard_config["enabled"]
        )
        if args.current_quota_manifest is not None and guard_config["enabled"]:
            # fail closed before writing anything when the materializer ran with a different
            # guard-aware setting than the launch expects
            check_calibration_guard_aware_manifest(
                json.loads(args.current_quota_manifest.read_text(encoding="utf-8")),
                expected=calibration_guard_aware,
                source=str(args.current_quota_manifest),
            )
        rows, summary = assemble(
            args.previous_jsonl,
            args.mined_jsonl,
            previous_sha256=args.previous_sha256,
            validation_paths=args.validation_jsonl,
            max_rows=args.max_rows,
            row_multiple=args.row_multiple,
            repetition_config=repetition_config,
            deficit_weights=deficit_weights,
            repetition_seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            media_root=args.media_root,
            anchor_config=anchor_config,
            anchor_seed=args.anchor_seed,
            coverage_config=coverage_config,
            coverage_seed=args.coverage_blend_seed,
            classification_calibration_path=args.classification_calibration_jsonl,
            empty_answer_guard=guard_config,
            calibration_guard_aware=calibration_guard_aware,
            anchor_overfill=args.anchor_overfill,
        )
        guard_report = summary["empty_answer_guard"]
        if guard_report["enabled"] and guard_report["mode"] == "enforce" and guard_report["status"] == "exceeded":
            # fail closed: leave the numbers on disk for diagnosis, never the corpus
            summary_path = args.summary or args.output.with_name("assemble_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps({**summary, "training_jsonl": None, "error": "empty_answer_guard_exceeded"}, indent=2) + "\n"
            )
            raise ValueError(
                "empty-answer guard: caps still exceeded after trimming this iteration's empty rows "
                f"({', '.join(guard_report['exceeded_after'])}); summary written to {summary_path}, "
                f"{args.output} not written"
            )
        if guard_report["status"] == "exceeded_untrimmable":
            print(
                "assemble_training_json: empty-answer guard: caps exceeded only by never-trimmed rows "
                f"({', '.join(guard_report['exceeded_after'])}); residual recorded in "
                "empty_answer_guard.untrimmable_excess, continuing",
                file=sys.stderr,
            )
        if summary["operator_attention"]:
            print(
                "assemble_training_json: operator_attention: "
                f"{', '.join(summary['operator_attention'])} "
                f"(anchor.cumulative_share_rows={summary['anchor']['cumulative_share_rows']:.4f}, "
                f"requested_share={summary['anchor'].get('requested_share')})",
                file=sys.stderr,
            )
        _write_jsonl(args.output, rows)
        summary = bind_summary(summary, args.output)
        if anchor_config["enabled"]:
            write_anchor_manifest(
                args.anchor_manifest or args.output.with_name("anchor_manifest.json"),
                {**summary["anchor"], "training_jsonl": summary["training_jsonl"]},
            )
        if coverage_config["enabled"]:
            write_coverage_manifest(
                args.coverage_blend_manifest or args.output.with_name("coverage_blend_manifest.json"),
                {**summary["coverage_blend"], "exposure_rows": summary["exposure_rows"],
                 "training_jsonl": summary["training_jsonl"]},
            )
        repetition_manifest = bind_repetition_manifest(
            summary["repetition_blend"], args.output
        )
        summary["repetition_blend"] = repetition_manifest
        quota_values = (
            args.current_quota_manifest,
            args.quota_manifest,
            args.epochs,
            args.global_batch,
        )
        if any(value is not None for value in quota_values):
            if any(value is None for value in quota_values):
                raise ValueError(
                    "cumulative quota binding requires current/final manifests, "
                    "epochs, and global batch"
                )
            current_quota = json.loads(
                args.current_quota_manifest.read_text(encoding="utf-8")
            )
            final_quota = bind_cumulative_manifest(
                current_quota,
                current_jsonl=args.mined_jsonl,
                training_jsonl=args.output,
                assembly_summary=summary,
                epochs=args.epochs,
                global_batch=args.global_batch,
            )
            args.quota_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.quota_manifest.write_text(
                json.dumps(final_quota, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        repetition_manifest_path = (
            args.repetition_manifest
            or args.output.with_name("repetition_blend_manifest.json")
        )
        repetition_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        repetition_manifest_path.write_text(
            json.dumps(repetition_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path = args.summary or args.output.with_name("assemble_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"assemble_training_json: {exc}", file=sys.stderr)
        return 2
    print(f"assemble_training_json: wrote {len(rows)} real records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
