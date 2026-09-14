#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Assemble monotonic, real-mining-only NVPAW training JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from answer_profile import EMPTY_DEFINITION, is_empty_ground_truth, profile_rows, row_profile
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
GUARD_MODES = ("enforce", "report")
GUARD_TRIM_ORDER = ("detection_calibration_negative", "mined_empty")
GUARD_NEVER_TRIMMED = (
    "previous_iteration",
    ANCHOR_SOURCE_KIND,
    COVERAGE_SOURCE_KIND,
    CLASSIFICATION_CALIBRATION_SOURCE_KIND,
)
_CLASSIFICATION_FORMATS = ("BCQ", "MCQ")


def parse_task_shares(values: list[str] | None) -> dict[str, float]:
    shares: dict[str, float] = {}
    for value in values or []:
        task, separator, share = value.partition("=")
        if not separator or not task or task in shares:
            raise ValueError(f"invalid task share {value!r}; expected TASK=SHARE")
        try:
            shares[task] = float(share)
        except ValueError as exc:
            raise ValueError(f"invalid task share {value!r}; expected TASK=SHARE") from exc
    return shares


def validate_empty_answer_guard_config(
    max_share: float | None,
    per_task: dict[str, float] | None,
    classification_share: float | None,
    mode: str | None,
) -> dict[str, Any]:
    """Resolve the guard caps; no cap = report only (``enabled`` false, ``mode`` None)."""

    def share(name: str, value: Any) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not 0.0 < number <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        return number

    overall = share("--max-empty-answer-share", max_share)
    tasks: dict[str, float] = {}
    for task, value in (per_task or {}).items():
        if task not in TASK_SPECS:
            raise ValueError(f"--max-empty-answer-share-task names an unsupported task: {task!r}")
        cap = share(f"--max-empty-answer-share-task {task}", value)
        assert cap is not None
        tasks[task] = cap
    classification = share("--max-classification-empty-share", classification_share)
    enabled = overall is not None or bool(tasks) or classification is not None
    if mode is not None and mode not in GUARD_MODES:
        raise ValueError("--empty-answer-guard-mode must be enforce or report")
    if mode is not None and not enabled:
        raise ValueError("--empty-answer-guard-mode requires at least one cap")
    return {
        "enabled": enabled,
        "mode": (mode or "enforce") if enabled else None,
        "max_empty_answer_share": overall,
        "max_empty_answer_share_task": tasks,
        "max_classification_empty_share": classification,
    }


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


def _guard_row_helps(profile: dict[str, Any], exceeded: list[str]) -> bool:
    task = profile["task_type"]
    if "overall" in exceeded or f"task:{task}" in exceeded:
        return True
    return profile["format"] in _CLASSIFICATION_FORMATS and (
        "classification" in exceeded or f"classification_task:{task}" in exceeded
    )


def _apply_empty_answer_guard(
    merged: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    row_multiple: int | None,
    anchor_fill: Callable[[int], tuple[list[dict[str, Any]], pathlib.Path]] | None,
    register: Callable[[dict[str, Any]], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Report the empty-answer shares and, in enforce mode, trim this iteration's empty rows.

    Trim order: detection calibration negatives, then mined empty rows; a row
    goes only when it lowers a cap that is still exceeded. Previous rows,
    anchors, coverage rows and classification calibration rows are never
    trimmed. Alignment is restored by rounding up with non-empty anchors or,
    without anchors, by rounding down with more empty rows; otherwise fail closed.
    """

    profiles = [row_profile(record, context=f"empty-answer guard row[{index}]") for index, record in enumerate(merged)]
    ledger = _GuardLedger(profiles)
    before = ledger.shares()
    exceeded_before = ledger.exceeded(config)

    def trim_source(index: int) -> str | None:
        if provenance[index]["source_kind"] != "current_mining" or not profiles[index]["empty"]:
            return None
        record = merged[index]
        if record.get(CALIBRATION_MARK) is True:
            if record.get(CALIBRATION_KIND_MARK) == CLASSIFICATION_CALIBRATION_KIND:
                return None
            return GUARD_TRIM_ORDER[0]
        return GUARD_TRIM_ORDER[1]

    # Trim from the tail of the current slice so the freshest corrective rows stay in front.
    candidates: dict[str, list[int]] = {source: [] for source in GUARD_TRIM_ORDER}
    for index in range(len(merged) - 1, -1, -1):
        source = trim_source(index)
        if source is not None:
            candidates[source].append(index)
    removed: set[int] = set()
    trimmed_by_source: Counter[str] = Counter()
    trimmed_by_task: Counter[str] = Counter()

    def trim(index: int, source: str) -> None:
        removed.add(index)
        ledger.remove(profiles[index])
        trimmed_by_source[source] += 1
        trimmed_by_task[profiles[index]["task_type"]] += 1

    if config["enabled"] and config["mode"] == "enforce":
        for source in GUARD_TRIM_ORDER:
            exceeded = ledger.exceeded(config)
            if not exceeded:
                break
            for index in candidates[source]:
                if not _guard_row_helps(profiles[index], exceeded):
                    continue
                trim(index, source)
                exceeded = ledger.exceeded(config)
                if not exceeded:
                    break
    alignment: dict[str, Any] = {
        "row_multiple": row_multiple,
        "policy": None,
        "fill_anchors": 0,
        "rows_trimmed_for_rounding": 0,
    }
    kept = [index for index in range(len(merged)) if index not in removed]
    fill_records: list[dict[str, Any]] = []
    fill_provenance: list[dict[str, Any]] = []
    if removed and row_multiple and len(kept) % row_multiple:
        gap = -(-len(kept) // row_multiple) * row_multiple - len(kept)
        if anchor_fill is not None:
            extra, anchor_source = anchor_fill(gap)
            if len(extra) == gap:
                for offset, picked in enumerate(extra):
                    record = {**picked, ANCHOR_MARK: True}
                    register(record)
                    fill_records.append(record)
                    fill_provenance.append(
                        {
                            "source_kind": ANCHOR_SOURCE_KIND,
                            "source": str(anchor_source),
                            "source_index": offset,
                            "id": record.get("id"),
                            "purpose_tags": ["anchor", "empty_answer_guard_fill"],
                        }
                    )
                    ledger.add(row_profile(record, context=f"empty-answer guard fill[{offset}]"))
                alignment.update(policy="round_up_fill_with_anchors", fill_anchors=gap)
        if not fill_records:
            need = len(kept) % row_multiple
            for source in GUARD_TRIM_ORDER:
                for index in candidates[source]:
                    if need == 0:
                        break
                    if index in removed:
                        continue
                    trim(index, source)
                    alignment["rows_trimmed_for_rounding"] += 1
                    need -= 1
                if need == 0:
                    break
            if need:
                raise ValueError(
                    "training materialization cannot align the corpus after the empty-answer guard trim: "
                    f"{len(kept)} rows, row_multiple={row_multiple}, {need} more trimmable empty rows needed "
                    "and no anchors to fill the gap"
                )
            alignment["policy"] = "round_down_trim_empty_rows"
            kept = [index for index in range(len(merged)) if index not in removed]
    new_merged = [merged[index] for index in kept]
    new_provenance = [provenance[index] for index in kept]
    if fill_records:
        insert_at = next(
            (position for position, item in enumerate(new_provenance) if item["source_kind"] == "previous_iteration"),
            len(new_merged),
        )
        new_merged[insert_at:insert_at] = fill_records
        new_provenance[insert_at:insert_at] = fill_provenance
    exceeded_after = ledger.exceeded(config)
    if not config["enabled"]:
        status = "within_caps"
    elif exceeded_after:
        status = "exceeded"
    elif removed:
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
        "trim_order": list(GUARD_TRIM_ORDER),
        "never_trimmed": list(GUARD_NEVER_TRIMMED),
        "before": before,
        "after": ledger.shares(),
        "exceeded_before": exceeded_before,
        "exceeded_after": exceeded_after,
        "trimmed": {
            "total": len(removed),
            "by_source": {source: trimmed_by_source[source] for source in GUARD_TRIM_ORDER if trimmed_by_source[source]},
            "by_task": dict(sorted(trimmed_by_task.items())),
        },
        "alignment": alignment,
        "status": status,
    }
    return new_merged, new_provenance, report


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor = anchor_config or validate_anchor_config(None, None, None, None)
    coverage = coverage_config or validate_coverage_config(None, None, None, None)
    guard = empty_answer_guard or validate_empty_answer_guard_config(None, None, None, None)
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

    guard_anchor_fill: Callable[[int], tuple[list[dict[str, Any]], pathlib.Path]] | None = None
    if anchor["enabled"]:
        anchor_source = pathlib.Path(anchor["source"]).expanduser().resolve(strict=True)
        anchor_candidates = load_records(anchor_source)
        non_anchor_rows = len(merged) - prior_anchor_rows
        anchor_total, anchor_new = slice_targets["anchor"]
        shares = task_shares_from_jsonl(pathlib.Path(anchor["task_shares"]))

        def guard_anchor_fill(gap: int) -> tuple[list[dict[str, Any]], pathlib.Path]:
            # alignment fill after the empty-answer guard trim: correct rows with a
            # non-empty answer only, so the fill cannot push a share back over its cap
            extra, _ = select_anchors(
                anchor_candidates,
                new_rows=gap,
                task_shares=shares,
                source_cap=anchor["source_cap"],
                seed=(17 if anchor_seed is None else anchor_seed) + 2,
                is_excluded=lambda record: excluded(record, anchor_source)
                or ("empty_ground_truth" if is_empty_ground_truth(record, context="guard anchor fill") else None),
            )
            return extra, anchor_source
        anchors, selection = select_anchors(
            anchor_candidates,
            new_rows=anchor_new,
            task_shares=shares,
            source_cap=anchor["source_cap"],
            seed=17 if anchor_seed is None else anchor_seed,
            is_excluded=lambda record: excluded(record, anchor_source),
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
    elif max_rows is not None or row_multiple is not None:
        materialized_rows = min(uncapped_records, max_rows or uncapped_records)
        if row_multiple is not None:
            materialized_rows -= materialized_rows % row_multiple
            if materialized_rows < row_multiple:
                raise ValueError(
                    "training materialization cannot form one complete global batch: "
                    f"available={uncapped_records}, row_multiple={row_multiple}"
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
        anchor_slots = 0
        coverage_slots = 0
        share_base_rows = max(0, materialized_rows - min(share_excluded_rows, materialized_rows))
        if anchor["enabled"]:
            wanted_total = int(round(share_base_rows * anchor["share"]))
            anchor_slots = max(0, min(len(anchors_idx), wanted_total - prior_anchor_rows))
        if coverage["enabled"]:
            wanted_coverage = int(round(materialized_rows * coverage["share"]))
            coverage_slots = max(0, min(len(coverage_idx), wanted_coverage - prior_coverage_rows))
        current_capacity = max(0, materialized_rows - len(prior))
        current_limit = min(len(current), current_capacity - anchor_slots - coverage_slots)
        if current and current_limit <= 0:
            raise ValueError(
                "training materialization cannot retain all previous iteration records "
                "and include current Mining data under the configured cap"
            )
        current_limit = max(0, current_limit)
        leftover = materialized_rows - len(prior) - anchor_slots - coverage_slots - current_limit
        if leftover > 0:
            extra_anchors = min(leftover, len(anchors_idx) - anchor_slots)
            anchor_slots += extra_anchors
            leftover -= extra_anchors
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
            fillable = anchor["enabled"] and row_multiple is not None
            rounded = (
                -(-(keep_rows + anchor_slots) // row_multiple) * row_multiple if fillable else keep_rows + anchor_slots
            )
            if not fillable or (max_rows is not None and rounded > max_rows):
                raise ValueError(
                    "training materialization cannot retain the calibration rows under the "
                    f"configured cap: calibration={len(protected)}, current_limit={current_limit}"
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
                    is_excluded=lambda record: excluded(record, anchor_source),
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
        merged = [merged[index] for index in selected]
        provenance = [provenance[index] for index in selected]
        tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)

    def register(record: dict[str, Any]) -> None:
        seen.add(_dedup_key(record))
        corpus_ids.add(str(record.get("id")))

    merged, provenance, guard_report = _apply_empty_answer_guard(
        merged,
        provenance,
        config=guard,
        row_multiple=row_multiple,
        anchor_fill=guard_anchor_fill,
        register=register,
    )
    tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)
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
        },
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
