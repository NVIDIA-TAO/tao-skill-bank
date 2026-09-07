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

from validate_sharegpt import load_records, resolve_image, target_path


DEFECT_DETECTION_TASK = "Defect Detection"
MAINTENANCE_TASK_TYPES = (
    "Component Classification",
    "Component Detection",
    "Defect Classification",
    "Ref_based Defect Classification",
    "Ref_based Defect Detection",
)
POSITIVE_EVIDENCE = {
    "hard_positive_proxy_false_negative",
    "hard_positive_best_overlap_0_lt_iou_lte_0p5",
}
NEGATIVE_EVIDENCE = {"hard_negative_proxy_false_positive"}
CALIBRATION_EMPTY_EVIDENCE = "calibration_empty_ground_truth"
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
    candidate: dict[str, Any], *, media_root: pathlib.Path
) -> tuple[str, str, str]:
    filepath = str(candidate["filepath"])
    path = resolve_image(filepath, media_root)
    content_sha = candidate.get("content_sha256")
    phash = candidate.get("perceptual_hash")
    if content_sha is None:
        content_sha = _sha256(path) if path.is_file() else hashlib.sha256(str(path).encode()).hexdigest()
    if phash is None:
        phash = _perceptual_hash(path) if path.is_file() else str(content_sha)[:16]
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise ValueError(f"candidate {filepath!r} has invalid content_sha256")
    if not isinstance(phash, str) or len(phash) != 16:
        raise ValueError(f"candidate {filepath!r} has invalid perceptual_hash")
    try:
        int(phash, 16)
    except ValueError as exc:
        raise ValueError(f"candidate {filepath!r} has non-hex perceptual_hash") from exc
    return str(path), content_sha, phash


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
    entries: list[dict[str, Any]], target: int, *, max_novel: int
) -> list[dict[str, Any]]:
    groups = {
        task: sorted(
            (item for item in entries if item["task_type"] == task),
            key=lambda item: (
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
    while len(selected) < target:
        advanced = False
        for task in MAINTENANCE_TASK_TYPES:
            position = positions[task]
            while (
                position < len(groups[task])
                and not groups[task][position]["is_replay"]
                and novel_count >= max_novel
            ):
                position += 1
            positions[task] = position
            if position >= len(groups[task]):
                continue
            chosen = groups[task][position]
            selected.append(chosen)
            novel_count += not chosen["is_replay"]
            positions[task] = position + 1
            advanced = True
            if len(selected) == target:
                break
        if not advanced:
            break
    return selected


def _without_visual_duplicates(
    entries: Iterable[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    hamming_distance: int,
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
        if hashes.has_within(item["perceptual_hash"], hamming_distance):
            counters["near_duplicates_excluded"] += 1
            continue
        output.append(item)
        paths.add(item["resolved_path"])
        content.add(item["content_sha256"])
        hashes.add(item["perceptual_hash"])
    return output


def _validation_identities(
    records: list[dict[str, Any]], media_root: pathlib.Path
) -> tuple[set[str], set[str], _HammingIndex, set[str]]:
    paths: set[str] = set()
    content: set[str] = set()
    phashes = _HammingIndex()
    fingerprints: set[str] = set()
    for record in records:
        path = resolve_image(target_path(record, context="validation"), media_root)
        fingerprints.add(_record_fingerprint(record))
        path_text = str(path)
        if path_text in paths:
            continue
        paths.add(path_text)
        if path.is_file():
            content.add(_sha256(path))
            phashes.add(_perceptual_hash(path))
    return paths, content, phashes, fingerprints


def materialize(
    *,
    candidate_rows: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    media_root: pathlib.Path,
    max_rows: int,
    minimum_rows: int | None = None,
    row_multiple: int,
    defect_detection_fraction: float,
    proxy_empty_rate: float,
    epochs: int,
    global_batch: int,
    near_duplicate_hamming_distance: int,
    novel_image_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min(max_rows, row_multiple, epochs, global_batch) <= 0:
        raise ValueError("row, epoch, and global-batch values must be positive")
    if minimum_rows is not None and minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive when supplied")
    if global_batch != row_multiple:
        raise ValueError("row_multiple must equal effective global_batch")
    if not 0.5 <= defect_detection_fraction <= 1.0:
        raise ValueError("Defect Detection minimum fraction must be in [0.5, 1.0]")
    if not 0.0 <= proxy_empty_rate <= 1.0:
        raise ValueError("Proxy empty-ground-truth rate must be in [0, 1]")
    if not 0 <= near_duplicate_hamming_distance <= 64:
        raise ValueError("near-duplicate Hamming distance must be in [0, 64]")
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
    target_rows = requested_target_rows
    dd_target = math.ceil(target_rows * defect_detection_fraction)
    empty_target = math.floor(dd_target * proxy_empty_rate + 0.5)
    positive_target = dd_target - empty_target
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
        path = str(resolve_image(target_path(record, context="source"), media_root))
        by_path.setdefault(path, []).append(record)
    validation_paths, validation_content, validation_phashes, validation_fingerprints = (
        _validation_identities(validation_records, media_root)
    )
    counters: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    seen_record_fingerprints: set[str] = set()
    for candidate in candidate_rows:
        route_tier = candidate.get("route_tier")
        if route_tier not in {"strict", "calibration"}:
            counters["non_strict_routes_excluded"] += 1
            continue
        if not isinstance(candidate.get("filepath"), str):
            raise ValueError("every candidate requires filepath")
        resolved_path, content_sha, phash = _candidate_visual_identity(
            candidate, media_root=media_root
        )
        if (
            resolved_path in validation_paths
            or content_sha in validation_content
            or validation_phashes.has_within(
                phash, near_duplicate_hamming_distance
            )
        ):
            counters["benchmark_or_proxy_leakage_excluded"] += 1
            continue
        routed = candidate.get("routed_task_types")
        if isinstance(routed, str):
            routed = json.loads(routed)
        if not isinstance(routed, list):
            raise ValueError("every candidate requires routed_task_types")
        if route_tier == "calibration" and routed != [DEFECT_DETECTION_TASK]:
            counters["invalid_calibration_routes_excluded"] += 1
            continue
        for record in by_path.get(resolved_path, []):
            task = str(record.get("task_type"))
            if task not in routed:
                continue
            fingerprint = _record_fingerprint(record)
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
            if route_tier == "calibration" and CALIBRATION_EMPTY_EVIDENCE not in evidence:
                counters["invalid_calibration_routes_excluded"] += 1
                continue
            entry = {
                "record": record,
                "record_id": str(record.get("id")),
                "task_type": task,
                "resolved_path": resolved_path,
                "content_sha256": content_sha,
                "perceptual_hash": phash,
                "similarity": float(candidate.get("max_cosine_similarity", 0.0)),
                "evidence": evidence,
                "is_replay": bool(candidate.get("is_replay", False)),
            }
            if task == DEFECT_DETECTION_TASK:
                objects = _ground_truth_objects(record)
                entry["objects"] = objects
                if objects:
                    if route_tier == "calibration":
                        counters["non_empty_calibration_rows_excluded"] += 1
                        continue
                    if not (
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
            entries.append(entry)
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
    eligible_empty = _without_visual_duplicates(
        empty, selected=[], hamming_distance=near_duplicate_hamming_distance, counters=counters
    )
    selected_empty: list[dict[str, Any]] = []
    empty_novel_limit = min(empty_target, novel_image_limit)
    empty_novel_count = 0
    for item in eligible_empty:
        if not item["is_replay"] and empty_novel_count >= empty_novel_limit:
            continue
        selected_empty.append(item)
        empty_novel_count += not item["is_replay"]
        if len(selected_empty) == empty_target:
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
        positive_target,
        max_novel=min(positive_target, novel_image_limit - selected_empty_novel),
    )
    selected_dd = selected_empty + selected_positive
    maintenance_unique = _without_visual_duplicates(
        maintenance,
        selected=selected_dd,
        hamming_distance=near_duplicate_hamming_distance,
        counters=counters,
    )
    selected_dd_novel = sum(not item["is_replay"] for item in selected_dd)
    selected_maintenance = _task_balanced(
        maintenance_unique,
        target_rows - len(selected_dd),
        max_novel=novel_image_limit - selected_dd_novel,
    )
    accepted_target_rows: int | None = None
    for candidate_target in range(
        requested_target_rows,
        minimum_rows_aligned - 1,
        -row_multiple,
    ):
        candidate_dd = math.ceil(candidate_target * defect_detection_fraction)
        candidate_empty = math.floor(candidate_dd * proxy_empty_rate + 0.5)
        candidate_positive = candidate_dd - candidate_empty
        candidate_maintenance = candidate_target - candidate_dd
        if (
            candidate_empty <= len(selected_empty)
            and candidate_positive <= len(selected_positive)
            and candidate_maintenance <= len(selected_maintenance)
        ):
            accepted_target_rows = candidate_target
            target_rows = candidate_target
            dd_target = candidate_dd
            empty_target = candidate_empty
            positive_target = candidate_positive
            selected_empty = selected_empty[:candidate_empty]
            selected_positive = selected_positive[:candidate_positive]
            selected_dd = selected_empty + selected_positive
            selected_maintenance = selected_maintenance[:candidate_maintenance]
            marginal_quotas = _positive_quota_report(
                positive_unique, selected_positive, candidate_positive
            )
            break
    selected_entries = selected_dd + selected_maintenance
    selected_records = [item["record"] for item in selected_entries]
    tasks = Counter(item["task_type"] for item in selected_entries)
    selected_paths = {item["resolved_path"] for item in selected_entries}
    selected_content = {item["content_sha256"] for item in selected_entries}
    selected_phashes = [item["perceptual_hash"] for item in selected_entries]
    verification_index = _HammingIndex()
    near_duplicate_pairs = 0
    for phash in selected_phashes:
        if verification_index.has_within(phash, near_duplicate_hamming_distance):
            near_duplicate_pairs += 1
        verification_index.add(phash)
    selected_empty_count = len(selected_empty)
    selected_positive_count = len(selected_positive)
    selected_replay_count = sum(item["is_replay"] for item in selected_entries)
    row_count = len(selected_entries)
    expected_steps = row_count // global_batch * epochs if row_count % global_batch == 0 else None
    maintenance_target = target_rows - dd_target
    maintenance_requested = {
        task: maintenance_target // len(MAINTENANCE_TASK_TYPES)
        + (index < maintenance_target % len(MAINTENANCE_TASK_TYPES))
        for index, task in enumerate(MAINTENANCE_TASK_TYPES)
    }
    maintenance_available = Counter(item["task_type"] for item in maintenance_unique)
    maintenance_selected = Counter(item["task_type"] for item in selected_maintenance)
    verification = {
        "target_rows_reached": row_count == target_rows,
        "minimum_rows_reached": row_count >= minimum_rows_aligned,
        "defect_detection_quota_reached": len(selected_dd) >= dd_target,
        "empty_rate_matched": selected_empty_count == empty_target,
        "unique_target_images": len(selected_paths) == row_count,
        "unique_image_content": len(selected_content) == row_count,
        "near_duplicate_free": near_duplicate_pairs == 0,
        "optimizer_boundary_aligned": expected_steps is not None,
        "novel_image_limit_respected": row_count - selected_replay_count <= novel_image_limit,
        "task_strict_with_authorized_empty_calibration_only": True,
        "all_five_maintenance_tasks_present": all(
            maintenance_selected[task] > 0 for task in MAINTENANCE_TASK_TYPES
        ),
    }
    manifest = {
        "schema_version": "defect_detection_quota_manifest_v1",
        "selection_policy": "defect_detection_primary_task_strict_v1",
        "verified": all(verification.values()),
        "verification": verification,
        "configuration": {
            "materialization_cap": max_rows,
            "requested_target_rows_after_global_batch_alignment": requested_target_rows,
            "target_rows_after_global_batch_alignment": target_rows,
            "minimum_rows_requested": minimum_rows_requested,
            "minimum_rows_batch_aligned": minimum_rows_aligned,
            "defect_detection_minimum_fraction": defect_detection_fraction,
            "near_duplicate_hamming_distance": near_duplicate_hamming_distance,
            "novel_mining_pool_image_limit": novel_image_limit,
            "calibration_policy": "direct_empty_ground_truth_only_when_proxy_fp_hard_negatives_do_not_fill_proxy_matched_empty_quota",
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
            "defect_detection": len(selected_dd),
            "defect_detection_target": dd_target,
            "maintenance": len(selected_maintenance),
            "maintenance_target": maintenance_target,
            "by_task": dict(sorted(tasks.items())),
            "novel_mining_pool_images": row_count - selected_replay_count,
            "replayed_mining_pool_images": selected_replay_count,
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
        "empty_ground_truth": {
            "proxy_rate": proxy_empty_rate,
            "target_empty": empty_target,
            "selected_empty": selected_empty_count,
            "selected_non_empty": selected_positive_count,
            "selected_rate": selected_empty_count / len(selected_dd) if selected_dd else None,
        },
        "positive_marginal_quotas": marginal_quotas,
        "uniqueness": {
            "unique_rows": len({_record_fingerprint(row) for row in selected_records}),
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
    }
    return selected_records, manifest


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


def verify_bound_manifest(
    manifest_path: pathlib.Path,
    *,
    training_jsonl: pathlib.Path,
    expected_rows: int,
    epochs: int,
    global_batch: int,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.expanduser().resolve(strict=True).read_text())
    if payload.get("schema_version") != "defect_detection_quota_manifest_v1":
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
    parser.add_argument("--defect-detection-fraction", default=0.5, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--global-batch", required=True, type=int)
    parser.add_argument("--near-duplicate-hamming-distance", default=3, type=int)
    parser.add_argument("--novel-image-limit", type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        proxy = load_records(args.proxy_annotations)
        empty_rate, proxy_empty, proxy_rows = _proxy_empty_rate(proxy)
        validations = proxy[:]
        for path in args.validation_jsonl:
            validations.extend(load_records(path))
        rows, manifest = materialize(
            candidate_rows=_read_parquet(args.candidate_parquet),
            source_records=load_records(args.source_annotations),
            validation_records=validations,
            media_root=args.media_root,
            max_rows=args.max_rows,
            minimum_rows=args.minimum_rows,
            row_multiple=args.row_multiple,
            defect_detection_fraction=args.defect_detection_fraction,
            proxy_empty_rate=empty_rate,
            epochs=args.epochs,
            global_batch=args.global_batch,
            near_duplicate_hamming_distance=args.near_duplicate_hamming_distance,
            novel_image_limit=args.novel_image_limit,
        )
        manifest["empty_ground_truth"]["proxy_empty_rows"] = proxy_empty
        manifest["empty_ground_truth"]["proxy_defect_detection_rows"] = proxy_rows
        _write_jsonl(args.output, rows)
        manifest = bind_manifest(manifest, args.output)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not manifest["verified"]:
            raise ValueError(
                "Defect Detection materialization quota is not verified; inspect the manifest"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"defect_detection_ablation: {exc}", file=sys.stderr)
        return 2
    print(
        f"defect_detection_ablation: wrote {len(rows)} rows; "
        f"Defect Detection={manifest['row_counts']['defect_detection']} verified=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
