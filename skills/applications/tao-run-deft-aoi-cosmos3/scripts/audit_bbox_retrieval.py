#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and score the proxy-only ``audit_bbox_retrieval_v1`` audit.

The retrieval unit is a ground-truth object, but every ranked and emitted
identity is the original full-image parent. The frozen Benchmark is
deliberately absent from this tool's CLI: its only accepted Benchmark evidence
is a previously sealed split-attestation JSON file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import pathlib
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageStat

from atomic_samples import identity_for_paths
from nvpaw_annotations import validate_bbox
from validate_sharegpt import image_paths, prompt_and_response, resolve_image


SCHEMA_VERSION = "audit_bbox_retrieval_v1"
CROP_POLICY_VERSION = "bbox_square_mean_pad_rgb224_bicubic_v1"
ARMS = (
    "whole_image_current",
    "bbox_context_1p5",
    "bbox_context_3p0",
    "bbox_multiscale_max",
)
KS = (1, 5, 10, 20)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260908
IOU_THRESHOLD = 0.5

PHENOTYPE_MAP = {
    "PCB Conductor Open / Copper Loss": "conductor_open",
    "PCB Conductor Short / Excess Copper": "conductor_short",
    "PCB Hole / Annular-Ring Defect": "hole_annular_ring",
    "PCB Surface / Conductor Damage": "surface_conductor_damage",
    "Foreign Material / Contamination": "foreign_material",
    "Missing Component": "component_missing",
    "Wrong / Unexpected Component": "component_wrong_unexpected",
    "Shift / Skew / Rotation": "component_misalignment",
    "Tombstoning": "component_orientation",
    "Billboarding": "component_orientation",
    "Overturned Component": "component_orientation",
    "Reverse Polarity": "reverse_polarity",
    "Component Damage": "component_damage",
    "Open / Poor Solder Joint": "solder_open",
    "Excess Solder / Solder Bridge": "solder_excess_bridge",
    "Other": "other",
    # Free-form NVPAW labels below are frozen into broader visual phenotypes.
    "Cable blocking connector.": "foreign_material",
    "Component sticks up": "component_orientation",
    "Contaminated - please check": "foreign_material",
    "Contamination": "foreign_material",
    "Corrosion": "corrosion",
    "Corrosion -Region 940": "corrosion",
    "Corrosion.": "corrosion",
    "Debris , please remove": "foreign_material",
    "Debris , please remove - Resized region": "foreign_material",
    "Debris - Remove": "foreign_material",
    "Debris Found": "foreign_material",
    "Debris, please remove": "foreign_material",
    "Debris, please remove - Resized region": "foreign_material",
    "Debris, please remove.": "foreign_material",
    "Exposed spring": "component_orientation",
    "Extra object in connector": "foreign_material",
    "Foreign Object - Please review": "foreign_material",
    "Foreign object - Please remove.": "foreign_material",
    "Foreign object - Please review.": "foreign_material",
    "Glue leakage": "foreign_material",
    "Irregular print.": "surface_conductor_damage",
    "Label fell off - Please review.": "component_missing",
    "Label sticks up": "component_orientation",
    "Missing component": "component_missing",
    "Missing label": "component_missing",
    "Missing label sticker.": "component_missing",
    "Missing screw": "component_missing",
    "Missing sticker label": "component_missing",
    "Remove Debris": "foreign_material",
    "Remove Foreign object": "foreign_material",
    "Remove debris": "foreign_material",
    "Remove debris.": "foreign_material",
    "Remove foreign object": "foreign_material",
    "Remove foreign object.": "foreign_material",
    "Remove label": "component_missing",
    "Remove label sticker.": "component_missing",
    "Screw hole corrosion": "corrosion",
    "Spring Position Incorrect": "component_misalignment",
    "[ASSY]Missing label": "component_missing",
    "[ASSY]Missing label sticker - Please apply.": "component_missing",
    "[ASSY]Missing label.": "component_missing",
    "[ASSY]Protective case damage": "component_damage",
    "[ASSY]Screw not properly sitting": "component_misalignment",
    "[ASSY]Screw spring exposed_R": "component_orientation",
    "[ASSY]Screw spring_D": "component_orientation",
    "[ASSY]Spring Expose_R": "component_orientation",
    "[ASSY]Spring Exposed_L": "component_orientation",
    "[ASSY]Spring exposing": "component_orientation",
    "[ASSY][保護蓋遺失-TBD]Missing Orange Protective Casing-Top": "component_missing",
    "[ASSY][螺絲彈簧突出-TBD]": "component_orientation",
    "[Damage] Heat sink heavily damaged": "component_damage",
    "[Damage] Heavy damage on heat sink": "component_damage",
    "[Damage]Corrosion": "corrosion",
    "[Damage]Corrosion.": "corrosion",
    "[Damage]Damaged battery holder": "component_damage",
    "[Damage]Damaged heat sink": "component_damage",
    "[Damage]Damaged hole.": "hole_annular_ring",
    "[Damage]Exposed copper.": "conductor_open",
    "[Damage]Heat Sink heavily damage, please check with QC": "component_damage",
    "[Damage]Heavily damaged heat sink": "component_damage",
    "[Damage]Heavy corrosion on copper plate": "corrosion",
    "[Damage][保護蓋受損-TBD] Connector Protector Damaged": "component_damage",
    "[Damage][連接器金屬框外推, NG批退至FA]_Connector metal frame pushed out": "component_misalignment",
    "[Damage][錫點受損, 批退至FA補錫]_Solder points damaged, please check": "solder_open",
    "[Damage]damaged  heat sink": "component_damage",
    "[Damage]damaged heat sink": "component_damage",
    "[NTF]Debris on pin.": "foreign_material",
    "[NTF]Debris.": "foreign_material",
    "[NTF]Extra label.": "foreign_material",
    "[NTF]Foreign object, please remove and retest.": "foreign_material",
    "[Pollution] Contaminated screw": "foreign_material",
    "[Pollution] Please remove debris from copper plate": "foreign_material",
    "[Pollution] Remove debris": "foreign_material",
    "[Pollution]Contamination.": "foreign_material",
    "[Pollution]Flux leakage": "foreign_material",
    "[Pollution]Remove foreign object": "foreign_material",
    "connector not being connected": "component_misalignment",
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


def canonical_phenotype(label: str) -> str:
    if label not in PHENOTYPE_MAP:
        raise ValueError(f"unmapped defect label: {label!r}")
    return PHENOTYPE_MAP[label]


def extract_labeled_boxes(text: str, *, context: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: ground-truth answer is not JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{context}: ground-truth answer must be a JSON array")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{context}: object {index} must be a JSON object")
        label = item.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}: object {index} requires a label")
        bbox = validate_bbox(item.get("bbox_2d"), record_id=f"{context}: object {index}")
        output.append(
            {
                "bbox_2d": bbox,
                "raw_label": label,
                "canonical_phenotype": canonical_phenotype(label),
            }
        )
    return output


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def minimum_cost_assignment(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Dependency-free rectangular Hungarian assignment, matching the evaluator."""

    if not costs or not costs[0]:
        return []
    row_count = len(costs)
    column_count = len(costs[0])
    if any(len(row) != column_count for row in costs):
        raise ValueError("assignment cost matrix must be rectangular")
    transposed = row_count > column_count
    matrix = [list(row) for row in zip(*costs)] if transposed else [list(row) for row in costs]
    row_count = len(matrix)
    column_count = len(matrix[0])
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row_for_column = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row_for_column[0] = row_index
        minimum_reduced_cost = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        active_column = 0
        while True:
            used[active_column] = True
            active_row = matched_row_for_column[active_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    matrix[active_row - 1][column - 1]
                    - row_potential[active_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_reduced_cost[column]:
                    minimum_reduced_cost[column] = reduced_cost
                    predecessor[column] = active_column
                if minimum_reduced_cost[column] < delta:
                    delta = minimum_reduced_cost[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row_for_column[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_reduced_cost[column] -= delta
            active_column = next_column
            if matched_row_for_column[active_column] == 0:
                break
        while True:
            previous_column = predecessor[active_column]
            matched_row_for_column[active_column] = matched_row_for_column[previous_column]
            active_column = previous_column
            if active_column == 0:
                break
    pairs: list[tuple[int, int]] = []
    for column in range(1, column_count + 1):
        row = matched_row_for_column[column]
        if row:
            pairs.append((column - 1, row - 1) if transposed else (row - 1, column - 1))
    return sorted(pairs)


def unmatched_ground_truth_indices(
    ground_truth: list[Iterable[float]],
    predictions: list[Iterable[float]],
    *,
    threshold: float = IOU_THRESHOLD,
) -> tuple[list[int], tuple[int, int, int]]:
    if not ground_truth:
        return [], (0, len(predictions), 0)
    if not predictions:
        return list(range(len(ground_truth))), (0, 0, len(ground_truth))
    matrix = [[box_iou(gt, pred) for pred in predictions] for gt in ground_truth]
    match_bonus = min(len(ground_truth), len(predictions)) + 1.0
    rewards = [
        [score + (match_bonus if score > threshold else 0.0) for score in row]
        for row in matrix
    ]
    assignments = minimum_cost_assignment([[-score for score in row] for row in rewards])
    matched_gt = {
        gt_index
        for gt_index, pred_index in assignments
        if matrix[gt_index][pred_index] > threshold
    }
    true_positives = len(matched_gt)
    counts = (
        true_positives,
        len(predictions) - true_positives,
        len(ground_truth) - true_positives,
    )
    return [index for index in range(len(ground_truth)) if index not in matched_gt], counts


def load_display_image(path: pathlib.Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def normalized_box_to_pixels(
    bbox: Iterable[int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    values = validate_bbox(list(bbox), record_id="normalized bbox")
    x1 = max(0, min(width, math.floor(values[0] * width / 1000.0)))
    y1 = max(0, min(height, math.floor(values[1] * height / 1000.0)))
    x2 = max(0, min(width, math.ceil(values[2] * width / 1000.0)))
    y2 = max(0, min(height, math.ceil(values[3] * height / 1000.0)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("normalized bbox has no positive area after pixel conversion")
    return x1, y1, x2, y2


def render_context_crop(
    image: Image.Image,
    bbox: Iterable[int],
    *,
    scale: float,
) -> tuple[Image.Image, dict[str, Any]]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("crop scale must be positive and finite")
    image = image.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = normalized_box_to_pixels(bbox, width=width, height=height)
    native_width = x2 - x1
    native_height = y2 - y1
    side = max(1, math.ceil(max(native_width, native_height) * scale))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    right = left + side
    bottom = top + side
    visible_box = (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )
    if visible_box[2] <= visible_box[0] or visible_box[3] <= visible_box[1]:
        raise ValueError("crop has no visible image region")
    visible = image.crop(visible_box).convert("RGB")
    mean = tuple(round(value) for value in ImageStat.Stat(visible).mean[:3])
    canvas = Image.new("RGB", (side, side), color=mean)
    canvas.paste(visible, (visible_box[0] - left, visible_box[1] - top))
    resized = canvas.resize((224, 224), resample=Image.Resampling.BICUBIC)
    visible_area = (visible_box[2] - visible_box[0]) * (visible_box[3] - visible_box[1])
    gray_stats = ImageStat.Stat(resized.convert("L"))
    return resized, {
        "bbox_pixel": [x1, y1, x2, y2],
        "image_width": width,
        "image_height": height,
        "native_width": native_width,
        "native_height": native_height,
        "native_area": native_width * native_height,
        "context_scale": float(scale),
        "context_side": side,
        "padding_fraction": 1.0 - visible_area / float(side * side),
        "local_contrast": float(gray_stats.stddev[0]),
        "crop_policy_version": CROP_POLICY_VERSION,
    }


def _existing_crop_metadata(
    image: Image.Image,
    bbox: Iterable[int],
    *,
    scale: float,
    crop_path: pathlib.Path,
) -> dict[str, Any] | None:
    """Recover geometry cheaply for a verified crop from an interrupted prepare."""

    try:
        with Image.open(crop_path) as stored:
            stored.verify()
        with Image.open(crop_path) as stored:
            rendered = stored.convert("RGB")
            if rendered.size != (224, 224):
                return None
            local_contrast = float(ImageStat.Stat(rendered.convert("L")).stddev[0])
    except (OSError, ValueError):
        return None
    width, height = image.size
    x1, y1, x2, y2 = normalized_box_to_pixels(bbox, width=width, height=height)
    native_width = x2 - x1
    native_height = y2 - y1
    side = max(1, math.ceil(max(native_width, native_height) * scale))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    visible_width = max(0, min(width, left + side) - max(0, left))
    visible_height = max(0, min(height, top + side) - max(0, top))
    if not visible_width or not visible_height:
        return None
    return {
        "bbox_pixel": [x1, y1, x2, y2],
        "image_width": width,
        "image_height": height,
        "native_width": native_width,
        "native_height": native_height,
        "native_area": native_width * native_height,
        "context_scale": float(scale),
        "context_side": side,
        "padding_fraction": 1.0 - visible_width * visible_height / float(side * side),
        "local_contrast": local_contrast,
        "crop_policy_version": CROP_POLICY_VERSION,
    }


def derive_source_group_id(dataset: str, parent_filepath: str) -> str:
    """Group a parent with common crop/photometric derivative path variants."""

    path = pathlib.PurePosixPath(parent_filepath.replace("\\", "/"))
    ignored_directories = {
        "aug",
        "augment",
        "augmented",
        "crop",
        "crops",
        "gen_aug",
        "gen_qwen",
        "generated",
        "synthetic",
    }
    parents = [part.casefold() for part in path.parts[:-1] if part.casefold() not in ignored_directories]
    stem = path.stem.casefold()
    previous = None
    while previous != stem:
        previous = stem
        stem = re.sub(r"^(?:rotation|rotate)[_-]?\d+[_-]+", "", stem)
        stem = re.sub(r"^light[_-]?\d+[_-]+", "", stem)
        stem = re.sub(r"^(?:flip|flipped)[_-]?(?:h|v|horizontal|vertical)?[_-]+", "", stem)
        stem = re.sub(r"^(?:blur|noise|brightness|contrast)[_-]?[a-z0-9.]+[_-]+", "", stem)
    stem = re.sub(r"(?:__|[_-])crop[_-]?\d+(?:_[0-9a-f]{8,32})?$", "", stem)
    normalized = "/".join([*parents, stem, path.suffix.casefold()])
    digest = hashlib.sha256(f"{dataset.casefold()}\0{normalized}".encode()).hexdigest()
    return f"source_group:{digest}"


def _object_id(prefix: str, record_id: str, object_index: int) -> str:
    digest = hashlib.sha256(f"{prefix}\0{record_id}\0{object_index}".encode()).hexdigest()
    return f"{prefix}:{digest}"


def _crop_path(
    crop_root: pathlib.Path,
    *,
    role: str,
    object_id: str,
    scale: float,
) -> pathlib.Path:
    digest = hashlib.sha256(
        f"{CROP_POLICY_VERSION}\0{role}\0{object_id}\0{scale:.6f}".encode()
    ).hexdigest()
    return crop_root / role / f"context_{scale:g}" / digest[:2] / f"{digest}.png"


def _crop_descriptors(
    image: Image.Image,
    bbox: list[int],
    *,
    crop_root: pathlib.Path,
    role: str,
    object_id: str,
    context_scales: Iterable[float],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scale in context_scales:
        path = _crop_path(
            crop_root,
            role=role,
            object_id=object_id,
            scale=float(scale),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = (
            _existing_crop_metadata(image, bbox, scale=float(scale), crop_path=path)
            if path.is_file()
            else None
        )
        if metadata is None:
            rendered, metadata = render_context_crop(image, bbox, scale=float(scale))
            rendered.save(path, format="PNG")
        key = f"{float(scale):.1f}"
        output[key] = {
            **metadata,
            "filepath": str(path.resolve()),
            "sha256": _sha256(path),
        }
    return output


def _base_object(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    object_index: int,
    object_id: str,
    media_root: pathlib.Path,
    crop_root: pathlib.Path,
    role: str,
    context_scales: Iterable[float],
    parent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_id = str(record["id"])
    paths = image_paths(record, context=record_id)
    if len(paths) != 1:
        raise ValueError(f"{record_id}: bbox audit accepts only single-image rows")
    parent_path = resolve_image(paths[0], media_root)
    if parent_context is None:
        if not parent_path.is_file():
            raise ValueError(f"{record_id}: missing parent image {parent_path}")
        parent_context = {
            "path": parent_path,
            "image": load_display_image(parent_path),
            "sha256": _sha256(parent_path),
            "parent_id": identity_for_paths("single_image", [str(parent_path)]),
        }
    if pathlib.Path(parent_context["path"]) != parent_path:
        raise ValueError(f"{record_id}: parent context does not match source image")
    image = parent_context["image"]
    parent_id = str(parent_context["parent_id"])
    crops = _crop_descriptors(
        image,
        item["bbox_2d"],
        crop_root=crop_root,
        role=role,
        object_id=object_id,
        context_scales=context_scales,
    )
    first_crop = crops[sorted(crops, key=float)[0]]
    return {
        "record_id": record_id,
        "dataset": str(record.get("dataset", "unknown")),
        "parent_id": parent_id,
        "parent_filepath": str(parent_path),
        "parent_sha256": str(parent_context["sha256"]),
        "source_group_id": derive_source_group_id(
            str(record.get("dataset", "unknown")), str(parent_path)
        ),
        "object_index": object_index,
        "bbox_2d": item["bbox_2d"],
        "raw_label": item["raw_label"],
        "canonical_phenotype": item["canonical_phenotype"],
        "native_area": first_crop["native_area"],
        "native_short_side": min(first_crop["native_width"], first_crop["native_height"]),
        "image_width": first_crop["image_width"],
        "image_height": first_crop["image_height"],
        "crops": crops,
        "crop_policy_version": CROP_POLICY_VERSION,
    }


def build_candidate_objects(
    records: Iterable[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    crop_root: pathlib.Path,
    context_scales: Iterable[float] = (1.5, 3.0),
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("candidate crop workers must be at least one")
    if workers > 1:
        build_one = functools.partial(
            _build_candidate_record,
            media_root=media_root,
            crop_root=crop_root,
            context_scales=tuple(context_scales),
        )
        output: list[dict[str, Any]] = []
        detection_records = (
            record for record in records if record.get("task_type") == "Defect Detection"
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for record_objects in executor.map(
                build_one,
                detection_records,
                chunksize=8,
            ):
                output.extend(record_objects)
        return output
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("task_type") != "Defect Detection":
            continue
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Defect Detection Mining row requires an id")
        _, answer = prompt_and_response(record, context=record_id)
        paths = image_paths(record, context=record_id)
        if len(paths) != 1:
            raise ValueError(f"{record_id}: bbox audit accepts only single-image rows")
        parent_path = resolve_image(paths[0], media_root)
        if not parent_path.is_file():
            raise ValueError(f"{record_id}: missing parent image {parent_path}")
        parent_context = {
            "path": parent_path,
            "image": load_display_image(parent_path),
            "sha256": _sha256(parent_path),
            "parent_id": identity_for_paths("single_image", [str(parent_path)]),
        }
        for object_index, item in enumerate(
            extract_labeled_boxes(answer, context=record_id)
        ):
            candidate_id = _object_id("candidate", record_id, object_index)
            output.append(
                {
                    "candidate_id": candidate_id,
                    **_base_object(
                        record,
                        item,
                        object_index=object_index,
                        object_id=candidate_id,
                        media_root=media_root,
                        crop_root=crop_root,
                        role="candidate",
                        context_scales=context_scales,
                        parent_context=parent_context,
                    ),
                }
            )
    return output


def _build_candidate_record(
    record: dict[str, Any],
    *,
    media_root: pathlib.Path,
    crop_root: pathlib.Path,
    context_scales: tuple[float, ...],
) -> list[dict[str, Any]]:
    return build_candidate_objects(
        [record],
        media_root=media_root,
        crop_root=crop_root,
        context_scales=context_scales,
        workers=1,
    )


def extract_fn_objects(
    records: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    evaluator: Any,
    variant: str,
    media_root: pathlib.Path,
    crop_root: pathlib.Path,
    context_scales: Iterable[float] = (1.5, 3.0),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_index: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        prediction_id = prediction.get("id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError("prediction row requires an id")
        if prediction_id in prediction_index:
            raise ValueError(f"duplicate prediction id: {prediction_id!r}")
        prediction_index[prediction_id] = prediction
    output: list[dict[str, Any]] = []
    totals = Counter()
    detection_rows = 0
    mismatches = 0
    for record in records:
        if record.get("task_type") != "Defect Detection":
            continue
        detection_rows += 1
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Defect Detection Proxy row requires an id")
        if record_id not in prediction_index:
            raise ValueError(f"missing prediction for Proxy row {record_id!r}")
        _, answer = prompt_and_response(record, context=record_id)
        objects = extract_labeled_boxes(answer, context=record_id)
        paths = image_paths(record, context=record_id)
        if len(paths) != 1:
            raise ValueError(f"{record_id}: bbox audit accepts only single-image rows")
        parent_path = resolve_image(paths[0], media_root)
        if not parent_path.is_file():
            raise ValueError(f"{record_id}: missing parent image {parent_path}")
        parent_context = {
            "path": parent_path,
            "image": load_display_image(parent_path),
            "sha256": _sha256(parent_path),
            "parent_id": identity_for_paths("single_image", [str(parent_path)]),
        }
        gt_boxes = [tuple(float(value) for value in item["bbox_2d"]) for item in objects]
        raw_prediction = str(prediction_index[record_id].get("raw_prediction", ""))
        native_predictions, parse_ok = evaluator.parse_boxes(raw_prediction)
        predicted_boxes = (
            evaluator.canonicalize_prediction_boxes(native_predictions, "xyxy")
            if parse_ok
            else []
        )
        unmatched, counts = unmatched_ground_truth_indices(
            gt_boxes, predicted_boxes, threshold=IOU_THRESHOLD
        )
        evaluator_counts = evaluator.one_to_one_detection_counts(
            gt_boxes, predicted_boxes, IOU_THRESHOLD
        )
        if tuple(evaluator_counts) != counts:
            mismatches += 1
            raise ValueError(
                f"{record_id}: FN assignment does not reconcile with recorded evaluator: "
                f"audit={counts} evaluator={evaluator_counts}"
            )
        totals.update({"true_positives": counts[0], "false_positives": counts[1], "false_negatives": counts[2]})
        for object_index in unmatched:
            query_id = _object_id(f"query-{variant}", record_id, object_index)
            output.append(
                {
                    "query_id": query_id,
                    "query_variant": variant,
                    "prediction_id": record_id,
                    "match_threshold": IOU_THRESHOLD,
                    "match_operator": ">",
                    **_base_object(
                        record,
                        objects[object_index],
                        object_index=object_index,
                        object_id=_object_id("query", record_id, object_index),
                        media_root=media_root,
                        crop_root=crop_root,
                        role="query",
                        context_scales=context_scales,
                        parent_context=parent_context,
                    ),
                }
            )
    return output, {
        "variant": variant,
        "detection_rows": detection_rows,
        "queries": len(output),
        "true_positives": totals["true_positives"],
        "false_positives": totals["false_positives"],
        "false_negatives": totals["false_negatives"],
        "evaluator_count_mismatches": mismatches,
        "iou_threshold": IOU_THRESHOLD,
        "iou_operator": ">",
    }


def parent_topk(
    candidates: list[dict[str, Any]], scores: np.ndarray, *, top_k: int
) -> list[dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordering = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), str(candidates[index]["candidate_id"])),
    )
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for index in ordering:
        candidate = candidates[index]
        parent_id = str(candidate["parent_id"])
        if parent_id in seen:
            continue
        seen.add(parent_id)
        output.append({**candidate, "score": float(scores[index]), "rank": len(output) + 1})
        if len(output) == top_k:
            break
    return output


def parent_only_rows(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "filepath": str(row["parent_filepath"]),
            "atomic_sample_id": str(row["parent_id"]),
            "candidate_id": str(row["candidate_id"]),
            "routed_task_types": ["Defect Detection"],
        }
        for row in ranked
    ]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _dataset_balanced(
    queries: list[dict[str, Any]],
    per_query: dict[str, dict[str, Any]],
    field: str,
) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for query in queries:
        by_dataset[str(query["dataset"])].append(float(per_query[str(query["query_id"])][field]))
    return _mean(_mean(values) for values in by_dataset.values())


def compute_arm_metrics(
    queries: list[dict[str, Any]],
    neighbors: dict[str, list[dict[str, Any]]],
    *,
    relevant_parent_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    per_query: dict[str, dict[str, Any]] = {}
    unique_relevant: set[str] = set()
    unique_relevant_groups: set[str] = set()
    retrieved_parents: dict[str, str] = {}
    retrieved_parent_groups: dict[str, str] = {}
    for query in queries:
        query_id = str(query["query_id"])
        phenotype = str(query["canonical_phenotype"])
        rows = neighbors.get(query_id, [])
        relevance = [str(row["candidate_phenotype"]) == phenotype for row in rows]
        metrics: dict[str, Any] = {}
        for k in KS:
            metrics[f"p_at_{k}"] = sum(relevance[:k]) / float(k)
        first = next((index + 1 for index, relevant in enumerate(relevance) if relevant), None)
        metrics["reciprocal_rank"] = 1.0 / first if first is not None else 0.0
        dcg = sum(
            1.0 / math.log2(index + 2)
            for index, relevant in enumerate(relevance[:20])
            if relevant
        )
        relevant_available = (
            int(relevant_parent_counts.get(phenotype, 0))
            if relevant_parent_counts is not None
            else len(
                {
                    str(row["parent_id"])
                    for row, relevant in zip(rows, relevance, strict=True)
                    if relevant
                }
            )
        )
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(20, relevant_available)))
        metrics["ndcg_at_20"] = dcg / ideal if ideal else 0.0
        metrics["coverage"] = float(any(relevance[:20]))
        relevant_at_10 = [
            str(row["parent_id"])
            for row, relevant in zip(rows[:10], relevance[:10], strict=True)
            if relevant
        ]
        metrics["relevant_parent_ids_at_10"] = relevant_at_10
        unique_relevant.update(relevant_at_10)
        for row, relevant in zip(rows[:10], relevance[:10], strict=True):
            parent_id = str(row["parent_id"])
            group_id = str(row.get("source_group_id", parent_id))
            retrieved_parents.setdefault(parent_id, str(row["candidate_dataset"]))
            retrieved_parent_groups.setdefault(parent_id, group_id)
            if relevant:
                unique_relevant_groups.add(group_id)
        per_query[query_id] = metrics
    dataset_balanced = {
        field: _dataset_balanced(queries, per_query, field)
        for field in [
            *(f"p_at_{k}" for k in KS),
            "reciprocal_rank",
            "ndcg_at_20",
            "coverage",
        ]
    }
    source_counts = Counter(retrieved_parents.values())
    retrieved_count = len(retrieved_parents)
    source_shares = {
        source: count / retrieved_count for source, count in sorted(source_counts.items())
    }
    source_group_counts = Counter(retrieved_parent_groups.values())
    source_group_shares = {
        group: count / retrieved_count for group, count in sorted(source_group_counts.items())
    }
    return {
        "dataset_balanced_macro": dataset_balanced,
        "per_query": per_query,
        "unique_relevant_parents": len(unique_relevant),
        "unique_relevant_source_groups": len(unique_relevant_groups),
        "retrieved_unique_parents_at_10": retrieved_count,
        "source_share": source_shares,
        "max_source_share": max(source_shares.values(), default=0.0),
        "source_group_share": source_group_shares,
        "max_source_group_share": max(source_group_shares.values(), default=0.0),
    }


def paired_bootstrap_p10(
    queries: list[dict[str, Any]],
    challenger_per_query: dict[str, dict[str, Any]],
    baseline_per_query: dict[str, dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    query_ids = [str(query["query_id"]) for query in queries]
    if set(query_ids) != set(challenger_per_query) or set(query_ids) != set(baseline_per_query):
        raise ValueError("bootstrap inputs do not cover the same queries")
    point_challenger = _dataset_balanced(queries, challenger_per_query, "p_at_10")
    point_baseline = _dataset_balanced(queries, baseline_per_query, "p_at_10")
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    positions_by_dataset: dict[str, list[int]] = defaultdict(list)
    for position, query in enumerate(queries):
        positions_by_dataset[str(query["dataset"])].append(position)
    for sample_index in range(resamples):
        dataset_means: list[float] = []
        for positions in positions_by_dataset.values():
            sampled = generator.choice(positions, size=len(positions), replace=True)
            dataset_means.append(
                _mean(
                    float(challenger_per_query[str(queries[int(position)]["query_id"])]["p_at_10"])
                    - float(baseline_per_query[str(queries[int(position)]["query_id"])]["p_at_10"])
                    for position in sampled
                )
            )
        differences[sample_index] = _mean(dataset_means)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "resamples": resamples,
        "seed": seed,
        "delta": point_challenger - point_baseline,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def evaluate_gates(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    bootstrap: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, bool]:
    baseline_unique = int(baseline["unique_relevant_parents"])
    challenger_unique = int(challenger["unique_relevant_parents"])
    unique_gate = challenger_unique >= 1.5 * baseline_unique if baseline_unique else challenger_unique > 0
    gates = {
        "primary_precision": (
            float(bootstrap["delta"]) >= 0.15
            and float(bootstrap["ci95_lower"]) > 0.0
        ),
        "unique_relevant_parents": unique_gate,
        "source_share": float(challenger["max_source_share"]) <= 0.60,
        "lineage": (
            float(lineage["lineage_rate"]) == 1.0
            and int(lineage["shared_image_leaks"]) == 0
            and int(lineage["crop_emitted_count"]) == 0
        ),
    }
    gates["go"] = all(gates.values())
    return gates


def audit_lineage(
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    benchmark_attestation: dict[str, int],
) -> dict[str, Any]:
    required_query = {"query_id", "parent_id", "parent_filepath", "parent_sha256"}
    required_candidate = {"candidate_id", "parent_id", "parent_filepath", "parent_sha256"}
    total = len(queries) + len(candidates)
    traced = sum(required_query.issubset(item) for item in queries) + sum(
        required_candidate.issubset(item) for item in candidates
    )
    proxy_paths = {str(item.get("parent_filepath")) for item in queries}
    source_paths = {str(item.get("parent_filepath")) for item in candidates}
    proxy_hashes = {str(item.get("parent_sha256")) for item in queries}
    source_hashes = {str(item.get("parent_sha256")) for item in candidates}
    path_leaks = len(proxy_paths & source_paths)
    content_leaks = len(proxy_hashes & source_hashes)
    benchmark_leaks = int(benchmark_attestation.get("benchmark:mining", -1))
    if benchmark_leaks < 0:
        raise ValueError("split attestation lacks benchmark:mining overlap")
    return {
        "lineage_rows": total,
        "lineage_traceable_rows": traced,
        "lineage_rate": traced / total if total else 0.0,
        "proxy_mining_parent_path_leaks": path_leaks,
        "proxy_mining_content_sha_leaks": content_leaks,
        "benchmark_mining_attested_leaks": benchmark_leaks,
        "shared_image_leaks": path_leaks + content_leaks + benchmark_leaks,
        "crop_emitted_count": 0,
    }


def _iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            yield value


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_evaluator(path: pathlib.Path) -> Any:
    resolved = path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location("bbox_audit_recorded_evaluator", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recorded evaluator: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "parse_boxes",
        "canonicalize_prediction_boxes",
        "one_to_one_detection_counts",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"recorded evaluator lacks required helpers: {missing}")
    return module


def _rank_quartiles(rows: list[dict[str, Any]], field: str, output: str) -> None:
    if not rows:
        return
    ordering = sorted(
        range(len(rows)),
        key=lambda index: (float(rows[index][field]), str(rows[index].get("query_id", index))),
    )
    for rank, index in enumerate(ordering):
        rows[index][output] = f"Q{min(4, rank * 4 // len(rows) + 1)}"


def _read_parquet_filepaths(path: pathlib.Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for bbox retrieval audit") from exc
    table = pq.read_table(path, columns=["filepath"])
    values = table.column("filepath").to_pylist()
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{path}: invalid filepath column")
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate filepath values")
    return set(values)


def _write_embedding_inputs(path: pathlib.Path, filepaths: Iterable[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for bbox retrieval audit") from exc
    values = sorted(set(filepaths))
    if not values:
        raise ValueError("bbox audit embedding input is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"filepath": values}), path, compression="zstd")


def stage_query_parent_images(
    queries: Iterable[dict[str, Any]],
    *,
    stage_root: pathlib.Path,
) -> int:
    """Copy full-image query inputs beneath the audit root for container access.

    Proxy manifests can legitimately point outside the submitter's container
    mounts.  The original parent path remains the emitted identity; only the
    encoder input uses this byte-identical staged copy.
    """

    stage_root.mkdir(parents=True, exist_ok=True)
    staged: dict[pathlib.Path, pathlib.Path] = {}
    for query in queries:
        parent = pathlib.Path(str(query["parent_filepath"])).resolve(strict=True)
        expected_sha256 = str(query["parent_sha256"])
        suffix = parent.suffix.casefold() or ".image"
        destination = stage_root / expected_sha256[:2] / f"{expected_sha256}{suffix}"
        if destination not in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or _sha256(destination) != expected_sha256:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{destination.name}.",
                    dir=destination.parent,
                    delete=False,
                ) as temporary:
                    temporary_path = pathlib.Path(temporary.name)
                try:
                    shutil.copyfile(parent, temporary_path)
                    if _sha256(temporary_path) != expected_sha256:
                        raise ValueError(
                            f"staged query parent content changed while copying: {parent}"
                        )
                    os.replace(temporary_path, destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
            staged[destination] = parent
        query["embedding_filepath"] = str(destination.resolve())
    return len(staged)


def _verify_preregistered_report(path: pathlib.Path, expected_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("pre-registered report SHA-256 must be 64 lowercase hex characters")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            "pre-registered report changed before result computation: "
            f"expected={expected_sha256} observed={observed}"
        )


def prepare_audit(
    *,
    proxy: pathlib.Path,
    reference_predictions: pathlib.Path,
    zero_shot_predictions: pathlib.Path,
    mining: pathlib.Path,
    evaluator_path: pathlib.Path,
    media_root: pathlib.Path,
    a0_source_embeddings: pathlib.Path,
    encoder_manifest: pathlib.Path,
    split_attestation: pathlib.Path,
    output_dir: pathlib.Path,
    preregistered_report_sha256: str,
    crop_workers: int = 1,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    report_path = output_dir / "REPORT.md"
    _verify_preregistered_report(report_path, preregistered_report_sha256)
    resolved_inputs = {
        "proxy": proxy.expanduser().resolve(strict=True),
        "reference_predictions": reference_predictions.expanduser().resolve(strict=True),
        "zero_shot_predictions": zero_shot_predictions.expanduser().resolve(strict=True),
        "mining": mining.expanduser().resolve(strict=True),
        "evaluator": evaluator_path.expanduser().resolve(strict=True),
        "a0_source_embeddings": a0_source_embeddings.expanduser().resolve(strict=True),
        "encoder_manifest": encoder_manifest.expanduser().resolve(strict=True),
        "split_attestation": split_attestation.expanduser().resolve(strict=True),
    }
    media_root = media_root.expanduser().resolve()
    split_payload = json.loads(resolved_inputs["split_attestation"].read_text())
    overlap = split_payload.get("target_overlap")
    if not isinstance(overlap, dict):
        raise ValueError("split attestation lacks target_overlap")
    for pair in ("proxy:mining", "benchmark:mining"):
        if overlap.get(pair) != 0:
            raise ValueError(f"split attestation does not seal zero {pair} overlap")
    evaluator = _load_evaluator(resolved_inputs["evaluator"])
    proxy_rows = list(_iter_jsonl(resolved_inputs["proxy"]))
    reference_rows = list(_iter_jsonl(resolved_inputs["reference_predictions"]))
    zero_shot_rows = list(_iter_jsonl(resolved_inputs["zero_shot_predictions"]))
    crop_root = output_dir / "crops"
    candidates = build_candidate_objects(
        _iter_jsonl(resolved_inputs["mining"]),
        media_root=media_root,
        crop_root=crop_root,
        workers=crop_workers,
    )
    if not candidates:
        raise ValueError("Mining contains no positive single-image Defect Detection objects")
    reference_queries, reference_reconciliation = extract_fn_objects(
        proxy_rows,
        reference_rows,
        evaluator=evaluator,
        variant="reference",
        media_root=media_root,
        crop_root=crop_root,
    )
    zero_shot_queries, zero_shot_reconciliation = extract_fn_objects(
        proxy_rows,
        zero_shot_rows,
        evaluator=evaluator,
        variant="zero_shot",
        media_root=media_root,
        crop_root=crop_root,
    )
    queries = [*reference_queries, *zero_shot_queries]
    if not reference_queries or not zero_shot_queries:
        raise ValueError("both reference and zero-shot predictions must yield FN queries")
    staged_query_parents = stage_query_parent_images(
        queries,
        stage_root=output_dir / "embedding_sources" / "query_parents",
    )
    for variant in ("reference", "zero_shot"):
        variant_rows = [row for row in queries if row["query_variant"] == variant]
        _rank_quartiles(variant_rows, "native_area", "bbox_area_quartile")
        for row in variant_rows:
            row["padding_fraction"] = float(row["crops"]["1.5"]["padding_fraction"])
        _rank_quartiles(variant_rows, "padding_fraction", "padding_fraction_quartile")
    cached_paths = _read_parquet_filepaths(resolved_inputs["a0_source_embeddings"])
    candidate_parent_paths = {str(row["parent_filepath"]) for row in candidates}
    missing_a0 = sorted(candidate_parent_paths - cached_paths)
    if missing_a0:
        raise ValueError(
            "A0 whole-image cache does not cover every candidate parent: "
            f"missing={missing_a0[:10]}"
        )
    embedding_filepaths = {
        str(crop["filepath"])
        for row in [*candidates, *queries]
        for crop in row["crops"].values()
    }
    embedding_filepaths.update(str(row["embedding_filepath"]) for row in queries)
    candidate_path = output_dir / "candidate_objects.jsonl"
    query_path = output_dir / "query_objects.jsonl"
    embedding_input_path = output_dir / "embedding_inputs.parquet"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(query_path, queries)
    _write_embedding_inputs(embedding_input_path, embedding_filepaths)
    lineage = audit_lineage(
        queries,
        candidates,
        benchmark_attestation={"benchmark:mining": int(overlap["benchmark:mining"])},
    )
    encoder_payload = json.loads(resolved_inputs["encoder_manifest"].read_text())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": "PREPARED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_read": False,
        "preregistered_report_sha256": preregistered_report_sha256,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in resolved_inputs.items()
        },
        "encoder": encoder_payload.get("encoder", encoder_payload),
        "crop_policy_version": CROP_POLICY_VERSION,
        "crop_workers": crop_workers,
        "phenotype_mapping": PHENOTYPE_MAP,
        "candidates": len(candidates),
        "candidate_parents": len({row["parent_id"] for row in candidates}),
        "candidate_source_groups": len({row["source_group_id"] for row in candidates}),
        "queries": {
            "reference": reference_reconciliation,
            "zero_shot": zero_shot_reconciliation,
        },
        "embedding_inputs": len(embedding_filepaths),
        "query_parent_images_staged": staged_query_parents,
        "a0_candidate_cache_rows_reused": len(candidate_parent_paths),
        "a0_candidate_rows_reembedded": 0,
        "lineage": lineage,
        "artifacts": {
            "candidate_objects": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
            "query_objects": {"path": str(query_path), "sha256": _sha256(query_path)},
            "embedding_inputs": {"path": str(embedding_input_path), "sha256": _sha256(embedding_input_path)},
        },
    }
    _atomic_json(output_dir / "prepare_manifest.json", payload)
    return payload


def _load_required_embeddings(
    path: pathlib.Path,
    required_filepaths: Iterable[str],
) -> dict[str, np.ndarray]:
    """Load only requested vectors, without materialising a large cache at once."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for bbox retrieval audit") from exc
    required = set(required_filepaths)
    if not required:
        return {}
    found: dict[str, np.ndarray] = {}
    parquet = pq.ParquetFile(path.expanduser().resolve(strict=True))
    for batch in parquet.iter_batches(
        batch_size=2048,
        columns=["filepath", "embedding"],
    ):
        values = batch.to_pydict()
        for filepath, embedding in zip(
            values["filepath"], values["embedding"], strict=True
        ):
            if filepath not in required:
                continue
            if filepath in found:
                raise ValueError(f"{path}: duplicate embedding filepath {filepath!r}")
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"{path}: invalid embedding for {filepath!r}")
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                raise ValueError(f"{path}: zero-norm embedding for {filepath!r}")
            found[filepath] = vector / norm
    missing = sorted(required - set(found))
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} required embeddings: {missing[:10]}")
    dimensions = {vector.shape[0] for vector in found.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{path}: inconsistent embedding dimensions: {sorted(dimensions)}")
    return found


def _embedding_matrix(
    filepaths: Iterable[str],
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    values = [embeddings[str(filepath)] for filepath in filepaths]
    if not values:
        raise ValueError("cannot construct an empty embedding matrix")
    return np.stack(values).astype(np.float32, copy=False)


def _candidate_relevant_parent_counts(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    parents: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        parents[str(candidate["canonical_phenotype"])].add(str(candidate["parent_id"]))
    return {phenotype: len(values) for phenotype, values in sorted(parents.items())}


def rank_retrieval_arms(
    candidates: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    a0_embeddings: dict[str, np.ndarray],
    crop_embeddings: dict[str, np.ndarray],
    top_k: int = 20,
    query_batch_size: int = 64,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if query_batch_size <= 0:
        raise ValueError("query batch size must be positive")
    candidate_matrices = {
        "whole_image_current": _embedding_matrix(
            (row["parent_filepath"] for row in candidates), a0_embeddings
        ),
        "bbox_context_1p5": _embedding_matrix(
            (row["crops"]["1.5"]["filepath"] for row in candidates), crop_embeddings
        ),
        "bbox_context_3p0": _embedding_matrix(
            (row["crops"]["3.0"]["filepath"] for row in candidates), crop_embeddings
        ),
    }
    query_matrices = {
        "whole_image_current": _embedding_matrix(
            (row["embedding_filepath"] for row in queries), crop_embeddings
        ),
        "bbox_context_1p5": _embedding_matrix(
            (row["crops"]["1.5"]["filepath"] for row in queries), crop_embeddings
        ),
        "bbox_context_3p0": _embedding_matrix(
            (row["crops"]["3.0"]["filepath"] for row in queries), crop_embeddings
        ),
    }
    dimensions = {
        matrix.shape[1] for matrix in [*candidate_matrices.values(), *query_matrices.values()]
    }
    if len(dimensions) != 1:
        raise ValueError(f"A0 and crop embedding dimensions differ: {sorted(dimensions)}")
    output = {arm: {} for arm in ARMS}
    for start in range(0, len(queries), query_batch_size):
        stop = min(len(queries), start + query_batch_size)
        score_blocks = {
            arm: query_matrices[arm][start:stop] @ candidate_matrices[arm].T
            for arm in ("whole_image_current", "bbox_context_1p5", "bbox_context_3p0")
        }
        score_blocks["bbox_multiscale_max"] = np.maximum(
            score_blocks["bbox_context_1p5"],
            score_blocks["bbox_context_3p0"],
        )
        for local_index, query in enumerate(queries[start:stop]):
            query_id = str(query["query_id"])
            for arm in ARMS:
                ranked = parent_topk(candidates, score_blocks[arm][local_index], top_k=top_k)
                output[arm][query_id] = [
                    {
                        "rank": rank,
                        "parent_id": str(row["parent_id"]),
                        "parent_filepath": str(row["parent_filepath"]),
                        "candidate_id": str(row["candidate_id"]),
                        "candidate_dataset": str(row["dataset"]),
                        "candidate_phenotype": str(row["canonical_phenotype"]),
                        "candidate_bbox_2d": list(row["bbox_2d"]),
                        "source_group_id": str(row["source_group_id"]),
                        "score": float(row["score"]),
                    }
                    for rank, row in enumerate(ranked, start=1)
                ]
    return output


def _metric_without_per_query(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "per_query"}


def _slice_metrics(
    queries: list[dict[str, Any]],
    neighbors: dict[str, list[dict[str, Any]]],
    *,
    relevant_parent_counts: dict[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    fields = (
        "dataset",
        "canonical_phenotype",
        "bbox_area_quartile",
        "padding_fraction_quartile",
    )
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for field in fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for query in queries:
            grouped[str(query[field])].append(query)
        output[field] = {}
        for value, group in sorted(grouped.items()):
            group_metrics = compute_arm_metrics(
                group,
                neighbors,
                relevant_parent_counts=relevant_parent_counts,
            )
            output[field][value] = {
                "queries": len(group),
                **_metric_without_per_query(group_metrics),
            }
    return output


def _stratified_human_query_ids(
    queries: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[str]:
    if len(queries) < count:
        raise ValueError(f"human-check kit requires {count} queries, found {len(queries)}")
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for query in queries:
        strata[
            (
                str(query["dataset"]),
                str(query["canonical_phenotype"]),
                str(query["bbox_area_quartile"]),
            )
        ].append(str(query["query_id"]))
    generator = np.random.default_rng(seed)
    for values in strata.values():
        generator.shuffle(values)
    keys = sorted(strata)
    generator.shuffle(keys)
    selected: list[str] = []
    position = 0
    while len(selected) < count:
        added = False
        for key in keys:
            values = strata[key]
            if position < len(values):
                selected.append(values[position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("stratified selection exhausted before requested count")
        position += 1
    return selected


def _panel_image(
    path: pathlib.Path,
    bbox: Iterable[int],
    *,
    size: tuple[int, int] = (224, 224),
) -> Image.Image:
    source = load_display_image(path)
    width, height = source.size
    x1, y1, x2, y2 = normalized_box_to_pixels(bbox, width=width, height=height)
    fitted = ImageOps.contain(source, size, method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, color=(32, 32, 32))
    offset_x = (size[0] - fitted.width) // 2
    offset_y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    scale = min(size[0] / width, size[1] / height)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [
            offset_x + round(x1 * scale),
            offset_y + round(y1 * scale),
            offset_x + round(x2 * scale),
            offset_y + round(y2 * scale),
        ],
        outline=(255, 48, 48),
        width=3,
    )
    return canvas


def render_human_pairs(
    output_dir: pathlib.Path,
    queries: list[dict[str, Any]],
    rankings: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    count: int = 200,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = _stratified_human_query_ids(queries, count=count, seed=seed)
    query_index = {str(query["query_id"]): query for query in queries}
    generator = np.random.default_rng(seed)
    shuffled_arms = list(ARMS)
    generator.shuffle(shuffled_arms)
    blind_labels = {
        f"Arm {chr(ord('A') + index)}": arm for index, arm in enumerate(shuffled_arms)
    }
    index_rows: list[dict[str, str]] = []
    for grid_index, query_id in enumerate(selected_ids, start=1):
        query = query_index[query_id]
        canvas = Image.new("RGB", (4 * 240, 5 * 260), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        query_panel = _panel_image(
            pathlib.Path(query["parent_filepath"]), query["bbox_2d"]
        )
        canvas.paste(query_panel, (8, 26))
        draw.text((8, 6), "QUERY", fill=(0, 0, 0))
        draw.text(
            (248, 35),
            f"dataset: {query['dataset']}\nphenotype: {query['canonical_phenotype']}\n"
            f"area: {query['bbox_area_quartile']}",
            fill=(0, 0, 0),
        )
        for arm_position, (blind_label, arm) in enumerate(blind_labels.items(), start=1):
            y = arm_position * 260
            draw.text((8, y + 6), blind_label, fill=(0, 0, 0))
            neighbors = rankings[arm][query_id][:3]
            if len(neighbors) != 3:
                raise ValueError(f"{arm}/{query_id}: fewer than three parent neighbors")
            for neighbor_index, neighbor in enumerate(neighbors):
                panel = _panel_image(
                    pathlib.Path(neighbor["parent_filepath"]),
                    neighbor["candidate_bbox_2d"],
                )
                x = (neighbor_index + 1) * 240 + 8
                canvas.paste(panel, (x, y + 26))
                draw.text(
                    (x, y + 6),
                    f"rank {neighbor_index + 1} | {neighbor['candidate_dataset']}",
                    fill=(0, 0, 0),
                )
        filename = f"pair_{grid_index:03d}.png"
        canvas.save(output_dir / filename, format="PNG")
        index_rows.append(
            {
                "grid": filename,
                "query_id": query_id,
                "target_dataset": str(query["dataset"]),
                "canonical_phenotype": str(query["canonical_phenotype"]),
                "bbox_area_quartile": str(query["bbox_area_quartile"]),
            }
        )
    index_path = output_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    key_path = output_dir / "blind_key.json"
    _atomic_json(
        key_path,
        {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "mapping": blind_labels,
            "note": "Keep this key separate while scoring the PNG grids.",
        },
    )
    return {
        "count": count,
        "seed": seed,
        "index": {"path": str(index_path), "sha256": _sha256(index_path)},
        "blind_key": {"path": str(key_path), "sha256": _sha256(key_path)},
        "grid_sha256": {
            row["grid"]: _sha256(output_dir / row["grid"]) for row in index_rows
        },
    }


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _render_results_report(
    report_path: pathlib.Path,
    *,
    preregistered_sha256: str,
    metrics: dict[str, Any],
) -> None:
    _verify_preregistered_report(report_path, preregistered_sha256)
    original = report_path.read_text(encoding="utf-8")
    marker = "## Results\n\n_Pending computation._\n\n## Decision\n\n_Pending computation._\n"
    if marker not in original:
        raise ValueError("pre-registered report lacks the untouched result placeholder")
    lines = [
        "## Results",
        "",
        f"Computed at `{metrics['computed_at']}` from Proxy queries and Mining candidates only. "
        "The frozen Benchmark was not read.",
        "",
        "| Query variant | Arm | P@1 | P@5 | P@10 | P@20 | MRR | nDCG@20 | Coverage | Unique relevant parents | Max source share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("reference", "zero_shot"):
        for arm in ARMS:
            arm_metrics = metrics["variants"][variant]["arms"][arm]
            macro = arm_metrics["dataset_balanced_macro"]
            lines.append(
                f"| {variant} | {arm} | {_format_metric(macro['p_at_1'])} | "
                f"{_format_metric(macro['p_at_5'])} | {_format_metric(macro['p_at_10'])} | "
                f"{_format_metric(macro['p_at_20'])} | {_format_metric(macro['reciprocal_rank'])} | "
                f"{_format_metric(macro['ndcg_at_20'])} | {_format_metric(macro['coverage'])} | "
                f"{arm_metrics['unique_relevant_parents']} | "
                f"{_format_metric(arm_metrics['max_source_share'])} |"
            )
    lines.extend(["", "### Reference-arm paired bootstrap and sealed gates", ""])
    lines.append("| Arm | P@10 delta vs A0 | 95% CI | Precision | Unique parents | Source share | Lineage | Overall |")
    lines.append("|---|---:|---:|---|---|---|---|---|")
    for arm in ARMS[1:]:
        bootstrap = metrics["variants"]["reference"]["bootstrap_vs_a0"][arm]
        gates = metrics["gates"][arm]
        lines.append(
            f"| {arm} | {_format_metric(bootstrap['delta'])} | "
            f"[{_format_metric(bootstrap['ci95_lower'])}, {_format_metric(bootstrap['ci95_upper'])}] | "
            f"{gates['primary_precision']} | {gates['unique_relevant_parents']} | "
            f"{gates['source_share']} | {gates['lineage']} | {gates['go']} |"
        )
    lineage = metrics["lineage"]
    lines.extend(
        [
            "",
            "### Integrity and review kit",
            "",
            f"- Lineage: `{lineage['lineage_traceable_rows']}/{lineage['lineage_rows']}` "
            f"(`{_format_metric(lineage['lineage_rate'])}`).",
            f"- Shared-image leaks: `{lineage['shared_image_leaks']}`; emitted crop training samples: "
            f"`{lineage['crop_emitted_count']}`.",
            f"- Human grids: `{metrics['human_pairs']['count']}` deterministic blinded query-to-top-3 grids.",
            f"- Stage B: `{metrics['stage_b']['status']}` — {metrics['stage_b']['reason']}",
            "",
            "Detailed source shares and all required dataset/class/area/padding slices are in `metrics.json`.",
            "",
            "## Decision",
            "",
            f"**{metrics['go_no_go']}**",
            "",
        ]
    )
    report_path.write_text(original.replace(marker, "\n".join(lines)), encoding="utf-8")


def _validate_crop_embedding_spec(
    path: pathlib.Path,
    *,
    prepare_manifest: dict[str, Any],
    crop_embeddings: pathlib.Path,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to verify the crop embedding spec") from exc
    path = path.expanduser().resolve(strict=True)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("crop embedding spec must contain a YAML mapping")
    if payload.get("model") != "SigLIP":
        raise ValueError("crop embedding spec must use model: SigLIP")
    source_encoder = prepare_manifest.get("encoder", {})
    model_id = source_encoder.get("model_id")
    snapshot = source_encoder.get("snapshot_revision")
    model_path = payload.get("model_path")
    if not all(isinstance(value, str) and value for value in (model_id, snapshot, model_path)):
        raise ValueError("source cache or crop spec lacks a pinned encoder identity")
    resolved_model_path = pathlib.Path(model_path).expanduser().resolve(strict=True)
    if not resolved_model_path.is_dir() or resolved_model_path.name != snapshot:
        raise ValueError(
            "crop embedding spec does not use the A0 encoder snapshot: "
            f"expected model_id={model_id!r} snapshot={snapshot!r}, observed={model_path!r}"
        )
    expected_input = pathlib.Path(
        prepare_manifest["artifacts"]["embedding_inputs"]["path"]
    ).resolve()
    expected_output = crop_embeddings.resolve()
    configured_input = pathlib.Path(str(payload.get("input_parquet", ""))).expanduser().resolve()
    configured_output = pathlib.Path(str(payload.get("output_parquet", ""))).expanduser().resolve()
    if configured_input != expected_input or configured_output != expected_output:
        raise ValueError(
            "crop embedding spec paths do not match the prepared input and scored output"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "model": payload["model"],
        "model_path": str(resolved_model_path),
        "batch_size": payload.get("batch_size", 64),
    }


def compute_audit(
    *,
    output_dir: pathlib.Path,
    crop_embeddings: pathlib.Path,
    crop_embedding_spec: pathlib.Path,
    job_id: str,
    preregistered_report_sha256: str,
    human_pair_count: int = 200,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=True)
    report_path = output_dir / "REPORT.md"
    _verify_preregistered_report(report_path, preregistered_report_sha256)
    prepare_manifest_path = output_dir / "prepare_manifest.json"
    prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    if prepare_manifest.get("state") != "PREPARED":
        raise ValueError("prepare manifest is not in PREPARED state")
    if prepare_manifest.get("preregistered_report_sha256") != preregistered_report_sha256:
        raise ValueError("prepare manifest and compute report seal differ")
    if prepare_manifest.get("benchmark_read") is not False:
        raise ValueError("prepare manifest does not attest benchmark_read=false")
    for artifact in prepare_manifest["artifacts"].values():
        path = pathlib.Path(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"prepared artifact changed before compute: {path}")
    candidate_path = pathlib.Path(prepare_manifest["artifacts"]["candidate_objects"]["path"])
    query_path = pathlib.Path(prepare_manifest["artifacts"]["query_objects"]["path"])
    candidates = list(_iter_jsonl(candidate_path))
    queries = list(_iter_jsonl(query_path))
    a0_path = pathlib.Path(prepare_manifest["inputs"]["a0_source_embeddings"]["path"])
    crop_embeddings = crop_embeddings.expanduser().resolve(strict=True)
    crop_spec = _validate_crop_embedding_spec(
        crop_embedding_spec,
        prepare_manifest=prepare_manifest,
        crop_embeddings=crop_embeddings,
    )
    candidate_parent_paths = {str(row["parent_filepath"]) for row in candidates}
    embedded_paths = {
        str(row["embedding_filepath"]) for row in queries
    } | {
        str(crop["filepath"])
        for row in [*candidates, *queries]
        for crop in row["crops"].values()
    }
    a0_embeddings = _load_required_embeddings(a0_path, candidate_parent_paths)
    crop_embedding_index = _load_required_embeddings(crop_embeddings, embedded_paths)
    rankings = rank_retrieval_arms(
        candidates,
        queries,
        a0_embeddings=a0_embeddings,
        crop_embeddings=crop_embedding_index,
    )
    relevant_parent_counts = _candidate_relevant_parent_counts(candidates)
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_read": False,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "variants": {},
        "lineage": prepare_manifest["lineage"],
        "candidate_relevant_parent_counts": relevant_parent_counts,
    }
    for variant in ("reference", "zero_shot"):
        variant_queries = [row for row in queries if row["query_variant"] == variant]
        arm_payload: dict[str, Any] = {}
        for arm in ARMS:
            arm_metrics = compute_arm_metrics(
                variant_queries,
                rankings[arm],
                relevant_parent_counts=relevant_parent_counts,
            )
            arm_payload[arm] = {
                **_metric_without_per_query(arm_metrics),
                "slices": _slice_metrics(
                    variant_queries,
                    rankings[arm],
                    relevant_parent_counts=relevant_parent_counts,
                ),
            }
            arm_payload[arm]["per_query"] = arm_metrics["per_query"]
        bootstrap = {
            arm: paired_bootstrap_p10(
                variant_queries,
                arm_payload[arm]["per_query"],
                arm_payload["whole_image_current"]["per_query"],
            )
            for arm in ARMS[1:]
        }
        metrics["variants"][variant] = {
            "queries": len(variant_queries),
            "arms": arm_payload,
            "bootstrap_vs_a0": bootstrap,
        }
    reference = metrics["variants"]["reference"]
    gates = {
        arm: evaluate_gates(
            reference["arms"]["whole_image_current"],
            reference["arms"][arm],
            reference["bootstrap_vs_a0"][arm],
            metrics["lineage"],
        )
        for arm in ARMS[1:]
    }
    passing = [arm for arm in ARMS[1:] if gates[arm]["go"]]
    winner = max(
        passing,
        key=lambda arm: (
            reference["arms"][arm]["dataset_balanced_macro"]["p_at_10"],
            reference["arms"][arm]["unique_relevant_parents"],
            -ARMS.index(arm),
        ),
        default=None,
    )
    metrics["gates"] = gates
    metrics["winning_arm"] = winner
    metrics["stage_b"] = {
        "status": "eligible_not_run" if winner else "not_eligible",
        "reason": (
            f"Stage-A winner {winner} must be evaluated with DINOv3/C-RADIO in a separately authorized audit."
            if winner
            else "No SigLIP bbox arm passed every sealed Stage-A gate."
        ),
    }
    metrics["go_no_go"] = (
        f"GO: {winner} passed every sealed Stage-A gate; Stage B is required before adoption."
        if winner
        else "NO-GO: no bbox arm passed every sealed Stage-A gate; do not launch SFT."
    )
    human_queries = [row for row in queries if row["query_variant"] == "reference"]
    metrics["human_pairs"] = render_human_pairs(
        output_dir / "human_pairs",
        human_queries,
        rankings,
        count=human_pair_count,
    )
    rankings_path = output_dir / "rankings.jsonl"
    _write_jsonl(
        rankings_path,
        (
            {
                "query_id": query["query_id"],
                "query_variant": query["query_variant"],
                "arm": arm,
                "neighbors": rankings[arm][str(query["query_id"])],
                "emitted_parents": [
                    {
                        "filepath": row["parent_filepath"],
                        "atomic_sample_id": row["parent_id"],
                        "candidate_id": row["candidate_id"],
                        "routed_task_types": ["Defect Detection"],
                    }
                    for row in rankings[arm][str(query["query_id"])]
                ],
            }
            for query in queries
            for arm in ARMS
        ),
    )
    metrics_path = output_dir / "metrics.json"
    _atomic_json(metrics_path, metrics)
    _render_results_report(
        report_path,
        preregistered_sha256=preregistered_report_sha256,
        metrics=metrics,
    )
    artifact_paths = [
        report_path,
        metrics_path,
        rankings_path,
        candidate_path,
        query_path,
        pathlib.Path(prepare_manifest["artifacts"]["embedding_inputs"]["path"]),
        *sorted((output_dir / "human_pairs").glob("*")),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "COMPLETE",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_read": False,
        "go_no_go": metrics["go_no_go"],
        "job_id": job_id,
        "preregistered_report_sha256": preregistered_report_sha256,
        "encoder": prepare_manifest["encoder"],
        "crop_policy_version": CROP_POLICY_VERSION,
        "a0": {
            "cache_reused": prepare_manifest["inputs"]["a0_source_embeddings"],
            "candidate_parent_rows_reembedded": 0,
        },
        "crop_embeddings": {
            "path": str(crop_embeddings),
            "sha256": _sha256(crop_embeddings),
        },
        "crop_embedding_spec": crop_spec,
        "inputs": prepare_manifest["inputs"],
        "artifacts": {
            str(path.relative_to(output_dir)): {"path": str(path), "sha256": _sha256(path)}
            for path in artifact_paths
            if path.is_file()
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=SCHEMA_VERSION,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="build proxy FN queries, Mining object candidates, and crop embedding input",
    )
    prepare.add_argument("--proxy", type=pathlib.Path, required=True)
    prepare.add_argument("--reference-predictions", type=pathlib.Path, required=True)
    prepare.add_argument("--zero-shot-predictions", type=pathlib.Path, required=True)
    prepare.add_argument("--mining", type=pathlib.Path, required=True)
    prepare.add_argument("--evaluator", type=pathlib.Path, required=True)
    prepare.add_argument("--media-root", type=pathlib.Path, required=True)
    prepare.add_argument("--a0-source-embeddings", type=pathlib.Path, required=True)
    prepare.add_argument("--encoder-manifest", type=pathlib.Path, required=True)
    prepare.add_argument("--split-attestation", type=pathlib.Path, required=True)
    prepare.add_argument("--output-dir", type=pathlib.Path, required=True)
    prepare.add_argument("--preregistered-report-sha256", required=True)
    prepare.add_argument("--crop-workers", type=int, default=8)
    compute = subparsers.add_parser(
        "compute",
        help="rank the four arms and write metrics, report, manifest, and review grids",
    )
    compute.add_argument("--output-dir", type=pathlib.Path, required=True)
    compute.add_argument("--crop-embeddings", type=pathlib.Path, required=True)
    compute.add_argument("--crop-embedding-spec", type=pathlib.Path, required=True)
    compute.add_argument("--job-id", required=True)
    compute.add_argument("--preregistered-report-sha256", required=True)
    compute.add_argument("--human-pair-count", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_audit(
            proxy=args.proxy,
            reference_predictions=args.reference_predictions,
            zero_shot_predictions=args.zero_shot_predictions,
            mining=args.mining,
            evaluator_path=args.evaluator,
            media_root=args.media_root,
            a0_source_embeddings=args.a0_source_embeddings,
            encoder_manifest=args.encoder_manifest,
            split_attestation=args.split_attestation,
            output_dir=args.output_dir,
            preregistered_report_sha256=args.preregistered_report_sha256,
            crop_workers=args.crop_workers,
        )
        print(
            f"BBOX_AUDIT_PREPARED candidates={payload['candidates']} "
            f"reference_queries={payload['queries']['reference']['queries']} "
            f"embedding_inputs={payload['embedding_inputs']}"
        )
        return 0
    manifest = compute_audit(
        output_dir=args.output_dir,
        crop_embeddings=args.crop_embeddings,
        crop_embedding_spec=args.crop_embedding_spec,
        job_id=args.job_id,
        preregistered_report_sha256=args.preregistered_report_sha256,
        human_pair_count=args.human_pair_count,
    )
    print(manifest["go_no_go"])
    print("BBOX_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
