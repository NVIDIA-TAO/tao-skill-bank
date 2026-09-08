#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build Proxy RCCA gap candidates with the recorded exact evaluator logic.

This script does not calculate an application KPI. It dynamically loads the
workspace evaluator recorded in state and reuses its parsers/matching helpers
to assign record-level correctness for mining selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import statistics
import sys
from typing import Any

import yaml

from atomic_samples import identity_for_paths
from gap_analysis.config import load_profile, validate_config
from gap_analysis.runner import run_selection
from validate_sharegpt import (
    image_paths,
    load_records,
    prompt_and_response,
    resolve_image,
    target_path,
)


def _classification_phenotypes(source: dict[str, Any]) -> list[str]:
    prompt, response = prompt_and_response(source, context=str(source.get("id")))
    mapping = {
        match.group(1): match.group(2).strip().rstrip(".")
        for line in prompt.splitlines()
        for match in [re.match(r"^\s*([A-Z])[.)]\s*(.+?)\s*$", line)]
        if match is not None
    }
    values = [part for part in re.split(r"""[\s,\[\]'"]+""", response) if part]
    return sorted({mapping.get(value, value) for value in values}) or ["__empty__"]


def _detection_strata(
    source: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], float | None]:
    _, response = prompt_and_response(source, context=str(source.get("id")))
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        objects = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source.get('id')}: detection ground truth is not JSON") from exc
    if not isinstance(objects, list):
        raise ValueError(f"{source.get('id')}: detection ground truth is not an array")
    labels = sorted(
        {
            str(item.get("label"))
            for item in objects
            if isinstance(item, dict) and isinstance(item.get("label"), str)
        }
    ) or ["__empty__"]
    areas = []
    for item in objects:
        box = item.get("bbox_2d") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{source.get('id')}: invalid detection bbox")
        areas.append(
            max(0.0, float(box[2]) - float(box[0]))
            * max(0.0, float(box[3]) - float(box[1]))
        )
    area = math.log(max(1e-12, statistics.median(areas))) if areas else None
    return labels, objects, area


def _proxy_local_contrast(
    source: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    media_root: pathlib.Path | None,
) -> float | None:
    supplied = source.get("local_contrast")
    if isinstance(supplied, (int, float)) and not isinstance(supplied, bool):
        return float(supplied)
    if not objects or media_root is None:
        return None
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:
        raise ValueError("Pillow is required to build Proxy local-contrast strata") from exc
    context = str(source.get("id"))
    path = resolve_image(target_path(source, context=context), media_root)
    if not path.is_file():
        raise ValueError(f"{context}: Proxy image is missing: {path}")
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
            inside = gray.crop(box)
            outside = gray.crop(outer)
            inside_sum = ImageStat.Stat(inside).sum[0]
            outside_sum = ImageStat.Stat(outside).sum[0]
            inside_pixels = inside.width * inside.height
            ring_pixels = outside.width * outside.height - inside_pixels
            inside_mean = inside_sum / max(1, inside_pixels)
            ring_mean = (
                (outside_sum - inside_sum) / ring_pixels
                if ring_pixels > 0
                else inside_mean
            )
            contrasts.append(abs(inside_mean - ring_mean) / 255.0)
    if not contrasts:
        raise ValueError(f"{context}: no measurable Proxy detection boxes")
    return statistics.mean(contrasts)


def _rank_proxy_quartiles(
    rows: list[dict[str, Any]], numeric_field: str, quartile_field: str
) -> None:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get(numeric_field) is not None:
            by_task.setdefault(str(row["task_type"]), []).append(row)
    for entries in by_task.values():
        ordered = sorted(
            entries, key=lambda row: (float(row[numeric_field]), str(row["id"]))
        )
        for index, row in enumerate(ordered):
            row[quartile_field] = f"Q{min(4, index * 4 // len(ordered) + 1)}"


def _load_evaluator(path: pathlib.Path) -> Any:
    resolved = path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location("deft_recorded_exact_f1", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load exact evaluator: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "build_classification_examples",
        "build_detection_examples",
        "parse_choice_labels",
        "parse_direct_bcq",
        "parse_boxes",
        "canonicalize_prediction_boxes",
        "box_iou",
        "one_to_one_detection_counts",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"exact evaluator lacks required helpers: {missing}")
    return module


def _score_row(
    evaluator: Any,
    source: dict[str, Any],
    prediction: dict[str, Any],
) -> tuple[float, bool, str, dict[str, Any] | None]:
    row_id = str(source["id"])
    raw = str(prediction.get("raw_prediction", ""))
    classification = evaluator.build_classification_examples({row_id: source})
    if classification:
        example = classification[0]
        if example["answer_format"] == "direct_bcq":
            labels, parse_ok = evaluator.parse_direct_bcq(raw)
        else:
            labels, parse_ok = evaluator.parse_choice_labels(
                raw, example["option_letters"], example["option_text"]
            )
        correct = bool(parse_ok and labels == example["gt_labels"])
        return float(correct), bool(parse_ok), "classification_mismatch", None
    detection = evaluator.build_detection_examples({row_id: source}, 1.0)
    if not detection:
        raise ValueError(f"exact evaluator does not classify source row {row_id!r}")
    example = detection[0]
    native, parse_ok = evaluator.parse_boxes(raw)
    predicted = evaluator.canonicalize_prediction_boxes(native, "xyxy") if parse_ok else []
    if parse_ok:
        tp, fp, fn = evaluator.one_to_one_detection_counts(
            example["gt_boxes"], predicted, 0.5
        )
    else:
        tp, fp, fn = 0, 0, len(example["gt_boxes"])
    correct = bool(parse_ok and fp == 0 and fn == 0)
    evidence = _detection_evidence(
        evaluator,
        example["gt_boxes"],
        predicted,
        parse_ok=bool(parse_ok),
        threshold=0.5,
    )
    return float(correct), bool(parse_ok), "detection_mismatch", evidence


def _detection_evidence(
    evaluator: Any,
    gt_boxes: list[tuple[float, ...]],
    predicted_boxes: list[tuple[float, ...]],
    *,
    parse_ok: bool,
    threshold: float,
) -> dict[str, Any]:
    """Classify proxy box errors without changing the packaged KPI evaluator."""

    if parse_ok and callable(getattr(evaluator, "one_to_one_detection_counts", None)):
        tp, fp, fn = evaluator.one_to_one_detection_counts(
            gt_boxes, predicted_boxes, threshold
        )
    elif parse_ok:
        matched = sum(
            max(
                (evaluator.box_iou(gt_box, predicted) for predicted in predicted_boxes),
                default=0.0,
            )
            > threshold
            for gt_box in gt_boxes
        )
        tp = min(matched, len(predicted_boxes))
        fp = len(predicted_boxes) - tp
        fn = len(gt_boxes) - tp
    else:
        tp, fp, fn = 0, 0, len(gt_boxes)
    best_overlaps = [
        max(
            (evaluator.box_iou(gt_box, predicted) for predicted in predicted_boxes),
            default=0.0,
        )
        for gt_box in gt_boxes
    ]
    partial = sum(0.0 < value <= threshold for value in best_overlaps)
    evidence_types: list[str] = []
    if fp:
        evidence_types.append("hard_negative_proxy_false_positive")
    if partial:
        evidence_types.append("hard_positive_best_overlap_0_lt_iou_lte_0p5")
    if fn:
        evidence_types.append("hard_positive_proxy_false_negative")
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "best_overlap_0_lt_iou_lte_0p5_count": partial,
        "best_iou_by_ground_truth_box": best_overlaps,
        "evidence_types": evidence_types,
    }


def build_candidates(
    evaluator_path: pathlib.Path,
    source_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    media_root: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    evaluator = _load_evaluator(evaluator_path)
    predictions = {str(row.get("id", "")): row for row in prediction_rows}
    if "" in predictions or len(predictions) != len(prediction_rows):
        raise ValueError("prediction JSONL has missing or duplicate IDs")
    source_ids = {str(row.get("id", "")) for row in source_rows}
    if "" in source_ids or len(source_ids) != len(source_rows):
        raise ValueError("Proxy JSONL has missing or duplicate IDs")
    missing = sorted(source_ids - predictions.keys())
    unknown = sorted(predictions.keys() - source_ids)
    if missing or unknown:
        raise ValueError(f"prediction coverage mismatch: missing={missing[:10]}, unknown={unknown[:10]}")
    candidates: list[dict[str, Any]] = []
    for source in source_rows:
        row_id = str(source["id"])
        score, parse_ok, gap_type, detection_evidence = _score_row(
            evaluator, source, predictions[row_id]
        )
        paths = image_paths(source, context=row_id)
        target = target_path(source, context=row_id)
        sample_kind = "reference_pair" if len(paths) == 2 else "single_image"
        is_detection = "Detection" in str(source["task_type"])
        if is_detection:
            phenotypes, objects, log_bbox_area = _detection_strata(source)
            gt_count = len(objects)
            local_contrast = _proxy_local_contrast(
                source, objects, media_root=media_root
            )
        else:
            phenotypes = _classification_phenotypes(source)
            gt_count = 0 if phenotypes == ["__empty__"] else 1
            log_bbox_area = None
            local_contrast = None
        candidate = {
                "id": row_id,
                "evaluation_role": "proxy",
                "task_type": str(source["task_type"]),
                "metric_family": (
                    "detection" if "detection" in str(source["task_type"]).casefold()
                    else "classification"
                ),
                "reference_cohort": (
                    "reference_based" if len(paths) == 2 else "non_reference_based"
                ),
                "dataset": str(source.get("dataset", "unknown")),
                "canonical_phenotypes": phenotypes,
                "gt_count_bin": (
                    "0" if gt_count == 0 else "1" if gt_count == 1 else "2-3" if gt_count <= 3 else "4+"
                ),
                "log_bbox_area": log_bbox_area,
                "log_bbox_area_quartile": "NA",
                "local_contrast": local_contrast,
                "local_contrast_quartile": "NA",
                # Several task rows may address the same physical target.
                # Default to the target path so routing embeds that image once.
                "target_id": str(source.get("target_id", target)),
                "target_path": target,
                "reference_path": paths[0] if len(paths) == 2 else None,
                "image_paths": paths,
                "sample_kind": sample_kind,
                "atomic_sample_id": identity_for_paths(sample_kind, paths),
                "sample_score": score,
                "parse_ok": parse_ok,
                "gap_type": gap_type,
                "raw_prediction": predictions[row_id].get("raw_prediction"),
                # Keep a stable parquet schema even when the first Proxy row is
                # classification; only exact Defect Detection rows carry
                # nonzero evidence.
                "defect_detection_evidence": detection_evidence
                or {
                    "true_positive_count": 0,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                    "best_overlap_0_lt_iou_lte_0p5_count": 0,
                    "best_iou_by_ground_truth_box": [],
                    "evidence_types": [],
                },
            }
        candidates.append(candidate)
    _rank_proxy_quartiles(candidates, "log_bbox_area", "log_bbox_area_quartile")
    _rank_proxy_quartiles(candidates, "local_contrast", "local_contrast_quartile")
    return candidates


def _write_parquet(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for RCCA parquet artifacts") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--predictions", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--gap-analysis-profile", default="deficit_weighted_round_robin")
    choice.add_argument("--gap-analysis-config", type=pathlib.Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        if args.gap_analysis_config:
            config = validate_config(yaml.safe_load(args.gap_analysis_config.read_text()))
        else:
            config = load_profile(args.gap_analysis_profile)
        if args.budget is not None:
            config["budget"] = args.budget
        if args.seed is not None:
            config["seed"] = args.seed
        config = validate_config(config)
        candidates = build_candidates(
            args.evaluator,
            load_records(args.source),
            load_records(args.predictions),
            media_root=(
                args.media_root.expanduser().resolve()
                if args.media_root is not None
                else None
            ),
        )
        selected, selection = run_selection(candidates, config)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet(args.output_dir / "gap_candidates.parquet", candidates)
        _write_parquet(args.output_dir / "selected_gaps.parquet", selected)
        summary = {
            "schema_version": 1,
            "evaluation_role": "proxy",
            "evaluator": str(args.evaluator.expanduser().resolve()),
            "samples": len(candidates),
            "incorrect_samples": sum(row["sample_score"] < 1.0 for row in candidates),
            "parse_failures": sum(not row["parse_ok"] for row in candidates),
            "selection": selection,
        }
        (args.output_dir / "gaps_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"analyze_gaps: {exc}", file=sys.stderr)
        return 2
    print(
        f"analyze_gaps: samples={summary['samples']} "
        f"incorrect={summary['incorrect_samples']} selected={len(selected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
