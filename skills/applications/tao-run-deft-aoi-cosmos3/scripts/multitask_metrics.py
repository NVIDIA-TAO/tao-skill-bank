#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-aware classification and detection metrics for NVPaw annotations."""

from __future__ import annotations

import ast
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from nvpaw_annotations import DIRECT_CLASS_LABELS, TASK_SPECS, parse_detection_answer


def _field(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.I)
    return match.group(1).strip() if match else cleaned


def _structured(text: str) -> tuple[bool, Any]:
    cleaned = _strip_fence(text)
    for parser in (json.loads, ast.literal_eval):
        try:
            return True, parser(cleaned)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return False, None


def _flatten_choices(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        bracketed = re.fullmatch(
            r"\[\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\]", cleaned, re.I
        )
        if bracketed:
            return [part.strip().upper() for part in bracketed.group(1).split(",")]
        if re.fullmatch(r"[A-Z](?:\s*,\s*[A-Z])+", cleaned, re.I):
            return [part.strip().upper() for part in cleaned.split(",")]
        return [cleaned]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_choices(item))
        return result
    if isinstance(value, dict):
        for key in ("labels", "choices", "answer", "label", "choice"):
            if key in value:
                return _flatten_choices(value[key])
    return []


def parse_classification_prediction(
    text: Any, *, option_map: dict[str, str]
) -> tuple[set[str], bool]:
    if not isinstance(text, str):
        return set(), False
    cleaned = _strip_fence(text).strip()
    if not cleaned:
        return set(), False
    if cleaned in DIRECT_CLASS_LABELS:
        return {DIRECT_CLASS_LABELS[cleaned]}, True
    leading = re.match(r"^(yes|no)\b", cleaned, flags=re.I)
    if leading and not option_map:
        return {"defect" if leading.group(1).casefold() == "yes" else "no_defect"}, True
    parsed, value = _structured(cleaned)
    if not parsed:
        value = cleaned
    tokens = _flatten_choices(value)
    if parsed and isinstance(value, list) and not value:
        return set(), True
    semantic_by_text = {
        re.sub(r"\s+", " ", value).strip().casefold(): value
        for value in option_map.values()
    }
    labels: set[str] = set()
    for token in tokens:
        normalized_token = token.strip()
        letter = normalized_token.upper()
        if letter in option_map:
            labels.add(option_map[letter])
            continue
        marked = re.fullmatch(
            r"(?:answer|option|choice)?\s*[:=-]?\s*([A-Z])(?:\s*[.)].*)?",
            normalized_token,
            re.I,
        )
        if marked and marked.group(1).upper() in option_map:
            labels.add(option_map[marked.group(1).upper()])
            continue
        semantic_key = re.sub(r"\s+", " ", normalized_token).strip().casefold()
        if semantic_key in semantic_by_text:
            labels.add(semantic_by_text[semantic_key])
            continue
        return set(), False
    if tokens or (parsed and isinstance(value, list)):
        return labels, True
    return set(), False


Box = tuple[int, int, int, int]


def box_iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union else 0.0


def minimum_cost_assignment(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Dependency-free O(n^3) Hungarian assignment for rectangular matrices."""
    if not costs or not costs[0]:
        return []
    rows = len(costs)
    columns = len(costs[0])
    if any(len(row) != columns for row in costs):
        raise ValueError("assignment cost matrix must be rectangular")
    transposed = rows > columns
    matrix = [list(row) for row in zip(*costs)] if transposed else [list(row) for row in costs]
    rows, columns = len(matrix), len(matrix[0])
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        matched_row[0] = row_index
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        active = 0
        while True:
            used[active] = True
            active_row = matched_row[active]
            delta = math.inf
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[active_row - 1][column - 1]
                    - row_potential[active_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = active
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            active = next_column
            if matched_row[active] == 0:
                break
        while True:
            previous = predecessor[active]
            matched_row[active] = matched_row[previous]
            active = previous
            if active == 0:
                break
    pairs: list[tuple[int, int]] = []
    for column in range(1, columns + 1):
        row = matched_row[column]
        if row:
            pairs.append((column - 1, row - 1) if transposed else (row - 1, column - 1))
    return sorted(pairs)


def detection_counts(
    ground_truth: list[dict[str, Any]],
    prediction: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    if not ground_truth:
        return 0, len(prediction), 0
    if not prediction:
        return 0, 0, len(ground_truth)
    ious = [
        [
            box_iou(tuple(gt["bbox_2d"]), tuple(pred["bbox_2d"]))
            if gt["label"].strip().casefold() == pred["label"].strip().casefold()
            else 0.0
            for pred in prediction
        ]
        for gt in ground_truth
    ]
    bonus = min(len(ground_truth), len(prediction)) + 1.0
    rewards = [
        [iou + (bonus if iou > iou_threshold else 0.0) for iou in row]
        for row in ious
    ]
    pairs = minimum_cost_assignment([[-value for value in row] for row in rewards])
    tp = sum(ious[gt_index][pred_index] > iou_threshold for gt_index, pred_index in pairs)
    return tp, len(prediction) - tp, len(ground_truth) - tp


def _f1(tp: int, fp: int, fn: int, *, both_empty: bool = False) -> float:
    if both_empty:
        return 1.0
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _classification_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(len(set(row["ground_truth_labels"]) & set(row["prediction_labels"])) for row in rows)
    fp = sum(len(set(row["prediction_labels"]) - set(row["ground_truth_labels"])) for row in rows)
    fn = sum(len(set(row["ground_truth_labels"]) - set(row["prediction_labels"])) for row in rows)
    label_universe = sorted(
        set().union(
            *(set(row["ground_truth_labels"]) | set(row["prediction_labels"]) for row in rows)
        )
    )
    per_label: dict[str, float] = {}
    for label in label_universe:
        label_tp = sum(label in row["ground_truth_labels"] and label in row["prediction_labels"] for row in rows)
        label_fp = sum(label not in row["ground_truth_labels"] and label in row["prediction_labels"] for row in rows)
        label_fn = sum(label in row["ground_truth_labels"] and label not in row["prediction_labels"] for row in rows)
        per_label[label] = _f1(label_tp, label_fp, label_fn)
    macro = (
        sum(per_label.values()) / len(per_label)
        if per_label
        else sum(row["sample_score"] for row in rows) / len(rows)
    )
    return {
        "support": len(rows),
        "metric_family": "classification",
        "primary_metric": "macro_f1",
        "primary_value": macro,
        "micro_f1": _f1(tp, fp, fn, both_empty=(tp == fp == fn == 0)),
        "macro_f1": macro,
        "exact_match_accuracy": sum(row["exact_match"] for row in rows) / len(rows),
        "per_label_f1": per_label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _detection_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    value = _f1(tp, fp, fn, both_empty=(tp == fp == fn == 0))
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    return {
        "support": len(rows),
        "metric_family": "detection",
        "primary_metric": "box_micro_f1",
        "primary_value": value,
        "box_micro_f1": value,
        "box_micro_precision": precision,
        "box_micro_recall": recall,
        "record_macro_f1": sum(row["sample_score"] for row in rows) / len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _target_path(annotation: dict[str, Any]) -> str:
    roles = annotation["image_roles"]
    return annotation["images"][roles.index("target")]


def evaluate(
    samples: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    evaluation_role: str = "benchmark",
    kpi_threshold: float = 1.0,
    iou_threshold: float = 0.5,
    kpi_profile: str = "task_balanced_v1",
    min_group_support: int = 1,
    required_tasks: set[str] | None = None,
) -> dict[str, Any]:
    if evaluation_role not in {"proxy", "benchmark"}:
        raise ValueError("evaluation_role must be proxy or benchmark")
    if not 0.0 <= kpi_threshold <= 1.0:
        raise ValueError("kpi_threshold must be in [0, 1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if kpi_profile not in {"task_balanced_v1", "task_dataset_balanced_v1"}:
        raise ValueError(f"unsupported rich KPI profile {kpi_profile!r}")
    if type(min_group_support) is not int or min_group_support < 1:
        raise ValueError("min_group_support must be a positive integer")

    annotation_index: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        record_id = annotation.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("every rich annotation requires a non-empty id")
        if record_id in annotation_index:
            raise ValueError(f"duplicate annotation id {record_id!r}")
        annotation_index[record_id] = annotation

    received: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    unknown_count = 0
    for sample in samples:
        sample_id = _field(sample, ("video_id", "sample_id", "id", "image_id"))
        sample_id = str(sample_id) if sample_id is not None else ""
        if sample_id not in annotation_index:
            unknown_count += 1
            continue
        if sample_id in received:
            duplicate_count += 1
            continue
        received[sample_id] = sample

    parse_failures = 0
    rows: list[dict[str, Any]] = []
    for record_id, annotation in annotation_index.items():
        sample = received.get(record_id)
        response = (
            _field(sample, ("response", "prediction", "pred", "model_response", "output"))
            if sample is not None
            else None
        )
        answer = annotation.get("answer")
        if not isinstance(answer, dict):
            raise ValueError(f"annotation {record_id!r} has no canonical answer")
        metric_family = annotation.get("metric_family")
        parse_ok = sample is not None
        tp = fp = fn = 0
        exact_match = False
        ground_truth_labels: list[str] = []
        prediction_labels: list[str] = []
        prediction_value: Any = []
        if metric_family == "classification":
            ground_truth_labels = list(answer.get("labels", []))
            predicted, parsed = parse_classification_prediction(
                response, option_map=annotation.get("option_map", {})
            )
            parse_ok = parse_ok and parsed
            prediction_labels = sorted(predicted) if parsed else []
            gt_set = set(ground_truth_labels)
            pred_set = set(prediction_labels)
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            exact_match = parse_ok and gt_set == pred_set
            score = _f1(tp, fp, fn, both_empty=parse_ok and not gt_set and not pred_set)
            prediction_value = {"kind": "choice_set", "labels": prediction_labels}
        elif metric_family == "detection":
            ground_truth_objects = answer.get("objects", [])
            try:
                parsed_answer = parse_detection_answer(
                    response if isinstance(response, str) else "",
                    record_id=record_id,
                )
                prediction_objects = parsed_answer["objects"]
                parsed = True
            except ValueError:
                prediction_objects = []
                parsed = False
            parse_ok = parse_ok and parsed
            tp, fp, fn = detection_counts(
                ground_truth_objects,
                prediction_objects,
                iou_threshold=iou_threshold,
            )
            exact_match = parse_ok and fp == 0 and fn == 0
            score = _f1(
                tp,
                fp,
                fn,
                both_empty=parse_ok and not ground_truth_objects and not prediction_objects,
            )
            prediction_value = {"kind": "detections", "objects": prediction_objects}
        else:
            raise ValueError(f"annotation {record_id!r} has unsupported metric_family")
        if sample is None:
            gap_type = "missing_prediction"
            score = 0.0
        elif not parse_ok:
            gap_type = "parse_failure"
            score = 0.0
            parse_failures += 1
        elif score == 1.0:
            gap_type = "correct"
        else:
            gap_type = f"{metric_family}_error"
        row: dict[str, Any] = {
            "id": record_id,
            "source_id": annotation.get("source_id", record_id),
            "target_id": annotation.get("target_id", _target_path(annotation)),
            "target_path": _target_path(annotation),
            "image_paths": list(annotation.get("images", [])),
            "evaluation_role": evaluation_role,
            "task_type": annotation.get("task_type"),
            "metric_family": metric_family,
            "reference_cohort": annotation.get("reference_cohort"),
            "dataset": annotation.get("dataset", "unknown"),
            "sample_score": score,
            "weakness_score": 1.0 - score,
            "gap_type": gap_type,
            "parse_ok": parse_ok,
            "exact_match": exact_match,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ground_truth_labels": ground_truth_labels,
            "prediction_labels": prediction_labels,
            "prediction_json": json.dumps(prediction_value, sort_keys=True),
            "ground_truth_json": json.dumps(answer, sort_keys=True),
            "metadata_json": json.dumps(
                {
                    "prompt_format": annotation.get("prompt_format"),
                    "prompt_variant": annotation.get("prompt_variant"),
                },
                sort_keys=True,
            ),
        }
        confidence = _field(sample or {}, ("confidence", "score", "probability"))
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            if math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0:
                row["uncertainty"] = 1.0 - float(confidence)
        rows.append(row)

    coverage = {
        "expected_predictions": len(annotations),
        "received_prediction_rows": len(samples),
        "missing_predictions": len(annotation_index) - len(received),
        "duplicate_prediction_ids": duplicate_count,
        "unknown_prediction_ids": unknown_count,
        "parse_failures": parse_failures,
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    task_metrics: dict[str, dict[str, Any]] = {}
    for task_type, task_rows in sorted(grouped.items()):
        families = {str(row["metric_family"]) for row in task_rows}
        if len(families) != 1:
            raise ValueError(f"task {task_type!r} mixes metric families")
        family = next(iter(families))
        task_metrics[task_type] = (
            _classification_group(task_rows)
            if family == "classification"
            else _detection_group(task_rows)
        )

    expected_tasks = set(TASK_SPECS) if required_tasks is None else set(required_tasks)
    unknown_tasks = set(task_metrics) - set(TASK_SPECS)
    missing_tasks = expected_tasks - set(task_metrics)
    if unknown_tasks:
        raise ValueError(f"unexpected task groups: {sorted(unknown_tasks)}")
    if missing_tasks:
        raise ValueError(f"missing required task groups: {sorted(missing_tasks)}")

    dataset_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_groups[(str(row["task_type"]), str(row["dataset"]))].append(row)
    dataset_metrics: dict[str, dict[str, Any]] = {}
    for (task_type, dataset), group_rows in sorted(dataset_groups.items()):
        family = str(group_rows[0]["metric_family"])
        metrics = (
            _classification_group(group_rows)
            if family == "classification"
            else _detection_group(group_rows)
        )
        metrics.update(
            {
                "task_type": task_type,
                "dataset": dataset,
                "sample_macro_f1": sum(row["sample_score"] for row in group_rows)
                / len(group_rows),
            }
        )
        dataset_metrics[f"{task_type}|{dataset}"] = metrics

    cohort_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cohort_rows[str(row["reference_cohort"])].append(row)
        family_rows[str(row["metric_family"])].append(row)
    cohort_metrics = {
        cohort: {
            "support": len(group_rows),
            "sample_macro_f1": sum(row["sample_score"] for row in group_rows)
            / len(group_rows),
        }
        for cohort, group_rows in sorted(cohort_rows.items())
    }
    family_metrics = {
        family: (
            _classification_group(group_rows)
            if family == "classification"
            else _detection_group(group_rows)
        )
        for family, group_rows in sorted(family_rows.items())
    }

    attainments: dict[str, float] = {}
    if kpi_profile == "task_balanced_v1":
        gate_groups = {
            task_type: {
                "value": float(metrics["primary_value"]),
                "support": int(metrics["support"]),
            }
            for task_type, metrics in task_metrics.items()
        }
    else:
        gate_groups = {
            name: {
                "value": float(metrics["primary_value"]),
                "support": int(metrics["support"]),
            }
            for name, metrics in dataset_metrics.items()
        }
    required_groups = sorted(gate_groups)
    insufficient_support = sum(
        group["support"] < min_group_support for group in gate_groups.values()
    )
    for group_name, group in gate_groups.items():
        value = float(group["value"])
        attainment = 1.0 if kpi_threshold == 0.0 else min(value / kpi_threshold, 1.0)
        group["target"] = kpi_threshold
        group["attainment"] = attainment
        group["met"] = value >= kpi_threshold and group["support"] >= min_group_support
        attainments[group_name] = attainment
    for task_type, metrics in task_metrics.items():
        value = float(metrics["primary_value"])
        metrics["target"] = kpi_threshold
        metrics["attainment"] = (
            1.0 if kpi_threshold == 0.0 else min(value / kpi_threshold, 1.0)
        )
        metrics["met"] = value >= kpi_threshold
    balanced_score = min(attainments.values()) if attainments else 0.0
    macro_attainment = sum(attainments.values()) / len(attainments) if attainments else 0.0
    spread = max(attainments.values()) - min(attainments.values()) if attainments else 0.0
    constraint_values = {
        name: coverage[name]
        for name in (
            "missing_predictions",
            "duplicate_prediction_ids",
            "unknown_prediction_ids",
            "parse_failures",
        )
    }
    if kpi_profile == "task_dataset_balanced_v1":
        constraint_values["insufficient_support_groups"] = insufficient_support
    constraints_met = all(value == 0 for value in constraint_values.values())
    gate_eligible = evaluation_role == "benchmark"
    kpi_met = balanced_score >= 1.0 and constraints_met if gate_eligible else None
    metric_result = {
        "name": "balanced_score",
        "value": balanced_score,
        "unit": "",
        "kpi_profile": kpi_profile,
        "group_metric_target": kpi_threshold,
        "min_group_support": min_group_support,
        "required_groups": required_groups,
        "constraints": constraint_values,
        "tie_breakers": {
            "macro_attainment": macro_attainment,
            "attainment_spread": spread,
            "coverage_failures": sum(constraint_values.values()),
        },
        "task_metrics": task_metrics,
        "gate_groups": gate_groups,
        "dataset_metrics": dataset_metrics,
        "reference_cohort_metrics": cohort_metrics,
        "metric_family_metrics": family_metrics,
    }
    summary = {
        "annotation_profile": "nvpaw_multitask_v1",
        "evaluation_role": evaluation_role,
        "samples": len(rows),
        "metrics": {
            "balanced_score": balanced_score,
            "macro_attainment": macro_attainment,
            "attainment_spread": spread,
        },
        "kpi": {
            "profile": kpi_profile,
            "metric": "balanced_score",
            "threshold": 1.0 if gate_eligible else None,
            "value": balanced_score,
            "met": kpi_met,
            "gate_eligible": gate_eligible,
        },
    }
    return {
        "summary": summary,
        "metric_result": metric_result,
        "task_metrics": task_metrics,
        "gate_groups": gate_groups,
        "min_group_support": min_group_support,
        "dataset_metrics": dataset_metrics,
        "reference_cohort_metrics": cohort_metrics,
        "metric_family_metrics": family_metrics,
        "sample_metrics": rows,
        "coverage": coverage,
        "gap_candidates": list(rows) if evaluation_role == "proxy" else [],
    }


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_parquet(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to write multi-task metric artifacts") from exc
    materialized = list(rows)
    table = pa.Table.from_pylist(materialized) if materialized else pa.table({"id": pa.array([], type=pa.string())})
    pq.write_table(table, path)


def write_artifacts(result: dict[str, Any], output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics_summary.json", result["summary"])
    _write_json(output_dir / "metric_result.json", result["metric_result"])
    _write_json(
        output_dir / "task_metrics.json",
        {
            "tasks": result["task_metrics"],
            "task_dataset": result["dataset_metrics"],
            "reference_cohort": result["reference_cohort_metrics"],
            "metric_family": result["metric_family_metrics"],
        },
    )
    _write_json(output_dir / "prediction_coverage.json", result["coverage"])
    _write_parquet(output_dir / "sample_metrics.parquet", result["sample_metrics"])
    if result["summary"]["evaluation_role"] == "proxy":
        _write_parquet(output_dir / "gap_candidates.parquet", result["gap_candidates"])
        _write_json(output_dir / "gaps_summary.json", result["summary"])
