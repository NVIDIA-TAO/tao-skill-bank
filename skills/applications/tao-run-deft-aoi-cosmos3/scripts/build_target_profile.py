#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Profile canonical NVPAW annotations or compare profiles; never modify a loop."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
import pathlib
import posixpath
import re
import sys
from typing import Any, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import image_paths, prompt_and_response

TASKS = tuple(task for task, spec in TASK_SPECS.items()
              if spec["metric_family"] in {"classification", "detection"})
SIZE_LABELS = ("<16", "16-33", "33-66", "66-130", "130-260", ">260")
COUNT_LABELS = ("0", "1", "2-3", "4-9", "10+")
CELL_FIELDS = ("task_type", "empty_status", "count_bin", "size_bin", "pair_kind")
PROFILE_SCHEMA = "nvpaw_target_profile_v1"


def read_rows(path: pathlib.Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: JSONL row must be an object")
                yield value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_path(value: str) -> str:
    return posixpath.normpath(value.replace("\\", "/"))


def dataset_family(record: dict) -> str:
    paths = ["/" + normalize_path(path).lstrip("/")
             for path in image_paths(record, context=str(record.get("id")))]
    for path in reversed(paths):
        match = re.search(r"/NVPAW_pair/([^/]+)/", path)
        if match:
            return "pair:" + match.group(1)
    for path in reversed(paths):
        match = re.search(r"/datasets/([^/]+)/", path)
        if match:
            return match.group(1)
    return str(record.get("dataset") or "unknown")


def pair_kind(paths: list[str]) -> str:
    paths = [normalize_path(path) for path in paths]
    if len(paths) < 2:
        return "single"
    if any("/gen_Qwen/" in "/" + path or "qwen_edit" in path for path in paths):
        return "generated_edit"
    if any("/gen_aug/" in "/" + path for path in paths) or re.search(r"rotation|light|aug", paths[-1]):
        return "augmented_view"
    return "identical_path" if paths[0] == paths[-1] else "different_photo"


def count_bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4-9" if count <= 9 else "10+"


def size_bin(area: float) -> str:
    # Operator reference /tmp/box_profile.py: effective side = sqrt(area).
    for edge, label in zip((16, 33, 66, 130, 260), SIZE_LABELS):
        if math.sqrt(area) < edge:
            return label
    return SIZE_LABELS[-1]


def ground_truth_boxes(answer: str, *, context: str) -> list[dict]:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.I)
    try:
        values = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid detection ground truth") from exc
    if not isinstance(values, list):
        raise ValueError(f"{context}: detection ground truth must be a list")
    result = []
    for item in values:
        box = item.get("bbox_2d", item.get("box_2d")) if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000
            for value in box
        ):
            raise ValueError(f"{context}: ground truth coordinates must be in [0, 1000]")
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            raise ValueError(f"{context}: ground truth box must have positive area")
        result.append({"bbox_2d": box, "label": str(item.get("label", "unknown")),
                       "area": width * height, "aspect": width / height})
    return result


def row_features(record: dict) -> dict:
    task = str(record.get("task_type"))
    if task not in TASKS:
        raise ValueError(f"unsupported analysis task: {task}")
    context = str(record.get("id", "row"))
    _, answer = prompt_and_response(record, context=context)
    paths = image_paths(record, context=context)
    if len(paths) != len(TASK_SPECS[task]["image_roles"]):
        raise ValueError(f"{context}: incorrect image count for {task}")
    detection = TASK_SPECS[task]["metric_family"] == "detection"
    boxes = ground_truth_boxes(answer, context=context) if detection else []
    # Classification has no boxes. Only explicit empty labels / direct negative
    # BCQ are an empty stratum; raw annotation labels are not KPI predictions.
    empty = not boxes if detection else answer.strip() in {
        "[]", "{}", "No, the target image does not contain any defects.",
    }
    return {"task_type": task, "dataset": dataset_family(record),
            "empty_status": "empty" if empty else "non_empty",
            "count_bin": count_bin(len(boxes)) if detection else "NA",
            "size_bin": size_bin(max(item["area"] for item in boxes)) if boxes else "NA",
            "pair_kind": pair_kind(paths), "gt_boxes": len(boxes), "boxes": boxes,
            "labels": [item["label"] for item in boxes] if detection else [answer.strip()]}


def _quantile(ordered: list[float], fraction: float) -> float | None:
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def build_profile(rows: Iterable[dict]) -> dict:
    totals, ignored, cells = {}, Counter(), Counter()
    for record in rows:
        task = str(record.get("task_type"))
        if task not in TASKS:
            ignored[task] += 1
            continue
        feature = row_features(record)
        stats = totals.setdefault(task, {"rows": 0, "empty_rows": 0, "boxes": 0,
            "box_size_bins": Counter(), "count_bins": Counter(), "label_histogram": Counter(),
            "pair_kind": Counter(), "dataset_histogram": Counter(),
            "areas": array("d"), "aspects": array("d")})
        stats["rows"] += 1
        stats["empty_rows"] += feature["empty_status"] == "empty"
        stats["count_bins"][feature["count_bin"]] += 1
        stats["label_histogram"].update(feature["labels"])
        stats["pair_kind"][feature["pair_kind"]] += 1
        stats["dataset_histogram"][feature["dataset"]] += 1
        cells[tuple(feature[field] for field in (*CELL_FIELDS, "dataset"))] += 1
        for item in feature["boxes"]:
            stats["boxes"] += 1
            stats["box_size_bins"][size_bin(item["area"])] += 1
            stats["areas"].append(item["area"] / 1_000_000.0)
            stats["aspects"].append(item["aspect"])
    for task, stats in totals.items():
        detection = TASK_SPECS[task]["metric_family"] == "detection"
        areas, aspects = sorted(stats.pop("areas")), sorted(stats.pop("aspects"))
        nonempty = stats["rows"] - stats["empty_rows"]
        stats["empty_rate"] = stats["empty_rows"] / stats["rows"]
        stats["boxes_per_nonempty_row"] = stats["boxes"] / nonempty if detection and nonempty else None
        stats["relative_area_quantiles"] = {str(q): _quantile(areas, q / 100) for q in (10, 25, 50, 75, 90)}
        stats["aspect_ratio_median"] = _quantile(aspects, 0.5)
        stats["count_bins"] = {key: stats["count_bins"][key] for key in (COUNT_LABELS if detection else ("NA",))}
        stats["nonempty_count_share"] = {key: stats["count_bins"][key] / nonempty if nonempty else 0.0
                                         for key in COUNT_LABELS[1:]} if detection else {}
        stats["box_size_bins"] = {key: stats["box_size_bins"][key] for key in SIZE_LABELS}
        for field in ("label_histogram", "pair_kind", "dataset_histogram"):
            stats[field] = dict(sorted(stats[field].items()))
    return {"schema_version": PROFILE_SCHEMA, "analysis_only": True,
            "rows": sum(stats["rows"] for stats in totals.values()),
            "ignored_task_rows": dict(sorted(ignored.items())),
            "definitions": {"coordinate_frame": [0, 1000], "box_side": "sqrt(area)",
                "size_intervals": "left-inclusive, right-exclusive; final bin includes 260",
                "row_size": "largest_box_effective_side; NA for empty/classification",
                "relative_area": "area / 1000000", "quantiles": "linear interpolation over all boxes",
                "classification_labels": "raw canonical ground-truth answer", "shares": "fractions, not percent"},
            "tasks": dict(sorted(totals.items())),
            "cells": [{**dict(zip((*CELL_FIELDS, "dataset"), key)), "rows": count}
                      for key, count in sorted(cells.items())]}


def integer_targets(counts: dict[tuple, int], budget: int) -> dict[tuple, int]:
    if budget < 0 or not counts or sum(counts.values()) <= 0:
        raise ValueError("positive target support and non-negative budget are required")
    total = sum(counts.values())
    targets = {key: budget * value // total for key, value in counts.items()}
    order = sorted(counts, key=lambda key: (-(budget * counts[key] % total), key))
    for key in order[:budget - sum(targets.values())]:
        targets[key] += 1
    return targets


def compare_profiles(target: dict, supply: dict, *, budget: int) -> list[dict]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    counts = []
    for payload in (target, supply):
        if payload.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError("incompatible target profile schema")
        values = Counter()
        for cell in payload["cells"]:
            values[tuple(cell[field] for field in CELL_FIELDS)] += cell["rows"]
        counts.append(values)
    wanted, available = counts
    targets = integer_targets(wanted, budget)
    return [{**dict(zip(CELL_FIELDS, key)),
        "target_share": wanted[key] / sum(wanted.values()),
        "supply_share": available[key] / max(1, sum(available.values())),
        "requested_rows": targets.get(key, 0), "available_rows": available[key],
        "achievable_rows": min(targets.get(key, 0), available[key]),
        "shortage_rows": max(0, targets.get(key, 0) - available[key])}
        for key in sorted(wanted.keys() | available.keys())]


def markdown(payload: dict) -> str:
    lines = ["# Target profile (analysis only)", "", "| Task | Rows | Empty % | Boxes | Boxes/nonempty |", "|---|---:|---:|---:|---:|"]
    for task, stats in payload["tasks"].items():
        lines.append(f"| {task} | {stats['rows']} | {100 * stats['empty_rate']:.2f} | {stats['boxes']} | {stats['boxes_per_nonempty_row']} |")
    lines.extend(["", "Coordinates are already 0–1000; size = sqrt(area). Detailed histograms and cells are in JSON.", ""])
    return "\n".join(lines)


def write_outputs(outputs: dict[pathlib.Path, str]) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis artifact: {path}")
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=pathlib.Path)
    mode.add_argument("--compare", nargs=2, type=pathlib.Path, metavar=("TARGET", "SUPPLY"))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    try:
        if args.compare:
            if args.budget is None:
                raise ValueError("--compare requires --budget")
            comparison = compare_profiles(*(json.loads(path.read_text()) for path in args.compare), budget=args.budget)
            print("| Cell | Target share | Supply share | Requested | Achievable | Shortage |")
            print("|---|---:|---:|---:|---:|---:|")
            for cell in comparison:
                key = " / ".join(cell[field] for field in CELL_FIELDS)
                print(f"| {key} | {cell['target_share']:.4f} | {cell['supply_share']:.4f} | {cell['requested_rows']} | {cell['achievable_rows']} | {cell['shortage_rows']} |")
        else:
            name = args.name or args.input.stem
            if pathlib.Path(name).name != name or name in {".", ".."}:
                raise ValueError("--name must be a filename stem")
            payload = build_profile(read_rows(args.input))
            payload["input"] = {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)}
            text = markdown(payload)
            write_outputs({args.output_dir / f"{name}_profile.json": json.dumps(payload, indent=2) + "\n",
                           args.output_dir / f"{name}_profile.md": text})
            print(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"build_target_profile: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
