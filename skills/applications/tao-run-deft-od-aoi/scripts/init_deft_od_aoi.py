#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate normalized DEFT OD AOI roles and freeze a real-only run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULTS = Path(__file__).resolve().parents[1] / "assets" / "default_policy.yaml"
ROLES = ("kpi", "test", "real", "clean")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def _image_path(images: Path, row: dict[str, Any]) -> Path:
    source = str(row.get("source_path") or "").strip()
    path = Path(source) if source else images / str(row.get("file_name") or "")
    return path.expanduser().resolve()


def _role(name: str, value: dict[str, Any]) -> dict[str, Any]:
    images = Path(str(value.get("images") or "")).expanduser().resolve()
    coco_path = Path(str(value.get("coco") or "")).expanduser().resolve()
    if not images.is_dir() or not coco_path.is_file():
        raise ValueError(f"{name} images/COCO are missing")
    coco = json.loads(coco_path.read_text())
    categories = coco.get("categories", [])
    if not categories or {str(row.get("name")) for row in categories} != {"defect"}:
        raise ValueError(f"{name} must declare only the defect category")
    category_ids = {int(row["id"]) for row in categories}
    image_rows = coco.get("images", [])
    image_ids = {int(row["id"]) for row in image_rows}
    if not image_rows or len(image_ids) != len(image_rows):
        raise ValueError(f"{name} has no images or duplicate image ids")
    counts = {image_id: 0 for image_id in image_ids}
    for annotation in coco.get("annotations", []):
        image_id, category = int(annotation["image_id"]), int(annotation["category_id"])
        if image_id not in counts or category not in category_ids:
            raise ValueError(f"{name} annotation references unknown image/category")
        x, y, width, height = map(float, annotation["bbox"])
        if min(x, y) < 0 or width <= 0 or height <= 0:
            raise ValueError(f"{name} contains an invalid bbox")
        counts[image_id] += 1
    paths = [_image_path(images, row) for row in image_rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{name} references missing images; first={missing[0]}")
    if name == "clean" and any(counts.values()):
        raise ValueError("clean role must have zero annotations")
    if name != "clean" and any(count == 0 for count in counts.values()):
        raise ValueError(f"{name} role contains boxless images")
    return {"images": str(images), "coco": str(coco_path), "coco_sha256": _sha(coco_path),
            "image_count": len(paths), "annotation_count": sum(counts.values()),
            "identities": {str(path) for path in paths}}


def initialize(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    user = yaml.safe_load(config_path.read_text())
    if not isinstance(user, dict):
        raise ValueError("config must be a YAML mapping")
    policy = _merge(yaml.safe_load(DEFAULTS.read_text()), user)
    if not isinstance(policy.get("max_iterations"), int) or policy["max_iterations"] < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not str(policy.get("platform") or "").strip():
        raise ValueError("platform must be selected before initialization")
    checkpoint = Path(str(policy.get("base_checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError("base_checkpoint must be a trainable RT-DETR file")
    role_reports = {name: _role(name, policy["sources"][name]) for name in ROLES}
    owners: dict[str, str] = {}
    for name, report in role_reports.items():
        for identity in report.pop("identities"):
            if identity in owners:
                raise ValueError(f"image overlaps {owners[identity]} and {name}: {identity}")
            owners[identity] = name
    gap = policy["gap"]
    if not (0 <= gap["inference_confidence"] <= gap["loose_confidence"]
            <= gap["strict_confidence"] <= 1):
        raise ValueError("inference/loose/strict confidence thresholds are inconsistent")
    if not (0 <= gap["background_iou_upper"] < gap["near_miss_iou_upper"] <= 1):
        raise ValueError("background and near-miss IoU thresholds are inconsistent")
    if policy["class_name"] != "defect":
        raise ValueError("DEFT OD AOI has one foreground class named defect")
    synthesis = policy.get("synthesis", {})
    if synthesis.get("enabled"):
        if not Path(str(synthesis.get("pool_dataset_root") or "")).is_dir():
            raise ValueError("enabled synthesis needs pool_dataset_root")
        if not Path(str(synthesis.get("defect_spec") or "")).is_file():
            raise ValueError("enabled synthesis needs defect_spec")
        if not synthesis.get("routes"):
            raise ValueError("enabled synthesis needs at least one dataset route")
        for name, route in synthesis["routes"].items():
            if not Path(str(route.get("checkpoint") or "")).is_file() or not Path(
                    str(route.get("recipe") or "")).is_file():
                raise ValueError(f"synthesis route {name} needs checkpoint and recipe")
    output.mkdir(parents=True)
    policy["base_checkpoint"] = str(checkpoint)
    policy["sources"] = {name: {"images": role_reports[name]["images"],
                                 "coco": role_reports[name]["coco"]} for name in ROLES}
    frozen = output / "deft_od_aoi_policy.yaml"
    frozen.write_text(yaml.safe_dump(policy, sort_keys=False))
    classmap = output / "inference_classmap.txt"
    classmap.write_text("background\ndefect\n")
    synthesis_enabled = bool(policy.get("synthesis", {}).get("enabled"))
    state = {"schema_version": 1, "status": "READY",
             "mode": "rtdetr_with_synthesis" if synthesis_enabled else "rtdetr_real_only",
             "synthesis_enabled": synthesis_enabled,
             "current_iteration": 0, "next_stage": "candidate_cache",
             "max_iterations": policy["max_iterations"], "platform": policy["platform"],
             "base_checkpoint": str(checkpoint), "policy": str(frozen.resolve()),
             "policy_sha256": _sha(frozen), "classmap": str(classmap.resolve()),
             "roles": role_reports, "iterations": {}}
    _json(output / "deft_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(args.config.resolve(), args.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
