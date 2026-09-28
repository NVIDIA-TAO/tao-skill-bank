#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate normalized DEFT OD AOI roles and freeze a real-only run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULTS = Path(__file__).resolve().parents[1] / "assets" / "default_policy.yaml"
# Normalized handoff roles: KPI/test are held out, ``real`` is the
# defective-real mining pool, and ``clean`` is the verified-clean mining pool.
ROLES = ("kpi", "test", "real", "clean")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


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


def _validate_bbox(value: Any, image: dict[str, Any], role: str) -> None:
    x, y, width, height = map(float, value)
    image_width = float(image.get("width") or 0)
    image_height = float(image.get("height") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"{role} normalized COCO image dimensions must be positive")
    if width <= 0 or height <= 0:
        raise ValueError(f"{role} normalized COCO bbox dimensions must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"{role} normalized COCO bbox exceeds image bounds")


def _has_synthesis_clean_reference(pool: Path) -> bool:
    return any(
        image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
        for texture in pool.iterdir() if texture.is_dir()
        for clean_dir in (texture / "clean_image",) if clean_dir.is_dir()
        for image in clean_dir.iterdir()
    )


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
    images_by_id = {int(row["id"]): row for row in image_rows}
    image_ids = set(images_by_id)
    if not image_rows and name != "clean":
        raise ValueError(f"{name} has no images")
    if len(image_ids) != len(image_rows):
        raise ValueError(f"{name} has duplicate image ids")
    counts = {image_id: 0 for image_id in image_ids}
    for annotation in coco.get("annotations", []):
        image_id, category = int(annotation["image_id"]), int(annotation["category_id"])
        if image_id not in counts or category not in category_ids:
            raise ValueError(f"{name} annotation references unknown image/category")
        _validate_bbox(annotation["bbox"], images_by_id[image_id], name)
        counts[image_id] += 1
    paths = [_image_path(images, row) for row in image_rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{name} references missing images; first={missing[0]}")
    if name == "clean" and any(counts.values()):
        raise ValueError("clean role must have zero annotations")
    if name == "real" and any(count == 0 for count in counts.values()):
        raise ValueError("defective-real role contains a boxless image")
    return {"images": str(images), "coco": str(coco_path), "coco_sha256": _sha(coco_path),
            "image_count": len(paths), "annotation_count": sum(counts.values()),
            "identities": {str(path) for path in paths}}


def _validate_kpi_retrieval_metadata(coco_path: Path) -> None:
    document = json.loads(coco_path.read_text())
    boxed_image_ids = {
        int(annotation["image_id"]) for annotation in document.get("annotations", [])
    }
    fields = ("dataset_id", "texture_id", "defect_class")
    for row in document.get("images", []):
        if int(row["id"]) not in boxed_image_ids:
            continue
        metadata = row.get("deft_od_aoi") or {}
        missing = [
            key for key in fields
            if not str(metadata.get(key) or row.get(key) or "").strip()
        ]
        if missing:
            raise ValueError(
                f"KPI image {row['id']} lacks canonical retrieval metadata: {missing}"
            )


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
    baseline_mode = str(policy.get("baseline_mode") or "")
    if baseline_mode not in {"cold_start", "checkpoint"}:
        raise ValueError("baseline_mode must be cold_start or checkpoint")
    role_reports = {name: _role(name, policy["sources"][name]) for name in ROLES}
    _validate_kpi_retrieval_metadata(Path(role_reports["kpi"]["coco"]))
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
    profile = str(((policy.get("retrieval") or {}).get("preprocessing") or {}).get(
        "profile", ""
    ))
    if profile not in {"tight_context", "square_context"}:
        raise ValueError(
            "retrieval.preprocessing.profile must be tight_context or square_context"
        )
    if (profile == "square_context"
            and int(policy["retrieval"].get("output_size", 0)) < 1):
        raise ValueError("retrieval.output_size must be positive")
    synthesis = policy.get("synthesis", {})
    if synthesis.get("enabled"):
        pool = Path(str(synthesis.get("pool_dataset_root") or "")).expanduser().resolve()
        if not pool.is_dir():
            raise ValueError("enabled synthesis needs pool_dataset_root")
        if not _has_synthesis_clean_reference(pool):
            raise ValueError("enabled synthesis needs at least one clean reference image")
        if not Path(str(synthesis.get("defect_spec") or "")).is_file():
            raise ValueError("enabled synthesis needs defect_spec")
        if not synthesis.get("routes"):
            raise ValueError("enabled synthesis needs at least one dataset route")
        for name, route in synthesis["routes"].items():
            ready = Path(str(route.get("checkpoint") or "")).is_file() and Path(
                str(route.get("recipe") or "")).is_file()
            finetune = route.get("finetune") or {}
            if not ready and any(not str(finetune.get(key) or "").strip() for key in
                                 ("dataset_root", "validation_testcase", "base_checkpoint",
                                  "vae_path", "checkpoint_root", "result_handoff")):
                raise ValueError(f"synthesis route {name} needs checkpoint/recipe or finetune inputs")
    output.mkdir(parents=True)
    policy["base_checkpoint"] = str(checkpoint)
    policy["sources"] = {name: {"images": role_reports[name]["images"],
                                 "coco": role_reports[name]["coco"]} for name in ROLES}
    frozen = output / "deft_od_aoi_policy.yaml"
    frozen.write_text(yaml.safe_dump(policy, sort_keys=False))
    classmap = output / "inference_classmap.txt"
    classmap.write_text("background\ndefect\n")
    synthesis_enabled = bool(policy.get("synthesis", {}).get("enabled"))
    bootstrap_required = synthesis_enabled and any(
        not (Path(str(route.get("checkpoint") or "")).is_file()
             and Path(str(route.get("recipe") or "")).is_file())
        for route in policy.get("synthesis", {}).get("routes", {}).values()
    )
    retrieval_capabilities = {
        role: {"status": "AVAILABLE" if role_reports[role]["image_count"] else "UNAVAILABLE",
               "reason": None if role_reports[role]["image_count"] else "empty_source_role",
               "source_image_count": role_reports[role]["image_count"]}
        for role in ("real", "clean")
    }
    warnings = [{
        "code": "empty_retrieval_source_role",
        "role": role,
        "message": f"{role} retrieval source role is empty; that producer starts exhausted",
    } for role, evidence in retrieval_capabilities.items()
        if evidence["status"] == "UNAVAILABLE"]
    state = {"schema_version": 1, "status": "READY",
             "mode": "rtdetr_with_synthesis" if synthesis_enabled else "rtdetr_real_only",
             "baseline_mode": baseline_mode,
             "synthesis_enabled": synthesis_enabled,
             "synthesis_bootstrap_required": bootstrap_required,
             "current_iteration": 0,
             "next_stage": "synthesis_bootstrap" if bootstrap_required else "candidate_cache",
             "max_iterations": policy["max_iterations"], "platform": policy["platform"],
             "base_checkpoint": str(checkpoint), "policy": str(frozen.resolve()),
             "policy_sha256": _sha(frozen), "classmap": str(classmap.resolve()),
             "roles": role_reports,
             "capabilities": {"retrieval": retrieval_capabilities},
             "warnings": warnings, "iterations": {}}
    _json(output / "deft_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = initialize(args.config.resolve(), args.output_dir.resolve())
    for warning in state.get("warnings", []):
        print(f"WARNING: {warning['message']}", file=sys.stderr)
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
