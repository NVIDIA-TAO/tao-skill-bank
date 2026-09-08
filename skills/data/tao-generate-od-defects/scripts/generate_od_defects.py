#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run AnomalyGenNext inference and publish merged OD defect labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_manifest(root: Path) -> None:
    manifest = root / "prepared_anomalygennext_inputs" / "prepared_inputs_manifest.json"
    if not manifest.is_file():
        return
    value = json.loads(manifest.read_text())
    if value.get("status") != "COMPLETE" or not value.get("generation_ready"):
        raise ValueError("prepared-input manifest is not generation-ready")
    for artifact in value.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.is_file() or _sha(path) != artifact["sha256"]:
            raise ValueError(f"prepared-input artifact changed or is missing: {path}")


def _normalize(group: dict[str, Any]) -> dict[str, Any]:
    result = {"dataset_id": str(group.get("dataset_id") or "default"),
              "testcase": str(group["testcase"]), "checkpoint": str(group["checkpoint"]),
              "recipe": str(group["recipe"]),
              "anomaly_types": [str(value) for value in group.get("anomaly_types", [])]}
    if not SAFE_ID.fullmatch(result["dataset_id"]):
        raise ValueError(f"unsafe dataset id: {result['dataset_id']!r}")
    for key in ("testcase", "checkpoint", "recipe"):
        if not Path(result[key]).is_file():
            raise FileNotFoundError(f"{key} not found: {result[key]}")
    rows = _rows(Path(result["testcase"]))
    if not rows:
        raise ValueError("generation testcase is empty")
    if not result["anomaly_types"]:
        result["anomaly_types"] = list(dict.fromkeys(str(row.get("anomaly_type") or "") for row in rows))
    if not all(result["anomaly_types"]):
        raise ValueError("generation rows need anomaly_type")
    recipe = yaml.safe_load(Path(result["recipe"]).read_text())
    recipe_types = {f"{pair[0]}+{pair[1]}" for pair in recipe.get("anomaly_types", [])}
    if set(result["anomaly_types"]) - recipe_types:
        raise ValueError("generation anomaly types are absent from the recipe")
    for row in rows:
        if str(row.get("anomaly_type") or "") not in result["anomaly_types"]:
            raise ValueError("testcase contains an undeclared anomaly type")
        for key in ("image_filename", "mask_filename"):
            if not Path(str(row.get(key) or "")).is_file():
                raise FileNotFoundError(f"testcase {key} is missing: {row.get(key)}")
    requested = int(group.get("requested_rows", len(rows)))
    if requested != len(rows):
        raise ValueError(f"requested/testcase row mismatch: {requested} != {len(rows)}")
    result["requested_rows"] = requested
    return result


def groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.inputs_dir) == bool(args.input_data_path):
        raise ValueError("pass exactly one of --inputs-dir and --input-data-path")
    if args.inputs_dir:
        root = args.inputs_dir.resolve()
        _validate_manifest(root)
        plan = root / "prepared_anomalygennext_inputs" / "anomalygen_next_generation_plan.json"
        if not plan.is_file():
            plan = root / "anomalygen_next_generation_plan.json"
        values = json.loads(plan.read_text())
        if not isinstance(values, list) or not values:
            raise ValueError("generation plan is empty")
        if args.datasets:
            requested = {value.strip() for value in args.datasets.split(",") if value.strip()}
            available = {str(row.get("dataset_id")) for row in values}
            if requested - available:
                raise ValueError(f"unknown dataset ids: {sorted(requested - available)}")
            values = [row for row in values if str(row.get("dataset_id")) in requested]
    else:
        if not args.checkpoint or not args.recipe:
            raise ValueError("native input requires --checkpoint and --recipe")
        values = [{"dataset_id": args.dataset_id, "testcase": str(args.input_data_path),
                   "checkpoint": str(args.checkpoint), "recipe": str(args.recipe),
                   "anomaly_types": [value.strip() for value in args.anomaly_types.split(",")
                                     if value.strip()]}]
    result = [_normalize(value) for value in values]
    if len({group["dataset_id"] for group in result}) != len(result):
        raise ValueError("generation plan contains duplicate dataset ids")
    return result


def _csv_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _run_group(group: dict[str, Any], output: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw, labels = output / "raw", output / "pseudo_labels"
    raw.mkdir(parents=True)
    command = ["torchrun", f"--nproc_per_node={args.num_gpus}",
               str(args.repo / "anomalygen/scripts/texture/generate.py"),
               "--checkpoint", group["checkpoint"], "--recipe", group["recipe"],
               "--base_checkpoint", str(args.base_checkpoint),
               "--input_data_path", group["testcase"], "--output_dir", str(raw)]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(args.repo / "anomalygen/scripts/texture/pseudo_label.py"),
                    "--gen_root", str(raw), "--output_dir", str(labels), "--no_caption"], check=True)
    generated = _csv_count(raw / "texture_ft_generation_result.csv")
    blocked = _csv_count(raw / "guardrail_blocked.csv")
    if generated + blocked != group["requested_rows"]:
        raise ValueError("generated + guardrail-blocked does not equal requested rows")
    coco = json.loads((labels / "coco_annotations.json").read_text())
    if len(coco.get("images", [])) != generated:
        raise ValueError("pseudo-label image count does not equal generated count")
    return {"group": group, "coco": coco, "generated": generated, "blocked": blocked,
            "image_root": raw / "reconstructed_image"}


def _merge(results: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    images, annotations, category_ids, status = [], [], {}, []
    next_image = next_annotation = 1
    for result in results:
        group, coco = result["group"], result["coco"]
        expected = set(group["anomaly_types"])
        categories = {int(row["id"]): str(row["name"]) for row in coco.get("categories", [])}
        if set(categories.values()) - expected:
            raise ValueError("pseudo-label output contains unexpected categories")
        local_images = {int(row["id"]): row for row in coco.get("images", [])}
        image_map = {}
        per_image = Counter()
        for old_id, row in sorted(local_images.items()):
            source = Path(str(row["file_name"]))
            if not source.is_file():
                source = result["image_root"] / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            image_map[old_id] = next_image
            images.append({**row, "id": next_image, "file_name": str(source.resolve()),
                           "dataset_id": group["dataset_id"]})
            next_image += 1
        for row in coco.get("annotations", []):
            old_image, old_category = int(row["image_id"]), int(row["category_id"])
            if old_image not in image_map or old_category not in categories:
                raise ValueError("annotation references an unknown image or category")
            x, y, width, height = map(float, row["bbox"])
            image = local_images[old_image]
            if min(x, y) < 0 or width <= 0 or height <= 0 or x + width > image["width"] or y + height > image["height"]:
                raise ValueError("annotation bbox is invalid or outside its image")
            name = categories[old_category]
            category = category_ids.setdefault(name, len(category_ids) + 1)
            annotations.append({**row, "id": next_annotation, "image_id": image_map[old_image],
                                "category_id": category})
            next_annotation += 1
            per_image[old_image] += 1
        if any(per_image[old_id] == 0 for old_id in local_images):
            raise ValueError("one or more generated images have no annotation")
        status.append({"dataset_id": group["dataset_id"], "requested": group["requested_rows"],
                       "generated": result["generated"], "guardrail_blocked": result["blocked"]})
    labels = output / "pseudo_labels"
    native = {"images": images, "annotations": annotations,
              "categories": [{"id": value, "name": key} for key, value in category_ids.items()]}
    binary = {"images": images, "annotations": [{**row, "category_id": 1} for row in annotations],
              "categories": [{"id": 1, "name": "defect"}]}
    _json(labels / "coco_annotations.json", native)
    _json(labels / "coco_annotations_od_defect.json", binary)
    report = {"status": "COMPLETE", "groups": status, "generated": len(images),
              "annotations": len(annotations), "training_pool_mutated": False}
    _json(output / "validation_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--input-data-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--anomaly-types", default="")
    parser.add_argument("--dataset-id", default="default")
    parser.add_argument("--datasets")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/paidf-anomalygen"))
    args = parser.parse_args()
    if args.output_dir.exists() or args.num_gpus < 1 or not args.base_checkpoint.exists():
        raise ValueError("output must be new; GPU count positive; base checkpoint must exist")
    if not args.repo.is_dir():
        raise FileNotFoundError(args.repo)
    env = os.environ
    if args.hf_cache:
        if not args.hf_cache.is_dir():
            raise FileNotFoundError(args.hf_cache)
        env["HF_HOME"] = str(args.hf_cache.resolve())
    selected = groups(args)
    args.output_dir.mkdir(parents=True)
    try:
        report = _merge([_run_group(group, args.output_dir / group["dataset_id"], args)
                         for group in selected], args.output_dir)
        _json(args.output_dir / "status.json", {"status": "COMPLETE"})
    except Exception as exc:
        _json(args.output_dir / "status.json", {"status": "ERROR", "message": str(exc)})
        raise
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
