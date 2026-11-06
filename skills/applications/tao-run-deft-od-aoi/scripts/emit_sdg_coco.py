#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate shared AnomalyGen outputs and emit OD training COCO projections.

This application converter never executes generation, training, or pseudo labeling.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SHARED = next(p for p in (Path(__file__).resolve().parents[2] / "tao-generate-anomalies/scripts", Path(__file__).resolve().parents[3] / "data/tao-generate-anomalies/scripts") if p.is_dir())
sys.path.insert(0, str(SHARED))
from anomalygen_contract import read_jsonl, sha256
from anomalygen_consumer import load_generation, validate_prepared_ancestry


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _merge(
    results: list[dict[str, Any]], output: Path, *, guardrail_enabled: bool
) -> dict[str, Any]:
    images, annotations, category_ids, status = [], [], {}, []
    next_image = next_annotation = 1
    for result in results:
        group, coco = result["group"], result["coco"]
        expected = set(group["anomaly_types"])
        categories = {
            int(row["id"]): str(row["name"]) for row in coco.get("categories", [])
        }
        if set(categories.values()) - expected:
            raise ValueError("pseudo-label output contains unexpected categories")
        if result["generated"] + result["blocked"] != group["requested_rows"]:
            raise ValueError("generated + blocked does not match requested rows")
        image_rows = coco.get("images", [])
        if len(image_rows) != result["generated"]:
            raise ValueError("COCO image count does not match generated count")
        if len({int(row["id"]) for row in image_rows}) != len(image_rows):
            raise ValueError("duplicate COCO image identity")
        if len({str(row["file_name"]) for row in image_rows}) != len(image_rows):
            raise ValueError("duplicate COCO image file")
        local_images = {int(row["id"]): row for row in coco.get("images", [])}
        image_map = {}
        per_image = Counter()
        for old_id, row in sorted(local_images.items()):
            source = Path(str(row["file_name"]))
            if not source.is_file():
                source = result["image_root"] / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.resolve().parent != Path(result["image_root"]).resolve():
                raise ValueError("COCO image is outside the verified generation bucket")
            image_map[old_id] = next_image
            images.append(
                {
                    **row,
                    "id": next_image,
                    "file_name": str(source.resolve()),
                    "dataset_id": group["dataset_id"],
                    **({"anomalygen_prepared": result["prepared_binding"]} if result.get("prepared_binding") else {}),
                    **({"anomalygen_lineage": result["lineage"][source.name]} if "lineage" in result else {}),
                }
            )
            next_image += 1
        for row in coco.get("annotations", []):
            old_image, old_category = int(row["image_id"]), int(row["category_id"])
            if old_image not in image_map or old_category not in categories:
                raise ValueError("annotation references an unknown image or category")
            x, y, width, height = map(float, row["bbox"])
            image = local_images[old_image]
            if (
                min(x, y) < 0
                or width <= 0
                or height <= 0
                or x + width > image["width"]
                or y + height > image["height"]
            ):
                raise ValueError("annotation bbox is invalid or outside its image")
            name = categories[old_category]
            category = category_ids.setdefault(name, len(category_ids) + 1)
            annotations.append(
                {
                    **row,
                    "id": next_annotation,
                    "image_id": image_map[old_image],
                    "category_id": category,
                }
            )
            next_annotation += 1
            per_image[old_image] += 1
        if any(per_image[old_id] == 0 for old_id in local_images):
            raise ValueError("one or more generated images have no annotation")
        status.append(
            {
                "dataset_id": group["dataset_id"],
                "requested": group["requested_rows"],
                "generated": result["generated"],
                "guardrail_blocked": result["blocked"],
            }
        )
    labels = output / "pseudo_labels"
    native = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": value, "name": key} for key, value in category_ids.items()
        ],
    }
    binary = {
        "images": images,
        "annotations": [{**row, "category_id": 1} for row in annotations],
        "categories": [{"id": 1, "name": "defect"}],
    }
    _json(labels / "coco_annotations.json", native)
    _json(labels / "coco_annotations_od_defect.json", binary)
    report = {
        "status": "COMPLETE",
        "groups": status,
        "generated": len(images),
        "annotations": len(annotations),
        "guardrail_enabled": guardrail_enabled,
        "training_pool_mutated": False,
    }
    _json(output / "validation_summary.json", report)
    return report


def validate_prepared(path):
    value = json.loads(Path(path).read_text())
    if value.get("status") != "COMPLETE" or value.get("generation_ready") is not True:
        raise ValueError("prepared inputs are not complete")
    artifacts = value.get("artifacts", [])
    if not artifacts:
        raise ValueError("prepared inputs have no artifact bindings")
    for artifact in artifacts:
        source = Path(artifact["path"])
        if not source.is_file() or sha256(source) != artifact["sha256"]:
            raise ValueError("prepared input artifact changed")
    groups = value.get("generator_groups", [])
    if not groups or len({g["dataset_id"] for g in groups}) != len(groups):
        raise ValueError("prepared inputs have invalid dataset groups")
    return {g["dataset_id"]: g for g in groups}


def inspect_runs(runs_manifest, prepared_manifest=None, exclude_stems=None):
    entries = json.loads(Path(runs_manifest).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("runs manifest must be a nonempty list")
    prepared = validate_prepared(prepared_manifest) if prepared_manifest else None
    results, seen = [], set()
    for entry in entries:
        dataset_id = entry["dataset_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset_id) or dataset_id in seen:
            raise ValueError("unsafe or duplicate dataset_id")
        seen.add(dataset_id)
        manifest = json.loads(Path(entry["manifest"]).read_text())
        counts, rows = load_generation(entry["manifest"], exclude_stems)
        testcase = Path(manifest["testcase"])
        cases = read_jsonl(testcase)
        bucket = Path(manifest["selected_output"])
        prepared_binding = None
        if prepared is not None:
            prepared_binding = {"manifest": str(Path(prepared_manifest).resolve()), "manifest_sha256": sha256(prepared_manifest), "dataset_id": dataset_id}
            validate_prepared_ancestry(prepared_binding, exclude_stems=exclude_stems)
            group = prepared.get(dataset_id)
            if group is None or read_jsonl(group["testcase"]) != cases:
                raise ValueError("generation testcase differs from prepared group")
            if any(str(Path(manifest[key]).resolve()) != str(Path(group[key]).resolve()) for key in ("checkpoint", "recipe")):
                raise ValueError("generation checkpoint/recipe differs from prepared group")
            if group["requested_rows"] != counts["testcases"]:
                raise ValueError("prepared requested count mismatch")
        coco = json.loads(Path(manifest["pseudo_labels"]).read_text()) if counts["generated"] else {"images": [], "annotations": [], "categories": []}
        expected_files = {row["output_filename"] for row in rows}
        if {Path(row["file_name"]).name for row in coco["images"]} != expected_files:
            raise ValueError("COCO file identities differ from native generated identities")
        results.append({"prepared_binding": prepared_binding, "group": {"dataset_id": dataset_id, "requested_rows": counts["testcases"], "anomaly_types": sorted({case["anomaly_type"] for case in cases})}, "generated": counts["generated"], "blocked": counts["guardrail_blocked"], "image_root": bucket / "reconstructed_image", "coco": coco, "lineage": {row["output_filename"]: row["lineage"] for row in rows}})
    if prepared is not None and seen != set(prepared):
        raise ValueError("generation runs do not cover prepared groups")
    return results


def publish(runs_manifest: Path, output: Path, prepared_manifest=None, exclude_stems=None):
    if output.exists():
        raise FileExistsError("output must be new")
    results = inspect_runs(runs_manifest, prepared_manifest, exclude_stems)
    report = _merge(results, output, guardrail_enabled=True)
    report.update(training_ready=report["generated"] > 0,
                  guardrail_blocked=sum(row["blocked"] for row in results),
                  runs_manifest=str(Path(runs_manifest).resolve()), runs_manifest_sha256=sha256(runs_manifest),
                  prepared_manifest=str(Path(prepared_manifest).resolve()) if prepared_manifest else None,
                  prepared_manifest_sha256=sha256(prepared_manifest) if prepared_manifest else None,
                  exclude_stems=str(Path(exclude_stems).resolve()) if exclude_stems else None)
    _json(output / "validation_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        required=True,
        help="JSON list of {dataset_id, manifest} shared-run entries",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, help="Required for application synthesis; omit only for standalone native smoke conversion")
    parser.add_argument("--exclude-stems", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.runs_manifest, args.output_dir, args.prepared_manifest, args.exclude_stems), sort_keys=True))


if __name__ == "__main__":
    main()
