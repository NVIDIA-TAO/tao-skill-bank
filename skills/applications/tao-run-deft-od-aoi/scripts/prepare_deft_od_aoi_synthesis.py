#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize DEFT AOI false negatives into AnomalyGenNext preparation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FIELDS = ("dataset_id", "texture_id", "defect_class", "fn_mask_source")


def _source(images: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("source_path") or "")) if row.get("source_path") else (
        images / str(row["file_name"])
    )
    return path.resolve()


def _xyxy(value: Any) -> tuple[float, float, float, float]:
    values = tuple(map(float, value))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"invalid xyxy gap box: {value}")
    return values


def _iou(left: tuple[float, ...], annotation: dict[str, Any]) -> float:
    x, y, width, height = map(float, annotation["bbox"])
    right = (x, y, x + width, y + height)
    ix1, iy1, ix2, iy2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + width * height - intersection
    return intersection / union if union else 0.0


def prepare(policy_path: Path, strict_gaps: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    synthesis = policy["synthesis"]
    if not synthesis.get("enabled"):
        raise ValueError("synthesis is disabled in the frozen policy")
    routes = synthesis.get("routes") or {}
    pool, defect_spec = Path(str(synthesis["pool_dataset_root"])), Path(str(synthesis["defect_spec"]))
    if not pool.is_dir() or not defect_spec.is_file() or not routes:
        raise ValueError("synthesis pool, defect spec, and routes are required")
    for name, route in routes.items():
        if not Path(str(route.get("checkpoint") or "")).is_file() or not Path(str(route.get("recipe") or "")).is_file():
            raise ValueError(f"route {name} needs an existing checkpoint and recipe")
    kpi = policy["sources"]["kpi"]
    images_root = Path(kpi["images"])
    coco = json.loads(Path(kpi["coco"]).read_text())
    images = {str(row["id"]): row for row in coco["images"]}
    annotations: dict[str, list[dict[str, Any]]] = {}
    for row in coco.get("annotations", []):
        annotations.setdefault(str(row["image_id"]), []).append(row)
    gaps = pd.read_parquet(strict_gaps)
    required = {"image_id", "filepath", "gap_type", "bbox", "class"}
    if not required.issubset(gaps):
        raise ValueError(f"strict gaps lack {sorted(required - set(gaps))}")
    rows = []
    for gap in gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].to_dict("records"):
        image_id, box = str(gap["image_id"]), _xyxy(gap["bbox"])
        if image_id not in images:
            raise ValueError(f"FN image id is absent from KPI COCO: {image_id}")
        ranked = sorted(((_iou(box, row), row) for row in annotations.get(image_id, [])),
                        key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.999:
            raise ValueError(f"FN box has no exact KPI annotation match: {image_id} {box}")
        annotation, image = ranked[0][1], images[image_id]
        metadata = {}
        for source in (image, image.get("deft_od_aoi", {}), annotation,
                       annotation.get("deft_od_aoi", {})):
            metadata.update({key: source[key] for key in FIELDS if key in source})
        missing = [key for key in FIELDS if not str(metadata.get(key) or "").strip()]
        if missing:
            raise ValueError(f"FN metadata is incomplete for {image_id}: {missing}")
        dataset = str(metadata["dataset_id"])
        if dataset not in routes:
            raise ValueError(f"FN dataset has no synthesis route: {dataset}")
        mask = Path(str(metadata["fn_mask_source"])).expanduser().resolve()
        if not mask.is_file():
            raise ValueError(f"FN pixel mask is missing: {mask}")
        metadata["fn_mask_source"] = str(mask)
        source_path = _source(images_root, image)
        if source_path != Path(str(gap["filepath"])).resolve():
            raise ValueError(f"gap filepath does not match frozen KPI image: {image_id}")
        rows.append({**gap, "filepath": str(source_path), "split": "kpi", **metadata,
                     "anomaly_type": f"{metadata['texture_id']}+{metadata['defect_class']}"})
    if not rows:
        raise ValueError("strict gaps contain no synthesis-eligible false negatives")
    output.mkdir(parents=True)
    normalized = output / "normalized_fn_gaps.parquet"
    pd.DataFrame(rows).to_parquet(normalized, index=False)
    config = {"source_tag": "deft_od_aoi", "gap_parquet": str(normalized),
              "pool_dataset_root": str(pool.resolve()), "defect_spec": str(defect_spec.resolve()),
              "datasets": {name: {"checkpoint": str(Path(route["checkpoint"]).resolve()),
                                    "recipe": str(Path(route["recipe"]).resolve())}
                           for name, route in routes.items()},
              "selection": {"mode": "all_eligible", "datasets": sorted(routes),
                            "mask_sample_seed": 42},
              "embedding": {"model": policy["retrieval"]["model"],
                            "model_path": policy["retrieval"]["model_path"], "batch_size": 64},
              "retrieval": {"metric": "cosine", "candidate_topn":
                            int(policy["retrieval"]["candidate_overfetch"]),
                            "max_neighbors_per_fn": int(synthesis["max_neighbors_per_fn"]),
                            "min_similarity": float(synthesis["min_similarity"]),
                            "prior_clean_exclusion_manifest": ""},
              "amp": {"model_id": synthesis["amp_model_id"], "seed": 43}}
    config_path = output / "anomalygen_filtering.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {"status": "COMPLETE", "fn_count": len(rows), "config": str(config_path.resolve())}
    (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--strict-gaps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.policy.resolve(), args.strict_gaps.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
