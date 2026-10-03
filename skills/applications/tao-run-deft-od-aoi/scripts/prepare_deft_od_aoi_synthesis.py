#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize DEFT AOI false negatives into AnomalyGenNext preparation inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _image_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index canonical COCO ids and filename-stem ids emitted by gap analysis."""
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        aliases = {str(row.get("id"))}
        file_name = str(row.get("file_name") or "").strip()
        if file_name:
            aliases.add(Path(file_name).stem)
        for alias in aliases:
            if not alias or alias == "None":
                raise ValueError("KPI COCO image lacks an id")
            existing = output.get(alias)
            if existing is not None and existing is not row:
                raise ValueError(f"duplicate KPI image identity: {alias}")
            output[alias] = row
    return output


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
            raise ValueError(f"route {name} needs resolved checkpoint and recipe; run synthesis bootstrap")
    kpi = policy["sources"]["kpi"]
    images_root = Path(kpi["images"])
    coco = json.loads(Path(kpi["coco"]).read_text())
    images = _image_index(coco["images"])
    annotations: dict[str, list[dict[str, Any]]] = {}
    for row in coco.get("annotations", []):
        annotations.setdefault(str(row["image_id"]), []).append(row)
    gaps = pd.read_parquet(strict_gaps)
    required = {"image_id", "filepath", "gap_type", "bbox", "class"}
    if not required.issubset(gaps):
        raise ValueError(f"strict gaps lack {sorted(required - set(gaps))}")
    rows = []
    skipped_unrouted: Counter[str] = Counter()
    observed_datasets: set[str] = set()
    for gap in gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].to_dict("records"):
        image_id, box = str(gap["image_id"]), _xyxy(gap["bbox"])
        if image_id not in images:
            raise ValueError(f"FN image id is absent from KPI COCO: {image_id}")
        image = images[image_id]
        annotation_image_id = str(image["id"])
        ranked = sorted(((_iou(box, row), row)
                         for row in annotations.get(annotation_image_id, [])),
                        key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.999:
            raise ValueError(f"FN box has no exact KPI annotation match: {image_id} {box}")
        annotation = ranked[0][1]
        metadata = {}
        for source in (image, image.get("deft_od_aoi", {}), annotation,
                       annotation.get("deft_od_aoi", {})):
            metadata.update({key: source[key] for key in FIELDS if key in source})
        dataset = str(metadata.get("dataset_id") or "").strip()
        if not dataset:
            raise ValueError(
                f"KPI FN {image_id} has no dataset_id; every KPI sample must retain its "
                "real nonempty dataset_id. synthesis.routes is an allowlist, and a missing "
                "dataset_id is malformed metadata, not a synthesis opt-out"
            )
        observed_datasets.add(dataset)
        if dataset not in routes:
            skipped_unrouted[dataset] += 1
            continue
        missing = [key for key in FIELDS[1:] if not str(metadata.get(key) or "").strip()]
        if missing:
            raise ValueError(f"FN metadata is incomplete for {image_id}: {missing}")
        mask = Path(str(metadata["fn_mask_source"])).expanduser().resolve()
        if not mask.is_file():
            raise ValueError(f"FN pixel mask is missing: {mask}")
        metadata["fn_mask_source"] = str(mask)
        source_path = _source(images_root, image)
        gap_path = Path(str(gap["filepath"])).expanduser().resolve()
        try:
            same_file = source_path.samefile(gap_path)
        except OSError:
            same_file = False
        if not same_file:
            raise ValueError(f"gap filepath does not match frozen KPI image: {image_id}")
        rows.append({**gap, "filepath": str(source_path), "split": "kpi", **metadata,
                     "anomaly_type": f"{metadata['texture_id']}+{metadata['defect_class']}"})
    output.mkdir(parents=True)
    if not rows:
        configured_routes = sorted(map(str, routes))
        skipped_count = sum(skipped_unrouted.values())
        report = {
            "status": "SKIPPED", "reason": "no_routed_false_negatives", "fn_count": 0,
            "skipped_unrouted_fn_count": skipped_count,
            "skipped_unrouted_by_dataset": dict(sorted(skipped_unrouted.items())),
            "observed_dataset_ids": sorted(observed_datasets),
            "configured_route_keys": configured_routes,
            "message": (
                f"Skipped synthesis: all {skipped_count} strict false negatives use "
                f"unrouted dataset IDs {sorted(observed_datasets)}; configured synthesis "
                f"routes are {configured_routes}."
            ),
            "config": "",
        }
        (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
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
    report = {
        "status": "COMPLETE",
        "fn_count": len(rows),
        "skipped_unrouted_fn_count": sum(skipped_unrouted.values()),
        "skipped_unrouted_by_dataset": dict(sorted(skipped_unrouted.items())),
        "observed_dataset_ids": sorted(observed_datasets),
        "configured_route_keys": sorted(map(str, routes)),
        "config": str(config_path.resolve()),
    }
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
