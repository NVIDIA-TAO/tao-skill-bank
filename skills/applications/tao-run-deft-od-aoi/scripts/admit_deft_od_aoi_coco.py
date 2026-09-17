#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit mined source images and publish cumulative real-only binary COCO."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _vectors(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or width-mismatched")
    matrix = np.stack(rows)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norm == 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norm


def _selected(role: str, candidate_root: Path, retrieval_root: Path,
              minimum: float) -> list[dict[str, Any]]:
    mined = retrieval_root / f"mine_{role}" / "final_unique_files.parquet"
    if not mined.is_file():
        raise FileNotFoundError(f"enabled {role} mining output is missing: {mined}")
    chosen = pd.read_parquet(mined)
    if chosen.empty or "filepath" not in chosen:
        raise ValueError(f"enabled {role} mining output is empty")
    candidates = pd.read_parquet(candidate_root / f"{role}_candidate_embeddings.parquet")
    queries = pd.read_parquet(retrieval_root / f"{role}_query_embeddings.parquet")
    required = {"filepath", "source_filepath", "source_image_id", "embedding"}
    if not required.issubset(candidates) or not {"filepath", "embedding"}.issubset(queries):
        raise ValueError(f"{role} embedding outputs lack routing columns")
    candidates = chosen[["filepath"]].merge(candidates, on="filepath", how="left", validate="one_to_one")
    if candidates.embedding.isna().any():
        raise ValueError(f"{role} mined paths do not match candidate embeddings")
    candidate_vectors, query_vectors = _vectors(candidates.embedding), _vectors(queries.embedding)
    if candidate_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(f"{role} candidate/query embedding widths differ")
    candidates["similarity"] = (candidate_vectors @ query_vectors.T).max(axis=1)
    candidates = candidates[candidates.similarity >= minimum].sort_values(
        ["similarity", "source_filepath"], ascending=[False, True]
    )
    return candidates.drop_duplicates("source_filepath").to_dict("records")


def _source_index(policy: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    source = policy["sources"][role]
    images = Path(source["images"])
    coco = json.loads(Path(source["coco"]).read_text())
    annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    result = {}
    for image in coco["images"]:
        path = Path(str(image.get("source_path") or "")) if image.get("source_path") else (
            images / str(image["file_name"])
        )
        result[str(path.resolve())] = {"image": image,
                                      "annotations": annotations.get(int(image["id"]), [])}
    return result


def _place(source: Path, images: Path, mode: str) -> Path:
    name = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:20] + source.suffix.lower()
    target = images / name
    if target.exists():
        return target
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        os.link(source, target)
    return target


def _synthetic_candidates(document: dict[str, Any], images_root: Path,
                          minimum_area: float, maximum_aspect: float
                          ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    categories = document.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise ValueError("synthetic COCO must declare exactly one category")
    category_id = categories[0].get("id")
    annotations: dict[int, list[dict[str, Any]]] = {}
    for row in document.get("annotations", []):
        if row.get("category_id") != category_id:
            raise ValueError("synthetic COCO contains an unexpected category")
        annotations.setdefault(int(row["image_id"]), []).append(row)
    quality = collections.Counter({
        "input_images": 0, "input_annotations": 0,
        "rejected_annotations_small": 0, "rejected_annotations_aspect": 0,
        "rejected_annotations_full_frame": 0,
        "rejected_images_no_eligible_boxes": 0,
    })
    candidates = []
    for image in document.get("images", []):
        width, height = int(image["width"]), int(image["height"])
        if width < 1 or height < 1:
            raise ValueError("synthetic COCO image has invalid dimensions")
        quality["input_images"] += 1
        rows = []
        for row in annotations.get(int(image["id"]), []):
            quality["input_annotations"] += 1
            try:
                x, y, box_width, box_height = map(float, row["bbox"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("synthetic COCO contains an invalid bbox") from error
            if (not all(math.isfinite(value) for value in (x, y, box_width, box_height))
                    or min(x, y) < 0 or box_width <= 0 or box_height <= 0
                    or x + box_width > width or y + box_height > height):
                raise ValueError("synthetic COCO bbox is invalid or outside its image")
            if box_width * box_height < minimum_area:
                quality["rejected_annotations_small"] += 1
                continue
            if max(box_width / box_height, box_height / box_width) > maximum_aspect:
                quality["rejected_annotations_aspect"] += 1
                continue
            if x == 0 and y == 0 and box_width == width and box_height == height:
                quality["rejected_annotations_full_frame"] += 1
                continue
            rows.append(row)
        if not rows:
            quality["rejected_images_no_eligible_boxes"] += 1
            continue
        source = Path(str(image.get("source_path") or image["file_name"]))
        if not source.is_absolute():
            source = images_root / source
        if not source.is_file():
            raise FileNotFoundError(f"synthetic image is missing: {source}")
        candidates.append({"source": source.resolve(), "image": image, "rows": rows,
                           "stratum": str(image.get("dataset_id") or "unknown")})
    quality["eligible_images"] = len(candidates)
    quality["eligible_annotations"] = sum(len(row["rows"]) for row in candidates)
    return candidates, dict(quality)


def _stratified_synthetic(candidates: list[dict[str, Any]], limit: int
                          ) -> list[dict[str, Any]]:
    if limit <= 0 or not candidates:
        return []
    unique = {str(row["source"]): row for row in candidates}
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in unique.values():
        groups[row["stratum"]].append(row)
    if limit >= len(unique):
        return [unique[key] for key in sorted(unique)]
    exact = {name: limit * len(rows) / len(unique) for name, rows in groups.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remaining = limit - sum(allocation.values())
    for name in sorted(groups, key=lambda value: (-(exact[value] % 1), value)):
        if remaining <= 0:
            break
        allocation[name] += 1
        remaining -= 1
    selected = []
    for name, rows in groups.items():
        ranked = sorted(rows, key=lambda row: (
            hashlib.sha256(str(row["source"]).encode()).hexdigest(), str(row["source"]),
        ))
        selected.extend(ranked[:allocation[name]])
    return sorted(selected, key=lambda row: str(row["source"]))


def admit(policy_path: Path, candidate_root: Path, retrieval_root: Path, output: Path,
          previous_path: Path | None, mode: str, synthetic_coco: Path | None = None,
          synthetic_images: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    manifest = json.loads((retrieval_root / "query_manifest.json").read_text())
    enabled = set(manifest["enabled_roles"])
    additions = {role: _selected(role, candidate_root, retrieval_root,
                                  float(policy["retrieval"]["minimum_similarity"]))
                 for role in enabled}
    previous = ({"images": [], "annotations": [], "categories": []} if previous_path is None
                else json.loads(previous_path.read_text()))
    existing_sources = {str(row.get("source_path") or Path(str(row["file_name"])).resolve())
                        for row in previous.get("images", [])}
    by_kind = {kind: sum(row.get("deft_kind") == kind for row in previous.get("images", []))
               for kind in ("real_defect", "clean_negative", "synthetic_defect")}
    for role in additions:
        additions[role] = [row for row in additions[role]
                           if str(Path(row["source_filepath"]).resolve()) not in existing_sources]
    real_total = by_kind["real_defect"] + len(additions.get("real", []))
    clean_limit = int(real_total * float(policy["routing"]["clean_cumulative_cap_per_real"]))
    clean_room = max(0, clean_limit - by_kind["clean_negative"])
    additions["clean"] = additions.get("clean", [])[:clean_room]
    if enabled and not any(additions.get(role) for role in enabled) and not previous.get("images"):
        raise ValueError("mining admitted no source images")

    output.mkdir(parents=True)
    images_root = output / "images"
    images_root.mkdir()
    images, annotations = [], []
    next_image = next_annotation = 1

    def append(source: Path, image: dict[str, Any], rows: list[dict[str, Any]], kind: str,
               similarity: float | None) -> None:
        nonlocal next_image, next_annotation
        target = _place(source, images_root, mode)
        images.append({**image, "id": next_image, "file_name": target.name,
                       "source_path": str(source.resolve()), "deft_kind": kind,
                       "retrieval_similarity": similarity})
        for row in rows:
            annotations.append({**row, "id": next_annotation, "image_id": next_image,
                                "category_id": 1})
            next_annotation += 1
        next_image += 1

    old_annotations: dict[int, list[dict[str, Any]]] = {}
    for row in previous.get("annotations", []):
        old_annotations.setdefault(int(row["image_id"]), []).append(row)
    for image in previous.get("images", []):
        source = Path(str(image.get("source_path") or image["file_name"]))
        if not source.is_file() and previous_path:
            source = previous_path.parent / "images" / Path(str(image["file_name"])).name
        append(source, image, old_annotations.get(int(image["id"]), []),
               str(image["deft_kind"]), image.get("retrieval_similarity"))
    admitted_rows = []
    for role, kind in (("real", "real_defect"), ("clean", "clean_negative")):
        index = _source_index(policy, role)
        for selected in additions.get(role, []):
            source = Path(str(selected["source_filepath"])).resolve()
            if str(source) not in index:
                raise ValueError(f"selected {role} source is absent from its frozen COCO: {source}")
            row = index[str(source)]
            append(source, row["image"], row["annotations"], kind, float(selected["similarity"]))
            admitted_rows.append({"source_filepath": str(source), "kind": kind,
                                  "similarity": float(selected["similarity"])})
    synthetic_admitted = 0
    synthetic_quality: dict[str, int] = {}
    synthetic_requested = 0
    admitted_by_stratum: dict[str, int] = {}
    if bool(synthetic_coco) != bool(synthetic_images):
        raise ValueError("pass both synthetic COCO and synthetic images, or neither")
    if synthetic_coco and synthetic_images:
        document = json.loads(synthetic_coco.read_text())
        admission = policy.get("admission") or {}
        candidates, synthetic_quality = _synthetic_candidates(
            document, synthetic_images,
            float(admission.get("minimum_box_area_px", 64)),
            float(admission.get("maximum_box_aspect", 25.0)),
        )
        unique = {}
        for candidate in candidates:
            key = str(candidate["source"])
            if key in existing_sources:
                continue
            if key in unique and unique[key]["rows"] != candidate["rows"]:
                raise ValueError(f"conflicting duplicate synthetic source: {key}")
            unique.setdefault(key, candidate)
        synthetic_requested = len(unique)
        limit = int(real_total * float(policy["synthesis"]["cumulative_fraction_of_real_defects"]))
        room = max(0, limit - by_kind["synthetic_defect"])
        admitted = _stratified_synthetic(list(unique.values()), room)
        admitted_by_stratum = dict(sorted(collections.Counter(
            row["stratum"] for row in admitted
        ).items()))
        for candidate in admitted:
            source = candidate["source"]
            append(source, candidate["image"], candidate["rows"], "synthetic_defect", None)
            existing_sources.add(str(source))
            synthetic_admitted += 1
            admitted_rows.append({"source_filepath": str(source),
                                  "kind": "synthetic_defect", "similarity": None})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "defect"}]}
    _json(output / "train.json", coco)
    pd.DataFrame(admitted_rows, columns=["source_filepath", "kind", "similarity"]).to_parquet(
        output / "admitted_sources.parquet", index=False
    )
    report = {"status": "COMPLETE", "iteration": int(manifest["iteration"]),
              "retained_previous_images": len(previous.get("images", [])),
              "admitted": {"real": len(additions.get("real", [])),
                           "clean": len(additions.get("clean", [])),
                           "synthetic": synthetic_admitted},
              "total_images": len(images), "total_annotations": len(annotations),
              "by_kind": {kind: sum(row["deft_kind"] == kind for row in images)
                          for kind in ("real_defect", "clean_negative", "synthetic_defect")},
              "synthetic_admission": {
                  "requested_new": synthetic_requested,
                  "admitted_new": synthetic_admitted,
                  "excluded_by_cap": synthetic_requested - synthetic_admitted,
                  "admitted_by_stratum": admitted_by_stratum,
                  "quality_filter": synthetic_quality,
              },
              "training_pool_mutated": False}
    _json(output / "admission_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-coco", type=Path)
    parser.add_argument("--synthetic-coco", type=Path)
    parser.add_argument("--synthetic-images", type=Path)
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    result = admit(args.policy.resolve(), args.candidate_root.resolve(),
                   args.retrieval_root.resolve(), args.output_dir.resolve(),
                   args.previous_coco.resolve() if args.previous_coco else None, args.link_mode,
                   args.synthetic_coco.resolve() if args.synthetic_coco else None,
                   args.synthetic_images.resolve() if args.synthetic_images else None)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
