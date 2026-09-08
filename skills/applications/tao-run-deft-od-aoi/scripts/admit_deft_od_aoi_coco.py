#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit mined source images and publish cumulative real-only binary COCO."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    if bool(synthetic_coco) != bool(synthetic_images):
        raise ValueError("pass both synthetic COCO and synthetic images, or neither")
    if synthetic_coco and synthetic_images:
        document = json.loads(synthetic_coco.read_text())
        synthetic_annotations: dict[int, list[dict[str, Any]]] = {}
        for row in document.get("annotations", []):
            synthetic_annotations.setdefault(int(row["image_id"]), []).append(row)
        limit = int(real_total * float(policy["synthesis"]["cumulative_fraction_of_real_defects"]))
        room = max(0, limit - by_kind["synthetic_defect"])
        for image in document.get("images", []):
            source = Path(str(image["file_name"]))
            if not source.is_file():
                source = synthetic_images / source.name
            rows = synthetic_annotations.get(int(image["id"]), [])
            if (synthetic_admitted >= room or not rows or not source.is_file()
                    or str(source.resolve()) in existing_sources):
                continue
            append(source.resolve(), image, rows, "synthetic_defect", None)
            existing_sources.add(str(source.resolve()))
            synthetic_admitted += 1
            admitted_rows.append({"source_filepath": str(source.resolve()),
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
