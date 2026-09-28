#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare reusable candidate crops or per-iteration DEFT AOI gap queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageOps


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _id(*values: Any) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()[:16]


def _source(images: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("source_path") or "")) if row.get("source_path") else (
        images / str(row["file_name"])
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _box(value: Any, width: int, height: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    x, y, w, h = map(float, value)
    if min(x, y) < 0 or w <= 0 or h <= 0:
        raise ValueError(f"invalid xywh box: {value}")
    cx, cy, w, h = x + w / 2, y + h / 2, w * scale, h * scale
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(width, int(cx + w / 2 + 0.999)), min(height, int(cy + h / 2 + 0.999))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box clips empty: {value}")
    return x1, y1, x2, y2


def _gap_box(value: Any, width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid xyxy gap box: {value}")
    return _box((x1, y1, x2 - x1, y2 - y1), width, height, scale)


def _crop(source: Path, box: tuple[int, int, int, int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).convert("RGB").crop(box).save(output, format="PNG")


def _size(source: Path) -> tuple[int, int]:
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).size


def _embedding_spec(policy: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    retrieval = policy["retrieval"]
    return {"input_parquet": str(input_path), "output_parquet": str(output),
            "model": retrieval["model"], "model_path": retrieval["model_path"],
            "model_config_path": "", "batch_size": 64}


def _kpi_pockets(policy: dict[str, Any]) -> dict[str, dict[str, str]]:
    source = policy["sources"]["kpi"]
    images = Path(source["images"])
    document = json.loads(Path(source["coco"]).read_text())
    result = {}
    for row in document["images"]:
        path = _source(images, row)
        metadata = row.get("deft_od_aoi") or {}
        values = {
            "dataset": str(metadata.get("benchmark") or row.get("benchmark") or "unknown"),
            "texture": str(metadata.get("texture") or "unknown"),
            "defect": str(metadata.get("defect_type") or "unknown"),
        }
        values["pocket"] = "/".join((values["dataset"], values["texture"], values["defect"]))
        result[str(path)] = values
    return result


def candidates(policy_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    output.mkdir(parents=True)
    result: dict[str, int] = {}
    for role in ("real", "clean"):
        source = policy["sources"][role]
        images = Path(source["images"])
        coco = json.loads(Path(source["coco"]).read_text())
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        rows = []
        for image_row in coco["images"]:
            source_path = _source(images, image_row)
            width, height = _size(source_path)
            if role == "real":
                for annotation in annotations.get(int(image_row["id"]), []):
                    box = _box(annotation["bbox"], width, height,
                               float(policy["retrieval"]["defect_context_scale"]))
                    crop_id = "real-" + _id(source_path, annotation["id"], box)
                    crop = output / "crops" / role / f"{crop_id}.png"
                    _crop(source_path, box, crop)
                    rows.append({"filepath": str(crop), "source_filepath": str(source_path),
                                 "source_image_id": int(image_row["id"]), "role": role,
                                 "candidate_id": crop_id, "source_bbox": annotation["bbox"]})
            else:
                for grid in policy["retrieval"]["clean_grids"]:
                    grid = int(grid)
                    if grid < 1:
                        raise ValueError("clean grid values must be positive")
                    for row in range(grid):
                        for column in range(grid):
                            box = (column * width // grid, row * height // grid,
                                   (column + 1) * width // grid, (row + 1) * height // grid)
                            crop_id = "clean-" + _id(source_path, grid, row, column)
                            crop = output / "crops" / role / f"{crop_id}.png"
                            _crop(source_path, box, crop)
                            rows.append({"filepath": str(crop), "source_filepath": str(source_path),
                                         "source_image_id": int(image_row["id"]), "role": role,
                                         "candidate_id": crop_id, "source_bbox": None})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"{role} produced no candidate crops")
        parquet = output / f"{role}_candidates.parquet"
        frame.to_parquet(parquet, index=False)
        spec = _embedding_spec(policy, parquet, output / f"{role}_candidate_embeddings.parquet")
        (output / f"embed_{role}_candidates.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        result[role] = len(frame)
    _json(output / "candidate_manifest.json", {"status": "COMPLETE", "counts": result,
                                                "encoder": policy["retrieval"]})
    return result


def _previous_sources(previous_path: Path | None) -> dict[str, set[Path]]:
    result = {"real": set(), "clean": set()}
    if previous_path is None:
        return result
    previous = json.loads(previous_path.read_text())
    kinds = {"real_defect": "real", "clean_negative": "clean"}
    for row in previous.get("images", []):
        role = kinds.get(str(row.get("deft_kind") or ""))
        raw = row.get("original_source_path") or row.get("source_path")
        if role and raw:
            result[role].add(Path(str(raw)).resolve())
    return result


def _exclude_candidates(candidate_root: Path, role: str, prior_sources: set[Path],
                        output: Path) -> tuple[Path | None, int, int]:
    if not prior_sources:
        return None, 0, 0
    candidates = pd.read_parquet(candidate_root / f"{role}_candidates.parquet")
    required = {"filepath", "source_filepath"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"{role} candidates lack {sorted(required - set(candidates.columns))}")
    source_paths = candidates.source_filepath.map(lambda value: Path(str(value)).resolve())
    excluded = candidates.loc[source_paths.isin(prior_sources), ["filepath"]].drop_duplicates()
    if excluded.empty:
        return None, 0, 0
    exclude_path = output / f"exclude_{role}_candidate_crops.parquet"
    excluded.to_parquet(exclude_path, index=False)
    matched_sources = set(source_paths[source_paths.isin(prior_sources)])
    return exclude_path, len(excluded), len(matched_sources)


def queries(policy_path: Path, strict_path: Path, loose_path: Path, iteration: int,
            output: Path, candidate_root: Path, real_factor: int | None,
            previous_path: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    strict, loose = pd.read_parquet(strict_path), pd.read_parquet(loose_path)
    required = {"filepath", "gap_type", "bbox", "best_iou"}
    for label, frame in (("strict", strict), ("loose", loose)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} gaps lack {sorted(required - set(frame.columns))}")
    events = []
    for row in strict[strict.gap_type.astype(str).str.upper().eq("FN")].to_dict("records"):
        events.append(("real", "fn", row))
    gap = policy["gap"]
    for row in loose[loose.gap_type.astype(str).str.upper().eq("FP")].to_dict("records"):
        iou = float(row["best_iou"])
        if iou < gap["background_iou_upper"]:
            events.append(("clean", "background_fp", row))
        elif iou < gap["near_miss_iou_upper"]:
            events.append(("real", "near_miss_fp", row))
    output.mkdir(parents=True)
    pockets = _kpi_pockets(policy)
    counts, frames, targets, requested = {}, {}, {}, {}
    prior_sources = _previous_sources(previous_path)
    excluded_candidate_crops, excluded_source_images = {}, {}
    for role in ("real", "clean"):
        rows = []
        for index, (_, reason, event) in enumerate(item for item in events if item[0] == role):
            source = Path(str(event["filepath"])).resolve()
            if str(source) not in pockets:
                raise ValueError(f"gap image is absent from the frozen KPI role: {source}")
            if role == "real" and "unknown" in pockets[str(source)].values():
                raise ValueError(f"gap image lacks frozen pocket metadata: {source}")
            width, height = _size(source)
            box = _gap_box(event["bbox"], width, height,
                           float(policy["retrieval"]["defect_context_scale"]))
            query_id = f"iter{iteration}-{role}-" + _id(source, event["bbox"], reason, index)
            crop = output / "crops" / role / f"{query_id}.png"
            _crop(source, box, crop)
            rows.append({"filepath": str(crop), "query_id": query_id, "role": role,
                         "reason": reason, "source_filepath": str(source),
                         "source_bbox": event["bbox"], "best_iou": float(event["best_iou"]),
                         **pockets[str(source)]})
        counts[role], frames[role] = len(rows), pd.DataFrame(rows)
        if not rows:
            continue
        parquet = output / f"{role}_queries.parquet"
        frames[role].to_parquet(parquet, index=False)
        embedded = output / f"{role}_query_embeddings.parquet"
        (output / f"embed_{role}_queries.yaml").write_text(
            yaml.safe_dump(_embedding_spec(policy, parquet, embedded), sort_keys=False)
        )
        routing = policy["routing"]
        if role == "real":
            factor = real_factor or int(routing["real_mine_factor_min"])
            if not int(routing["real_mine_factor_min"]) <= factor <= int(routing["real_mine_factor_max"]):
                raise ValueError("real factor is outside the frozen policy range")
            strict_count = sum(row["reason"] == "fn" for row in rows)
            near = pd.DataFrame(row for row in rows if row["reason"] == "near_miss_fp")
            near_target = (0 if near.empty else sum(
                min(len(group) * int(routing["near_miss_real_factor"]),
                    int(routing["near_miss_real_cap"]))
                for _, group in near.groupby("pocket")
            ))
            targets[role] = {"fn": strict_count * factor, "near_miss_fp": near_target}
        else:
            targets[role] = {"background_fp": len(rows) * int(routing["clean_factor"])}
        desired = sum(targets[role].values())
        candidate_count = len(pd.read_parquet(candidate_root / f"{role}_candidates.parquet"))
        requested[role] = min(candidate_count, desired * int(policy["retrieval"]["candidate_overfetch"]))
        mining = {"source_path": str(candidate_root / f"{role}_candidate_embeddings.parquet"),
                  "target_path": str(embedded), "output_dir": str(output / f"mine_{role}"),
                  "desired_unique_count": requested[role], "allocation_policy": "global",
                  "distance_metric": "cosine", "candidate_expansion_factor": int(policy["retrieval"]["candidate_overfetch"])}
        exclude_path, crop_count, source_count = _exclude_candidates(
            candidate_root, role, prior_sources[role], output
        )
        excluded_candidate_crops[role] = crop_count
        excluded_source_images[role] = source_count
        if exclude_path is not None:
            mining["exclude_path"] = str(exclude_path)
        (output / f"mine_{role}.yaml").write_text(yaml.safe_dump(mining, sort_keys=False))
    report = {"status": "COMPLETE", "iteration": iteration, "query_counts": counts,
              "enabled_roles": [role for role, count in counts.items() if count],
              "admission_targets": targets, "requested_crop_counts": requested,
              "excluded_candidate_crops": excluded_candidate_crops,
              "excluded_source_images": excluded_source_images}
    _json(output / "query_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidates")
    candidate.add_argument("--policy", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    query = sub.add_parser("queries")
    query.add_argument("--policy", type=Path, required=True)
    query.add_argument("--strict-gaps", type=Path, required=True)
    query.add_argument("--loose-gaps", type=Path, required=True)
    query.add_argument("--iteration", type=int, required=True)
    query.add_argument("--output-dir", type=Path, required=True)
    query.add_argument("--candidate-root", type=Path, required=True)
    query.add_argument("--real-factor", type=int)
    query.add_argument("--previous-coco", type=Path)
    args = parser.parse_args()
    result = (candidates(args.policy.resolve(), args.output_dir.resolve()) if args.command == "candidates"
              else queries(args.policy.resolve(), args.strict_gaps.resolve(), args.loose_gaps.resolve(),
                           args.iteration, args.output_dir.resolve(), args.candidate_root.resolve(),
                           args.real_factor,
                           args.previous_coco.resolve() if args.previous_coco else None))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
