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
from PIL import Image


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
        image.convert("RGB").crop(box).save(output, format="PNG")


def _embedding_spec(policy: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    retrieval = policy["retrieval"]
    return {"input_parquet": str(input_path), "output_parquet": str(output),
            "model": retrieval["model"], "model_path": retrieval["model_path"],
            "model_config_path": "", "batch_size": 64}


def _path_values(path: Path, label: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} exclusion input is missing: {path}")
    frame = pd.read_parquet(path)
    if "filepath" not in frame:
        raise ValueError(f"{label} exclusion input lacks filepath")
    return {str(Path(value).expanduser().resolve()) for value in frame.filepath.astype(str)}


def _history_sources(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"previous COCO is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value.get("images"), list):
        raise ValueError("previous COCO lacks images")
    return {
        str(Path(str(row.get("source_path") or path.parent / "images" / row["file_name"])).resolve())
        for row in value["images"]
    }


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
            with Image.open(source_path) as image:
                width, height = image.size
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


def queries(policy_path: Path, strict_path: Path, loose_path: Path, iteration: int,
            output: Path, candidate_root: Path, real_factor: int | None,
            previous_coco: Path | None = None,
            exclusion_paths: dict[str, Path] | None = None) -> dict[str, Any]:
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
    history = _history_sources(previous_coco)
    configured_exclusions = exclusion_paths or {}
    counts, frames, role_status = {}, {}, {}
    for role in ("real", "clean"):
        rows = []
        for index, (_, reason, event) in enumerate(item for item in events if item[0] == role):
            source = Path(str(event["filepath"])).resolve()
            with Image.open(source) as image:
                box = _gap_box(event["bbox"], image.width, image.height,
                               float(policy["retrieval"]["defect_context_scale"]))
            query_id = f"iter{iteration}-{role}-" + _id(source, event["bbox"], reason, index)
            crop = output / "crops" / role / f"{query_id}.png"
            _crop(source, box, crop)
            rows.append({"filepath": str(crop), "query_id": query_id, "role": role,
                         "reason": reason, "source_filepath": str(source),
                         "source_bbox": event["bbox"], "best_iou": float(event["best_iou"])})
        counts[role], frames[role] = len(rows), pd.DataFrame(rows)
        if not rows:
            role_status[role] = {"status": "NO_QUERIES", "query_count": 0,
                                 "candidate_count": 0, "excluded_count": 0,
                                 "remaining_candidate_count": 0}
            continue
        parquet = output / f"{role}_queries.parquet"
        frames[role].to_parquet(parquet, index=False)
        candidate_path = candidate_root / f"{role}_candidates.parquet"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"{role} candidate manifest is missing: {candidate_path}")
        candidates = pd.read_parquet(candidate_path)
        required_candidates = {"filepath", "source_filepath"}
        if candidates.empty or not required_candidates.issubset(candidates):
            raise ValueError(
                f"{role} candidate manifest lacks usable {sorted(required_candidates)}"
            )
        candidate_files = {
            str(Path(value).expanduser().resolve()) for value in candidates.filepath.astype(str)
        }
        candidate_sources = {
            str(Path(value).expanduser().resolve()) for value in candidates.source_filepath.astype(str)
        }
        explicit = set()
        if role in configured_exclusions:
            explicit = _path_values(configured_exclusions[role], role)
            unmatched = explicit - candidate_files - candidate_sources
            if unmatched:
                raise ValueError(
                    f"{role} exclusion input contains {len(unmatched)} paths absent from candidates"
                )
        excluded = candidates[
            candidates.source_filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve()) in history | explicit
            )
            | candidates.filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve()) in explicit
            )
        ]
        exclusion_file = output / f"exclude_{role}_candidates.parquet"
        excluded[["filepath"]].drop_duplicates().to_parquet(exclusion_file, index=False)
        remaining = len(candidate_files - {
            str(Path(value).expanduser().resolve()) for value in excluded.filepath.astype(str)
        })
        role_status[role] = {
            "status": "READY" if remaining else "EXHAUSTED",
            "query_count": len(rows), "candidate_count": len(candidate_files),
            "excluded_count": int(excluded.filepath.nunique()),
            "remaining_candidate_count": remaining,
            "exclusion_manifest": str(exclusion_file.resolve()),
            "history_source_count": len(history & candidate_sources),
            "explicit_exclusion_count": len(explicit),
        }
        if not remaining:
            continue
        embedded = output / f"{role}_query_embeddings.parquet"
        (output / f"embed_{role}_queries.yaml").write_text(
            yaml.safe_dump(_embedding_spec(policy, parquet, embedded), sort_keys=False)
        )
        routing = policy["routing"]
        factor = (real_factor or int(routing["real_mine_factor_min"])) if role == "real" else int(routing["clean_factor"])
        desired = min(remaining, max(1, len(rows) * factor))
        mining = {"source_path": str(candidate_root / f"{role}_candidate_embeddings.parquet"),
                  "target_path": str(embedded), "output_dir": str(output / f"mine_{role}"),
                  "desired_unique_count": desired, "allocation_policy": "global",
                  "distance_metric": "cosine", "candidate_expansion_factor": int(policy["retrieval"]["candidate_overfetch"])}
        if role_status[role]["excluded_count"]:
            mining["exclude_path"] = str(exclusion_file.resolve())
        (output / f"mine_{role}.yaml").write_text(yaml.safe_dump(mining, sort_keys=False))
    enabled = [role for role, evidence in role_status.items() if evidence["status"] == "READY"]
    synthesis_pending = bool(policy.get("synthesis", {}).get("enabled")) and any(
        strict.gap_type.astype(str).str.upper().eq("FN")
    )
    report = {"status": "COMPLETE", "iteration": iteration, "query_counts": counts,
              "enabled_roles": enabled, "role_status": role_status,
              "converged": not enabled and not synthesis_pending,
              "synthesis_pending": synthesis_pending}
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
    query.add_argument("--real-exclusions", type=Path)
    query.add_argument("--clean-exclusions", type=Path)
    args = parser.parse_args()
    result = (candidates(args.policy.resolve(), args.output_dir.resolve()) if args.command == "candidates"
              else queries(args.policy.resolve(), args.strict_gaps.resolve(), args.loose_gaps.resolve(),
                           args.iteration, args.output_dir.resolve(), args.candidate_root.resolve(),
                           args.real_factor,
                           args.previous_coco.resolve() if args.previous_coco else None,
                           {role: getattr(args, f"{role}_exclusions").resolve()
                            for role in ("real", "clean")
                            if getattr(args, f"{role}_exclusions", None)}))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
