#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze FN masks and embedding inputs for AnomalyGenNext preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IDENTITY_COLUMNS = {
    "dataset_id", "texture_id", "defect_class", "anomaly_type", "fn_mask_source"
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_id(*values: Any) -> str:
    text = "\0".join(str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _images(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _defect_specs(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("defect_type") or row.get("anomaly_type") or row.get("name") or "").strip()
        if not name or name in result:
            raise ValueError("defect_spec rows need unique anomaly_type values")
        if row.get("spatial_dependency") == "text" and not str(
            row.get("roi_prompt_defect_location") or ""
        ).strip():
            raise ValueError(f"text-routed defect lacks roi_prompt_defect_location: {name}")
        result[name] = row
    return result


def _recipe_types(path: Path) -> set[str]:
    value = yaml.safe_load(path.read_text())
    rows = value.get("anomaly_types") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"recipe lacks anomaly_types: {path}")
    return {f"{row[0]}+{row[1]}" for row in rows if isinstance(row, list) and len(row) == 2}


def _isolate_mask(mask_path: Path, image_path: Path, bbox: Any, output: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"))
        if mask.shape != (image.height, image.width):
            raise ValueError(f"mask/image dimensions differ: {mask_path}")
    box = np.asarray(bbox, dtype=float).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError(f"invalid bbox: {bbox!r}")
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(mask.shape[1], int(np.ceil(x2))), min(mask.shape[0], int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty clipped bbox: {bbox!r}")
    isolated = np.zeros_like(mask)
    isolated[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    if not np.any(isolated):
        raise ValueError(f"FN bbox contains no mask pixels: {mask_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(isolated).save(output)


def _select(rows: pd.DataFrame, selection: dict[str, Any]) -> pd.DataFrame:
    datasets = [str(value) for value in selection.get("datasets", [])]
    eligible = rows[rows.dataset_id.astype(str).isin(datasets)].copy()
    eligible = eligible.sort_values(["dataset_id", "image_id", "filepath", "fn_id"])
    mode = selection.get("mode", "all_eligible")
    if mode == "all_eligible":
        selected = eligible
    elif mode == "per_dataset":
        split, count = str(selection["split"]), int(selection.get("per_dataset", 1))
        selected = pd.concat(
            [eligible[(eligible.dataset_id == name) & (eligible.split == split)].head(count)
             for name in datasets], ignore_index=True
        )
        if any(len(eligible[(eligible.dataset_id == name) & (eligible.split == split)]) < count
               for name in datasets):
            raise ValueError("not enough eligible FNs for per_dataset selection")
    else:
        raise ValueError(f"unsupported selection.mode: {mode}")
    if selected.empty:
        raise ValueError("selection produced no eligible false negatives")
    return selected.reset_index(drop=True)


def prepare(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    gaps = pd.read_parquet(Path(config["gap_parquet"]).expanduser().resolve())
    required = {"image_id", "filepath", "gap_type", "bbox", "class", "split"} | IDENTITY_COLUMNS
    missing = sorted(required - set(gaps.columns))
    if missing:
        raise ValueError(f"gap parquet lacks normalized columns: {missing}")
    gaps = gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].copy()
    specs = _defect_specs(Path(config["defect_spec"]).expanduser().resolve())
    pool_root = Path(config["pool_dataset_root"]).expanduser().resolve()
    datasets = config.get("datasets") or {}
    recipe_types: dict[str, set[str]] = {}
    for name, dataset in datasets.items():
        checkpoint = Path(dataset["checkpoint"]).expanduser().resolve()
        recipe = Path(dataset["recipe"]).expanduser().resolve()
        if not checkpoint.is_file() or not recipe.is_file():
            raise FileNotFoundError(f"dataset {name} checkpoint/recipe is missing")
        recipe_types[str(name)] = _recipe_types(recipe)

    normalized: list[dict[str, Any]] = []
    for row in gaps.to_dict("records"):
        dataset, texture = str(row["dataset_id"]), str(row["texture_id"])
        defect, anomaly = str(row["defect_class"]), str(row["anomaly_type"])
        if dataset not in datasets or anomaly != f"{texture}+{defect}":
            continue
        image = Path(str(row["filepath"])).expanduser().resolve()
        mask = Path(str(row["fn_mask_source"])).expanduser().resolve()
        clean_dir = pool_root / texture / "clean_image"
        sampled_dir = pool_root / texture / "mask" / defect
        if anomaly not in specs or anomaly not in recipe_types[dataset]:
            continue
        if not image.is_file() or not mask.is_file() or not clean_dir.is_dir() or not sampled_dir.is_dir():
            continue
        if not _images(clean_dir) or not _images(sampled_dir):
            continue
        bbox_json = json.dumps(np.asarray(row["bbox"]).reshape(-1).tolist(), separators=(",", ":"))
        normalized.append({**row, "filepath": str(image), "fn_mask_source": str(mask),
                           "clean_dir": str(clean_dir), "sampled_dir": str(sampled_dir),
                           "bbox_json": bbox_json,
                           "fn_id": "fn-" + _stable_id(dataset, row["image_id"], image, bbox_json)})
    if not normalized:
        raise ValueError("no eligible normalized false negatives")
    selected = _select(pd.DataFrame(normalized), config["selection"])

    root = output / "prepared_anomalygennext_inputs"
    manifests, specs_dir = output / "manifests", output / "specs"
    masks_root = root / "source_masks"
    for directory in (manifests, specs_dir, masks_root):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root / "filtering_config.yaml")
    mask_rows, query_rows, clean_rows = [], [], []
    clean_seen: set[tuple[str, str]] = set()
    sampled_used: dict[str, set[Path]] = defaultdict(set)
    seed = int(config["selection"].get("mask_sample_seed", 42))
    for order, row in selected.iterrows():
        fn_id, anomaly = str(row.fn_id), str(row.anomaly_type)
        mask_dir = masks_root / fn_id
        isolated = mask_dir / f"{fn_id}__fn_mask.png"
        _isolate_mask(Path(row.fn_mask_source), Path(row.filepath), row.bbox, isolated)
        candidates = _images(Path(row.sampled_dir))
        rng = random.Random(int(_stable_id(seed, fn_id), 16))
        available = [path for path in candidates if path not in sampled_used[anomaly]] or candidates
        sampled = available[rng.randrange(len(available))]
        sampled_used[anomaly].add(sampled)
        sampled_output = mask_dir / f"{fn_id}__same_type_sampled_mask.png"
        with Image.open(sampled) as image:
            image.convert("L").save(sampled_output)
        for branch, path in (("fn_mask", isolated), ("same_type_sampled_mask", sampled_output)):
            mask_rows.append({"fn_id": fn_id, "query_order": order, "dataset_id": row.dataset_id,
                              "anomaly_type": anomaly, "branch": branch, "mask_path": str(path)})
        query_rows.append({"filepath": row.filepath, "fn_id": fn_id, "query_order": order,
                           "dataset_id": row.dataset_id, "pool_key": row.texture_id,
                           "anomaly_type": anomaly, "od_category": row["class"]})
        for clean in _images(Path(row.clean_dir)):
            key = (str(clean), anomaly)
            if key not in clean_seen:
                clean_seen.add(key)
                clean_rows.append({"filepath": str(clean), "dataset_id": row.dataset_id,
                                   "pool_key": row.texture_id,
                                   "anomaly_type_eligibility": anomaly})
    selected.to_parquet(manifests / "fn_queries.parquet", index=False)
    query_frame = pd.DataFrame(query_rows)
    query_frame.to_parquet(manifests / "selected_fn_queries.parquet", index=False)
    query_frame.drop_duplicates("filepath").to_parquet(
        manifests / "fn_embedding_inputs.parquet", index=False
    )
    pd.DataFrame(mask_rows).to_parquet(manifests / "mask_selection.parquet", index=False)
    clean_frame = pd.DataFrame(clean_rows)
    clean_frame.to_parquet(manifests / "clean_pool.parquet", index=False)
    embedding = config["embedding"]
    for name, input_path in (("clean", manifests / "clean_pool.parquet"),
                             ("fn", manifests / "fn_embedding_inputs.parquet")):
        spec = {"input_parquet": str(input_path),
                "output_parquet": str(output / "embeddings" / f"{name}_embeddings.parquet"),
                "model": embedding["model"], "model_path": embedding["model_path"],
                "model_config_path": "", "batch_size": int(embedding.get("batch_size", 64))}
        (specs_dir / f"{name}_embeddings.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    contract = {"status": "COMPLETE", "source_tag": config.get("source_tag", "user_provided"),
                "selected_fn_count": len(selected), "clean_image_count": len(clean_frame),
                "source_mask_count": len(mask_rows), "training_pool_mutated": False}
    _write_json(root / "input_contract.json", contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.config).resolve(), Path(args.output_dir).resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
