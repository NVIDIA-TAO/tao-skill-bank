#!/usr/bin/env python3
"""Read-only integrity and distribution audit for the LAM segmentation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # The audit remains useful if OpenCV is absent.
    cv2 = None


CLASS_VALUES = (0, 85, 170, 255)
CLASS_NAMES = ("background", "mask_height_1", "mask_height_2", "trench_depth")
SLICE_RE = re.compile(r"[-_](\d{1,2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquisition(stem: str) -> str:
    return SLICE_RE.sub("", stem)


def slice_index(stem: str) -> int | None:
    match = SLICE_RE.search(stem)
    return int(match.group(1)) if match else None


def material(stem: str) -> str:
    if "SiGe" in stem:
        return "SiGe"
    if "Si" in stem:
        return "Si"
    return "unknown"


def dhash(image: Image.Image) -> int:
    resampling = getattr(Image, "Resampling", Image)
    gray = image.convert("L").resize((9, 8), resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.uint8)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def normalize_mask(path: Path) -> tuple[np.ndarray, list[int], str]:
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        if not bool((raw == raw[..., :1]).all()):
            raise ValueError(f"mask channels differ: {path}")
        raw = raw[..., 0]
    values = sorted(int(value) for value in np.unique(raw))
    if set(values).issubset(CLASS_VALUES):
        lut = np.full(256, 255, dtype=np.uint8)
        for index, value in enumerate(CLASS_VALUES):
            lut[value] = index
        return lut[raw.astype(np.uint8)], values, "intensity"
    if set(values).issubset({0, 1, 2, 3}):
        return raw.astype(np.uint8), values, "index"
    return raw.astype(np.int32), values, "invalid"


def load_coco(path: Path) -> tuple[dict[str, dict[int, int]], dict]:
    if not path.is_file():
        return {}, {"present": False}
    document = json.loads(path.read_text())
    areas: dict[str, dict[int, int]] = {}
    for annotation in document.get("annotations", []):
        per_class = Counter()
        for segment in annotation.get("segments_info", []):
            per_class[int(segment["category_id"])] += int(segment["area"])
        per_class[0] = 1024 * 1024 - sum(per_class.values())
        areas[annotation["file_name"]] = dict(per_class)
    return areas, {
        "present": True,
        "images": len(document.get("images", [])),
        "annotations": len(document.get("annotations", [])),
        "categories": [
            {"id": row.get("id"), "name": row.get("name")}
            for row in document.get("categories", [])
        ],
    }


def components(mask: np.ndarray, class_id: int) -> int | None:
    if cv2 is None:
        return None
    count, _ = cv2.connectedComponents((mask == class_id).astype(np.uint8), 8)
    return int(count - 1)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def audit_split(root: Path, split: str) -> tuple[list[dict], dict]:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    images = sorted(image_dir.glob("*.png"))
    masks = {path.stem: path for path in mask_dir.glob("*.png")} if mask_dir.is_dir() else {}
    coco_areas, coco_summary = load_coco(root / f"annotations_{split}.json")
    rows: list[dict] = []
    totals = np.zeros(4, dtype=np.int64)
    invalid_values = Counter()
    shape_counts = Counter()
    modes = Counter()
    coco_mismatch_images = []

    for image_path in images:
        with Image.open(image_path) as opened:
            shape_counts[f"{opened.width}x{opened.height}"] += 1
            modes[opened.mode] += 1
            gray = np.asarray(opened.convert("L"), dtype=np.float32)
            image_row = {
                "split": split,
                "file_name": image_path.name,
                "acquisition": acquisition(image_path.stem),
                "material": material(image_path.stem),
                "slice_index": slice_index(image_path.stem),
                "image_width": opened.width,
                "image_height": opened.height,
                "image_mode": opened.mode,
                "image_sha256": sha256(image_path),
                "image_dhash": f"{dhash(opened):016x}",
                "intensity_mean": float(gray.mean()),
                "intensity_std": float(gray.std()),
                "intensity_p01": float(np.quantile(gray, 0.01)),
                "intensity_p99": float(np.quantile(gray, 0.99)),
                "mask_present": image_path.stem in masks,
            }
        mask_path = masks.get(image_path.stem)
        if mask_path is not None:
            normalized, values, encoding = normalize_mask(mask_path)
            image_row.update(
                {
                    "mask_sha256": sha256(mask_path),
                    "mask_height": int(normalized.shape[0]),
                    "mask_width": int(normalized.shape[1]),
                    "mask_values": ",".join(str(value) for value in values),
                    "mask_encoding": encoding,
                    "image_mask_shape_match": (
                        normalized.shape[:2]
                        == (image_row["image_height"], image_row["image_width"])
                    ),
                }
            )
            valid = encoding != "invalid"
            counts = np.bincount(normalized.reshape(-1), minlength=4)[:4] if valid else np.zeros(4, dtype=np.int64)
            totals += counts
            for value in values:
                if value not in CLASS_VALUES and value not in (0, 1, 2, 3):
                    invalid_values[value] += int((normalized == value).sum())
            pixel_total = int(normalized.size)
            for class_id, name in enumerate(CLASS_NAMES):
                image_row[f"pixels_{name}"] = int(counts[class_id])
                image_row[f"fraction_{name}"] = float(counts[class_id] / pixel_total)
                image_row[f"components_{name}"] = components(normalized, class_id) if valid else None
            if valid:
                horizontal = normalized[:, 1:] != normalized[:, :-1]
                vertical = normalized[1:, :] != normalized[:-1, :]
                image_row["boundary_transitions"] = int(horizontal.sum() + vertical.sum())
            expected = coco_areas.get(image_path.name)
            mismatches = {}
            if expected and valid:
                for class_id in range(4):
                    difference = int(counts[class_id]) - int(expected.get(class_id, 0))
                    if difference:
                        mismatches[str(class_id)] = difference
            image_row["coco_area_differences"] = json.dumps(mismatches, sort_keys=True)
            if mismatches:
                coco_mismatch_images.append(image_path.name)
        rows.append(image_row)

    image_names = {path.stem for path in images}
    mask_names = set(masks)
    total_pixels = int(totals.sum())
    summary = {
        "images": len(images),
        "masks": len(masks),
        "image_without_mask": sorted(image_names - mask_names),
        "mask_without_image": sorted(mask_names - image_names),
        "image_shapes": dict(shape_counts),
        "image_modes": dict(modes),
        "invalid_mask_values": dict(invalid_values),
        "class_pixels": {CLASS_NAMES[i]: int(totals[i]) for i in range(4)},
        "class_fractions": {
            CLASS_NAMES[i]: float(totals[i] / total_pixels) if total_pixels else None
            for i in range(4)
        },
        "coco": coco_summary,
        "coco_mask_mismatch_count": len(coco_mismatch_images),
        "coco_mask_mismatch_images": coco_mismatch_images,
        "acquisition_groups": len({row["acquisition"] for row in rows}),
        "materials": dict(Counter(row["material"] for row in rows)),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_duplicates(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_sha256"]].append(row)
    output = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        output.append(
            {
                "sha256": digest,
                "count": len(members),
                "splits": ",".join(sorted({row["split"] for row in members})),
                "files": json.dumps(
                    [f"{row['split']}/{row['file_name']}" for row in members]
                ),
            }
        )
    return output


def near_duplicates(rows: list[dict], threshold: int = 4) -> list[dict]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    output = []
    for left in train:
        left_hash = int(left["image_dhash"], 16)
        for right in validation:
            distance = (left_hash ^ int(right["image_dhash"], 16)).bit_count()
            if distance <= threshold:
                output.append(
                    {
                        "train_file": left["file_name"],
                        "val_file": right["file_name"],
                        "hamming_distance": distance,
                        "same_acquisition": left["acquisition"] == right["acquisition"],
                    }
                )
    return output


def group_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in {"train", "val"}:
            grouped[(row["split"], row["acquisition"])].append(row)
    output = []
    for (split, group), members in sorted(grouped.items()):
        indices = [
            member["slice_index"]
            for member in members
            if member["slice_index"] is not None
        ]
        row = {
            "split": split,
            "acquisition": group,
            "images": len(members),
            "material": Counter(member["material"] for member in members).most_common(1)[0][0],
            "slice_min": min(indices) if indices else None,
            "slice_max": max(indices) if indices else None,
            "intensity_mean": float(np.mean([member["intensity_mean"] for member in members])),
        }
        for name in CLASS_NAMES:
            row[f"fraction_{name}"] = float(
                np.mean([member.get(f"fraction_{name}", 0.0) for member in members])
            )
        output.append(row)
    return output


def suspects(rows: list[dict]) -> list[dict]:
    training = [row for row in rows if row["split"] == "train" and row.get("mask_present")]
    medians = {}
    mads = {}
    for name in CLASS_NAMES:
        values = np.asarray([row[f"fraction_{name}"] for row in training], dtype=np.float64)
        median = float(np.median(values))
        medians[name] = median
        mads[name] = float(np.median(np.abs(values - median)))
    output = []
    for row in rows:
        if row["split"] not in {"train", "val"} or not row.get("mask_present"):
            continue
        reasons = []
        score = 0.0
        if row.get("mask_encoding") == "invalid":
            reasons.append("invalid_mask_values")
            score += 100.0
        if not row.get("image_mask_shape_match", False):
            reasons.append("image_mask_shape_mismatch")
            score += 100.0
        if row.get("coco_area_differences") not in (None, "{}"):
            reasons.append("coco_png_area_mismatch")
            score += 50.0
        robust = []
        for name in CLASS_NAMES:
            scale = max(1.4826 * mads[name], 1.0 / (1024 * 1024))
            robust.append(abs(row[f"fraction_{name}"] - medians[name]) / scale)
        max_robust = max(robust)
        if max_robust >= 6.0:
            reasons.append(f"class_fraction_outlier_z={max_robust:.2f}")
            score += max_robust
        missing = [name for name in CLASS_NAMES if row.get(f"pixels_{name}") == 0]
        if missing:
            reasons.append("empty_classes=" + ",".join(missing))
            score += len(missing)
        if reasons:
            output.append(
                {
                    "score": score,
                    "split": row["split"],
                    "file_name": row["file_name"],
                    "acquisition": row["acquisition"],
                    "reasons": ";".join(reasons),
                    **{f"fraction_{name}": row[f"fraction_{name}"] for name in CLASS_NAMES},
                }
            )
    return sorted(output, key=lambda row: (-row["score"], row["split"], row["file_name"]))


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    output = args.output.resolve()
    if not (root / "images/train").is_dir() or not (root / "masks/train").is_dir():
        raise RuntimeError(f"not a LAM segmentation dataset: {root}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    for split in ("train", "val", "test"):
        split_rows, split_summary = audit_split(root, split)
        all_rows.extend(split_rows)
        summaries[split] = split_summary
        print(f"audited {split}: {len(split_rows)} images", flush=True)

    duplicates = exact_duplicates(all_rows)
    perceptual = near_duplicates(all_rows)
    groups = group_summary(all_rows)
    suspect_rows = suspects(all_rows)
    train_groups = {row["acquisition"] for row in all_rows if row["split"] == "train"}
    val_groups = {row["acquisition"] for row in all_rows if row["split"] == "val"}
    val_shared = sum(
        row["acquisition"] in train_groups for row in all_rows if row["split"] == "val"
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(root),
        "read_only": True,
        "class_names": list(CLASS_NAMES),
        "class_values": list(CLASS_VALUES),
        "splits": summaries,
        "exact_duplicate_groups": len(duplicates),
        "near_duplicate_pairs_train_val_dhash_le_4": len(perceptual),
        "train_acquisition_groups": len(train_groups),
        "val_acquisition_groups": len(val_groups),
        "shared_acquisition_groups": len(train_groups & val_groups),
        "val_images_in_shared_acquisitions": val_shared,
        "suspect_count": len(suspect_rows),
        "opencv_available": cv2 is not None,
        "outputs": {
            "per_image": "per_image.csv",
            "exact_duplicates": "exact_duplicates.csv",
            "near_duplicates": "near_duplicates_train_val.csv",
            "groups": "group_summary.csv",
            "suspects": "annotation_suspects.csv",
        },
    }
    write_csv(output / "per_image.csv", all_rows)
    write_csv(output / "exact_duplicates.csv", duplicates)
    write_csv(output / "near_duplicates_train_val.csv", perceptual)
    write_csv(output / "group_summary.csv", groups)
    write_csv(output / "annotation_suspects.csv", suspect_rows)
    atomic_json(output / "audit_summary.json", summary)
    (output / "COMPLETE").write_text(summary["created_at"] + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
