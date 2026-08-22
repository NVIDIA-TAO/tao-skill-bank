#!/usr/bin/env python3
"""Read-only audit of the LAM semantic-segmentation dataset.

The audit deliberately follows the dataset symlinks and never edits source data.
It measures split integrity, label statistics, grayscale/color characteristics,
sequence-group overlap, and the label-information ceiling imposed by resizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_VALUES = (0, 85, 170, 255)
RESIZE_PROBES = (256, 512, 768, 1024, 1280, 1536, 1792, 2048)


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_key(path: Path) -> str:
    return re.sub(r"[-_]\d+$", "", path.stem)


def normalized_class_iou(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    scores = []
    for value in LABEL_VALUES:
        left = reference == value
        right = candidate == value
        union = np.logical_or(left, right).sum(dtype=np.int64)
        scores.append(float(np.logical_and(left, right).sum(dtype=np.int64) / union) if union else 1.0)
    return scores


def inspect_image(path: Path) -> dict:
    row = {
        "name": path.name,
        "resolved": str(path.resolve()),
        "broken": not path.exists(),
    }
    if row["broken"]:
        return row
    row["sha256"] = content_hash(path)
    with Image.open(path) as image:
        row.update({"mode": image.mode, "width": image.width, "height": image.height})
        rgb = np.asarray(image.convert("RGB"))
        row["rgb_channels_identical"] = bool(
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
        )
        row["channel_abs_diff_mean"] = float(
            (
                np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 1].astype(np.int16)).mean()
                + np.abs(rgb[..., 1].astype(np.int16) - rgb[..., 2].astype(np.int16)).mean()
                + np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 2].astype(np.int16)).mean()
            )
            / 3.0
        )
        tiny = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        row["thumbnail_hash"] = hashlib.sha256(np.asarray(tiny).tobytes()).hexdigest()
    return row


def inspect_mask(path: Path) -> dict:
    row = {
        "name": path.name,
        "resolved": str(path.resolve()),
        "broken": not path.exists(),
    }
    if row["broken"]:
        return row
    row["sha256"] = content_hash(path)
    with Image.open(path) as image:
        row.update({"mode": image.mode, "width": image.width, "height": image.height})
        array = np.asarray(image)
        if array.ndim == 3:
            row["mask_channels_identical"] = bool(
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2])
            )
            array = array[..., 0]
        unique, counts = np.unique(array, return_counts=True)
        row["label_counts"] = {str(int(value)): int(count) for value, count in zip(unique, counts)}
        row["unexpected_labels"] = [int(value) for value in unique if int(value) not in LABEL_VALUES]
        row["class_presence"] = [bool(np.any(array == value)) for value in LABEL_VALUES]
        horizontal = array[:, 1:] != array[:, :-1]
        vertical = array[1:, :] != array[:-1, :]
        row["boundary_edge_fraction"] = float(
            (horizontal.sum(dtype=np.int64) + vertical.sum(dtype=np.int64))
            / (horizontal.size + vertical.size)
        )
        resize_iou = {}
        for size in RESIZE_PROBES:
            down = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
            restored = np.asarray(down.resize(image.size, Image.Resampling.NEAREST))
            scores = normalized_class_iou(array, restored)
            resize_iou[str(size)] = {"class_iou": scores, "miou": float(np.mean(scores))}
        row["resize_roundtrip_iou"] = resize_iou
    return row


def duplicate_groups(rows: list[dict], key: str) -> list[list[str]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row["name"])
    return sorted((sorted(names) for names in groups.values() if len(names) > 1), key=lambda x: (-len(x), x))


def summarize_split(root: Path, split: str, workers: int) -> dict:
    images = sorted((root / "images" / split).iterdir())
    masks = sorted((root / "masks" / split).iterdir()) if split != "test" else []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        image_rows = list(pool.map(inspect_image, images))
        mask_rows = list(pool.map(inspect_mask, masks))
    image_names = {row["name"] for row in image_rows}
    mask_names = {row["name"] for row in mask_rows}
    label_counts = Counter()
    class_presence = np.zeros(len(LABEL_VALUES), dtype=np.int64)
    resize_mious = defaultdict(list)
    resize_class_ious = defaultdict(list)
    for row in mask_rows:
        label_counts.update({int(key): value for key, value in row.get("label_counts", {}).items()})
        class_presence += np.asarray(row.get("class_presence", [False] * len(LABEL_VALUES)), dtype=np.int64)
        for size, values in row.get("resize_roundtrip_iou", {}).items():
            resize_mious[size].append(values["miou"])
            resize_class_ious[size].append(values["class_iou"])
    total_pixels = sum(label_counts.values())
    frequencies = [label_counts[value] / total_pixels if total_pixels else 0.0 for value in LABEL_VALUES]
    inverse = np.asarray([1.0 / max(value, 1e-12) for value in frequencies], dtype=np.float64)
    sqrt_inverse = np.sqrt(inverse)
    median_frequency = float(np.median([value for value in frequencies if value > 0])) if total_pixels else 0.0
    return {
        "split": split,
        "image_count": len(image_rows),
        "mask_count": len(mask_rows),
        "broken_images": [row["name"] for row in image_rows if row["broken"]],
        "broken_masks": [row["name"] for row in mask_rows if row["broken"]],
        "images_without_masks": sorted(image_names - mask_names) if masks else [],
        "masks_without_images": sorted(mask_names - image_names),
        "image_modes": dict(Counter(row.get("mode") for row in image_rows if not row["broken"])),
        "mask_modes": dict(Counter(row.get("mode") for row in mask_rows if not row["broken"])),
        "image_sizes": {f"{w}x{h}": count for (w, h), count in Counter((row.get("width"), row.get("height")) for row in image_rows if not row["broken"]).items()},
        "mask_sizes": {f"{w}x{h}": count for (w, h), count in Counter((row.get("width"), row.get("height")) for row in mask_rows if not row["broken"]).items()},
        "rgb_channels_identical_count": sum(bool(row.get("rgb_channels_identical")) for row in image_rows),
        "channel_abs_diff_mean": float(np.mean([row["channel_abs_diff_mean"] for row in image_rows if "channel_abs_diff_mean" in row])),
        "label_counts": {str(value): int(label_counts[value]) for value in LABEL_VALUES},
        "unexpected_label_counts": {str(key): int(value) for key, value in label_counts.items() if key not in LABEL_VALUES},
        "class_pixel_frequencies": frequencies,
        "class_presence_image_counts": class_presence.tolist(),
        "suggested_weights": {
            "inverse_mean_one": (inverse / inverse.mean()).tolist() if total_pixels else [],
            "sqrt_inverse_mean_one": (sqrt_inverse / sqrt_inverse.mean()).tolist() if total_pixels else [],
            "median_frequency": [median_frequency / max(value, 1e-12) for value in frequencies] if total_pixels else [],
        },
        "boundary_edge_fraction_mean": float(np.mean([row["boundary_edge_fraction"] for row in mask_rows])) if mask_rows else None,
        "resize_roundtrip": {
            size: {
                "macro_image_miou_mean": float(np.mean(values)),
                "global_mean_class_iou": np.mean(np.asarray(resize_class_ious[size]), axis=0).tolist(),
                "minimum_image_miou": float(np.min(values)),
            }
            for size, values in sorted(resize_mious.items(), key=lambda item: int(item[0]))
        },
        "exact_image_duplicate_groups": duplicate_groups(image_rows, "sha256")[:100],
        "thumbnail_duplicate_groups": duplicate_groups(image_rows, "thumbnail_hash")[:100],
        "exact_mask_duplicate_groups": duplicate_groups(mask_rows, "sha256")[:100],
        "sequence_keys": sorted({sequence_key(Path(row["name"])) for row in image_rows}),
        "image_hashes": {row["name"]: row.get("sha256") for row in image_rows},
        "mask_hashes": {row["name"]: row.get("sha256") for row in mask_rows},
    }


def cross_split_duplicates(summaries: dict[str, dict], field: str) -> dict:
    hash_to_rows = defaultdict(list)
    for split, summary in summaries.items():
        for name, digest in summary[field].items():
            if digest:
                hash_to_rows[digest].append(f"{split}/{name}")
    return {
        digest: rows
        for digest, rows in hash_to_rows.items()
        if len({row.split("/", 1)[0] for row in rows}) > 1
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summaries = {split: summarize_split(args.root, split, args.workers) for split in ("train", "val", "test")}
    train_sequences = set(summaries["train"]["sequence_keys"])
    val_sequences = set(summaries["val"]["sequence_keys"])
    report = {
        "dataset_root": str(args.root),
        "splits": summaries,
        "cross_split": {
            "exact_image_duplicates": cross_split_duplicates(summaries, "image_hashes"),
            "exact_mask_duplicates": cross_split_duplicates({key: value for key, value in summaries.items() if key != "test"}, "mask_hashes"),
            "train_val_sequence_key_overlap": sorted(train_sequences & val_sequences),
            "train_sequence_key_count": len(train_sequences),
            "val_sequence_key_count": len(val_sequences),
        },
    }
    for summary in summaries.values():
        summary.pop("image_hashes", None)
        summary.pop("mask_hashes", None)
        summary.pop("sequence_keys", None)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
