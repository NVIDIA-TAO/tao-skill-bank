#!/usr/bin/env python3
"""Score held-out SegFormer predictions for model-driven DEFT mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


BACKBONES = ("fan_base", "fan_large", "mit_b5")
CLASS_VALUES = np.asarray([0, 85, 170, 255], dtype=np.int16)


def decode(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    distances = np.abs(array[..., None].astype(np.int16) - CLASS_VALUES)
    labels = np.argmin(distances, axis=-1).astype(np.uint8)
    if int(distances.min(axis=-1).max()) != 0:
        raise RuntimeError(f"unexpected palette value in {path}")
    return labels


def boundary(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def dilate_one(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            source_y = slice(max(0, -dy), min(mask.shape[0], mask.shape[0] - dy))
            source_x = slice(max(0, -dx), min(mask.shape[1], mask.shape[1] - dx))
            target_y = slice(max(0, dy), min(mask.shape[0], mask.shape[0] + dy))
            target_x = slice(max(0, dx), min(mask.shape[1], mask.shape[1] + dx))
            result[target_y, target_x] |= mask[source_y, source_x]
    return result


def score_pair(prediction: np.ndarray, target: np.ndarray) -> dict:
    if prediction.shape != target.shape:
        raise RuntimeError(f"shape mismatch: prediction={prediction.shape} target={target.shape}")
    class_ious = []
    for class_id in range(4):
        pred_class = prediction == class_id
        target_class = target == class_id
        union = np.count_nonzero(pred_class | target_class)
        if union:
            class_ious.append(np.count_nonzero(pred_class & target_class) / union)
    miou = float(np.mean(class_ious))

    pred_rare = (prediction == 1) | (prediction == 2)
    target_rare = (target == 1) | (target == 2)
    rare_count = np.count_nonzero(target_rare)
    rare_recall = (
        float(np.count_nonzero(pred_rare & target_rare) / rare_count)
        if rare_count
        else 1.0
    )

    pred_boundary = boundary(prediction)
    target_boundary = boundary(target)
    pred_count = np.count_nonzero(pred_boundary)
    target_count = np.count_nonzero(target_boundary)
    precision = (
        float(np.count_nonzero(pred_boundary & dilate_one(target_boundary)) / pred_count)
        if pred_count
        else (1.0 if target_count == 0 else 0.0)
    )
    recall = (
        float(np.count_nonzero(target_boundary & dilate_one(pred_boundary)) / target_count)
        if target_count
        else (1.0 if pred_count == 0 else 0.0)
    )
    boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "miou": miou,
        "rare_recall": rare_recall,
        "boundary_f1": boundary_f1,
    }


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    denominator = max(len(values) - 1, 1)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank / denominator
    return ranks


def score_backbone(
    backbone: str,
    manifest: dict,
    evaluations_root: Path,
    fold_data_root: Path,
) -> list[dict]:
    rows = []
    seen = set()
    for fold in manifest["folds"]:
        fold_index = fold["fold"]
        prediction_root = evaluations_root / f"deft_oof_{backbone}_fold{fold_index}"
        target_root = fold_data_root / f"fold{fold_index}/masks/val"
        for filename in fold["held_out"]:
            if filename in seen:
                raise RuntimeError(f"duplicate OOF prediction target: {filename}")
            seen.add(filename)
            metrics = score_pair(
                decode(prediction_root / filename),
                decode(target_root / filename),
            )
            rows.append({"filename": filename, "fold": fold_index, **metrics})
    if len(rows) != 316 or len(seen) != 316:
        raise RuntimeError(f"{backbone}: OOF coverage is {len(rows)} rows/{len(seen)} unique")

    miou_error_rank = percentile_ranks([1.0 - row["miou"] for row in rows])
    rare_error_rank = percentile_ranks([1.0 - row["rare_recall"] for row in rows])
    boundary_error_rank = percentile_ranks([1.0 - row["boundary_f1"] for row in rows])
    for row, mr, rr, br in zip(rows, miou_error_rank, rare_error_rank, boundary_error_rank):
        row.update(
            {
                "miou_error_rank": mr,
                "rare_error_rank": rr,
                "boundary_error_rank": br,
                "difficulty": 0.45 * mr + 0.35 * rr + 0.20 * br,
            }
        )
    return sorted(rows, key=lambda row: (-row["difficulty"], row["filename"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluations-root", type=Path, required=True)
    parser.add_argument("--fold-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("validation_used") is not False:
        raise RuntimeError("OOF manifest does not guarantee validation exclusion")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for backbone in BACKBONES:
        rows = score_backbone(
            backbone,
            manifest,
            args.evaluations_root,
            args.fold_data_root,
        )
        payload = {
            "schema_version": 1,
            "backbone": backbone,
            "method": "out_of_fold_segformer_error",
            "validation_used": False,
            "weights": {
                "miou_error_rank": 0.45,
                "rare_error_rank": 0.35,
                "boundary_error_rank": 0.20,
            },
            "sample_count": len(rows),
            "mean_oof_miou": float(np.mean([row["miou"] for row in rows])),
            "scores": rows,
        }
        output = args.output_root / f"{backbone}_oof_scores.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"{backbone} samples=316 mean_oof_miou={payload['mean_oof_miou']:.6f} "
            f"output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
