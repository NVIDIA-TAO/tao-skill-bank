#!/usr/bin/env python3
"""Evaluate same-backbone checkpoint interpolations on validation only."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

TAO_SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
if TAO_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, TAO_SITE_PACKAGES)

PATCH_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724/controller/patches"
)
for relative_path in (
    "nvidia_tao_pytorch/cv/segformer/utils/iou_metric.py",
    "nvidia_tao_pytorch/cv/segformer/model/segformer_pl_model.py",
):
    source = PATCH_ROOT / relative_path
    if source.is_file():
        destination = Path(TAO_SITE_PACKAGES) / relative_path
        if not destination.is_file():
            raise RuntimeError(f"runtime patch target is missing: {destination}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from nvidia_tao_pytorch.cv.segformer.dataloader.dataset import SFDataset
from nvidia_tao_pytorch.cv.segformer.dataloader.utils import (
    build_palette,
    build_target_class_list,
)
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def complete_runtime_sections(cfg) -> None:
    """Supply Hydra-default sections absent from serialized training specs."""
    if "evaluate" not in cfg:
        cfg.evaluate = {"vis_after_n_batches": 16}
    if "inference" not in cfg:
        cfg.inference = {"vis_after_n_batches": 16}


def candidates(model_rows: list[dict]) -> list[dict]:
    """Build a bounded nine-model interpolation design.

    We score every source alone, balanced source-family soups, the all-model
    soup, within-family best pairs, and cumulative soups in descending source
    validation order.  This exposes whether averaging helps without expanding
    into an impractical full nine-dimensional simplex.
    """
    model_count = len(model_rows)
    if model_count < 2:
        raise ValueError("checkpoint soup experiment requires at least two models")
    rows = []
    seen = set()

    def add(weights: list[float], source: str) -> None:
        total = sum(weights)
        if total <= 0.0:
            raise ValueError(f"soup weights have non-positive total: {weights}")
        weights = [value / total for value in weights]
        key = tuple(int(round(value * 1_000_000_000)) for value in weights)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "name": "w_" + "_".join(f"{value:.6f}" for value in weights),
                "weights": weights,
                "source": source,
            }
        )

    for index in range(model_count):
        weights = [0.0] * model_count
        weights[index] = 1.0
        add(weights, f"single_model_{index}")
    families: dict[str, list[int]] = {}
    for index, row in enumerate(model_rows):
        families.setdefault(str(row.get("source", "unspecified")), []).append(index)
    for family, indices in sorted(families.items()):
        weights = [0.0] * model_count
        for index in indices:
            weights[index] = 1.0 / len(indices)
        add(weights, f"uniform_family_{family}")
        ranked = sorted(
            indices,
            key=lambda index: float(model_rows[index].get("val_miou", -1.0)),
            reverse=True,
        )
        if len(ranked) >= 2:
            weights = [0.0] * model_count
            weights[ranked[0]] = 0.5
            weights[ranked[1]] = 0.5
            add(weights, f"top_pair_family_{family}")
    add([1.0 / model_count] * model_count, "uniform_all_models")

    # Balance families rather than models so a four-member family cannot
    # dominate the standalone-DEFT family merely by cardinality.
    family_balanced = [0.0] * model_count
    for indices in families.values():
        for index in indices:
            family_balanced[index] = 1.0 / len(families) / len(indices)
    add(family_balanced, "equal_source_families")

    ranked_all = sorted(
        range(model_count),
        key=lambda index: float(model_rows[index].get("val_miou", -1.0)),
        reverse=True,
    )
    for count in range(2, model_count + 1):
        weights = [0.0] * model_count
        for index in ranked_all[:count]:
            weights[index] = 1.0 / count
        add(weights, f"cumulative_top_{count}_by_source_val")
    return rows


def merged_state_dict(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    if any(set(state) != set(state_dicts[0]) for state in state_dicts[1:]):
        raise RuntimeError("checkpoint state-dict key sets differ")
    anchor = int(np.argmax(weights))
    merged = {}
    for key in state_dicts[0]:
        tensors = [state[key] for state in state_dicts]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise RuntimeError(f"checkpoint tensor shapes differ for {key}")
        if tensors[0].is_floating_point():
            value = torch.zeros_like(tensors[0], dtype=torch.float32)
            for weight, tensor in zip(weights, tensors):
                if weight:
                    value.add_(tensor.float(), alpha=float(weight))
            merged[key] = value.to(dtype=tensors[0].dtype)
        else:
            merged[key] = tensors[anchor].clone()
    return merged


def confusion(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    prediction = prediction.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = target != 255
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def metrics(matrix: np.ndarray) -> dict:
    intersection = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0, dtype=np.float64)
    labelled = matrix.sum(axis=1, dtype=np.float64)
    union = predicted + labelled - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    return {
        "miou": float(iou.mean()),
        "accuracy": float(intersection.sum() / max(labelled.sum(), 1.0)),
        "class_iou": [float(value) for value in iou],
        "confusion_matrix": matrix.astype(np.int64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("selection_split") != "val":
        raise ValueError("checkpoint soup weights must be selected on val")
    model_rows = manifest["models"]
    if len({row["backbone"] for row in model_rows}) != 1:
        raise ValueError("checkpoint averaging is only valid within one backbone")
    expected_samples = int(manifest.get("expected_samples", 1262))
    num_classes = int(manifest.get("num_classes", 4))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = OmegaConf.load(model_rows[0]["spec"])
    complete_runtime_sections(cfg)
    cfg.dataset.segment.root_dir = manifest["dataset_root"]
    target_classes = build_target_class_list(cfg.dataset.segment)
    _, _, _, color_map = build_palette(target_classes)
    dataset = SFDataset(
        root_dir=manifest["dataset_root"],
        augmentation=cfg.dataset.segment.augmentation,
        split="val",
        img_size=cfg.dataset.segment.img_size,
        label_transform=cfg.dataset.segment.label_transform,
        to_tensor=True,
        color_map=color_map,
        load_mask=True,
    )
    if len(dataset) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} validation samples, found {len(dataset)}"
        )
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    checkpoint_payloads = [
        torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        for row in model_rows
    ]
    state_dicts = [payload["state_dict"] for payload in checkpoint_payloads]
    candidate_rows = candidates(model_rows)
    results = []

    for candidate_index, candidate in enumerate(candidate_rows):
        lightning = SegFormerPlModel(cfg)
        state = merged_state_dict(state_dicts, candidate["weights"])
        missing, unexpected = lightning.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"state load mismatch for {candidate['name']}: "
                f"missing={missing} unexpected={unexpected}"
            )
        model = lightning.model.to(device).eval()
        matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                images = batch["img"].to(device, non_blocking=True)
                target = batch["mask"].long().to(device, non_blocking=True)
                logits = functional.interpolate(
                    model(images),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                matrix += confusion(logits.argmax(dim=1), target, num_classes)
                if rank == 0 and (batch_index + 1) % 50 == 0:
                    print(
                        f"SOUP_PROGRESS candidate={candidate_index + 1}/"
                        f"{len(candidate_rows)} local_samples={batch_index + 1}/"
                        f"{len(indices)}",
                        flush=True,
                    )
        if world_size > 1:
            dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        row = {**candidate, **metrics(matrix.cpu().numpy())}
        results.append(row)
        if rank == 0:
            print(
                f"SOUP_CANDIDATE name={candidate['name']} "
                f"val_miou={row['miou']:.12f}",
                flush=True,
            )
        del model, lightning, state, matrix
        torch.cuda.empty_cache()
        gc.collect()

    sample_count = torch.tensor([len(indices)], dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    if int(sample_count.item()) != expected_samples:
        raise RuntimeError(
            f"distributed sample count mismatch: {int(sample_count.item())}"
        )

    if rank == 0:
        results.sort(key=lambda row: (-row["miou"], row["name"]))
        best = results[0]
        best_state = merged_state_dict(state_dicts, best["weights"])
        anchor = int(np.argmax(best["weights"]))
        artifact = dict(checkpoint_payloads[anchor])
        artifact["state_dict"] = best_state
        artifact.pop("optimizer_states", None)
        artifact.pop("lr_schedulers", None)
        artifact["lam_soup_provenance"] = {
            "selection_split": "val",
            "test_used_for_selection": False,
            "models": model_rows,
            "weights": best["weights"],
            "val_miou": best["miou"],
        }
        args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.best_checkpoint)
        output = {
            "schema_version": 1,
            "experiment": manifest["experiment"],
            "selection_split": "val",
            "validation_only": True,
            "test_used_for_selection": False,
            "sample_count": int(sample_count.item()),
            "num_classes": num_classes,
            "models": model_rows,
            "best": best,
            "best_checkpoint": str(args.best_checkpoint),
            "best_checkpoint_bytes": args.best_checkpoint.stat().st_size,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            f"SOUP_COMPLETE experiment={manifest['experiment']} "
            f"best={best['name']} val_miou={best['miou']:.12f} "
            f"checkpoint={args.best_checkpoint}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
