#!/usr/bin/env python3
"""Score five prediction-fusion families on validation only.

The script is intended to run under ``torchrun`` with eight ranks.  Every rank
loads the same model set but scores a disjoint deterministic validation shard.
Only additive confusion matrices are reduced; validation predictions are never
used to train or modify a checkpoint.
"""

from __future__ import annotations

import argparse
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


def weight_candidates(model_count: int) -> list[dict]:
    """Return deterministic pairwise tenths plus a quarter-step simplex."""
    if model_count < 2:
        raise ValueError("prediction fusion requires at least two models")
    candidates: dict[tuple[int, ...], dict] = {}

    def add(weights: list[float], source: str) -> None:
        key = tuple(int(round(value * 100)) for value in weights)
        if sum(key) != 100:
            raise ValueError(f"fusion weights do not sum to 100: {weights}")
        row = candidates.setdefault(
            key,
            {
                "name": "w_" + "_".join(f"{value:03d}" for value in key),
                "weights": [value / 100.0 for value in key],
                "sources": [],
            },
        )
        row["sources"].append(source)

    for left, right in itertools.combinations(range(model_count), 2):
        for tenth in range(11):
            weights = [0.0] * model_count
            weights[left] = tenth / 10.0
            weights[right] = 1.0 - weights[left]
            add(weights, f"pair_{left}_{right}_tenths")

    if model_count == 3:
        for left in range(5):
            for middle in range(5 - left):
                right = 4 - left - middle
                add(
                    [left / 4.0, middle / 4.0, right / 4.0],
                    "three_model_quarter_simplex",
                )
        add([0.34, 0.33, 0.33], "near_uniform_three_model")

    return [candidates[key] for key in sorted(candidates)]


def supplied_weight_candidates(rows: list[dict], model_count: int) -> list[dict]:
    """Validate explicit candidates supplied by a hierarchical manifest."""
    validated = []
    names = set()
    for index, row in enumerate(rows):
        name = str(row.get("name") or f"candidate_{index:03d}")
        weights = [float(value) for value in row["weights"]]
        if name in names:
            raise ValueError(f"duplicate fusion candidate name: {name}")
        if len(weights) != model_count:
            raise ValueError(
                f"{name}: expected {model_count} weights, found {len(weights)}"
            )
        if any(value < 0.0 for value in weights):
            raise ValueError(f"{name}: fusion weights must be nonnegative")
        total = sum(weights)
        if abs(total - 1.0) > 1.0e-8:
            raise ValueError(f"{name}: fusion weights sum to {total}, not 1")
        names.add(name)
        validated.append(
            {
                "name": name,
                "weights": weights,
                "sources": list(row.get("sources", ["manifest_supplied"])),
            }
        )
    if not validated:
        raise ValueError("manifest supplied no fusion candidates")
    return validated


def confusion_matrix_batch(
    predictions: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute K confusion matrices for K prediction candidates."""
    candidate_count = predictions.shape[0]
    target = target.reshape(-1)
    predictions = predictions.reshape(candidate_count, -1)
    valid = target != 255
    target = target[valid]
    predictions = predictions[:, valid]
    offsets = (
        torch.arange(candidate_count, device=predictions.device, dtype=torch.int64)
        * num_classes
        * num_classes
    )[:, None]
    encoded = offsets + target[None, :] * num_classes + predictions
    return torch.bincount(
        encoded.reshape(-1),
        minlength=candidate_count * num_classes * num_classes,
    ).reshape(candidate_count, num_classes, num_classes)


def metrics(confusion: np.ndarray) -> dict:
    intersection = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0, dtype=np.float64)
    labelled = confusion.sum(axis=1, dtype=np.float64)
    union = predicted + labelled - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    accuracy = float(intersection.sum() / max(labelled.sum(), 1.0))
    return {
        "miou": float(iou.mean()),
        "accuracy": accuracy,
        "class_iou": [float(value) for value in iou],
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


def load_models(rows: list[dict], device: torch.device) -> list[torch.nn.Module]:
    loaded = []
    for row in rows:
        cfg = OmegaConf.load(row["spec"])
        complete_runtime_sections(cfg)
        lightning = SegFormerPlModel.load_from_checkpoint(
            row["checkpoint"],
            map_location="cpu",
            experiment_spec=cfg,
        )
        model = lightning.model.to(device).eval()
        loaded.append(model)
        del lightning
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scheme",
        choices=(
            "probability",
            "geometric_probability",
            "raw_logit",
            "class_rank_probability_tiebreak",
            "hard_vote_probability_tiebreak",
        ),
        required=True,
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    if payload.get("selection_split") != "val":
        raise ValueError("fusion weights must be selected on the val split")
    models_config = payload["models"]
    expected_samples = int(payload.get("expected_samples", 1262))
    num_classes = int(payload.get("num_classes", 4))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    data_cfg = OmegaConf.load(models_config[0]["spec"])
    complete_runtime_sections(data_cfg)
    data_cfg.dataset.segment.root_dir = payload["dataset_root"]
    target_classes = build_target_class_list(data_cfg.dataset.segment)
    _, _, _, color_map = build_palette(target_classes)
    dataset = SFDataset(
        root_dir=payload["dataset_root"],
        augmentation=data_cfg.dataset.segment.augmentation,
        split="val",
        img_size=data_cfg.dataset.segment.img_size,
        label_transform=data_cfg.dataset.segment.label_transform,
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

    models = load_models(models_config, device)
    if payload.get("weight_candidates"):
        candidates = supplied_weight_candidates(
            payload["weight_candidates"], len(models)
        )
    else:
        candidates = weight_candidates(len(models))
    weight_matrix = torch.tensor(
        [row["weights"] for row in candidates],
        dtype=torch.float32,
        device=device,
    )
    scheme_names = (args.scheme,)
    counts = torch.zeros(
        (len(scheme_names), len(candidates), num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["img"].to(device, non_blocking=True)
            target = batch["mask"].long().to(device, non_blocking=True)
            logits_rows = []
            probabilities = []
            for model in models:
                logits = model(images)
                logits = functional.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                logits_rows.append(logits.float())
                probabilities.append(torch.softmax(logits.float(), dim=1))
            logits_stack = torch.stack(logits_rows, dim=0)
            probability_stack = torch.stack(probabilities, dim=0)
            log_probability_stack = torch.log(
                probability_stack.clamp_min(torch.finfo(torch.float32).tiny)
            )
            class_ranks = torch.argsort(
                torch.argsort(probability_stack, dim=2), dim=2
            ).float()
            class_ranks /= max(num_classes - 1, 1)
            hard_votes = functional.one_hot(
                probability_stack.argmax(dim=2), num_classes=num_classes
            ).permute(0, 1, 4, 2, 3).float()

            scheme_scores = {
                "probability": probability_stack,
                "geometric_probability": log_probability_stack,
                "raw_logit": logits_stack,
                "class_rank_probability_tiebreak": class_ranks,
                "hard_vote_probability_tiebreak": hard_votes,
            }
            for scheme_index, scheme in enumerate(scheme_names):
                scores = scheme_scores[scheme]
                for start in range(0, len(candidates), 8):
                    stop = min(start + 8, len(candidates))
                    fused = torch.einsum(
                        "km,mbchw->kbchw",
                        weight_matrix[start:stop],
                        scores,
                    )
                    if scheme in (
                        "class_rank_probability_tiebreak",
                        "hard_vote_probability_tiebreak",
                    ):
                        # Weighted ranks/votes are the primary score.  Use an
                        # infinitesimal probability term only to resolve exact
                        # rank ties deterministically.
                        fused += 1.0e-6 * torch.einsum(
                            "km,mbchw->kbchw",
                            weight_matrix[start:stop],
                            probability_stack,
                        )
                    prediction = fused.argmax(dim=2).to(torch.int64)
                    counts[scheme_index, start:stop] += confusion_matrix_batch(
                        prediction,
                        target,
                        num_classes,
                    )
            if rank == 0 and (batch_index + 1) % 25 == 0:
                print(
                    f"FUSION_PROGRESS local_samples={batch_index + 1}/{len(indices)}",
                    flush=True,
                )

    sample_count = torch.tensor([len(indices)], dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    if int(sample_count.item()) != expected_samples:
        raise RuntimeError(
            f"distributed sample count mismatch: {int(sample_count.item())}"
        )

    if rank == 0:
        counts_cpu = counts.cpu().numpy()
        results = []
        for scheme_index, scheme in enumerate(scheme_names):
            for candidate_index, candidate in enumerate(candidates):
                row = {
                    "scheme": scheme,
                    **candidate,
                    **metrics(counts_cpu[scheme_index, candidate_index]),
                }
                results.append(row)
        results.sort(key=lambda row: (-row["miou"], row["scheme"], row["name"]))
        output = {
            "schema_version": 1,
            "experiment": payload["experiment"],
            "selection_split": "val",
            "validation_only": True,
            "test_used_for_selection": False,
            "sample_count": int(sample_count.item()),
            "num_classes": num_classes,
            "models": models_config,
            "candidate_count_per_scheme": len(candidates),
            "best": results[0],
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            f"FUSION_COMPLETE experiment={payload['experiment']} "
            f"best_scheme={results[0]['scheme']} best={results[0]['name']} "
            f"val_miou={results[0]['miou']:.12f}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
