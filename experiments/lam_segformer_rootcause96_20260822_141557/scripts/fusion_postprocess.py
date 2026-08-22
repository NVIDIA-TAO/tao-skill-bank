#!/usr/bin/env python3
"""Distributed validation-only fusion studies for the LAM SegFormer campaign.

The LAM ``test`` images duplicate ``val`` and have no independent masks, so all
reported metrics are explicitly validation or group-held-out validation metrics.
No result from this script is labeled as independent test performance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

# Installs every reviewed campaign patch before importing TAO modules.
import segformer_entrypoint  # noqa: F401

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.cv.segformer.dataloader.dataset import SFDataset
from nvidia_tao_pytorch.cv.segformer.dataloader.utils import build_palette, build_target_class_list
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_rootcause96_20260822_141557"
)
DATA_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research"
)
CATALOG_PATH = ROOT / "controller/post_model_catalog.json"
CACHE_ROOT = ROOT / "post/cache"
RESOLVED_PATH = CACHE_ROOT / "resolved_models.json"
NUM_CLASSES = 4
FOLDS = 5


def distributed_setup() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, torch.device("cuda", local_rank)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast(value, rank: int):
    values = [value if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def epoch_from_path(path: Path) -> int:
    match = re.search(r"(?:epoch|best)_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def best_validation_epoch(status_path: Path) -> tuple[int | None, float | None]:
    current_epoch = None
    best_epoch = None
    best_miou = None
    if not status_path.is_file():
        return best_epoch, best_miou
    for line in status_path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("epoch"), int):
            current_epoch = row["epoch"]
        kpi = row.get("kpi") or {}
        value = kpi.get("val_miou")
        if value is not None and (best_miou is None or float(value) > best_miou):
            best_miou = float(value)
            best_epoch = current_epoch
    return best_epoch, best_miou


def resolve_catalog() -> list[dict]:
    rows = json.loads(CATALOG_PATH.read_text())
    resolved = []
    for row in rows:
        result_dir = Path(row["results_dir"])
        checkpoints = sorted(
            result_dir.rglob("model*.pth"), key=lambda path: (epoch_from_path(path), path.name)
        )
        if not checkpoints:
            print(f"CACHE_SKIP_NO_CHECKPOINT label={row['label']}", flush=True)
            continue
        best_epoch, logged_best = best_validation_epoch(result_dir / "status.json")
        if best_epoch is None:
            selected = max(checkpoints, key=epoch_from_path)
            best_epoch = epoch_from_path(selected)
        else:
            selected = min(
                checkpoints,
                key=lambda path: (abs(epoch_from_path(path) - best_epoch), -epoch_from_path(path)),
            )
        resolved.append(
            {
                **row,
                "checkpoint": str(selected),
                "checkpoint_epoch": epoch_from_path(selected),
                "best_validation_epoch": best_epoch,
                "logged_best_val_miou": logged_best,
                "checkpoints": [str(path) for path in checkpoints],
            }
        )
    return resolved


def dataset_from_spec(spec_path: str) -> tuple[object, SFDataset]:
    cfg = OmegaConf.load(spec_path)
    cfg.dataset.segment.root_dir = str(DATA_ROOT)
    cfg.dataset.segment.validation_split = "val"
    cfg.dataset.segment.test_split = "val"
    cfg.dataset.segment.workers = 2
    target_classes = build_target_class_list(cfg.dataset.segment)
    _, _, _, color_map = build_palette(target_classes)
    dataset = SFDataset(
        root_dir=str(DATA_ROOT),
        augmentation=cfg.dataset.segment.augmentation,
        split="val",
        img_size=cfg.dataset.segment.img_size,
        label_transform=cfg.dataset.segment.label_transform,
        to_tensor=True,
        color_map=color_map,
    )
    return cfg, dataset


def local_loader(dataset, rank: int, world: int):
    indices = list(range(rank, len(dataset), world))
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    return indices, loader


def load_model(row: dict, cfg, device: torch.device, checkpoint: str | None = None):
    TLTPyTorchCookbook.set_passphrase(str(cfg.encryption_key))
    model = SegFormerPlModel.load_from_checkpoint(
        checkpoint or row["checkpoint"], map_location="cpu", experiment_spec=cfg
    )
    model.to(device).eval()
    return model


def confusion_update(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    valid = (target >= 0) & (target < NUM_CLASSES)
    encoded = target[valid].astype(np.int64) * NUM_CLASSES + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def reduce_confusion(confusion: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(confusion, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def scores(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0)
    total = confusion.sum()
    accuracy = float(diagonal.sum() / total) if total else 0.0
    return {
        "miou": float(iou.mean()),
        "accuracy": accuracy,
        "class_iou": iou.tolist(),
        "confusion": confusion.tolist(),
    }


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float32)
    values -= values.max(axis=0, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=0, keepdims=True)
    return values


def cache_paths(label: str, rank: int) -> tuple[Path, Path]:
    directory = CACHE_ROOT / "logits" / label
    return directory / f"rank{rank}.npy", directory / f"rank{rank}.complete"


def label_paths(rank: int) -> tuple[Path, Path, Path]:
    directory = CACHE_ROOT / "labels"
    return (
        directory / f"rank{rank}.npy",
        directory / f"rank{rank}.names.json",
        directory / f"rank{rank}.complete",
    )


def cache_labels(dataset, indices, loader, rank: int) -> None:
    label_path, names_path, complete_path = label_paths(rank)
    if complete_path.is_file():
        return
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.uint8, shape=(len(indices), 1024, 1024)
    )
    names = []
    for local_index, batch in enumerate(loader):
        labels[local_index] = batch["mask"][0, 0].numpy().astype(np.uint8)
        names.append(batch["name"][0])
    labels.flush()
    atomic_json(names_path, names)
    complete_path.write_text("complete\n")


def model_confusion_from_cache(label: str, rank: int) -> np.ndarray:
    logits_path, _ = cache_paths(label, rank)
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    logits = np.load(logits_path, mmap_mode="r")
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for index in range(len(labels)):
        confusion_update(confusion, np.asarray(logits[index]).argmax(0), np.asarray(labels[index]))
    return confusion


def cache_mode(rank: int, world: int, device: torch.device) -> None:
    resolved = resolve_catalog() if rank == 0 else None
    resolved = broadcast(resolved, rank)
    if not resolved:
        raise RuntimeError("no successful training checkpoint was found")

    _, label_dataset = dataset_from_spec(resolved[0]["spec"])
    indices, label_loader = local_loader(label_dataset, rank, world)
    cache_labels(label_dataset, indices, label_loader, rank)
    barrier()

    metric_rows = []
    for row in resolved:
        logits_path, complete_path = cache_paths(row["label"], rank)
        if not complete_path.is_file():
            cfg, dataset = dataset_from_spec(row["spec"])
            _, loader = local_loader(dataset, rank, world)
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            cache = np.lib.format.open_memmap(
                logits_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(indices), NUM_CLASSES, 1024, 1024),
            )
            model = load_model(row, cfg, device)
            with torch.inference_mode():
                for local_index, batch in enumerate(loader):
                    image = batch["img"].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model.model(image)
                        logits = F.interpolate(
                            logits, size=(1024, 1024), mode="bilinear", align_corners=False
                        )
                    cache[local_index] = logits[0].float().cpu().numpy().astype(np.float16)
            cache.flush()
            complete_path.write_text("complete\n")
            del model, cache
            gc.collect()
            torch.cuda.empty_cache()
        barrier()
        confusion = reduce_confusion(model_confusion_from_cache(row["label"], rank), device)
        if rank == 0:
            metric_rows.append({**row, "individual_validation": scores(confusion)})
            print(
                f"CACHE_MODEL_OK label={row['label']} "
                f"miou={metric_rows[-1]['individual_validation']['miou']:.6f}",
                flush=True,
            )
    if rank == 0:
        atomic_json(
            RESOLVED_PATH,
            {
                "metric_scope": "validation_only",
                "independent_test_available": False,
                "models": metric_rows,
            },
        )


def resolved_models() -> list[dict]:
    return json.loads(RESOLVED_PATH.read_text())["models"]


def basic_mode(rank: int, device: torch.device) -> None:
    rows = resolved_models()
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    maps = [np.load(cache_paths(row["label"], rank)[0], mmap_mode="r") for row in rows]
    methods = {name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for name in (
        "uniform_probability", "raw_logit", "geometric_probability", "majority_vote", "per_pixel_class_rank"
    )}
    for index in range(len(labels)):
        probability_sum = np.zeros((NUM_CLASSES, 1024, 1024), dtype=np.float32)
        log_probability_sum = np.zeros_like(probability_sum)
        logit_sum = np.zeros_like(probability_sum)
        votes = np.zeros_like(probability_sum, dtype=np.uint16)
        rank_sum = np.zeros_like(probability_sum, dtype=np.uint16)
        for mmap in maps:
            logits = np.asarray(mmap[index]).astype(np.float32)
            probability = softmax_numpy(logits)
            probability_sum += probability
            log_probability_sum += np.log(np.maximum(probability, 1e-7))
            logit_sum += logits
            prediction = logits.argmax(0)
            for class_id in range(NUM_CLASSES):
                votes[class_id] += prediction == class_id
            rank_sum += np.argsort(np.argsort(-logits, axis=0), axis=0).astype(np.uint16)
        target = np.asarray(labels[index])
        predictions = {
            "uniform_probability": probability_sum.argmax(0),
            "raw_logit": logit_sum.argmax(0),
            "geometric_probability": log_probability_sum.argmax(0),
            "majority_vote": votes.argmax(0),
            "per_pixel_class_rank": rank_sum.argmin(0),
        }
        for name, prediction in predictions.items():
            confusion_update(methods[name], prediction, target)
    output = {}
    for name, confusion in methods.items():
        output[name] = scores(reduce_confusion(confusion, device))
    if rank == 0:
        atomic_json(ROOT / "post/basic_fusions/results.json", {"scope": "validation", "methods": output})


def sequence_group(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[-_]\d+$", "", stem)


def fold_for_name(name: str) -> int:
    digest = hashlib.sha256(sequence_group(name).encode()).digest()
    return int.from_bytes(digest[:4], "little") % FOLDS


def sample_probabilities(rows: list[dict], rank: int) -> Path:
    output = CACHE_ROOT / "samples" / f"rank{rank}.npz"
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    names = json.loads(label_paths(rank)[1].read_text())
    maps = [np.load(cache_paths(row["label"], rank)[0], mmap_mode="r") for row in rows]
    probability_rows, target_rows, fold_rows = [], [], []
    for image_index, name in enumerate(names):
        target = np.asarray(labels[image_index]).reshape(-1)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))
        selected = []
        for class_id in range(NUM_CLASSES):
            candidates = np.flatnonzero(target == class_id)
            if candidates.size:
                selected.append(rng.choice(candidates, size=min(32, candidates.size), replace=False))
        if not selected:
            continue
        pixels = np.concatenate(selected)
        model_values = []
        for mmap in maps:
            probability = softmax_numpy(np.asarray(mmap[image_index]))
            model_values.append(probability.reshape(NUM_CLASSES, -1)[:, pixels].T)
        probability_rows.append(np.stack(model_values, axis=1).astype(np.float16))
        target_rows.append(target[pixels].astype(np.uint8))
        fold_rows.append(np.full(pixels.size, fold_for_name(name), dtype=np.uint8))
    np.savez(
        output,
        probabilities=np.concatenate(probability_rows),
        targets=np.concatenate(target_rows),
        folds=np.concatenate(fold_rows),
    )
    return output


def fit_weights(probabilities, targets, class_specific: bool, seed: int) -> np.ndarray:
    device = torch.device("cuda", 0)
    x = torch.from_numpy(probabilities.astype(np.float32)).to(device)
    y = torch.from_numpy(targets.astype(np.int64)).to(device)
    shape = (x.shape[1], NUM_CLASSES) if class_specific else (x.shape[1],)
    theta = torch.zeros(shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.08)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(250):
        if len(y) > 32768:
            selection = torch.randint(len(y), (32768,), generator=generator, device=device)
            batch_x, batch_y = x[selection], y[selection]
        else:
            batch_x, batch_y = x, y
        weights = torch.softmax(theta, dim=0)
        combined = (batch_x * (weights[None, :, :] if class_specific else weights[None, :, None])).sum(1)
        loss = F.nll_loss(torch.log(combined.clamp_min(1e-7)), batch_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.softmax(theta, dim=0).detach().cpu().numpy()


def cv_mode(rank: int, device: torch.device, class_specific: bool) -> None:
    rows = resolved_models()
    sample_probabilities(rows, rank)
    barrier()
    fitted = None
    if rank == 0:
        shards = [np.load(CACHE_ROOT / "samples" / f"rank{other}.npz") for other in range(dist.get_world_size() if dist.is_initialized() else 1)]
        probability = np.concatenate([shard["probabilities"] for shard in shards])
        target = np.concatenate([shard["targets"] for shard in shards])
        folds = np.concatenate([shard["folds"] for shard in shards])
        fold_weights = []
        for fold in range(FOLDS):
            keep = folds != fold
            fold_weights.append(fit_weights(probability[keep], target[keep], class_specific, 1234 + fold))
        full_weights = fit_weights(probability, target, class_specific, 4321)
        fitted = {"fold_weights": [value.tolist() for value in fold_weights], "full_weights": full_weights.tolist()}
    fitted = broadcast(fitted, rank)
    fold_weights = [np.asarray(value, dtype=np.float32) for value in fitted["fold_weights"]]
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    names = json.loads(label_paths(rank)[1].read_text())
    maps = [np.load(cache_paths(row["label"], rank)[0], mmap_mode="r") for row in rows]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for index, name in enumerate(names):
        weight = fold_weights[fold_for_name(name)]
        combined = np.zeros((NUM_CLASSES, 1024, 1024), dtype=np.float32)
        for model_index, mmap in enumerate(maps):
            probability = softmax_numpy(np.asarray(mmap[index]))
            if class_specific:
                combined += probability * weight[model_index, :, None, None]
            else:
                combined += probability * weight[model_index]
        confusion_update(confusion, combined.argmax(0), np.asarray(labels[index]))
    result = scores(reduce_confusion(confusion, device))
    if rank == 0:
        name = "class_specific_cv" if class_specific else "global_nonnegative_cv"
        atomic_json(
            ROOT / "post" / name / "results.json",
            {
                "scope": "sequence_group_oof_weight_fitting_on_validation",
                "checkpoint_selection_scope": "full_validation",
                "independent_test_available": False,
                "models": [row["label"] for row in rows],
                **fitted,
                "metrics": result,
            },
        )


def transformed(image: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    value = torch.rot90(image, rotation, dims=(-2, -1))
    return torch.flip(value, dims=(-1,)) if flip else value


def untransformed(logits: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    value = torch.flip(logits, dims=(-1,)) if flip else logits
    return torch.rot90(value, -rotation, dims=(-2, -1))


def d4_mode(rank: int, world: int, device: torch.device) -> None:
    rows = sorted(
        resolved_models(), key=lambda row: row["individual_validation"]["miou"], reverse=True
    )[:3]
    individual = {}
    for row in rows:
        cfg, dataset = dataset_from_spec(row["spec"])
        indices, loader = local_loader(dataset, rank, world)
        directory = ROOT / "post/d4_tta/probabilities" / row["label"]
        path, done = directory / f"rank{rank}.npy", directory / f"rank{rank}.complete"
        if not done.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            cache = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(len(indices), NUM_CLASSES, 1024, 1024)
            )
            model = load_model(row, cfg, device)
            with torch.inference_mode():
                for local_index, batch in enumerate(loader):
                    image = batch["img"].to(device, non_blocking=True)
                    probability = 0.0
                    for rotation in range(4):
                        for flip in (False, True):
                            variant = transformed(image, rotation, flip)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                logits = model.model(variant)
                                logits = F.interpolate(logits, (1024, 1024), mode="bilinear", align_corners=False)
                            probability = probability + untransformed(logits.softmax(1), rotation, flip)
                    cache[local_index] = (probability[0] / 8.0).float().cpu().numpy().astype(np.float16)
            cache.flush()
            done.write_text("complete\n")
            del model, cache
            gc.collect()
            torch.cuda.empty_cache()
        barrier()
        labels = np.load(label_paths(rank)[0], mmap_mode="r")
        probs = np.load(path, mmap_mode="r")
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for index in range(len(labels)):
            confusion_update(confusion, np.asarray(probs[index]).argmax(0), np.asarray(labels[index]))
        individual[row["label"]] = scores(reduce_confusion(confusion, device))
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    maps = [np.load(ROOT / "post/d4_tta/probabilities" / row["label"] / f"rank{rank}.npy", mmap_mode="r") for row in rows]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for index in range(len(labels)):
        total = sum(np.asarray(mmap[index], dtype=np.float32) for mmap in maps)
        confusion_update(confusion, total.argmax(0), np.asarray(labels[index]))
    ensemble = scores(reduce_confusion(confusion, device))
    if rank == 0:
        atomic_json(
            ROOT / "post/d4_tta/results.json",
            {"scope": "validation", "selected_models": [row["label"] for row in rows], "individual": individual, "uniform_ensemble": ensemble},
        )


def average_checkpoints(paths: list[str], output: Path) -> None:
    base = torch.load(paths[0], map_location="cpu", weights_only=False)
    state = base["state_dict"]
    accumulators = {
        key: value.float().clone() for key, value in state.items() if torch.is_tensor(value) and value.is_floating_point()
    }
    for path in paths[1:]:
        current = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        for key, accumulator in accumulators.items():
            accumulator.add_(current[key].float())
        del current
        gc.collect()
    for key, accumulator in accumulators.items():
        state[key] = (accumulator / len(paths)).to(dtype=state[key].dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)


def evaluate_checkpoint(row: dict, checkpoint: str, rank: int, world: int, device: torch.device) -> np.ndarray:
    cfg, dataset = dataset_from_spec(row["spec"])
    _, loader = local_loader(dataset, rank, world)
    model = load_model(row, cfg, device, checkpoint=checkpoint)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            image = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.model(image)
                logits = F.interpolate(logits, (1024, 1024), mode="bilinear", align_corners=False)
            confusion_update(
                confusion,
                logits[0].argmax(0).cpu().numpy(),
                batch["mask"][0, 0].numpy(),
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return reduce_confusion(confusion, device)


def soup_mode(rank: int, world: int, device: torch.device) -> None:
    rows = sorted(
        resolved_models(), key=lambda row: row["individual_validation"]["miou"], reverse=True
    )[:3]
    soups = []
    if rank == 0:
        for row in rows:
            candidates = sorted(
                row["checkpoints"],
                key=lambda path: abs(epoch_from_path(Path(path)) - row["best_validation_epoch"]),
            )[:5]
            if len(candidates) < 2:
                continue
            output = ROOT / "post/checkpoint_soups" / row["label"] / "soup.pth"
            if not output.is_file():
                average_checkpoints(candidates, output)
            soups.append({"row": row, "checkpoint": str(output), "members": candidates})
    soups = broadcast(soups, rank)
    results = []
    for soup in soups:
        confusion = evaluate_checkpoint(soup["row"], soup["checkpoint"], rank, world, device)
        if rank == 0:
            results.append({"label": soup["row"]["label"], "members": soup["members"], "metrics": scores(confusion)})
    if rank == 0:
        atomic_json(ROOT / "post/checkpoint_soups/results.json", {"scope": "validation", "soups": results})


def deft_all_mode(rank: int, device: torch.device) -> None:
    rows = resolved_models()
    deft = [row for row in rows if row["run_id"].startswith("D")]
    original = sorted(
        [row for row in rows if not row["run_id"].startswith("D")],
        key=lambda row: row["individual_validation"]["miou"], reverse=True,
    )[:5]
    selected = original + deft
    labels = np.load(label_paths(rank)[0], mmap_mode="r")
    maps = {row["label"]: np.load(cache_paths(row["label"], rank)[0], mmap_mode="r") for row in selected}
    method_members = {}
    for source in original:
        for deft_row in deft:
            method_members[f"pair:{source['label']}+{deft_row['label']}"] = [source["label"], deft_row["label"]]
    method_members["original_top5"] = [row["label"] for row in original]
    method_members["deft_only"] = [row["label"] for row in deft]
    method_members["all_selected"] = [row["label"] for row in selected]
    confusions = {name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for name in method_members}
    for index in range(len(labels)):
        probabilities = {label: softmax_numpy(np.asarray(mmap[index])) for label, mmap in maps.items()}
        for name, members in method_members.items():
            total = sum(probabilities[member] for member in members)
            confusion_update(confusions[name], total.argmax(0), np.asarray(labels[index]))
    results = {name: scores(reduce_confusion(value, device)) for name, value in confusions.items()}
    if rank == 0:
        atomic_json(
            ROOT / "post/deft_all_fusions/results.json",
            {"scope": "validation", "members": method_members, "metrics": results},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("cache", "basic", "global_cv", "class_cv", "d4_tta", "soup", "deft_all"),
    )
    args = parser.parse_args()
    rank, world, device = distributed_setup()
    try:
        if args.mode == "cache":
            cache_mode(rank, world, device)
        elif args.mode == "basic":
            basic_mode(rank, device)
        elif args.mode == "global_cv":
            cv_mode(rank, device, class_specific=False)
        elif args.mode == "class_cv":
            cv_mode(rank, device, class_specific=True)
        elif args.mode == "d4_tta":
            d4_mode(rank, world, device)
        elif args.mode == "soup":
            soup_mode(rank, world, device)
        else:
            deft_all_mode(rank, device)
        barrier()
        if rank == 0:
            print(f"FUSION_POSTPROCESS_OK mode={args.mode}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
