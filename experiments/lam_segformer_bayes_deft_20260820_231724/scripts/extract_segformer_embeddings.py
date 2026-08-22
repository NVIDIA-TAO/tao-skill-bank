#!/usr/bin/env python3
"""Extract deterministic SegFormer backbone embeddings with eight DDP ranks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def pool_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        pooled = feature.mean(dim=(2, 3))
    elif feature.ndim == 3:
        pooled = feature.mean(dim=1)
    elif feature.ndim == 2:
        pooled = feature
    else:
        raise RuntimeError(f"unsupported feature shape: {tuple(feature.shape)}")
    return functional.normalize(pooled.float(), p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="embed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = OmegaConf.load(args.spec)
    cfg.dataset.segment.root_dir = str(args.dataset_root)
    cfg.dataset.segment.predict_split = args.split
    cfg.dataset.segment.batch_size = 1
    cfg.dataset.segment.workers = 2
    target_classes = build_target_class_list(cfg.dataset.segment)
    _, _, _, color_map = build_palette(target_classes)
    dataset = SFDataset(
        root_dir=str(args.dataset_root),
        augmentation=cfg.dataset.segment.augmentation,
        split=args.split,
        img_size=cfg.dataset.segment.img_size,
        label_transform=cfg.dataset.segment.label_transform,
        to_tensor=True,
        color_map=color_map,
        load_mask=False,
    )
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    lightning_model = SegFormerPlModel.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        experiment_spec=cfg,
    )
    backbone = lightning_model.model.backbone.to(device).eval()
    del lightning_model

    names = []
    embeddings = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            pyramid = backbone.forward_feature_pyramid(images)
            if not isinstance(pyramid, (list, tuple)) or not pyramid:
                raise RuntimeError("backbone did not return a non-empty feature pyramid")
            embedding = pool_feature(pyramid[-1]).cpu().numpy()
            names.extend(batch["name"])
            embeddings.append(embedding)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = args.output_dir / f"rank{rank:02d}.npz"
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), np.float32)
    np.savez_compressed(shard, names=np.asarray(names), embeddings=matrix)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        combined = {}
        dimension = None
        for shard_rank in range(world_size):
            payload = np.load(args.output_dir / f"rank{shard_rank:02d}.npz")
            shard_names = payload["names"].tolist()
            shard_embeddings = payload["embeddings"]
            if len(shard_names) != len(shard_embeddings):
                raise RuntimeError(f"rank{shard_rank}: name/embedding count mismatch")
            for name, embedding in zip(shard_names, shard_embeddings):
                if name in combined:
                    raise RuntimeError(f"duplicate embedded sample: {name}")
                combined[name] = embedding
                dimension = int(embedding.shape[0])
        if len(combined) != 316:
            raise RuntimeError(f"expected 316 embeddings, found {len(combined)}")
        ordered_names = sorted(combined)
        ordered_embeddings = np.stack([combined[name] for name in ordered_names])
        norms = np.linalg.norm(ordered_embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError("embeddings are not L2-normalized")
        output = args.output_dir / "embeddings.npz"
        np.savez_compressed(
            output,
            names=np.asarray(ordered_names),
            embeddings=ordered_embeddings,
        )
        metadata = {
            "schema_version": 1,
            "backbone": args.backbone,
            "sample_count": len(ordered_names),
            "dimension": dimension,
            "normalized": True,
            "split": args.split,
            "validation_used": False,
            "artifact": str(output),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"EMBEDDINGS_COMPLETE backbone={args.backbone} samples=316 "
            f"dimension={dimension} output={output}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
