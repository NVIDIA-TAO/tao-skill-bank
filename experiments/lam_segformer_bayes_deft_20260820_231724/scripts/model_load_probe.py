#!/usr/bin/env python3
"""Verify that all requested checkpoints actually populate TAO backbones."""

from __future__ import annotations

import gc
from pathlib import Path
import sys

TAO_SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
if TAO_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, TAO_SITE_PACKAGES)

from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.segformer.model.segformer import SegFormer


MODELS = {
    "fan_base_16_p4_hybrid": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "lam_segformer_bayes_deft_20260820_231724/inputs/ptms/fan_base/"
        "fan_base_backbone_stripped.pth",
        [128, 256, 448, 448],
    ),
    "fan_large_16_p4_hybrid": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "lam_segformer_bayes_deft_20260820_231724/inputs/ptms/fan_large/"
        "fan_large_backbone_stripped.pth",
        [128, 256, 480, 480],
    ),
    "mit_b5": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "lam_segformer_bayes_deft_20260820_231724/inputs/ptms/mit_b5/"
        "mit_b5_backbone_stripped.pth",
        [64, 128, 320, 512],
    ),
}

DATASETS = {
    "original": (
        Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research"),
        316,
    ),
    "mix50": (
        Path(
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
            "lam_segformer_bayes_deft_20260820_231724/datasets/deft_mix50"
        ),
        474,
    ),
    "mix100": (
        Path(
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
            "lam_segformer_bayes_deft_20260820_231724/datasets/deft_mix100"
        ),
        632,
    ),
}


def probe_datasets() -> None:
    for name, (root, expected_train) in DATASETS.items():
        image_dir = root / "images" / "train"
        mask_dir = root / "masks" / "train"
        images = [path for path in image_dir.iterdir() if path.is_file()]
        masks = [path for path in mask_dir.iterdir() if path.is_file()]
        if len(images) != expected_train or len(masks) != expected_train:
            raise RuntimeError(
                f"{name}: expected {expected_train} train image/mask files, "
                f"found images={len(images)} masks={len(masks)}"
            )
        if any(not path.exists() for path in images + masks):
            raise RuntimeError(f"{name}: contains dangling train-data links")
        print(
            f"DATASET_PROBE name={name} train_images={len(images)} "
            f"train_masks={len(masks)}",
            flush=True,
        )


def main() -> None:
    probe_datasets()
    for name, (checkpoint, channels) in MODELS.items():
        model = SegFormer(
            output_nc=4,
            embed_dim=768,
            model=name,
            img_size=1024,
            in_channels=channels,
            feature_strides=[4, 8, 16, 32],
            pretrained_backbone_path=None,
        )
        target = model.backbone.state_dict()
        source = load_pretrained_weights(checkpoint)
        matched = {
            key
            for key, value in source.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        ratio = len(matched) / max(len(target), 1)
        print(
            f"PTM_MATCH name={name} matched={len(matched)} "
            f"target={len(target)} ratio={ratio:.6f}",
            flush=True,
        )
        if ratio < 0.90:
            raise RuntimeError(f"{name}: only {ratio:.1%} of backbone tensors match")
        del model, target, source, matched
        gc.collect()
    print("MODEL_LOAD_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
