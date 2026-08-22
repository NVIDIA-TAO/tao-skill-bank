#!/usr/bin/env python3
"""Expose model-construction/checkpoint errors hidden by the TAO CLI wrapper."""

from __future__ import annotations

import gc
import argparse
import faulthandler
import json
import traceback
from pathlib import Path

# Importing the campaign entrypoint installs the reviewed runtime overlay.
import segformer_entrypoint  # noqa: F401

from nvidia_tao_pytorch.cv.segformer.model.segformer import SegFormer


ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_rootcause96_20260822_141557"
)
MODELS = {
    "fan_large": (
        "fan_large_16_p4_hybrid",
        Path(
            "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
            "lam_segformer_bayes_deft_20260820_231724/inputs/ptms/"
            "fan_large/fan_large_backbone_stripped.pth"
        ),
        [128, 256, 480, 480],
    ),
    "dinov3_large": (
        "vit_large_dinov3",
        ROOT / "inputs/ptms/vit_large_dinov3.safetensors",
        [1024, 1024, 1024, 1024],
    ),
    "dinov3_huge_plus": (
        "vit_huge_plus_dinov3",
        ROOT / "inputs/ptms/vit_huge_plus_dinov3.safetensors",
        [1280, 1280, 1280, 1280],
    ),
    "vit5_large": (
        "vit5_large_patch16_224",
        ROOT / "inputs/ptms/vit5_large_patch16_224.pth",
        [1024, 1024, 1024, 1024],
    ),
}


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(MODELS))
    args = parser.parse_args()
    outcomes = []
    for name, (backbone, checkpoint, channels) in MODELS.items():
        if args.only and name != args.only:
            continue
        print(f"DIRECT_MODEL_START name={name} backbone={backbone}", flush=True)
        try:
            model = SegFormer(
                output_nc=4,
                embed_dim=768,
                model=backbone,
                img_size=1024,
                in_channels=channels,
                feature_strides=[4, 8, 16, 32],
                pretrained_backbone_path=str(checkpoint),
                activation_checkpoint=True,
                freeze_backbone=True,
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            outcomes.append(
                {"name": name, "status": "OK", "trainable": trainable, "total": total}
            )
            print(f"DIRECT_MODEL_OK name={name} trainable={trainable} total={total}", flush=True)
            del model
        except Exception as exc:  # diagnostic intentionally records every model
            outcomes.append(
                {
                    "name": name,
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"DIRECT_MODEL_ERROR name={name}", flush=True)
            traceback.print_exc()
        gc.collect()
    receipt = ROOT / "probes/direct_backbone_diagnostic.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    if any(row["status"] != "OK" for row in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
