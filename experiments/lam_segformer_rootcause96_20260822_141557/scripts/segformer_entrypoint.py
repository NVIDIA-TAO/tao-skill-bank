#!/usr/bin/env python3
"""Install reviewed SegFormer fixes into the ephemeral TAO container and run it."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


# Hydra otherwise collapses nested configuration/constructor failures into a
# terse one-line error, which made failed SLURM probes unnecessarily opaque.
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
for thread_cap in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_cap, "1")


TAO_SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
if str(TAO_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(TAO_SITE_PACKAGES))

PATCH_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_rootcause96_20260822_141557/controller/patches"
)

PATCHES = (
    "nvidia_tao_pytorch/cv/backbone_v2/radio.py",
    "nvidia_tao_pytorch/cv/segformer/dataloader/augmentation.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/adapter_modules.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/fan.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/radio.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/vit5.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/__init__.py",
    "nvidia_tao_pytorch/cv/segformer/model/segformer.py",
    "nvidia_tao_pytorch/cv/segformer/model/segformer_pl_model.py",
    "nvidia_tao_pytorch/cv/segformer/scripts/train.py",
    "nvidia_tao_pytorch/cv/segformer/utils/iou_metric.py",
    "nvidia_tao_pytorch/cv/segformer/utils/loss.py",
    "nvidia_tao_pytorch/config/segformer/default_config.py",
)


for relative_path in PATCHES:
    source = PATCH_ROOT / relative_path
    destination = TAO_SITE_PACKAGES / relative_path
    if not source.is_file():
        raise RuntimeError(f"required runtime patch is missing: {source}")
    if not destination.is_file() and relative_path != (
        "nvidia_tao_pytorch/cv/segformer/model/backbones/vit5.py"
    ):
        raise RuntimeError(f"runtime patch target is missing: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"TAO_RUNTIME_PATCH_INSTALLED={relative_path}", file=sys.stderr)

from nvidia_tao_pytorch.cv.segformer.entrypoint.segformer import main


if __name__ == "__main__":
    raise SystemExit(main())
