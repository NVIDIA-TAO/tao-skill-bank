#!/usr/bin/env python3
"""Launch TAO SegFormer despite the 7.1.0-pyt image's missing site path."""

from __future__ import annotations

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


def install_patch(relative_path: str) -> None:
    """Install a reviewed patch into this job's ephemeral container layer."""
    path = PATCH_ROOT / relative_path
    if not path.is_file():
        return
    destination = Path(TAO_SITE_PACKAGES) / relative_path
    if not destination.is_file():
        raise RuntimeError(f"runtime patch target is missing: {destination}")
    shutil.copyfile(path, destination)
    print(f"TAO_RUNTIME_PATCH_INSTALLED={relative_path}", file=sys.stderr)


install_patch(
    "nvidia_tao_pytorch/cv/segformer/utils/iou_metric.py",
)
install_patch(
    "nvidia_tao_pytorch/cv/segformer/model/segformer_pl_model.py",
)

from nvidia_tao_pytorch.cv.segformer.entrypoint.segformer import main


if __name__ == "__main__":
    raise SystemExit(main())
