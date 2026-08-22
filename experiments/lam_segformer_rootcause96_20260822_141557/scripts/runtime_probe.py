#!/usr/bin/env python3
"""Probe patched loss/config/backbone wiring inside the TAO image."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from timm.layers import trunc_normal_


TAO_SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
if str(TAO_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(TAO_SITE_PACKAGES))

from segformer_entrypoint import PATCHES, PATCH_ROOT, TAO_SITE_PACKAGES
from nvidia_tao_pytorch.config.segformer.default_config import SFModelConfig
from nvidia_tao_pytorch.cv.segformer.model.backbones import (
    cradio_vit_adapter_model_dict,
    vit_adapter_model_dict,
)
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def probe_loss(loss_name: str) -> dict:
    model = SegFormerPlModel.__new__(SegFormerPlModel)
    torch.nn.Module.__init__(model)
    model.train_config = type(
        "TrainConfig",
        (),
        {"segment": {"loss": loss_name, "iou_weight": 0.25}},
    )()
    model.n_class = 4
    model.weights = (0.815594, 3.042, 2.551302, 0.864234)
    model._build_criterion()
    prediction = torch.randn(1, 4, 16, 16, device="cuda", requires_grad=True)
    target = torch.randint(0, 4, (1, 16, 16), device="cuda")
    loss = model._compute_loss(prediction, target)
    loss.backward()
    return {
        "loss": float(loss.detach().cpu()),
        "finite": bool(torch.isfinite(loss).item()),
        "class_weights": list(model.class_weights),
    }


def main() -> None:
    initializer_probe = torch.empty(1, 1, 480)
    trunc_normal_(initializer_probe, std=0.02)
    if not torch.isfinite(initializer_probe).all():
        raise RuntimeError("timm trunc_normal_ produced non-finite values")
    print("TIMM_TRUNC_NORMAL_OK", flush=True)
    config = SFModelConfig()
    expected = {
        "c_radio_v4_vit_huge_patch16_224",
        "c_radio_v4_vit_so400m_patch16_224",
    }
    registered = expected.intersection(cradio_vit_adapter_model_dict)
    result = {
        "cuda_devices": torch.cuda.device_count(),
        "activation_checkpoint_default": config.activation_checkpoint,
        "registered_cradio_v4": sorted(registered),
        "registered_vit5": "vit5_large_patch16_224" in vit_adapter_model_dict,
        "patches": list(PATCHES),
        "patch_root": str(PATCH_ROOT),
        "loss_probes": {
            name: probe_loss(name)
            for name in ("ce_mmiou", "ce_lovasz", "ce_boundary")
        },
    }
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"expected 8 GPUs, found {torch.cuda.device_count()}")
    if registered != expected:
        raise RuntimeError(f"missing C-RADIOv4 registrations: {expected - registered}")
    if "vit5_large_patch16_224" not in vit_adapter_model_dict:
        raise RuntimeError("missing ViT-5-L SegFormer registration")
    if not all(probe["finite"] for probe in result["loss_probes"].values()):
        raise RuntimeError("a composite weighted loss is not finite")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
