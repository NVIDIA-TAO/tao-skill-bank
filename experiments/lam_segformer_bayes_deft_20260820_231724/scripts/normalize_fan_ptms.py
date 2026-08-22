#!/usr/bin/env python3
"""Extract FAN backbone tensors from NGC classification checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(
    "/localhome/local-rarunachalam/workspace/"
    "lam_segformer_bayes_deft_20260820_231724/ptm_download"
)
SOURCES = {
    "fan_base": ROOT / "fan_inspect/fan_hybrid_base_in22k_1k_384.pth",
    "fan_large": ROOT / "fan_inspect/fan_hybrid_large_in22k_1k_384.pth",
}
EXPECTED = {"fan_base": 599, "fan_large": 803}


def main() -> None:
    output_root = ROOT / "fan_normalized"
    output_root.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        backbone = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.") and not key.startswith("backbone.norm.")
        }
        if len(backbone) != EXPECTED[name]:
            raise RuntimeError(
                f"{name}: expected {EXPECTED[name]} backbone tensors, "
                f"found {len(backbone)}"
            )
        output = output_root / f"{name}_backbone_stripped.pth"
        torch.save(backbone, output)
        print(f"NORMALIZED name={name} tensors={len(backbone)} output={output}")


if __name__ == "__main__":
    main()
