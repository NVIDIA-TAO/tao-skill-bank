# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Qwen3-VL PatchEmbed helper for the TAO 7.2.0 Framework image.

Adapted from the Apache-2.0-licensed helper in NVIDIA cosmos-rl
(`cosmos_rl/policy/model/hf_models/patch.py`). The image's evaluation code
imports this API from cosmos_rl.policy, but its installed cosmos_rl package
omits the policy tree. Keep this module independent of the training/RL package
and its optional dependencies.
"""

from __future__ import annotations

import types
from typing import Any

import torch


def _linear_qwen3_vl_patch_embed_forward(
    self: Any, hidden_states: torch.Tensor
) -> torch.Tensor:
    import torch.nn.functional as functional

    patch_elements = (
        self.in_channels
        * self.temporal_patch_size
        * self.patch_size
        * self.patch_size
    )
    flat_input = hidden_states.reshape(-1, patch_elements).to(
        dtype=self.proj.weight.dtype
    )
    flat_weight = self.proj.weight.reshape(self.embed_dim, patch_elements)
    return functional.linear(flat_input, flat_weight, self.proj.bias)


_linear_qwen3_vl_patch_embed_forward._tao_channels_last_3d = True


def _auto_prefers_linear_patch_embed(device_capability: tuple[int, int] | None) -> bool:
    if device_capability is None:
        return False
    if tuple(device_capability) == (8, 0):
        return True
    return int(device_capability[0]) >= 10


def apply_qwen3_vl_patch_embed_compat(
    model: Any,
    mode: str,
    device_capability: tuple[int, int] | None = None,
) -> bool:
    """Replace Qwen3-VL Conv3d PatchEmbed with an equivalent linear path."""
    if mode not in {"auto", "linear", "conv3d"}:
        raise ValueError(f"Unsupported Qwen3-VL patch-embed mode: {mode!r}")
    if device_capability is None and torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
    enabled = mode == "linear" or (
        mode == "auto" and _auto_prefers_linear_patch_embed(device_capability)
    )
    if not enabled:
        return False

    patched = 0
    for module in model.modules():
        required = (
            "in_channels",
            "temporal_patch_size",
            "patch_size",
            "embed_dim",
            "proj",
        )
        if all(hasattr(module, name) for name in required) and isinstance(
            module.proj, torch.nn.Conv3d
        ):
            module.forward = types.MethodType(
                _linear_qwen3_vl_patch_embed_forward, module
            )
            patched += 1
    if patched != 1:
        raise RuntimeError(f"Expected one Qwen3-VL PatchEmbed module, found {patched}")
    return True
