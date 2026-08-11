# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for DINOv3 backbone options in downstream skill schemas (bug 6465432).

DINOv3 backbones are selectable for SegFormer and Visual ChangeNet. The shared
variants must remain in both generated schema sets; ViT-7B is currently a
Visual ChangeNet option only. This guards the committed skill schemas against
falling behind the tao-core config source again.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_GLOBS = [
    "skills/models/tao-train-segformer/schemas/*.schema.json",
    "skills/models/tao-train-visual-changenet/schemas/*.schema.json",
]

# DINOv3 backbones shared by both dense downstream tasks.
DINOV3_DOWNSTREAM_BACKBONES = [
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
]

VISUAL_CHANGENET_DINOV3_BACKBONES = DINOV3_DOWNSTREAM_BACKBONES + ["vit_7b_dinov3"]

def _schema_files():
    """Return the committed segformer/visual_changenet skill schema paths."""
    files = []
    for pattern in SCHEMA_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    return files


def _find_backbone_enum(node):
    """Return the backbone ``type`` enum list (the one containing vit_large_nvdinov2)."""
    if isinstance(node, dict):
        for value in node.values():
            if isinstance(value, list) and "vit_large_nvdinov2" in value:
                return value
            found = _find_backbone_enum(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_backbone_enum(value)
            if found is not None:
                return found
    return None


def test_downstream_schemas_exist():
    assert _schema_files(), "no segformer/visual_changenet skill schemas found"


@pytest.mark.parametrize("schema_path", _schema_files(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_dinov3_backbones_present(schema_path):
    enum = _find_backbone_enum(json.loads(schema_path.read_text()))
    assert enum is not None, f"{schema_path}: no backbone type enum found"
    expected = DINOV3_DOWNSTREAM_BACKBONES
    if "tao-train-visual-changenet" in schema_path.parts:
        expected = VISUAL_CHANGENET_DINOV3_BACKBONES
    missing = [backbone for backbone in expected if backbone not in enum]
    assert not missing, f"{schema_path.name} backbone enum missing DINOv3 options: {missing}"
