# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for the DINOv3 skill schemas (bug 6465432).

The committed ``skills/models/tao-train-dinov3/schemas/*.schema.json`` files are
generated from the TAO Core dataclass config (``nvidia_tao_core.config.dinov3``)
by ``scripts/generate_dataclass_schemas.py``. Guarded invariants:

1. The backbone enums in the committed schemas list every backbone TAO's DINOv3
   supports (always checked, stdlib-only — runs in the validate-skills CI job).
2. When a tao-core checkout is importable, it must carry the DINOv3 config with
   the same backbone set, and regenerating the schemas from it must reproduce
   the committed backbone options (skipped when tao-core is absent).

Note: a full byte-compare against a fresh regeneration is deliberately NOT done —
``dataclass2json_converter`` emits ``automl_disabled_parameters``/``popular`` in
hash-randomized set order, so byte output is not reproducible across processes.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODEL_DIR = REPO_ROOT / "skills" / "models" / "tao-train-dinov3"
ACTIONS = ("convert", "export", "inference", "train")

# Must match SUPPORTED_BACKBONES in nvidia_tao_core.config.dinov3.default_config
# (mirrored from nvidia_tao_pytorch.config.dinov3.default_config).
EXPECTED_BACKBONES = ["vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"]


def _backbone_properties(schema):
    return schema["properties"]["model"]["properties"]["backbone"]["properties"]


def _load_committed_schema(action):
    return json.loads((MODEL_DIR / "schemas" / f"{action}.schema.json").read_text())


@pytest.mark.parametrize("action", ACTIONS)
def test_committed_backbone_enums(action):
    """Every committed DINOv3 schema lists all supported backbones."""
    backbone = _backbone_properties(_load_committed_schema(action))
    assert backbone["teacher_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["student_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["teacher_type"]["default"] == "vit_b"
    assert backbone["student_type"]["default"] == "vit_b"


@pytest.fixture(scope="module")
def tao_core_dinov3_config():
    """The tao-core DINOv3 config module; skip without tao-core, fail if it drifted."""
    pytest.importorskip("nvidia_tao_core", reason="requires a tao-core checkout on PYTHONPATH")
    try:
        return importlib.import_module("nvidia_tao_core.config.dinov3.default_config")
    except ModuleNotFoundError:
        pytest.fail(
            "nvidia_tao_core is importable but has no config.dinov3 module: tao-core is "
            "behind tao-pytorch and the committed DINOv3 skill schemas cannot be "
            "regenerated from it (bug 6465432)."
        )


def test_tao_core_backbones_match(tao_core_dinov3_config):
    assert tao_core_dinov3_config.SUPPORTED_BACKBONES == EXPECTED_BACKBONES


def test_regenerated_backbone_enums_match_committed(tao_core_dinov3_config, tmp_path):
    """Schemas regenerated from tao-core carry the committed backbone options."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        generator = importlib.import_module("generate_dataclass_schemas")
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    scratch_model_dir = tmp_path / "skills" / "models" / "tao-train-dinov3"
    scratch_model_dir.mkdir(parents=True)
    (scratch_model_dir / "config.json").write_text(
        json.dumps(
            {
                "network_arch": "dinov3",
                "actions": {action: {} for action in ACTIONS},
            }
        )
    )

    manifest = generator.generate_for_model(scratch_model_dir, clean=False)
    assert not manifest["failures"], f"schema generation failed: {manifest['failures']}"

    for action in ACTIONS:
        regenerated = json.loads(
            (scratch_model_dir / "schemas" / f"{action}.schema.json").read_text()
        )
        committed_backbone = _backbone_properties(_load_committed_schema(action))
        regenerated_backbone = _backbone_properties(regenerated)
        for field in ("teacher_type", "student_type"):
            assert regenerated_backbone[field]["enum"] == committed_backbone[field]["enum"], (
                f"{action}.schema.json backbone options drifted from tao-core: re-run "
                "scripts/generate_dataclass_schemas.py for tao-train-dinov3."
            )
            assert regenerated_backbone[field]["default"] == committed_backbone[field]["default"]
