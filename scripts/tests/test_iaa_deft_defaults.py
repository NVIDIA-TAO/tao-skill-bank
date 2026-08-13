# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for IAA DEFT workflow defaults."""

import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_ROOT = REPO_ROOT / "skills/applications/tao-run-deft-iaa"


def _prepare_module():
    path = IAA_ROOT / "scripts/prepare_deft_config.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("prepare_deft_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_continual_model_is_the_template_and_cli_default():
    config = yaml.safe_load((IAA_ROOT / "specs/deft_config.yaml").read_text())
    assert config["training"]["continual_dataset"] is True
    assert config["training"]["continual_model"] is True

    parser = _prepare_module()._parser()
    action = next(item for item in parser._actions if item.dest == "continual_model")
    assert action.default is True
