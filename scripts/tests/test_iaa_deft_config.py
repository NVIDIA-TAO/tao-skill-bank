# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused configuration validation tests for IAA DEFT."""

import copy
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_ROOT = REPO_ROOT / "skills/applications/tao-run-deft-iaa"
sys.path.insert(0, str(IAA_ROOT / "scripts"))
from iaa_deft.config import IaaDeftConfig  # noqa: E402


def _write_config(tmp_path: Path, *, mining_pool_mode):
    payload = yaml.safe_load((IAA_ROOT / "specs/deft_config.yaml").read_text())
    payload = copy.deepcopy(payload)
    payload["iaa"]["mining_pool_mode"] = mining_pool_mode
    config = tmp_path / "deft_config.yaml"
    config.write_text(yaml.safe_dump(payload))
    return config


@pytest.mark.parametrize("mode", ["real", "augmented", "real_and_augmented"])
def test_mining_pool_mode_accepts_only_documented_values(tmp_path, mode):
    config = IaaDeftConfig(str(_write_config(tmp_path, mining_pool_mode=mode)))
    assert config.iaa_mining_pool_mode == mode


def test_mining_pool_mode_rejects_typo_by_full_key(tmp_path):
    config = _write_config(tmp_path, mining_pool_mode="raal")

    with pytest.raises(ValueError, match=r"iaa\.mining_pool_mode.*raal"):
        IaaDeftConfig(str(config))
