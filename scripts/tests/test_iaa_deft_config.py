# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema and value validation tests for the bundled IAA DEFT config."""

import copy
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_ROOT = REPO_ROOT / "skills/applications/tao-run-deft-iaa"
sys.path.insert(0, str(IAA_ROOT / "scripts"))
from iaa_deft.config import IaaDeftConfig  # noqa: E402


def _template():
    return yaml.safe_load((IAA_ROOT / "specs/deft_config.yaml").read_text())


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "deft_config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _set(payload, dotted: str, value):
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_shipped_config_passes_full_validation():
    config = IaaDeftConfig(str(IAA_ROOT / "specs/deft_config.yaml"))
    assert config.mining_topn == 25


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mining.topn", 0),
        ("mining.topn", None),
        ("mining.topn", ""),
        ("experiment.visualize", "not_a_number"),
        ("experiment.visualize_embeddings", -1),
        ("training.continual_model", "true"),
        ("iaa.eval_caption_dir", -1),
        ("iaa.mining_pool_mode", "raal"),
        ("gap_analysis.queries_per_slice", None),
        ("gap_analysis.query_types", -1),
        ("gap_analysis.caption_diversity.max_rows_per_image_path", 0),
        ("mining.history_aware.replay_fraction", 2.0),
    ],
)
def test_present_invalid_values_are_refused_by_full_key(tmp_path, key, value):
    payload = copy.deepcopy(_template())
    _set(payload, key, value)
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError) as error:
        IaaDeftConfig(str(path))

    assert str(path) in str(error.value)
    assert key in str(error.value)


@pytest.mark.parametrize("key", ["mining.typo_topn", "unknown_section"])
def test_unknown_keys_are_refused(tmp_path, key):
    payload = copy.deepcopy(_template())
    _set(payload, key, 99)
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=key):
        IaaDeftConfig(str(path))


def test_only_absent_optional_values_take_defaults(tmp_path):
    payload = copy.deepcopy(_template())
    del payload["mining"]["topn"]
    path = _write(tmp_path, payload)

    assert IaaDeftConfig(str(path)).mining_topn == 5


@pytest.mark.parametrize("mode", ["real", "augmented", "real_and_augmented"])
def test_mining_pool_mode_accepts_only_documented_values(tmp_path, mode):
    payload = copy.deepcopy(_template())
    payload["iaa"]["mining_pool_mode"] = mode
    config = IaaDeftConfig(str(_write(tmp_path, payload)))
    assert config.iaa_mining_pool_mode == mode
