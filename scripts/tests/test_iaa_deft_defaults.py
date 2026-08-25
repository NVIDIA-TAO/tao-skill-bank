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


def test_managed_endpoints_are_default_and_reuse_requires_explicit_flag():
    config = yaml.safe_load((IAA_ROOT / "specs/sdg_config.yaml").read_text())
    assert config["endpoints"]["ownership"] == "managed"
    assert config["endpoints"]["reuse_requested"] is False

    parser = _prepare_module()._parser()
    mode = next(item for item in parser._actions if item.dest == "sdg_endpoint_mode")
    reuse = next(item for item in parser._actions if item.dest == "reuse_external_endpoints")
    assert mode.default == "managed"
    assert reuse.default is False


def test_generation_nodes_default_is_one_and_template_records_topology():
    config = yaml.safe_load((IAA_ROOT / "specs/sdg_config.yaml").read_text())
    assert config["generation"]["generation_nodes"] == 1
    assert config["generation"]["gpus_per_generation_node"] == 1

    parser = _prepare_module()._parser()
    action = next(item for item in parser._actions if item.dest == "generation_nodes")
    assert action.default == 1


def test_generated_image_port_range_cannot_overlap_vlm_or_llm():
    parser = _prepare_module()._parser()
    defaults = {
        item.dest: item.default
        for item in parser._actions
        if item.dest in {"image_edit_port", "vlm_port", "llm_port"}
    }
    image_ports = set(range(defaults["image_edit_port"], defaults["image_edit_port"] + 4))
    assert image_ports.isdisjoint({defaults["vlm_port"], defaults["llm_port"]})
