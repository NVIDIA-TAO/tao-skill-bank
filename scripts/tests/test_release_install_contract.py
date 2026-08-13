# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the release delivered by the default marketplace."""

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_branch_publishes_7_2_images_and_new_plugin_version():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    assert "7.2.0" in images["pyt"]
    assert "7.2.0" in images["data_services"]

    manifests = [
        json.loads((REPO_ROOT / ".claude-plugin/plugin.json").read_text()),
        json.loads((REPO_ROOT / ".codex-plugin/plugin.json").read_text()),
    ]
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin/marketplace.json").read_text()
    )
    advertised = marketplace["metadata"]["version"]
    assert advertised == "0.1.13"
    assert {manifest["version"] for manifest in manifests} == {advertised}


def test_iaa_stamped_pins_match_published_versions():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    scripts = REPO_ROOT / "skills/applications/tao-run-deft-iaa/scripts"
    for path in (
        scripts / "prepare_deft_config.py",
        scripts / "init_deft_state.py",
        scripts / "audit_deft_run.py",
        scripts / "run_deft_container.py",
    ):
        text = path.read_text()
        assert images["pyt"] in text
        assert images["data_services"] in text
