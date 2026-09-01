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


def test_default_marketplace_exposes_the_canonical_codex_plugin_name():
    """A root marketplace add must expose the name Codex installs.

    The shared marketplace historically listed only the Claude-facing
    ``tao-skills`` alias. Codex reads its canonical name from plugin.json and
    then could not find that name in the marketplace it had just registered.
    Keep the alias for existing Claude installs, but require the canonical
    cross-client entry too.
    """
    codex_manifest = json.loads(
        (REPO_ROOT / ".codex-plugin/plugin.json").read_text()
    )
    shared_marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin/marketplace.json").read_text()
    )
    names = {entry["name"] for entry in shared_marketplace["plugins"]}

    assert codex_manifest["name"] in names
    assert "tao-skills" in names


def test_pas_stamped_pins_match_published_versions():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    scripts = REPO_ROOT / "skills/applications/tao-run-deft-pas/scripts"
    for path in (
        scripts / "prepare_deft_config.py",
        scripts / "init_deft_state.py",
        scripts / "audit_deft_run.py",
        scripts / "run_deft_container.py",
    ):
        text = path.read_text()
        assert images["deft_pas_pyt"] in text
        assert images["deft_pas_data_services"] in text


def test_unpublished_7_2_branch_has_a_runnable_install_path():
    """NVBug 6602338: release candidates must be reachable before tagging."""

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "/plugin marketplace add "
        "https://github.com/NVIDIA-TAO/tao-skill-bank.git#release/7.2.0"
    ) in readme
    assert (
        "codex plugin marketplace add NVIDIA-TAO/tao-skill-bank "
        "--ref release/7.2.0"
    ) in readme
    assert "immutable `@7.2.0` tag after it appears" in readme


def test_claude_marketplace_install_uses_https_for_every_documented_ref():
    """NVBug 6597811: Claude's GitHub shorthand must not select SSH."""

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "/plugin marketplace add NVIDIA-TAO/tao-skill-bank" not in readme
    assert "git@github.com:NVIDIA-TAO/tao-skill-bank" not in readme
    assert (
        "/plugin marketplace add "
        "https://github.com/NVIDIA-TAO/tao-skill-bank.git#7.1.0"
    ) in readme
    assert (
        "/plugin marketplace add "
        "https://github.com/NVIDIA-TAO/tao-skill-bank.git#<new-tag>"
    ) in readme
