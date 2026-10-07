# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the packaged bounded PAS intake discovery."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "skills/applications/tao-run-deft-pas/scripts/discover_pas_inputs.py"
SPEC = importlib.util.spec_from_file_location("discover_pas_inputs", SCRIPT)
assert SPEC and SPEC.loader
discovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = discovery
SPEC.loader.exec_module(discovery)


def _pair(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "images_raw.tar").write_bytes(b"images")
    (path / "meta.tar.gz").write_bytes(b"metadata")


def test_discovery_includes_depth_two_but_never_depth_three(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _pair(home / "pas" / "one" / "two")
    _pair(home / "pas" / "one" / "other" / "three")

    result = discovery.discover(workspace=workspace, home=home)

    paths = {item["path"] for item in result["candidates"]}
    assert str(home / "pas" / "one" / "two") in paths
    assert str(home / "pas" / "one" / "other" / "three") not in paths


def test_discovery_does_not_follow_symlinked_directories(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    _pair(outside)
    (home / "pas").mkdir(parents=True)
    os.symlink(outside, home / "pas" / "linked")

    result = discovery.discover(workspace=workspace, home=home)

    assert result["candidates"] == []
    assert result["broad_home_repository_search"] is False


def test_multiple_or_mixed_candidates_are_never_selected_by_search_order(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = home / "pas" / "first"
    second = home / "pas" / "second"
    _pair(first)
    _pair(second)
    (first / "images").mkdir()

    result = discovery.discover(workspace=workspace, home=home)

    assert result["selection"] is None
    assert result["selection_reason"] == "user-choice-required"
    assert {item["type"] for item in result["candidates"]} == {
        "archive-only",
        "archive-with-extracted-data",
    }


def test_explicit_pair_does_not_widen_search_after_validation_failure(tmp_path):
    images = tmp_path / "archives/images_raw.tar"
    images.parent.mkdir()
    images.write_bytes(b"images")
    metadata = tmp_path / "elsewhere/meta.tar.gz"
    metadata.parent.mkdir()
    metadata.write_bytes(b"metadata")

    try:
        discovery._explicit_pair(images, metadata)  # noqa: SLF001
    except ValueError as exc:
        assert "share one parent" in str(exc)
    else:
        raise AssertionError("an invalid explicit pair must not trigger discovery")
