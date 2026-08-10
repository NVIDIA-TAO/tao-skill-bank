# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for indexer.layout — locating a run's workspace inputs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexer import layout  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """A workspace laid out per references/data-layout.md."""
    w = tmp_path / "ws"
    (w / "kpi" / "images").mkdir(parents=True)
    (w / "train" / "base").mkdir(parents=True)
    (w / "kpi" / "testing_set.csv").write_text("x")
    (w / "train" / "base" / "training_set.csv").write_text("x")
    (w / "train" / "base" / "validation_set.csv").write_text("x")
    (w / "results" / "run_X").mkdir(parents=True)
    return w


def _state(ws):
    """deft_state.config as init_deft_state.py writes it."""
    return {"config": {
        "images_dir": str(ws / "kpi" / "images"),
        "kpi_test_csv": str(ws / "kpi" / "testing_set.csv"),
        "training_csv": str(ws / "train" / "base" / "training_set.csv"),
        "validation_csv": str(ws / "train" / "base" / "validation_set.csv"),
    }}


def test_recorded_paths_are_preferred_when_they_resolve(ws):
    r = layout.resolve(ws / "results" / "run_X", _state(ws))
    assert set(r["sources"].values()) == {"config"}
    assert r["images_dir"] == ws / "kpi" / "images"
    assert r["workspace"] == ws


def test_layout_is_used_when_there_is_no_state(ws):
    r = layout.resolve(ws / "results" / "run_X", None)
    assert set(r["sources"].values()) == {"layout"}
    assert r["images_dir"] == ws / "kpi" / "images"       # same answer, derived
    assert r["kpi_test_csv"] == ws / "kpi" / "testing_set.csv"


def test_a_recorded_path_that_no_longer_exists_falls_back(ws):
    # the signature of a relocated run: config still names the origin machine
    state = {"config": {"images_dir": "/gone/kpi/images",
                        "kpi_test_csv": "/gone/kpi/testing_set.csv",
                        "training_csv": "/gone/train/base/training_set.csv",
                        "validation_csv": "/gone/train/base/validation_set.csv"}}
    r = layout.resolve(ws / "results" / "run_X", state)
    assert set(r["sources"].values()) == {"layout"}
    assert r["images_dir"] == ws / "kpi" / "images"       # recovered locally


def test_each_path_is_decided_independently(ws):
    # a partially-stale config must not discard the entries that still resolve
    state = _state(ws)
    state["config"]["training_csv"] = "/gone/training_set.csv"
    r = layout.resolve(ws / "results" / "run_X", state)
    assert r["sources"]["images_dir"] == "config"
    assert r["sources"]["training_csv"] == "layout"
    assert r["training_csv"] == ws / "train" / "base" / "training_set.csv"


def test_a_non_standard_workspace_location_is_honoured(tmp_path):
    # if a run ever records inputs outside the documented tree, follow them
    odd = tmp_path / "somewhere" / "else"
    (odd / "kpi" / "images").mkdir(parents=True)
    rd = tmp_path / "results" / "run_X"
    rd.mkdir(parents=True)
    r = layout.resolve(rd, {"config": {"images_dir": str(odd / "kpi" / "images")}})
    assert r["images_dir"] == odd / "kpi" / "images"
    assert r["workspace"] == odd            # derived from images_dir, not position


def test_missing_config_keys_fall_back_individually(ws):
    r = layout.resolve(ws / "results" / "run_X", {"config": {}})
    assert set(r["sources"].values()) == {"layout"}


def test_describe_is_quiet_when_the_images_are_present(ws):
    assert layout.describe(layout.resolve(ws / "results" / "run_X", _state(ws))) is None


def test_describe_warns_when_nothing_resolves(tmp_path):
    rd = tmp_path / "moved" / "run_X"
    rd.mkdir(parents=True)
    msg = layout.describe(layout.resolve(rd, None))
    assert msg and "no KPI images" in msg
