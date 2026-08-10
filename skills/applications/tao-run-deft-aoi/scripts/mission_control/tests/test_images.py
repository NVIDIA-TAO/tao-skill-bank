# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for indexer.images — the crop-filename convention and join key."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexer import images  # noqa: E402


def _spec(classify):
    """Wrap a `dataset.classify` payload in the surrounding spec structure."""
    return {"dataset": {"classify": classify}}


# --------------------------------------------------------------------------- #
# read_light_ext — the happy path
# --------------------------------------------------------------------------- #

def test_reads_light_and_ext_from_spec():
    got = images.read_light_ext(_spec({"input_map": {"SolderLight": 0}, "image_ext": ".jpg"}))
    assert got == ("SolderLight", ".jpg")


def test_picks_channel_zero_light_by_input_map_value_not_key_order():
    # input_map values are channel indices; channel 0 is the one crops are named
    # after. Dict order must not decide it.
    spec = _spec({"input_map": {"WhiteLight": 2, "SolderLight": 0, "UVLight": 1}})
    assert images.read_light_ext(spec)[0] == "SolderLight"


def test_custom_extension_is_honoured():
    spec = _spec({"input_map": {"SolderLight": 0}, "image_ext": ".png"})
    assert images.read_light_ext(spec) == ("SolderLight", ".png")


def test_returns_plain_strings():
    light, ext = images.read_light_ext(_spec({"input_map": {"SolderLight": 0}}))
    assert type(light) is str and type(ext) is str


# --------------------------------------------------------------------------- #
# read_light_ext — degradation. Must return defaults, never raise.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", [
    pytest.param(None, id="spec-is-None"),
    pytest.param({}, id="empty-spec"),
    pytest.param({"dataset": {}}, id="no-classify-key"),
    pytest.param(_spec(None), id="classify-empty-in-yaml"),
    pytest.param(_spec("some-string"), id="classify-is-a-scalar"),
    pytest.param(_spec({"input_map": ["SolderLight"]}), id="input_map-is-a-list"),
    pytest.param(_spec({"input_map": {"A": "0", "B": 1}}), id="input_map-mixed-value-types"),
    pytest.param(_spec({"input_map": {}}), id="input_map-empty"),
])
def test_malformed_spec_falls_back_to_defaults(spec):
    assert images.read_light_ext(spec) == (images.DEFAULT_LIGHT, images.DEFAULT_EXT)


def test_empty_image_ext_falls_back_rather_than_producing_extensionless_paths():
    spec = _spec({"input_map": {"SolderLight": 0}, "image_ext": ""})
    assert images.read_light_ext(spec)[1] == images.DEFAULT_EXT


def test_missing_image_ext_key_uses_default():
    assert images.read_light_ext(_spec({"input_map": {"SolderLight": 0}}))[1] == images.DEFAULT_EXT


# --------------------------------------------------------------------------- #
# component_file — the join key for the entire map
# --------------------------------------------------------------------------- #

def test_builds_base_inputpath_objectname_light_ext():
    got = images.component_file("/ws/kpi/images", "board_A", "C1000@1", "SolderLight", ".jpg")
    assert got == "/ws/kpi/images/board_A/C1000@1_SolderLight.jpg"


def test_nested_input_path_is_preserved():
    got = images.component_file("/ws", "a/b/c", "C1", "L", ".jpg")
    assert got == "/ws/a/b/c/C1_L.jpg"


def test_absolute_input_path_stays_under_base():
    got = images.component_file("/ws/kpi/images", "/board_A", "C1", "L", ".jpg")
    assert got == "/ws/kpi/images/board_A/C1_L.jpg"


def test_trailing_slash_does_not_double_separator():
    got = images.component_file("/ws", "board_A/", "C1", "L", ".jpg")
    assert got == "/ws/board_A/C1_L.jpg"


def test_defaults_match_the_aoi_convention():
    assert images.component_file("/ws", "b", "C1") == "/ws/b/C1_SolderLight.jpg"


def test_symlinked_and_direct_paths_collapse_to_one_key(tmp_path):
    real = tmp_path / "real"
    (real / "board").mkdir(parents=True)
    (real / "board" / "C1_SolderLight.jpg").write_bytes(b"x")
    link = tmp_path / "via_link"
    link.symlink_to(real)

    direct = images.component_file(real, "board", "C1")
    through_link = images.component_file(link, "board", "C1")
    assert direct == through_link


def test_dot_dot_segments_are_normalized(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    straight = images.component_file(tmp_path / "a", "b", "C1")
    indirect = images.component_file(tmp_path / "a" / "b" / "..", "b", "C1")
    assert straight == indirect


def test_returned_path_is_absolute():
    assert os.path.isabs(images.component_file("relative/base", "b", "C1"))


# --------------------------------------------------------------------------- #
# exists — the filter collect() uses to drop rows that name absent files
# --------------------------------------------------------------------------- #

def test_exists_true_only_for_regular_files(tmp_path):
    f = tmp_path / "crop.jpg"
    f.write_bytes(b"x")
    assert images.exists(f) is True
    assert images.exists(tmp_path) is False            # a directory is not a crop
    assert images.exists(tmp_path / "missing.jpg") is False


# --------------------------------------------------------------------------- #
# read_lights_ext — the full lighting layout (input_map)
# --------------------------------------------------------------------------- #

def test_single_light_run_reports_one_condition():
    lights, ext = images.read_lights_ext(
        _spec({"input_map": {"SolderLight": 0}, "image_ext": ".jpg"}))
    assert (lights, ext) == (["SolderLight"], ".jpg")


def test_lights_are_ordered_by_channel_index_not_dict_order():
    lights, _ = images.read_lights_ext(
        _spec({"input_map": {"UVLight": 2, "SolderLight": 0, "WhiteLight": 1}}))
    assert lights == ["SolderLight", "WhiteLight", "UVLight"]


def test_channel_zero_accessor_agrees_with_the_full_list():
    spec = _spec({"input_map": {"WhiteLight": 1, "SolderLight": 0}})
    lights, ext = images.read_lights_ext(spec)
    assert images.read_light_ext(spec) == (lights[0], ext)


@pytest.mark.parametrize("spec", [None, {}, _spec(None), _spec({"input_map": []})])
def test_malformed_spec_degrades_to_one_default_light(spec):
    assert images.read_lights_ext(spec) == ([images.DEFAULT_LIGHT], images.DEFAULT_EXT)


# --------------------------------------------------------------------------- #
# swap_light — the same component under a different illumination
# --------------------------------------------------------------------------- #

def test_swap_light_targets_the_sibling_capture():
    got = images.swap_light("/ws/b/C1_SolderLight.jpg", "SolderLight", "WhiteLight", ".jpg")
    assert got == "/ws/b/C1_WhiteLight.jpg"


def test_swap_light_only_touches_the_suffix():
    got = images.swap_light("/ws/SolderLight/SolderLight_SolderLight.jpg",
                            "SolderLight", "UVLight", ".jpg")
    assert got == "/ws/SolderLight/SolderLight_UVLight.jpg"


def test_swap_light_leaves_non_conforming_paths_alone():
    p = "/ws/synthetic_iter1_ng/IC+bridge_00000.png"
    assert images.swap_light(p, "SolderLight", "WhiteLight", ".jpg") == p


def test_swapping_to_the_same_light_is_identity():
    p = "/ws/b/C1_SolderLight.jpg"
    assert images.swap_light(p, "SolderLight", "SolderLight", ".jpg") == p
