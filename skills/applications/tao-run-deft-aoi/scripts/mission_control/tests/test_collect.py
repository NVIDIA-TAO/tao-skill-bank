# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for indexer.collect — the worklist of images to embed."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexer import collect

LIGHT, EXT = "SolderLight", ".jpg"

COLS = ["input_path", "golden_path", "label", "object_name", "project"]


def _write_csv(path, rows, cols=COLS):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def _crop(images_dir, input_path, object_name):
    """Create the crop file at the layout collect() expects. Spelled out on
    purpose — see the module docstring."""
    d = Path(images_dir) / input_path
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{object_name}_{LIGHT}{EXT}"
    f.write_bytes(b"x")
    return f


@pytest.fixture
def ws(tmp_path):
    """A minimal workspace: <ws>/kpi/images/, <ws>/results/run_X/."""
    w = tmp_path / "ws"
    (w / "kpi" / "images").mkdir(parents=True)
    (w / "results" / "run_X").mkdir(parents=True)
    return w


# --------------------------------------------------------------------------- #
# _rows_from_component_csv — one CSV -> worklist rows
# --------------------------------------------------------------------------- #

def test_missing_csv_yields_no_rows(ws):
    rows = collect._rows_from_component_csv(
        ws / "kpi" / "nope.csv", ws / "kpi/images", "kpi", "kpi", LIGHT, EXT)
    assert rows == []


def test_header_only_csv_yields_no_rows(ws):
    csv = _write_csv(ws / "kpi/testing_set.csv", [])
    assert collect._rows_from_component_csv(csv, ws / "kpi/images", "kpi", "kpi", LIGHT, EXT) == []


def test_row_carries_every_map_column_and_its_tags(ws):
    csv = _write_csv(ws / "kpi/testing_set.csv",
                     [["board_A", "golden/b/", "PASS", "C1000@1", "proj_9"]])
    (row,) = collect._rows_from_component_csv(
        csv, ws / "kpi/images", "kpi", "kpi", LIGHT, EXT)

    assert set(row) == {"filepath", "kind", "split", "label",
                        "object_name", "input_path", "golden_path", "board"}
    assert row["filepath"].endswith(f"/board_A/C1000@1_{LIGHT}{EXT}")
    assert (row["kind"], row["split"]) == ("kpi", "kpi")
    assert (row["label"], row["object_name"]) == ("PASS", "C1000@1")
    assert row["board"] == "proj_9"          # sourced from the CSV's `project`
    assert row["golden_path"] == "golden/b/"


def test_tags_are_passed_through_verbatim(ws):
    csv = _write_csv(ws / "train/base/validation_set.csv",
                     [["board_A", "", "PASS", "C1", "p"]])
    (row,) = collect._rows_from_component_csv(
        csv, ws / "kpi/images", "pool", "val", LIGHT, EXT)
    assert (row["kind"], row["split"]) == ("pool", "val")


def test_optional_columns_absent_default_to_empty_string(ws):
    csv = _write_csv(ws / "kpi/testing_set.csv", [["board_A", "PASS", "C1"]],
                     cols=["input_path", "label", "object_name"])
    (row,) = collect._rows_from_component_csv(
        csv, ws / "kpi/images", "kpi", "kpi", LIGHT, EXT)
    assert row["golden_path"] == ""
    assert row["board"] == ""


def test_nan_cells_become_empty_string_not_a_literal_nan_path(ws):
    csv = _write_csv(ws / "kpi/testing_set.csv", [[None, None, "PASS", "C1", None]])
    (row,) = collect._rows_from_component_csv(
        csv, ws / "kpi/images", "kpi", "kpi", LIGHT, EXT)
    assert "nan" not in row["filepath"].split("/")
    assert row["golden_path"] == "" and row["board"] == ""


def test_light_and_ext_reach_the_filename(ws):
    csv = _write_csv(ws / "kpi/testing_set.csv", [["b", "", "PASS", "C1", "p"]])
    (row,) = collect._rows_from_component_csv(
        csv, ws / "kpi/images", "kpi", "kpi", "WhiteLight", ".png")
    assert row["filepath"].endswith("/b/C1_WhiteLight.png")


# --------------------------------------------------------------------------- #
# _rows_from_synth — AnomalyGen NG/OK pairs staged into training
# --------------------------------------------------------------------------- #

def _synth(ws, iter_n, ng_names, ok_names=()):
    rd = ws / "results" / "run_X"
    ng = rd / f"iter{iter_n}/dataset/images/synthetic_iter{iter_n}_ng"
    ng.mkdir(parents=True, exist_ok=True)
    for n in ng_names:
        (ng / f"{n}_{LIGHT}{EXT}").write_bytes(b"x")
    if ok_names:
        ok = rd / f"iter{iter_n}/dataset/images/synthetic_iter{iter_n}_ok"
        ok.mkdir(parents=True, exist_ok=True)
        for n in ok_names:
            (ok / f"{n}_{LIGHT}{EXT}").write_bytes(b"x")
    return rd


@pytest.mark.parametrize("stem, expected", [
    ("IC+bridge_00000", "bridge"),
    ("IC+missing_00001", "missing"),
    ("passive_component+excess_solder_00003", "excess_solder"),
    ("unparseable_00002", "ng"),
])
def test_defect_label_parsed_from_ng_filename(ws, stem, expected):
    rd = _synth(ws, 1, [stem])
    (row,) = collect._rows_from_synth(rd, LIGHT, EXT)
    assert row["label"] == expected
    assert row["object_name"] == stem


def test_ok_partners_are_labelled_pass(ws):
    rd = _synth(ws, 1, ["IC+bridge_00000"], ok_names=["IC+bridge_00000"])
    labels = sorted(r["label"] for r in collect._rows_from_synth(rd, LIGHT, EXT))
    assert labels == ["PASS", "bridge"]


def test_missing_ok_directory_still_yields_ng_rows(ws):
    rd = _synth(ws, 1, ["IC+bridge_00000"])          # no _ok dir at all
    rows = collect._rows_from_synth(rd, LIGHT, EXT)
    assert len(rows) == 1 and rows[0]["label"] == "bridge"


def test_synth_rows_are_tagged_pool_synth(ws):
    rd = _synth(ws, 1, ["IC+bridge_00000"])
    (row,) = collect._rows_from_synth(rd, LIGHT, EXT)
    assert (row["kind"], row["split"], row["board"]) == ("pool", "synth", "synthetic")


def test_synth_input_path_is_workspace_relative(ws):
    rd = _synth(ws, 2, ["IC+bridge_00000"])
    (row,) = collect._rows_from_synth(rd, LIGHT, EXT)
    assert row["input_path"] == "results/run_X/iter2/dataset/images/synthetic_iter2_ng"


def test_all_iterations_are_collected(ws):
    _synth(ws, 1, ["IC+bridge_00000"])
    rd = _synth(ws, 2, ["IC+missing_00000"])
    assert len(collect._rows_from_synth(rd, LIGHT, EXT)) == 2


def test_files_of_other_extensions_are_ignored(ws):
    rd = _synth(ws, 1, ["IC+bridge_00000"])
    (rd / "iter1/dataset/images/synthetic_iter1_ng/notes.txt").write_bytes(b"x")
    assert len(collect._rows_from_synth(rd, LIGHT, EXT)) == 1


# --------------------------------------------------------------------------- #
# collect — assembly, existence filter, KPI-wins de-dup
# --------------------------------------------------------------------------- #

def test_empty_workspace_returns_empty_frame(ws):
    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert df.empty


def test_gathers_kpi_train_and_val_splits(ws):
    images_dir = ws / "kpi/images"
    for board, obj in [("b1", "C1"), ("b2", "C2"), ("b3", "C3")]:
        _crop(images_dir, board, obj)
    _write_csv(ws / "kpi/testing_set.csv", [["b1", "", "PASS", "C1", "p"]])
    _write_csv(ws / "train/base/training_set.csv", [["b2", "", "PASS", "C2", "p"]])
    _write_csv(ws / "train/base/validation_set.csv", [["b3", "", "PASS", "C3", "p"]])

    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert sorted(df["split"]) == ["kpi", "train", "val"]
    assert sorted(df["kind"]) == ["kpi", "pool", "pool"]


def test_rows_naming_absent_files_are_dropped(ws):
    _crop(ws / "kpi/images", "b1", "C1")            # only C1 exists on disk
    _write_csv(ws / "kpi/testing_set.csv",
               [["b1", "", "PASS", "C1", "p"], ["b1", "", "PASS", "GHOST", "p"]])

    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert list(df["object_name"]) == ["C1"]


def test_same_crop_in_two_splits_collapses_to_one_kpi_row(ws):
    _crop(ws / "kpi/images", "b1", "C1")
    _write_csv(ws / "kpi/testing_set.csv", [["b1", "", "PASS", "C1", "p"]])
    _write_csv(ws / "train/base/training_set.csv", [["b1", "", "PASS", "C1", "p"]])

    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert len(df) == 1
    assert df.iloc[0]["kind"] == "kpi" and df.iloc[0]["split"] == "kpi"


def test_synthetic_images_join_the_worklist(ws):
    _crop(ws / "kpi/images", "b1", "C1")
    _write_csv(ws / "kpi/testing_set.csv", [["b1", "", "PASS", "C1", "p"]])
    _synth(ws, 1, ["IC+bridge_00000"])

    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert sorted(df["split"]) == ["kpi", "synth"]


def test_result_columns_and_index_are_stable(ws):
    _crop(ws / "kpi/images", "b1", "C1")
    _write_csv(ws / "kpi/testing_set.csv", [["b1", "g/", "PASS", "C1", "p"]])

    df = collect.collect(ws / "results/run_X", ws, LIGHT, EXT)
    assert list(df.columns) == ["filepath", "kind", "split", "label",
                                "object_name", "input_path", "golden_path", "board"]
    assert list(df.index) == list(range(len(df)))   # reset_index, no _kpi leftover
