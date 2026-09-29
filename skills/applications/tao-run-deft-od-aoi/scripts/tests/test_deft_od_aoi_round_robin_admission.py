# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "deft_od_aoi_round_robin_admission.py"
SPEC = importlib.util.spec_from_file_location("deft_od_aoi_round_robin_admission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _policy() -> dict:
    return {"admission": {
        "duplicate_global_cosine": 0.995,
        "duplicate_defect_cosine": 0.985,
        "duplicate_position_delta": 0.04,
        "clean_duplicate_global_cosine": 0.999,
        "clean_cluster_cosine": 0.97,
        "clean_cluster_minimum_cap": 2,
        "clean_cluster_quota_divisor": 4,
        "duplicate_gt_iou": 0.9,
        "minimum_box_area_px": 64,
        "maximum_box_aspect": 25.0,
    }}


def _image(path: Path, invert: bool = False) -> None:
    row, column = np.indices((32, 32))
    array = ((row * 7 + column * 11) % 255).astype(np.uint8)
    if invert:
        array = 255 - array
    Image.fromarray(array).save(path)


def test_real_admission_screens_boxes_and_records_survivors(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    _image(image)
    admission = MODULE.RoundRobinAdmission(_policy(), None)
    ranked = [{"candidate_id": "one", "source_filepath": str(image)}]

    selected = admission.admit(
        ranked, 1, False,
        lambda _: {"boxes": [[0, 0, 2, 2], [4, 4, 10, 10], [4, 4, 10, 10]]},
    )

    assert selected[0]["admission_boxes"] == [[4.0, 4.0, 10.0, 10.0]]
    assert admission.report["boxes_quarantined"] == 2
    assert admission.report["admitted"] == 1


def test_visual_duplicate_index_persists_between_iterations(tmp_path: Path) -> None:
    first, duplicate = tmp_path / "first.png", tmp_path / "duplicate.png"
    _image(first)
    _image(duplicate)
    boxes = [[4, 4, 10, 10]]
    admission = MODULE.RoundRobinAdmission(_policy(), None)
    assert admission.admit(
        [{"candidate_id": "first", "source_filepath": str(first)}],
        1, False, lambda _: {"boxes": boxes},
    )
    index = tmp_path / "admission_index.npy"
    admission.save(index)

    resumed = MODULE.RoundRobinAdmission(_policy(), index)
    selected = resumed.admit(
        [{"candidate_id": "duplicate", "source_filepath": str(duplicate)}],
        1, False, lambda _: {"boxes": boxes},
    )

    assert selected == []
    assert resumed.report["rejected_duplicate"] == 1


def test_admission_rejects_missing_previous_index(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npy"

    try:
        MODULE.RoundRobinAdmission(_policy(), missing)
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("missing previous admission index was accepted")
