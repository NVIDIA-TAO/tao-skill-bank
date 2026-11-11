# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline reader geometry on self-drawn masks with known answers."""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
SPEC = importlib.util.spec_from_file_location("offline_image", Path(__file__).resolve().parents[1] / "offline_image.py")
image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image)


def drawn():
    mask = np.zeros((40, 60), dtype=bool)
    mask[5:15, 10:50] = True      # bar: 400 px
    mask[20:35, 4:9] = True       # separate block: 75 px
    mask[8:12, 20:24] = False     # hole in the bar
    return mask


def test_rectangles_tile_the_mask_exactly():
    mask = drawn()
    rebuilt = np.zeros_like(mask)
    for x0, y0, x1, y1 in image.rectangles(mask):
        assert not rebuilt[y0:y1, x0:x1].any()   # no overlap
        rebuilt[y0:y1, x0:x1] = True
    assert (rebuilt == mask).all()


def test_components_and_largest_part():
    mask = drawn()
    parts = sorted(image.components(mask), key=lambda c: -c["area"])
    assert [c["area"] for c in parts] == [400 - 16, 75]
    assert parts[0]["bbox"] == [10, 5, 50, 15] and not parts[0]["touches_border"]
    assert image.largest_part(mask).sum() == 384
    assert len(image.components(~mask)) == 2     # background plus the enclosed hole


def test_region_facts_recover_size_and_axis():
    mask = np.zeros((80, 120), dtype=bool)
    mask[30:50, 20:100] = True                   # 80 x 20 axis-aligned rectangle
    facts = image.region_facts(mask)
    assert facts["largest_bbox_fill"] == 1.0
    assert facts["centroid_px"] == [59.5, 39.5]
    assert facts["major_axis_deg"] in (0.0, 180.0)
    assert facts["rect_equivalent_size_px"] == pytest.approx([80, 20], abs=0.1)


def test_score_detects_shift_and_ssim_is_one_for_identical():
    mask = drawn()
    assert image.score(mask, mask)["iou"] == 1.0
    shifted = np.roll(mask, 1, axis=1)
    result = image.score(mask, shifted)
    assert 0 < result["iou"] < 1 and result["mismatched_pixels"] > 0
    gray = np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.float64)
    assert image.ssim(gray, gray) == pytest.approx(1.0)
    assert image.ssim(gray, 255 - gray) < 0.5
