# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile, colour-cluster regions, per-region appearance and search helpers on self-drawn images."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL.Image")
SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image, search = load("offline_image"), load("appearance_search")
GREEN, WHITE, TAN = (10, 70, 50), (200, 205, 200), (175, 160, 125)


def scene(tmp_path, body=TAN, name="reference.png"):
    """Green board, white pads at x 20-50 and 110-140, tan body at x 50-110; rows 30-70."""
    rgb = np.zeros((100, 160, 3), dtype=np.uint8)
    rgb[:] = GREEN
    rgb[35:65, 20:50] = WHITE
    rgb[35:65, 110:140] = WHITE
    rgb[30:70, 50:110] = body
    path = tmp_path / name
    PIL.fromarray(rgb).save(path)
    return path


def render_dir(tmp_path, body=TAN, shift=0):
    out = tmp_path / f"render_{shift}_{body[0]}"
    out.mkdir()
    PIL.fromarray(np.asarray(PIL.open(scene(tmp_path, body, f"r_{shift}_{body[0]}.png")))).save(out / "rgb.png")
    ids = np.zeros((100, 160), dtype=np.uint16)
    ids[35:65, 20:50] = 2
    ids[35:65, 110:140] = 2
    ids[30:70, 50 + shift:110 + shift] = 1  # the body sits on top of the pads
    np.save(out / "entity_ids.npy", ids)
    (out / "render_receipt.json").write_text(json.dumps(
        {"entity_ids": {"entities": {"1": "component:X1:part1", "2": "solder:X1:1"}}}))
    return out


def run(func, capsys, **kwargs):
    func(argparse.Namespace(out_json=None, **kwargs))
    return json.loads(capsys.readouterr().out)


def test_profile_finds_material_boundaries(tmp_path, capsys):
    result = run(image.cmd_profile, capsys, image=str(scene(tmp_path)), window=[0, 0, 160, 100], axis="x",
                 band=[45, 55], k=8, min_run=4)
    assert [s["along_px"] for s in result["segments"]] == [[0, 20], [20, 50], [50, 110], [110, 140], [140, 160]]
    body = result["segments"][2]
    assert body["across_px"] == [30, 70] and body["band_mean_rgb"] == [175.0, 160.0, 125.0]
    assert result["segments"][1]["across_px"] == [35, 65]  # the pad is narrower than the body


def test_region_by_colour_cluster_scores_registration(tmp_path, capsys):
    common = dict(image=str(scene(tmp_path)), entity_prefix=["component:X1:part1"], gray_min=None, gray_max=255.0,
                  cluster_rgb=[list(map(float, TAN))], k=8, within=None, all_parts=False)
    exact = run(image.cmd_region, capsys, render=str(render_dir(tmp_path)), **common)
    moved = run(image.cmd_region, capsys, render=str(render_dir(tmp_path, shift=10)), **common)
    assert exact["iou"] == 1.0 and exact["centroid_offset_px"] == [0.0, 0.0]
    assert moved["iou"] == pytest.approx(50 / 70) and moved["centroid_offset_px"] == [10.0, 0.0]


def test_rois_loss_is_zero_for_equal_looks_and_sees_a_colour_change(tmp_path, capsys):
    reference = str(scene(tmp_path))
    common = dict(image=reference, roi=["body=component:X1:part1", "solder=solder:@2"], erode=2, min_pixels=50,
                  within=None, mask_render=None)
    same = run(image.cmd_rois, capsys, render=str(render_dir(tmp_path)), **common)
    dark = run(image.cmd_rois, capsys, render=str(render_dir(tmp_path, body=(60, 40, 30))), **common)
    assert same["loss"] == 0.0
    by_name = {r["roi"]: r for r in dark["rois"]}
    assert by_name["solder"]["loss"] == 0.0 and by_name["body"]["loss"] > 0.5
    assert dark["loss"] == pytest.approx(by_name["body"]["loss"] / 3, abs=1e-4)  # weights 1 and 2


def test_search_helpers_edit_only_the_named_keys():
    look = {"materials": {"solder": {"color": [1, 1, 1], "roughness": 0.2}}, "lighting": {"area": [{"power_w": 1.0}]}}
    search.put(look, "lighting/area/0/power_w", 0.5)
    search.put(look, "materials/overrides/component:C1:part0/roughness", 0.3)
    assert look["lighting"]["area"][0]["power_w"] == 0.5 and look["materials"]["solder"]["roughness"] == 0.2
    assert search.get(look, "materials/overrides/component:C1:part0/roughness") == 0.3
    assert search.linear([255, 0, 128]) == pytest.approx([1.0, 0.0, 0.2158], abs=1e-3)


def test_search_plan_may_only_touch_appearance_keys():
    stage = lambda path: {"stages": [{"method": "lhs", "params": [{"path": path, "range": [0, 1]}]}]}  # noqa: E731
    for path in ("materials/solder/roughness", "lighting/area/0/power_w", "render/exposure"):
        search.check_plan(stage(path), {"assets": [], "render": {"samples": 8}})
    for path in ("camera/ortho_scale_mm", "assets/0/amount", "render/resolution", "pad_alignment/mode"):
        with pytest.raises(SystemExit, match="Not appearance keys"):
            search.check_plan(stage(path), {})
    with pytest.raises(SystemExit, match="camera"):
        search.check_plan(stage("materials/solder/roughness"), {"camera": {"ortho_scale_mm": 5}})
    ratio = {"stages": [{"method": "ratio", "targets": [{"roi": "body", "path": "camera/target_mm"}]}]}
    with pytest.raises(SystemExit):
        search.check_plan(ratio, {})
