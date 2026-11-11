# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay comparison policy on self-made files: what gates and what is only reported."""
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
SPEC = importlib.util.spec_from_file_location("kicad_light", Path(__file__).resolve().parents[1] / "kicad_light.py")
light = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(light)


def verdict(tmp_path, name, old, new):
    a, b = tmp_path / ("old_" + name), tmp_path / ("new_" + name)
    old(a)
    new(b)
    result, gating, _ = light.compare(name, a, b)
    return result, gating, result in light.PASSING or not gating


def test_changed_entity_mask_fails_replay(tmp_path):
    ids = np.zeros((4, 5), dtype=np.uint16)
    changed = ids.copy()
    changed[2, 3] = 7
    assert verdict(tmp_path, "entity_ids.npy", lambda p: np.save(p, ids), lambda p: np.save(p, ids))[0] == "identical"
    assert verdict(tmp_path, "entity_ids.npy", lambda p: np.save(p, ids), lambda p: np.save(p, changed)) == \
        ("differs", True, False)
    assert verdict(tmp_path, "entity_ids.npy", lambda p: np.save(p, ids),
                   lambda p: np.save(p, ids.astype(np.uint8))) == ("differs", True, False)


def test_same_values_in_different_container_bytes_pass(tmp_path):
    ids = np.arange(20, dtype=np.uint16).reshape(4, 5)
    result = verdict(tmp_path, "entity_ids.npy", lambda p: np.save(p, ids),
                     lambda p: np.save(p, np.asfortranarray(ids)))
    assert result == ("array_identical", True, True)


def test_board_text_ignores_uuids_only(tmp_path):
    one = '(footprint "A" (uuid "11111111-1111-1111-1111-111111111111") (at 1 2))'
    two = '(footprint "A" (uuid "22222222-2222-2222-2222-222222222222") (at 1 2))'
    moved = two.replace("(at 1 2)", "(at 1 3)")
    assert verdict(tmp_path, "b.kicad_pcb", lambda p: p.write_text(one), lambda p: p.write_text(two)) == \
        ("equal_modulo_uuid", True, True)
    assert verdict(tmp_path, "b.kicad_pcb", lambda p: p.write_text(one), lambda p: p.write_text(moved)) == \
        ("differs", True, False)


def test_json_ignores_timing_but_not_measurements(tmp_path):
    base = {"iou": 0.9, "timing_seconds": {"render": 1.0}}
    slower = {"iou": 0.9, "timing_seconds": {"render": 2.0}}
    worse = {"iou": 0.8, "timing_seconds": {"render": 1.0}}
    dump = lambda value: (lambda p: p.write_text(json.dumps(value)))
    assert verdict(tmp_path, "r.json", dump(base), dump(slower)) == ("equal_modulo_volatile", True, True)
    assert verdict(tmp_path, "r.json", dump(base), dump(worse)) == ("differs", True, False)


def test_scene_containers_are_reported_not_gated(tmp_path):
    result = verdict(tmp_path, "scene.blend", lambda p: p.write_bytes(b"a"), lambda p: p.write_bytes(b"b"))
    assert result == ("bytes_differ_not_gating", False, True)
    assert verdict(tmp_path, "gone.png", lambda p: p.write_bytes(b"a"), lambda p: None) == ("missing", True, False)
