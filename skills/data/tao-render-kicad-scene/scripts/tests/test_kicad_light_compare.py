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


def test_run_directory_inside_text_artifacts_is_read_as_run(tmp_path):
    old_root, new_root = tmp_path / "run", tmp_path / "run-replay"
    for root in (old_root, new_root):
        root.mkdir()
        (root / "board.kicad_pcb").write_text(f'(model "{root}/models/part.step")\n')
        (root / "cad.json").write_text(json.dumps({"path": f"{root}/models/part.step", "seconds": 1 if root == old_root else 2}))
    for name, expected in (("board.kicad_pcb", "equal_modulo_uuid"), ("cad.json", "equal_modulo_volatile")):
        assert light.compare(name, old_root / name, new_root / name)[0] == "differs"            # without roots
        assert light.compare(name, old_root / name, new_root / name, (old_root, new_root))[0] == expected
    (new_root / "board.kicad_pcb").write_text(f'(model "{new_root}/models/other.step")\n')       # a real change
    assert light.compare("board.kicad_pcb", old_root / "board.kicad_pcb", new_root / "board.kicad_pcb",
                         (old_root, new_root))[0] == "differs"


def run_pair(tmp_path, reference_new_bytes, score_reference_hash=None):
    """An original run and a replay: a reference PNG rendered in the run, and a score JSON that names its hash."""
    PIL = pytest.importorskip("PIL.Image")
    pixels = np.full((4, 4, 3), 90, dtype=np.uint8)
    old_root, new_root = tmp_path / "run", tmp_path / "replay"
    for root in (old_root, new_root):
        root.mkdir()
    PIL.fromarray(pixels).save(old_root / "reference.png", compress_level=1)
    reference_new_bytes(PIL, pixels, new_root / "reference.png")
    old_hash, new_hash = light.sha(old_root / "reference.png"), light.sha(new_root / "reference.png")
    (old_root / "score.json").write_text(json.dumps({"reference_sha256": old_hash, "loss": 0.25}))
    (new_root / "score.json").write_text(json.dumps({"reference_sha256": score_reference_hash or new_hash, "loss": 0.25}))
    original = {"steps": [{"name": "reference", "outputs_sha256": {"reference.png": old_hash}},
                          {"name": "score", "outputs_sha256": {"score.json": light.sha(old_root / "score.json")}}]}
    rows, passed = light.compare_runs(original, old_root, new_root)
    return {r["artifact"]: r["verdict"] for r in rows}, passed, old_hash != new_hash


def test_hash_of_a_regenerated_reference_passes_only_through_a_proven_pixel_match(tmp_path):
    verdicts, passed, bytes_differ = run_pair(
        tmp_path, lambda PIL, px, path: PIL.fromarray(px).save(path, compress_level=9))
    assert bytes_differ and passed
    assert verdicts == {"reference.png": "pixel_identical", "score.json": "equal_modulo_volatile"}


def test_changed_reference_pixels_or_a_foreign_reference_hash_still_fail(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    verdicts, passed, _ = run_pair(tmp_path / "a", lambda PIL, px, path: PIL.fromarray(px + 40).save(path))
    assert not passed and verdicts == {"reference.png": "differs", "score.json": "differs"}
    verdicts, passed, _ = run_pair(tmp_path / "b", lambda PIL, px, path: PIL.fromarray(px).save(path, compress_level=9),
                                   score_reference_hash="0" * 64)   # the score names some other reference
    assert not passed and verdicts["score.json"] == "differs"


def test_input_from_outside_the_run_gates_and_skill_files_are_reported():
    original = {"steps": [{"name": "score", "inputs_sha256": {"/data/reference.jpg": "a" * 64,
                                                             "{skill}/scripts/x.py": "b" * 64, "{run}/01/board.glb": "c" * 64}}]}
    same = {"steps": [{"name": "score", "inputs_sha256": dict(original["steps"][0]["inputs_sha256"])}]}
    swapped = {"steps": [{"name": "score", "inputs_sha256": {"/data/reference.jpg": "d" * 64,
                                                            "{skill}/scripts/x.py": "e" * 64, "{run}/01/board.glb": "f" * 64}}]}
    assert light.input_changes(original, same) == ([], [])
    external, skill = light.input_changes(original, swapped)
    assert external == [{"step": "score", "input": "/data/reference.jpg"}]
    assert skill == [{"step": "score", "input": "{skill}/scripts/x.py"}]
