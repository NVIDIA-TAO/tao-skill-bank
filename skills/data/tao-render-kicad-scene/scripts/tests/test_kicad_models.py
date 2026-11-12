# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model binding through pcbnew, run in KiCad's own interpreter. Skipped without KICAD_APPDIR."""
import json
import os
from pathlib import Path
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
APPDIR = os.environ.get("KICAD_APPDIR")
pytestmark = pytest.mark.skipif(not APPDIR or not Path(APPDIR, "AppRun").is_file(), reason="KICAD_APPDIR not set")

PROBE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
out = Path(sys.argv[2])
import pcbnew
import kicad_sdk as k

k.seed_ids(11)
library = str(k.appdir() / "share/kicad/footprints/Resistor_SMD.pretty")
by_variable = "${KICAD10_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_0805_2012Metric.step"
absolute = str(k.model_dir() / "Resistor_SMD.3dshapes/R_0805_2012Metric.step")
result = {}


def footprint(ref, name="R_0603_1608Metric"):  # the model name never matches this footprint
    fp = pcbnew.FootprintLoad(library, name)
    fp.SetReference(ref)
    fp.Models().clear()
    return fp


# One footprint, two entries of the same file under different references and transforms.
board = k.new_board([(0, 0), (10, 0), (10, 10), (0, 10)])
fp = footprint("R1")
k.add_model(fp, by_variable, (1, 2, 3), (10, 20, 30), (1.1, 1.2, 1.3))
k.add_model(fp, absolute, (4, 5, 6), (40, 50, 60), (2.1, 2.2, 2.3))
k.bind_models(fp)
board.Add(fp)
# Two boards that both call their part R1, bound before either is saved.
other = k.new_board([(0, 0), (10, 0), (10, 10), (0, 10)])
fp2 = footprint("R1")
k.add_model(fp2, absolute)
k.bind_models(fp2)
other.Add(fp2)
_, meta = k.save(board, out / "two_entries")
_, meta_other = k.save(other, out / "same_refdes")
result["two_entries"] = meta["footprints"][0]["models"]
result["same_refdes"] = meta_other["footprints"][0]["models"]
result["stored"] = (out / "two_entries/board.kicad_pcb").read_text().count('(model "/')

# A missing file is refused; dropping it on purpose is recorded for that footprint only.
board = k.new_board([(0, 0), (10, 0), (10, 10), (0, 10)])
kept, dropped = footprint("R1"), footprint("R2")
k.add_model(kept, by_variable)
k.bind_models(kept)
k.add_model(dropped, "${KICAD10_3DMODEL_DIR}/Resistor_SMD.3dshapes/not_there.step")
try:
    k.bind_models(dropped)
    result["refusal"] = None
except RuntimeError as error:
    result["refusal"] = str(error)
k.bind_models(dropped, missing="drop")
board.Add(kept)
board.Add(dropped)
_, meta = k.save(board, out / "dropped")
result["dropped"] = {f["refdes"]: [len(f["models"]), f["models_dropped"]] for f in meta["footprints"]}

# A board that still names an unresolved model cannot be saved.
board = k.new_board([(0, 0), (10, 0), (10, 10), (0, 10)])
loose = footprint("R3")
k.add_model(loose, "${UNDEFINED_HERE}/part.step")
board.Add(loose)
try:
    k.save(board, out / "unresolved")
    result["save_refusal"] = None
except RuntimeError as error:
    result["save_refusal"] = str(error)
result["unresolved_board_left_behind"] = (out / "unresolved/board.kicad_pcb").exists()
(out / "result.json").write_text(json.dumps(result))
'''


def test_binding_keeps_each_entry_and_refuses_missing_files(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    done = subprocess.run([str(Path(APPDIR, "AppRun")), "python3.11", str(script), str(SCRIPTS), str(tmp_path)],
                          capture_output=True, text=True, env=dict(os.environ, KICAD_APPDIR=APPDIR))
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads((tmp_path / "result.json").read_text())

    first, second = result["two_entries"]
    assert first["declared"].startswith("${KICAD10_3DMODEL_DIR}/") and "runtime model library" in first["via"]
    assert second["declared"] == second["resolved_path"] and second["via"] == "absolute path"
    assert first["resolved_path"] == second["resolved_path"] and first["sha256"] == second["sha256"]
    assert first["offset_mm"] == [1, 2, 3] and second["rotation_deg"] == [40, 50, 60]
    assert first["scale"] == pytest.approx([1.1, 1.2, 1.3]) and second["scale"] == pytest.approx([2.1, 2.2, 2.3])
    assert result["stored"] == 2  # the board itself names the resolved files

    assert result["same_refdes"][0]["via"] == "absolute path"  # not overwritten by the other board's R1
    assert "not_there.step" in result["refusal"] and "file not found" in result["refusal"]
    assert result["dropped"]["R1"] == [1, []]
    assert result["dropped"]["R2"][0] == 0 and "not_there.step" in result["dropped"]["R2"][1][0]["declared"]
    assert "UNDEFINED_HERE" in result["save_refusal"] and not result["unresolved_board_left_behind"]
