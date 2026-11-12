#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create: footprints whose 3D models are not "one STEP named like the footprint".

U1 and U2 are different footprints that share one model file, named differently
from both and reached through a project variable and a relative path. U3 keeps
its library model and adds two more with their own offset, rotation and scale,
one of them VRML behind a user variable. U4 names a file that does not exist:
binding it is refused, and it is then dropped on purpose and recorded.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pcbnew  # noqa: E402

import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", required=True, help="directory written by make_project.py")
parser.add_argument("--output", required=True)
args = parser.parse_args()
project = Path(args.project).resolve()
k.seed_ids(20261113)
board = k.new_board([(-6, -4), (6, -4), (6, 4), (-6, 4)])


def load(library, name, ref, x, y):
    fp = pcbnew.FootprintLoad(str(k.appdir() / "share/kicad/footprints" / (library + ".pretty")), name)
    fp.SetReference(ref)
    fp.SetPosition(k.point(x, y))
    fp.Reference().SetVisible(False)
    fp.Value().SetVisible(False)
    return fp


u1 = load("Resistor_SMD", "R_0603_1608Metric", "U1", -4.0, -1.5)
u1.Models().clear()
k.add_model(u1, "${KIPRJMOD}/models/shared_chip.step")
k.bind_models(u1, project_dir=project)

u2 = load("Capacitor_SMD", "C_0603_1608Metric", "U2", -4.0, 1.5)
u2.Models().clear()
k.add_model(u2, "models/shared_chip.step")
k.bind_models(u2, project_dir=project)

u3 = load("Resistor_SMD", "R_1206_3216Metric", "U3", 0.5, 0.0)  # keeps the model its library entry declares
k.add_model(u3, str(project / "models/shared_chip.step"), offset_mm=(0, 0, 0.6), rotation_deg=(0, 0, 90), scale=(1, 1, 1.5))
k.add_model(u3, "${EXAMPLE_MODELS}/marker.wrl", offset_mm=(0, 2.0, 0))
k.bind_models(u3, project_dir=project, variables={"EXAMPLE_MODELS": project / "models"})

u4 = load("Resistor_SMD", "R_0805_2012Metric", "U4", 4.0, 0.0)
u4.Models().clear()
k.add_model(u4, "${KIPRJMOD}/models/not_in_this_project.step")
try:
    k.bind_models(u4, project_dir=project)
    refusal = None
except RuntimeError as error:
    refusal = str(error).replace(str(project), "<project>")
k.bind_models(u4, project_dir=project, missing="drop")

for fp in (u1, u2, u3, u4):
    board.Add(fp)
path, meta = k.save(board, args.output, extra={
    "synthetic": True, "parent": None, "missing_model_refusal": refusal,
    "purpose": "shared model under other names, several models with transforms, missing file refused then dropped"})
print(path)
