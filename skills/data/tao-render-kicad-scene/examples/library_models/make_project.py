#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make a small project directory with two model files for the library_models example.

`models/shared_chip.step` is a copy of a KiCad library model under a name no
footprint has; `models/marker.wrl` is a hand-written VRML box. Both are
created in the run directory, never in the repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
models = Path(args.output) / "models"
models.mkdir(parents=True)
source = Path(os.environ["KICAD_APPDIR"]) / "share/kicad/3dmodels/Resistor_SMD.3dshapes/R_0805_2012Metric.step"
shutil.copyfile(source, models / "shared_chip.step")
(models / "marker.wrl").write_text(
    "#VRML V2.0 utf8\n"
    "Shape { appearance Appearance { material Material { diffuseColor 0.8 0.1 0.1 } }\n"
    "        geometry Box { size 0.4 0.4 0.2 } }\n")  # KiCad reads VRML in units of 2.54 mm
facts = {name: hashlib.sha256((models / name).read_bytes()).hexdigest() for name in ("shared_chip.step", "marker.wrl")}
(Path(args.output) / "project.json").write_text(json.dumps({"models_sha256": facts, "copied_from": source.name}, indent=2) + "\n")
print(json.dumps(facts))
