#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that what the footprints declared is what was resolved, exported and rendered."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cad", required=True)
parser.add_argument("--receipt", required=True)
parser.add_argument("--out-json", required=True)
args = parser.parse_args()
cad = {f["refdes"]: f for f in json.loads(Path(args.cad).read_text())["footprints"]}
refusal = json.loads(Path(args.cad).read_text())["missing_model_refusal"]
receipt = json.loads(Path(args.receipt).read_text())
exported = {row["refdes"]: row["models_in_export"] for row in receipt["model_check"]}
parts = {}
for part in receipt["component_parts"]:
    parts.setdefault((part["refdes"], part["model_index"]), []).append(part["bounds_kicad_mm"])


def span(ref, index, axis):
    boxes = parts[(ref, index)]
    return min(b[axis][0] for b in boxes), max(b[axis][1] for b in boxes)


def size(ref, index, axis):
    low, high = span(ref, index, axis)
    return round(high - low, 3)


u1, u2, u3, u4 = (cad[r]["models"] for r in ("U1", "U2", "U3", "U4"))
flat = (size("U1", 0, "x"), size("U1", 0, "y"), size("U1", 0, "z"))
checks = {
    "model name differs from both footprints": all("shared_chip" in m[0]["resolved_path"] for m in (u1, u2)),
    "two footprints share one file": u1[0]["sha256"] == u2[0]["sha256"] == u3[1]["sha256"],
    "each entry keeps its own declared reference": [u1[0]["declared"], u2[0]["declared"]] ==
        ["${KIPRJMOD}/models/shared_chip.step", "models/shared_chip.step"] and u3[2]["declared"] == "${EXAMPLE_MODELS}/marker.wrl",
    "resolution route recorded": [u1[0]["via"], u2[0]["via"], u3[1]["via"], u3[2]["via"]] ==
        ["variable KIPRJMOD (project directory)", "relative to project directory", "absolute path",
         "variable EXAMPLE_MODELS (given)"] and "runtime model library" in u3[0]["via"],
    "every resolved model reached the export": exported == {"U1": 1, "U2": 1, "U3": 3, "U4": 0},
    "rotation 90 about z swaps the footprint of the shared model": (size("U3", 1, "x"), size("U3", 1, "y")) == (flat[1], flat[0]),
    "scale z 1.5 and offset z 0.6 applied": size("U3", 1, "z") == round(flat[2] * 1.5, 3)
        and round(span("U3", 1, "z")[0] - span("U1", 0, "z")[0], 3) == 0.6,
    "offset y 2.0 moves the VRML model up the screen": round(sum(span("U3", 2, "y")) / 2 - cad["U3"]["position_mm"][1], 3) == -2.0,
    "VRML unit is 2.54 mm": (size("U3", 2, "x"), size("U3", 2, "z")) == (1.016, 0.508),
    "missing file refused with its name": bool(refusal) and "not_in_this_project.step" in refusal and "file not found" in refusal,
    "drop is recorded, nothing substituted": u4 == [] and len(cad["U4"]["models_dropped"]) == 1 and cad["U1"]["models_dropped"] == [],
}
report = {"pass": all(checks.values()), "checks": checks,
          "sizes_mm": {"shared model as declared": flat, "U3 second model": [size("U3", 1, a) for a in "xyz"],
                       "U3 VRML model": [size("U3", 2, a) for a in "xyz"]}}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
sys.exit(0 if report["pass"] else 1)
