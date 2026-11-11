#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modify: load the parent board read-only, edit it with the SDK, save a new revision.

Three edits: shift/rotate C2 (placement), add an 0603 resistor (new footprint),
and route C1.2 -> R1.1 (copper). The parent file is never written.
"""
import argparse
from pathlib import Path
import sys

import pcbnew

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parent", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--c2-dx", type=float, default=0.25)
parser.add_argument("--c2-angle", type=float, default=90.0)
parser.add_argument("--seed", type=int, default=20261111)
args = parser.parse_args()
k.seed_ids(args.seed)

board = pcbnew.LoadBoard(args.parent)
c2 = k.find(board, "C2")
c2.SetPosition(k.point(pcbnew.ToMM(c2.GetPosition().x) + args.c2_dx, pcbnew.ToMM(c2.GetPosition().y)))
c2.SetOrientationDegrees(args.c2_angle)
r1 = k.place(board, "Resistor_SMD", "R_0603_1608Metric", "R1", 1.2, -1.0, 0.0)
c1_pad2 = next(p for p in k.find(board, "C1").Pads() if p.GetNumber() == "2")
r1_pad1 = next(p for p in r1.Pads() if p.GetNumber() == "1")
k.connect(board, "N1", [c1_pad2, r1_pad1], width_mm=0.2, via_points_mm=[(0.48, -1.0)])
path, meta = k.save(board, args.output, extra={
    "synthetic": True, "parent": {"file": Path(args.parent).name, "sha256": k.sha(args.parent)},
    "edits": [{"op": "move+rotate", "ref": "C2", "dx_mm": args.c2_dx, "angle_deg": args.c2_angle},
              {"op": "add_footprint", "ref": "R1", "footprint": "Resistor_SMD:R_0603_1608Metric"},
              {"op": "route", "net": "N1", "pads": ["C1:2", "R1:1"]}]})
print(path, len(meta["pads"]), "pads", len(meta["tracks"]), "tracks")
