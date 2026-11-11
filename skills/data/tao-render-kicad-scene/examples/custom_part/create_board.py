#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create: a board with one SDK-built SMD footprint, for a part the standard library does not match.

Pads are given as `number dx dy width height` in mm relative to the footprint
origin. The part's body is not in KiCad here: add it as a `box` asset in the
scene JSON. Use a library footprint instead whenever one fits.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--board-mm", type=float, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--ref", default="X1")
parser.add_argument("--at-mm", type=float, nargs=2, required=True, metavar=("X", "Y"))
parser.add_argument("--angle-deg", type=float, default=0.0)
parser.add_argument("--pad", nargs=5, action="append", required=True, metavar=("NUM", "DX", "DY", "W", "H"))
parser.add_argument("--scale-basis", default="synthetic fixture")
parser.add_argument("--seed", type=int, default=20261111)
args = parser.parse_args()
k.seed_ids(args.seed)

width, height = args.board_mm
board = k.new_board([(0, 0), (width, 0), (width, height), (0, height)])
pads = [(p[0], *map(float, p[1:])) for p in args.pad]
k.smd_footprint(board, args.ref, pads, args.at_mm[0], args.at_mm[1], args.angle_deg)
path, meta = k.save(board, args.output, extra={
    "synthetic": args.scale_basis == "synthetic fixture", "parent": None, "scale_basis": args.scale_basis,
    "representation": "SDK-built footprint (pads only); body is a custom asset in the scene JSON"})
print(path, [(p["id"], p["center_mm"], p["size_mm"]) for p in meta["pads"]])
