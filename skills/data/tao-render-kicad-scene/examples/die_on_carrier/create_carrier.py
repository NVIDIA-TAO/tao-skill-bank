#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create: a plain carrier board for a scene whose main object has no footprint (a die, a block).

KiCad owns only what is really board-like here: the carrier outline, thickness and an
optional die-attach pad as F.Cu copper. The die itself is a custom box asset in the
scene JSON; do not force it into a footprint.
"""
import argparse
from pathlib import Path
import sys

import pcbnew

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--width-mm", type=float, required=True)
parser.add_argument("--height-mm", type=float, required=True)
parser.add_argument("--attach-pad-mm", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
parser.add_argument("--scale-basis", default="synthetic fixture")
parser.add_argument("--seed", type=int, default=20261111)
args = parser.parse_args()
k.seed_ids(args.seed)

board = k.new_board([(0, 0), (args.width_mm, 0), (args.width_mm, args.height_mm), (0, args.height_mm)])
if args.attach_pad_mm:
    x0, y0, x1, y1 = args.attach_pad_mm
    pad = pcbnew.SHAPE_POLY_SET()
    pad.NewOutline()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        pad.Append(k.mm(x), k.mm(y))
    k.add_copper(board, pad)
path, meta = k.save(board, args.output, extra={
    "synthetic": args.scale_basis == "synthetic fixture", "parent": None, "scale_basis": args.scale_basis,
    "representation": "generic carrier; the die is a custom asset in the scene JSON, not a footprint"})
print(path, meta["outline_bbox_mm"])
