#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modify: inside each given box, replace the parent's copper with the donor board's copper.

Use it to reproduce a localized defect on a golden parent: the donor is a board
traced from the defective image and the boxes are where the defect is known to
be. Outside the boxes the parent is untouched. Boxes are x0 y0 x1 y1 in mm.
"""
import argparse
from pathlib import Path
import sys

import pcbnew

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parent", required=True)
parser.add_argument("--donor", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--box-mm", type=float, nargs=4, action="append", required=True)
parser.add_argument("--seed", type=int, default=20261113)
args = parser.parse_args()
k.seed_ids(args.seed)

windows = pcbnew.SHAPE_POLY_SET()
for x0, y0, x1, y1 in args.box_mm:
    windows.NewOutline()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        windows.Append(k.mm(x), k.mm(y))
windows.Simplify()
donor, _ = k.take_copper(pcbnew.LoadBoard(args.donor))
donor.BooleanIntersection(windows)
board = pcbnew.LoadBoard(args.parent)
copper, before = k.take_copper(board)
copper.BooleanSubtract(windows)
copper.BooleanAdd(donor)
islands = k.add_copper(board, copper)
path, meta = k.save(board, args.output, extra={
    "synthetic": None, "parent": {"file": Path(args.parent).name, "sha256": k.sha(args.parent)},
    "donor": {"file": Path(args.donor).name, "sha256": k.sha(args.donor)},
    "edits": [{"op": "transplant_copper", "box_mm": b} for b in args.box_mm],
    "copper_area_before_mm2": before})
print(path, islands, "copper islands", round(before, 4), "->", round(meta["copper"]["area_mm2"], 4), "mm2")
