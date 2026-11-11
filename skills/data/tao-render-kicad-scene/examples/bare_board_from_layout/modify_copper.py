#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modify: cut or add one copper rectangle on a parent bare board (open, short, spur...).

Rectangles are x0 y0 x1 y1 in millimetres in the board frame. The parent is
read-only; the edited board goes to a new directory.
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
parser.add_argument("--cut-mm", type=float, nargs=4, action="append", default=[])
parser.add_argument("--add-mm", type=float, nargs=4, action="append", default=[])
parser.add_argument("--seed", type=int, default=20261112)
args = parser.parse_args()
if not args.cut_mm and not args.add_mm:
    parser.error("give at least one --cut-mm or --add-mm rectangle")
k.seed_ids(args.seed)


def rect(x0, y0, x1, y1):
    poly = pcbnew.SHAPE_POLY_SET()
    poly.NewOutline()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        poly.Append(k.mm(x), k.mm(y))
    return poly


board = pcbnew.LoadBoard(args.parent)
copper, before = k.take_copper(board)
for box in args.add_mm:
    copper.BooleanAdd(rect(*box))
for box in args.cut_mm:
    copper.BooleanSubtract(rect(*box))
islands = k.add_copper(board, copper)
path, meta = k.save(board, args.output, extra={
    "synthetic": None, "parent": {"file": Path(args.parent).name, "sha256": k.sha(args.parent)},
    "edits": [{"op": "cut_copper", "rect_mm": b} for b in args.cut_mm] +
             [{"op": "add_copper", "rect_mm": b} for b in args.add_mm],
    "copper_area_before_mm2": before})
print(path, islands, "copper islands", round(before, 4), "->", round(meta["copper"]["area_mm2"], 4), "mm2")
