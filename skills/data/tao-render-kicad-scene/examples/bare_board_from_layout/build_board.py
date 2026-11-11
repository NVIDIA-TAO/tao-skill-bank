#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a bare board whose F.Cu copper is the union of traced pixel rectangles.

Input is the layout JSON from `offline_image.py layout`. Pixel (x, y) maps to
(x, y) * mm_per_px with the image's top-left corner at the board origin, so a
top-down render at the image's size lines up pixel for pixel. The result is a
visual layout hypothesis: it carries no netlist, and scale and copper polarity
are whatever the layout file states.
"""
import argparse
import json
from pathlib import Path
import sys

import pcbnew

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layout", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20261111)
args = parser.parse_args()
k.seed_ids(args.seed)

layout = json.loads(Path(args.layout).read_text())
scale = layout["mm_per_px"]
width, height = layout["size_px"]
board = k.new_board([(0, 0), (width * scale, 0), (width * scale, height * scale), (0, height * scale)])
copper = pcbnew.SHAPE_POLY_SET()
for x0, y0, x1, y1 in layout["rects_px"]:
    copper.NewOutline()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        copper.Append(k.mm(x * scale), k.mm(y * scale))
copper.Simplify()  # union: touching rectangles become islands with holes
islands = k.add_copper(board, copper)
path, meta = k.save(board, args.output, extra={
    "synthetic": not layout.get("derived_from_private_reference", False), "parent": None,
    "layout": {"sha256": k.sha(args.layout), "source_sha256": layout["source"]["sha256"],
               "mm_per_px": scale, "mm_per_px_basis": layout["mm_per_px_basis"], "copper": layout["copper"],
               "size_px": layout["size_px"], "rect_count": len(layout["rects_px"])},
    "representation": "2D copper layout hypothesis traced from one image; no netlist"})
print(path, islands, "copper islands", round(meta["copper"]["area_mm2"], 4), "mm2")
