#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw a self-made two-level bare-board picture (bright copper on dark laminate).

A stand-in for a binarised line-scan image so the trace -> board -> render ->
compare path can be replayed without any real reference. `--defect open` cuts
one trace to make a matching test image.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--size", type=int, default=320)
parser.add_argument("--defect", choices=("none", "open"), default="none")
args = parser.parse_args()

n = args.size
u = n / 320
image = Image.new("L", (n, n), 0)
draw = ImageDraw.Draw(image)
for index, y in enumerate((60, 110, 160)):                      # three traces with a diagonal jog
    draw.line([(20 * u, y * u), (140 * u, y * u), (190 * u, (y + 40) * u), (300 * u, (y + 40) * u)],
              fill=255, width=round(10 * u), joint="curve")
for x in (40, 100, 160):                                         # annular pads with drill holes
    cx, cy, outer, inner = x * u, 260 * u, 18 * u, 7 * u
    draw.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=255)
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=0)
draw.rectangle([220 * u, 230 * u, 300 * u, 300 * u], fill=255)  # a pour with an isolation slot
draw.rectangle([255 * u, 230 * u, 262 * u, 285 * u], fill=0)
if args.defect == "open":
    draw.rectangle([80 * u, 100 * u, 92 * u, 120 * u], fill=0)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
image.save(out)
print(out, image.size)
