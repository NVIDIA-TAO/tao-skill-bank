#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create: a self-generated coupon with two standard-library 0402 capacitors.

C2 is offset and rotated 37 degrees so unit, sign and rotation mistakes show up
in the pad-alignment check. Nothing here comes from a photograph.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kicad_sdk as k  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20261111)
args = parser.parse_args()
k.seed_ids(args.seed)

board = k.new_board([(-2, -2), (4.5, -2), (4.5, 3), (-2, 3)])
k.place(board, "Capacitor_SMD", "C_0402_1005Metric", "C1", 0.0, 0.0, 0.0)
k.place(board, "Capacitor_SMD", "C_0402_1005Metric", "C2", 2.0, 1.0, 37.0)
path, meta = k.save(board, args.output, extra={"synthetic": True, "parent": None})
print(path, len(meta["pads"]), "pads")
