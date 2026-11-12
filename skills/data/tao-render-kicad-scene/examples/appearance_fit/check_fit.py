#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the synthetic search: it stayed inside its plan and moved toward the look that made the reference."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", required=True)
parser.add_argument("--search-dir", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--out-json", required=True)
args = parser.parse_args()
plan = json.loads(Path(args.plan).read_text())
search = Path(args.search_dir)
history = json.loads((search / "history.json").read_text())
best, target = json.loads((search / "best.json").read_text()), json.loads(Path(args.target).read_text())
board = "board:board_soldermask"
checks = {
    "within budget": history["evaluations"] <= plan["budget"],
    "every evaluation kept": len(history["history"]) == history["evaluations"]
        and all((search / row["look"]).is_file() for row in history["history"]),
    "loss at least halved": history["best_loss"] <= 0.5 * history["initial_loss"],
    "light sample followed by a colour correction": any(r["stage"] == "light+ratio" for r in history["history"]),
    "board colour moved toward the target hue": best["materials"]["overrides"][board]["color"][2]
        > best["materials"]["overrides"][board]["color"][1],
    "geometry keys untouched": best["assets"] == target["assets"],
}
report = {"pass": all(checks.values()), "checks": checks,
          "initial_loss": history["initial_loss"], "best_loss": history["best_loss"],
          "best_evaluation": history["best_evaluation"], "stopped_by": history["stopped_by"],
          "found": {"world_strength": best["lighting"]["world_strength"], "area_power_w": best["lighting"]["area"][0]["power_w"],
                    "solder_roughness": best["materials"]["solder"]["roughness"],
                    "board_color": best["materials"]["overrides"][board]["color"]},
          "target": {"world_strength": target["lighting"]["world_strength"], "area_power_w": target["lighting"]["area"][0]["power_w"],
                     "solder_roughness": target["materials"]["solder"]["roughness"],
                     "board_color": target["materials"]["overrides"][board]["color"]},
          "note": "a low loss means matching regional statistics; several looks can share them, so the found "
                  "parameters need not equal the target"}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
sys.exit(0 if report["pass"] else 1)
