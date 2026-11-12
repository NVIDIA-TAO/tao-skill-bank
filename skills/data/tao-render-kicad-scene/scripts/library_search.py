#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List standard-library footprints that fit what was observed, with their 3D models.

Run with KiCad's bundled interpreter. Filters are the facts a picture can
support without a scale: pad count, mounting style and the length/width ratio
of the body outline on F.Fab. Each row says whether the footprint's declared
models resolve to files in this install, so a candidate without a usable model
is visible before anything is rendered.

  kicad_light.py step ... --tool kicad-python -- library_search.py \
      --library Capacitor_SMD --pads 2 --body-aspect 1.2 1.7 --out-json candidates.json
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcbnew  # noqa: E402

import kicad_sdk as k  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", action="append", required=True, help="library name, e.g. Capacitor_SMD")
    parser.add_argument("--pads", type=int, help="exact number of pads with distinct numbers")
    parser.add_argument("--name", help="regular expression the footprint name must match")
    parser.add_argument("--body-aspect", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="body length / width on F.Fab, longer side first")
    parser.add_argument("--mount", choices=("smd", "tht"))
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows, scanned = [], 0
    for library in args.library:
        directory = k.appdir() / "share/kicad/footprints" / (library + ".pretty")
        if not directory.is_dir():
            raise SystemExit(f"No such library: {directory}")
        for name in sorted(pcbnew.FootprintEnumerate(str(directory))):
            scanned += 1
            if args.name and not re.search(args.name, name):
                continue
            fp = pcbnew.FootprintLoad(str(directory), name)
            pads = list(fp.Pads())
            numbers = {p.GetNumber() for p in pads if p.GetNumber()}
            if args.pads is not None and len(numbers) != args.pads:
                continue
            smd = all(p.GetAttribute() == pcbnew.PAD_ATTRIB_SMD for p in pads) if pads else False
            if args.mount and (args.mount == "smd") != smd:
                continue
            body = k._body_bbox(fp)
            size = sorted((body[2] - body[0], body[3] - body[1]), reverse=True) if body else None
            aspect = round(size[0] / size[1], 4) if size and size[1] > 0 else None
            if args.body_aspect and (aspect is None or not args.body_aspect[0] <= aspect <= args.body_aspect[1]):
                continue
            boxes = [p.GetBoundingBox() for p in pads]
            models = k.footprint_models(fp)
            rows.append({
                "id": f"{library}:{name}", "pads": len(numbers), "mount": "smd" if smd else "tht or mixed",
                "body_size_mm": [round(v, 4) for v in size] if size else None, "body_aspect": aspect,
                "pad_size_mm": sorted({(round(pcbnew.ToMM(p.GetSize().x), 4), round(pcbnew.ToMM(p.GetSize().y), 4))
                                       for p in pads}),
                "pad_extent_mm": [round(pcbnew.ToMM(max(b.GetRight() for b in boxes) - min(b.GetLeft() for b in boxes)), 4),
                                  round(pcbnew.ToMM(max(b.GetBottom() for b in boxes) - min(b.GetTop() for b in boxes)), 4)]
                if boxes else None,
                "models": [{key: m[key] for key in ("declared", "resolved_path", "sha256", "problem")} for m in models],
                "model_usable": bool(models) and all(m["problem"] is None for m in models)})
    result = {"kicad_version": pcbnew.GetBuildVersion(), "libraries": args.library, "footprints_scanned": scanned,
              "filters": {"pads": args.pads, "name": args.name, "body_aspect": args.body_aspect, "mount": args.mount},
              "matches": len(rows), "matches_with_usable_model": sum(r["model_usable"] for r in rows),
              "body_source": "F.Fab straight graphics; a ratio, not a measured part", "rows": rows}
    Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("footprints_scanned", "matches", "matches_with_usable_model")}))


if __name__ == "__main__":
    main()
