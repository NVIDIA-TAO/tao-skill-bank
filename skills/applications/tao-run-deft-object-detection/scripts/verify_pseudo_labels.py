#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail the prep phase when Co-DETR wrote label files but no boxes.

``codetr inference`` exits 0 and prints ``Execution status: PASS`` whether or not
the checkpoint's weights actually loaded. When the spec's architecture does not
match the checkpoint, the layers are randomly initialised, every image yields
nothing above ``conf_threshold``, and the run produces one empty label file per
image. Every later stage then behaves as though the pool genuinely contains no
objects: ``annotations convert`` succeeds on an empty COCO, mining has nothing to
match, and the first hard error arrives stages later with no connection to the
cause.

This is the gate that turns that into an immediate, named failure. It reads only
the KITTI label directory, so it costs a directory walk rather than the conversion
it precedes.

Inputs:  --labels-dir, --expect-images, --min-nonempty-frac (default 0.5)
Output:  counts per class on stdout; --report-json for the run record

Exits 1 when the non-empty fraction is below the floor, when the file count does
not match ``--expect-images``, or when the directory holds no label files at all.
The message names the architecture mismatch as the first thing to check, because
it is the only cause that produces a *uniformly* empty directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", required=True,
                        help="KITTI label directory written by codetr inference.")
    parser.add_argument("--expect-images", type=int, default=None,
                        help="Pool image count. One label file per image is expected.")
    parser.add_argument("--min-nonempty-frac", type=float, default=0.5,
                        help="Floor on the fraction of label files carrying at least one "
                             "box. Default 0.5. A correctly paired checkpoint on a traffic "
                             "pool sits near 1.0; the failure mode this catches sits at 0.0.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        labels_dir = Path(args.labels_dir).expanduser().resolve()
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"--labels-dir is not a directory: {labels_dir}")

        files = sorted(labels_dir.glob("*.txt"))
        if not files:
            raise ValueError(f"{labels_dir}: no .txt label files at all")

        nonempty = 0
        boxes = 0
        per_class: Counter[str] = Counter()
        for path in files:
            rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if rows:
                nonempty += 1
                boxes += len(rows)
                for row in rows:
                    per_class[row.split()[0]] += 1

        frac = nonempty / len(files)
        report = {
            "labels_dir": str(labels_dir),
            "label_files": len(files),
            "nonempty_files": nonempty,
            "nonempty_frac": round(frac, 4),
            "boxes": boxes,
            "per_class": dict(per_class.most_common()),
        }
        if args.report_json:
            Path(args.report_json).expanduser().resolve().write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"label files:  {len(files)}")
        print(f"non-empty:    {nonempty} ({frac:.1%})")
        print(f"boxes:        {boxes}")
        for name, count in per_class.most_common():
            print(f"  {name}: {count}")

        failures = []
        if args.expect_images is not None and len(files) != args.expect_images:
            failures.append(
                f"expected one label file per image ({args.expect_images}), found {len(files)}")
        if frac < args.min_nonempty_frac:
            failures.append(
                f"only {frac:.1%} of label files carry a box, below the {args.min_nonempty_frac:.0%} "
                f"floor.\n"
                f"  Check the checkpoint/spec architecture pairing FIRST: model.backbone, "
                f"model.num_queries,\n"
                f"  model.num_feature_levels and dataset.num_classes must match the checkpoint. "
                f"A mismatch\n"
                f"  loads no weights, exits 0, prints PASS, and empties every file. Search the "
                f"inference log\n"
                f"  for 'Skipping size-mismatched key'. If the pairing is right, "
                f"inference.conf_threshold\n"
                f"  may simply be too high for this pool.")
        if failures:
            for item in failures:
                print(f"ERROR: {item}", file=sys.stderr)
            return 1

        print("OK: pseudo-labels are populated")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"verify_pseudo_labels: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
