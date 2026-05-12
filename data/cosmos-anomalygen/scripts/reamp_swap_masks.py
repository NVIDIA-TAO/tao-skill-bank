#!/usr/bin/env python3
"""Swap mask_filename paths in a JSONL to point at a fresh AMP output.

Assumes `run_auto_roi_amp.py` was invoked with `--n_seeds <K> --seed <N>`
into --new-amp-dir, producing <new-amp-dir>/<name>/<TEXTURE>+<ANOMALY>/seed{0..K-1}.png
for every record the original AMP pass emitted. For each JSONL row, derives
(name, full_type, seed_index) from the existing mask_filename and rewrites it
to the matching path under --new-amp-dir, **preserving the original seed
index**. This keeps the seed0 vs seed1 (etc.) split that prep-testcase
introduced for within-pair diversity intact under re-AMP.

This is used by sdg-refine's run_round.sh when --reamp-seed is set: the
(clean_image, submask) pairs stay the same, but the AMP augmentation is
re-rolled with a fresh base seed.
"""
import argparse
import json
import pathlib
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-jsonl", required=True, type=pathlib.Path)
    p.add_argument("--new-amp-dir", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--seed-index", type=int, default=None,
                   help="If set, force every row to this seed index (overrides per-row preservation). "
                        "Default: preserve each row's original seed index from its mask_filename.")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.base_jsonl.read_text().splitlines() if l.strip()]
    written, missing = 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fp:
        for row in rows:
            old = pathlib.Path(row["mask_filename"])
            # old path is <amp>/<name>/<full_type>/seed<N>.png
            name, full_type = old.parent.parent.name, old.parent.name
            seed_stem = old.stem if args.seed_index is None else f"seed{args.seed_index}"
            new = args.new_amp_dir / name / full_type / f"{seed_stem}.png"
            if not new.exists():
                print(f"warn: missing re-AMPed mask {new}", file=sys.stderr)
                missing += 1
                continue
            row["mask_filename"] = str(new)
            fp.write(json.dumps(row) + "\n")
            written += 1
    print(f"rewrote {written} rows, dropped {missing}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
