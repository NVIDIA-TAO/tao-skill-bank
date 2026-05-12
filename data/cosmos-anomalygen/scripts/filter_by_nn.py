#!/usr/bin/env python3
"""Filter a searched/ (or original/) SDG bucket by per-sample nn_score.

Reads a per-sample CSV (from `run_eval.sh --per-sample-csv`), keeps only
rows whose nn_score >= --threshold, and copies their outputs to
--filtered-dir. A reduced SDG_result.csv is written to the same place so
downstream tools still work on the filtered bucket.

Layout (both --searched-dir and --filtered-dir):
    reconstructed_image/<basename>
    original_mask/<basename>
    overlay_image/<basename>
    original_image/<basename>
    SDG_result.csv
"""
import argparse
import csv
import math
import pathlib
import shutil
import sys


_KINDS = ("reconstructed_image", "original_mask", "overlay_image", "original_image")


def _load_nn(per_sample_csv):
    rows = {}
    with open(per_sample_csv) as f:
        for r in csv.DictReader(f):
            m = r["nn_score"]
            rows[r["path"]] = float(m) if m else float("nan")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--searched-dir", required=True, type=pathlib.Path,
                   help="Source bucket with reconstructed_image/ etc and SDG_result.csv.")
    p.add_argument("--per-sample-csv", required=True, type=pathlib.Path,
                   help="Per-sample CSV (from run_eval.sh --per-sample-csv).")
    p.add_argument("--threshold", required=True, type=float,
                   help="Keep rows with nn_score >= threshold. NaN is dropped.")
    p.add_argument("--filtered-dir", required=True, type=pathlib.Path,
                   help="Destination for kept samples.")
    args = p.parse_args()

    if not (args.searched_dir / "SDG_result.csv").exists():
        print(f"error: {args.searched_dir}/SDG_result.csv not found", file=sys.stderr)
        sys.exit(1)

    nn = _load_nn(args.per_sample_csv)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)

    with open(args.searched_dir / "SDG_result.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        src_rows = list(reader)

    kept_rows = []
    dropped_rows = []
    for row in src_rows:
        name = row["output_filename"]
        score = nn.get(name, float("nan"))
        if not math.isnan(score) and score >= args.threshold:
            for kind in _KINDS:
                src_f = args.searched_dir / kind / name
                if src_f.exists():
                    (args.filtered_dir / kind).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, args.filtered_dir / kind / name)
            kept_rows.append({**row, "nn_score": f"{score:.6f}"})
        else:
            dropped_rows.append((name, score))

    out_fields = list(fieldnames)
    if "nn_score" not in out_fields:
        out_fields.append("nn_score")
    with (args.filtered_dir / "SDG_result.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in kept_rows:
            w.writerow({k: r.get(k, "") for k in out_fields})

    print(f"kept {len(kept_rows)}/{len(src_rows)} samples (nn_score >= {args.threshold})")
    print(f"output: {args.filtered_dir}")
    if dropped_rows[:3]:
        preview = ", ".join(f"{n} ({s:.3f})" if not math.isnan(s) else f"{n} (nan)"
                            for n, s in dropped_rows[:3])
        print(f"first dropped: {preview}")


if __name__ == "__main__":
    main()
