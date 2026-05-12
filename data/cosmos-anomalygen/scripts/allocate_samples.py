#!/usr/bin/env python3
"""Distribute num_SDG across defect types proportionally to training mask counts.

Scans `<mask_path>/<TEXTURE>/mask/<ANOMALY>/` per `TEXTURE+ANOMALY` type and
allocates via Hamilton largest-remainder so the totals sum exactly to
num_SDG.

Usage (CLI):
    allocate_samples.py --num-sdg N --defect-types t1 t2 ... \
        --mask-path <dataset_dir> --output alloc.json
"""
import argparse
import json
import math
import pathlib
import sys

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def count_masks(mask_path, defect_types):
    counts = {}
    for full_type in defect_types:
        texture, anomaly = full_type.split("+", 1)
        d = pathlib.Path(mask_path) / texture / "mask" / anomaly
        if not d.is_dir():
            counts[full_type] = 0
            continue
        counts[full_type] = sum(1 for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
    return counts


def allocate(num_sdg, defect_types, counts):
    """Hamilton largest-remainder apportionment. Sum(counts.values()) > 0."""
    if num_sdg < 0:
        raise ValueError(f"num_sdg must be >= 0 (got {num_sdg})")
    if not defect_types:
        raise ValueError("defect_types must be non-empty")
    total = float(sum(counts.get(t, 0) for t in defect_types))
    if total <= 0:
        raise ValueError(f"sum of mask counts must be > 0 (got {counts})")

    raw = {t: num_sdg * counts.get(t, 0) / total for t in defect_types}
    floors = {t: int(math.floor(r)) for t, r in raw.items()}
    remainder = num_sdg - sum(floors.values())
    order = sorted(defect_types, key=lambda t: raw[t] - floors[t], reverse=True)
    for t in order[:remainder]:
        floors[t] += 1

    # KPI floor: training validation needs ≥3 entries per defect, so refuse
    # to silently allocate 0 to a type. Suggest the smallest num_sdg that
    # would satisfy the floor (per finetune/SKILL.md):
    #     num_sdg = max(N_total, ceil(3 * N_total / N_min))
    zero_types = [t for t in defect_types if floors[t] == 0]
    if zero_types:
        n_total = int(total)
        positive = [counts.get(t, 0) for t in defect_types if counts.get(t, 0) > 0]
        n_min = min(positive) if positive else 0
        suggested = max(num_sdg, math.ceil(3 * n_total / n_min)) if n_min > 0 else num_sdg
        raise ValueError(
            f"validation JSONL coverage broken: types {zero_types} got 0 entries "
            f"with num_sdg={num_sdg}. Increase num_sdg to >= {suggested} "
            f"(KPI floor: max(N_total, ceil(3*N_total/N_min))) or trim defect_spec."
        )
    return floors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-sdg", type=int, required=True)
    p.add_argument("--defect-types", nargs="+", required=True)
    p.add_argument("--mask-path", type=pathlib.Path, required=True,
                   help="Dataset root — scans <mask_path>/<TEXTURE>/mask/<ANOMALY>/.")
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()

    counts = count_masks(args.mask_path, args.defect_types)
    print(f"derived per-type mask counts: {counts}", file=sys.stderr)
    alloc = allocate(args.num_sdg, args.defect_types, counts)
    args.output.write_text(json.dumps(alloc, indent=2))
    print(f"wrote {args.output} (sum={sum(alloc.values())})", file=sys.stderr)


if __name__ == "__main__":
    main()
