#!/usr/bin/env python3
"""Build the per-sample JSON consumed by scripts/run_auto_roi_amp.py.

Pair budget is **1 clean per submask** (not the clean × submask cartesian).
The max pair budget per defect is therefore `num_submasks[d]`. When
`allocation[d]` exceeds that budget, we fall back to n_seeds > 1 so that
AMP produces multiple placements per (clean, submask) record.

n_seeds is computed globally from the allocation:

    n_seeds = max_d ⌈allocation[d] / num_submasks[d]⌉

With proportional allocation this equals `⌈num_SDG / total_training_masks⌉`
(e.g., num_SDG=75 over 75 training masks → n_seeds=1; num_SDG=10000 over
75 masks → n_seeds=134).

Per defect d we emit `⌈allocation[d] / n_seeds⌉` records. Each record pairs
submask i with clean image `clean_imgs[i % len(clean_imgs)]`, so every
training submask is used before any submask repeats, and cleans rotate
across submasks to distribute the pool without enumerating the cartesian.

Record schema (consumed by scripts/run_auto_roi_amp.py):

    {"clean_image":     <path>,
     "defect_type":     "TEXTURE+ANOMALY",
     "submask":         <path to training mask>,
     "name":            "<clean_stem>__<submask_stem>",
     "cad_mask":        <path>  if spatial_dependency=="cad" else null,
     "cad_mask_label":  <path>  if spatial_dependency=="cad" else null}

Note: AMP's preprocess_submask defaults submask_split_largest=True, so a
disconnected training submask is reduced to its largest connected
component before placement. We do not set the field here — the True
default applies.

n_seeds is written to a sidecar file next to the output JSON so
prep_testcase.sh can pass it to run_auto_roi_amp.py without re-deriving it.
"""
import argparse
import json
import math
import pathlib
import sys

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _images(d):
    d = pathlib.Path(d)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _clean_pool(clean_dir, texture):
    nested = clean_dir / texture / "clean_image"
    if nested.is_dir():
        return _images(nested)
    tex = clean_dir / texture
    return _images(tex) if tex.is_dir() else _images(clean_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    p.add_argument("--clean-dir", required=True, type=pathlib.Path)
    p.add_argument("--defect-spec", required=True, type=pathlib.Path)
    p.add_argument("--allocation", required=True, type=pathlib.Path,
                   help="JSON {defect_type: n}.")
    p.add_argument("--output", required=True, type=pathlib.Path,
                   help="Where to write the sample list JSON. n_seeds is "
                        "written to <output>.n_seeds alongside it.")
    args = p.parse_args()

    entries = [json.loads(l) for l in args.defect_spec.read_text().splitlines() if l.strip()]
    labels_path = args.dataset_dir / "semantic_segmentation_labels.json"
    allocation = json.loads(args.allocation.read_text())

    # Pass 1: gather per-defect pools and compute n_seeds.
    pools = {}  # defect_type -> (submasks, clean_imgs, sd, texture)
    for e in entries:
        full = e["defect_type"]
        sd = e.get("spatial_dependency", "free")
        texture, anomaly = full.split("+", 1)

        n = allocation.get(full, 0)
        if n == 0:
            continue

        submasks = _images(args.dataset_dir / texture / "mask" / anomaly)
        if not submasks:
            print(f"warn: no submasks for {full}", file=sys.stderr); continue
        clean_imgs = _clean_pool(args.clean_dir, texture)
        if not clean_imgs:
            print(f"warn: no clean images for {full}", file=sys.stderr); continue
        pools[full] = (submasks, clean_imgs, sd, texture)

    n_seeds = 1
    for full, (submasks, _, _, _) in pools.items():
        need = math.ceil(allocation[full] / len(submasks))
        if need > n_seeds:
            n_seeds = need

    # Pass 2: emit records — one per (submask, clean) with clean round-robin.
    def _cad_fields(sd, texture, clean_img):
        if sd != "cad":
            return None, None
        cad_mask_path = args.dataset_dir / texture / "cad_mask" / f"{clean_img.stem}.png"
        if not cad_mask_path.exists():
            return "__SKIP__", None
        return str(cad_mask_path), str(labels_path)

    records = []
    for full, (submasks, clean_imgs, sd, texture) in pools.items():
        n_records = math.ceil(allocation[full] / n_seeds)
        for i in range(n_records):
            submask = submasks[i % len(submasks)]
            clean_img = clean_imgs[i % len(clean_imgs)]
            cad_mask, cad_mask_label = _cad_fields(sd, texture, clean_img)
            if cad_mask == "__SKIP__":
                continue
            records.append({
                "clean_image":    str(clean_img),
                "defect_type":    full,
                "submask":        str(submask),
                "name":           f"{clean_img.stem}__{submask.stem}",
                "cad_mask":       cad_mask,
                "cad_mask_label": cad_mask_label,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    sidecar = args.output.with_suffix(args.output.suffix + ".n_seeds")
    sidecar.write_text(str(n_seeds))
    print(f"wrote {len(records)} records to {args.output} (n_seeds={n_seeds})")


if __name__ == "__main__":
    main()
