#!/usr/bin/env python3
"""Cross-check the AMP input triple (dataset_dir, clean_dir, defect_spec).

Fails fast with an itemised report of every problem, so prep-testcase doesn't
crash mid-AMP. Exits 1 when any check fails.

Checks (per defect in defect_spec):
  ✓ <dataset_dir>/<TEXTURE>/mask/<ANOMALY>/ exists          (submask source)
  ✓ <clean_dir>/<TEXTURE>/clean_image/ or <clean_dir>/<TEXTURE>/ has ≥1 image
    (or flat <clean_dir>/ as final fallback)
  spatial_dependency=="text":
    ✓ roi_prompt_defect_location present and non-empty
  spatial_dependency=="cad":
    ✓ <dataset_dir>/semantic_segmentation_labels.json exists
    ✓ <dataset_dir>/<TEXTURE>/cad_mask/ exists
    ✓ <dataset_dir>/<TEXTURE>/cad_mask/<stem>.png exists for every clean image
"""
import argparse
import json
import pathlib
import sys

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(d):
    d = pathlib.Path(d)
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS]


def validate(dataset_dir, clean_dir, defect_spec):
    dataset_dir = pathlib.Path(dataset_dir)
    clean_dir = pathlib.Path(clean_dir)
    entries = [json.loads(l) for l in pathlib.Path(defect_spec).read_text().splitlines() if l.strip()]

    errors = []
    has_cad = any(e.get("spatial_dependency") == "cad" for e in entries)
    if has_cad:
        labels = dataset_dir / "semantic_segmentation_labels.json"
        if not labels.exists():
            errors.append(f"missing {labels} (required for spatial_dependency=cad)")

    for e in entries:
        full = e["defect_type"]
        sd = e.get("spatial_dependency")
        if "+" not in full:
            errors.append(f"{full}: defect_type must be TEXTURE+ANOMALY")
            continue
        texture, anomaly = full.split("+", 1)

        submask_dir = dataset_dir / texture / "mask" / anomaly
        if not submask_dir.is_dir():
            errors.append(f"{full}: missing submask source {submask_dir}")

        clean_nested = clean_dir / texture / "clean_image"
        clean_tex = clean_dir / texture
        if clean_nested.is_dir():
            clean_pool = clean_nested
        elif clean_tex.is_dir():
            clean_pool = clean_tex
        else:
            clean_pool = clean_dir
        clean_imgs = _list_images(clean_pool)
        if not clean_imgs:
            errors.append(f"{full}: no clean images under {clean_pool}")

        if sd == "text":
            prompt = (e.get("roi_prompt_defect_location") or "").strip()
            if not prompt:
                errors.append(f"{full}: spatial_dependency={sd} requires non-empty roi_prompt_defect_location")
        elif sd == "cad":
            cad_dir = dataset_dir / texture / "cad_mask"
            if not cad_dir.is_dir():
                errors.append(f"{full}: missing {cad_dir}")
                continue
            missing_cad = [p.stem for p in clean_imgs if not (cad_dir / f"{p.stem}.png").exists()]
            if missing_cad:
                errors.append(f"{full}: {len(missing_cad)} clean image(s) without matching cad_mask "
                              f"(first: {missing_cad[0]}.png)")
        elif sd in (None, "free"):
            pass
        else:
            errors.append(f"{full}: unknown spatial_dependency={sd!r}")

    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    p.add_argument("--clean-dir", type=pathlib.Path, required=True)
    p.add_argument("--defect-spec", type=pathlib.Path, required=True)
    args = p.parse_args()

    errors = validate(args.dataset_dir, args.clean_dir, args.defect_spec)
    if errors:
        print("AMP input validation failed:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1)
    print("AMP inputs OK")


if __name__ == "__main__":
    main()
