#!/usr/bin/env python3
"""Build SDG JSONL by pairing AMP-placed masks with their clean images.

Allocation dictates N per defect_type (from allocate_samples.py). The AMP
output layout produced by `scripts/run_auto_roi_amp.py` with n_seeds=1 is:

    <amp-output-dir>/<clean_stem>__<submask_stem>/<TEXTURE>+<ANOMALY>/seed0.png

(Legacy layout `<amp-output-dir>/<clean_stem>/<TEXTURE>+<ANOMALY>/seed<N>.png`
is also accepted for backwards compatibility.)

Since build_amp_samples.py emits exactly `allocation[t]` records per defect,
this script finds exactly that many masks. It errors if fewer exist (an AMP
failure, not a planning error).
"""
import argparse
import json
import pathlib
import sys


def _texture(full):
    return full.split("+", 1)[0] if "+" in full else ""


def _defect_short(full):
    return full.split("+", 1)[1] if "+" in full else full


def _clean_image_for(stem, texture, clean_dir):
    for base in (clean_dir / texture / "clean_image", clean_dir / texture, clean_dir):
        for ext in ("jpg", "png", "jpeg", "bmp"):
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
    return None


def _enumerate_amp_masks(amp_output_dir, full_type):
    """Yield (clean_stem, mask_path) pairs for one defect_type.

    Walks <amp_output_dir>/<clean_stem>/<full_type>/<submask_stem>__seed<i>.png
    matching the layout produced by scripts/run_auto_roi_amp.py.
    """
    amp_output_dir = pathlib.Path(amp_output_dir)
    if not amp_output_dir.is_dir():
        return
    for name_dir in sorted(p for p in amp_output_dir.iterdir() if p.is_dir()):
        type_dir = name_dir / full_type
        if not type_dir.is_dir():
            continue
        clean_stem = name_dir.name
        for mask in sorted(type_dir.glob("*__seed*.png")):
            yield clean_stem, mask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp-output-dir", type=pathlib.Path, required=True)
    p.add_argument("--clean-dir", type=pathlib.Path, required=True)
    p.add_argument("--allocation", type=pathlib.Path, required=True,
                   help="JSON: {anomaly_type: count}")
    p.add_argument("--defect-types", nargs="+", required=True,
                   help="Full TEXTURE+TYPE names matching allocation keys.")
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--crop-ratio", type=float, default=2.0)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()

    allocation = json.loads(args.allocation.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as fp:
        for full_type in args.defect_types:
            n = allocation.get(full_type, 0)
            if n == 0:
                continue

            pairs = list(_enumerate_amp_masks(args.amp_output_dir, full_type))
            if not pairs:
                print(f"warn: no AMP masks for {full_type}", file=sys.stderr); continue

            if len(pairs) < n:
                print(
                    f"error: allocation requested {n} masks for {full_type} "
                    f"but only {len(pairs)} AMP outputs exist. This means "
                    f"AMP failed on some records — check run_auto_roi_amp.py "
                    f"logs for NO_DETECTION / FAILED statuses.",
                    file=sys.stderr,
                )
                sys.exit(1)
            pairs = pairs[:n]

            texture = _texture(full_type)
            for stem, mask_path in pairs:
                clean_img = _clean_image_for(stem, texture, args.clean_dir)
                if clean_img is None:
                    print(f"warn: no clean image for {texture}/{stem}", file=sys.stderr); continue
                fp.write(json.dumps({
                    "image_filename": str(clean_img),
                    "mask_filename":  str(mask_path),
                    "anomaly_type":   full_type,
                    "guidance":       args.guidance,
                    "num_steps":      args.num_steps,
                    "crop_and_paste": True,
                    "crop_ratio":     args.crop_ratio,
                    "crop_grid_X":    "none",
                    "crop_grid_Y":    "none",
                    "num_generated_images": 1,
                    "poisson_blend":  False,
                    "iteration_generation_max_instance": 5,
                }) + "\n")
                written += 1
    print(f"wrote {written} entries to {args.output}")


if __name__ == "__main__":
    main()
