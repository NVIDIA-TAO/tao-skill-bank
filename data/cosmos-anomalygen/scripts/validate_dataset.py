#!/usr/bin/env python3
"""Validate AnomalyGen training dataset structure.

Scans <dataset_dir>/<TEXTURE>/{anomaly_image,mask}/<ANOMALY>/ and prints:
  - detected TEXTURE+ANOMALY types with image/mask counts
  - warnings for missing directories or unpaired image/mask files

Exits 1 if no anomaly types are detected; otherwise 0.
"""
import argparse
import os
import pathlib
import sys

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def validate(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    anomaly_types, issues = [], []

    for texture in sorted(os.listdir(dataset_dir)):
        texture_path = dataset_dir / texture
        if not texture_path.is_dir():
            continue
        img_root = texture_path / "anomaly_image"
        mask_root = texture_path / "mask"

        if not img_root.is_dir():
            issues.append(f"{texture}/ missing anomaly_image/"); continue
        if not mask_root.is_dir():
            issues.append(f"{texture}/ missing mask/"); continue

        for atype in sorted(os.listdir(img_root)):
            atype_img = img_root / atype
            atype_mask = mask_root / atype
            if not atype_img.is_dir():
                continue
            images = [p for p in atype_img.iterdir() if p.suffix.lower() in _IMG_EXTS]
            if not atype_mask.is_dir():
                issues.append(f"{texture}/{atype} has anomaly_image/ but no mask/"); continue
            masks = [p for p in atype_mask.iterdir() if p.suffix.lower() in _IMG_EXTS]

            anomaly_types.append([texture, atype])
            print(f"  [{texture}, {atype}]: {len(images)} images, {len(masks)} masks")

            # Strict pairing: anomaly_dataset.py asserts mask stem == f"{img}_mask".
            # Loose `startswith` would let `image_001_masked.png` slip past here
            # and crash training mid-iteration.
            mask_stems = {p.stem for p in masks}
            for img in images:
                expected = f"{img.stem}_mask"
                if expected not in mask_stems:
                    issues.append(
                        f"no mask for {texture}/{atype}/{img.name} "
                        f"(expected stem: {expected})"
                    )

    return anomaly_types, issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir", type=pathlib.Path)
    args = p.parse_args()

    print(f"=== Dataset Validation Summary ===")
    print(f"Dataset: {args.dataset_dir}")
    print("Detected anomaly types:")
    anomaly_types, issues = validate(args.dataset_dir)
    print(f"Issues: {len(issues)}")
    for msg in issues:
        print(f"  WARNING: {msg}")

    if not anomaly_types:
        print("error: no anomaly types detected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
