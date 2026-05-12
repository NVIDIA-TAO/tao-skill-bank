#!/usr/bin/env python3
"""Verify image/mask path existence and resize mismatched masks in place.

Rewrites the input JSONL so mismatched masks point at resized caches
under --cache-dir. Uses PIL.Image.NEAREST to preserve the 0/255 invariant
the SDG pipeline enforces.
"""
import argparse
import json
import os
import pathlib
import sys
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=pathlib.Path, required=True)
    p.add_argument("--cache-dir", type=pathlib.Path, required=True)
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]

    resized = 0
    missing = 0
    for entry in entries:
        if not pathlib.Path(entry["image_filename"]).exists():
            print(f"error: missing image {entry['image_filename']}", file=sys.stderr); missing += 1; continue
        if not pathlib.Path(entry["mask_filename"]).exists():
            print(f"error: missing mask {entry['mask_filename']}", file=sys.stderr); missing += 1; continue

        img_w, img_h = Image.open(entry["image_filename"]).size
        with Image.open(entry["mask_filename"]) as m:
            if m.size == (img_w, img_h):
                continue
            resized_im = m.resize((img_w, img_h), Image.NEAREST)

        key = os.path.relpath(entry["mask_filename"]).replace(os.sep, "__")
        stem, ext = os.path.splitext(key)
        out_path = args.cache_dir / f"{stem}__{img_w}x{img_h}{ext}"
        if not out_path.exists():
            resized_im.save(out_path)
        entry["mask_filename"] = str(out_path)
        resized += 1

    args.jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    print(f"verified {len(entries)} entries (resized={resized}, missing={missing})")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
