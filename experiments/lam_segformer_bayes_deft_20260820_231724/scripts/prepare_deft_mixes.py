#!/usr/bin/env python3
"""Build deterministic DEFT-style hard-example oversampling manifests.

Scoring is computed only from the 316 training masks. Validation images and
masks are linked unchanged and never participate in ranking or mixing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    denom = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        ranks[index] = rank / denom
    return ranks


def score_masks(mask_dir: Path) -> list[dict]:
    import numpy as np
    from PIL import Image

    rows = []
    for path in sorted(mask_dir.glob("*.png")):
        mask = np.asarray(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        labels, counts = np.unique(mask, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(labels, counts)}
        total = float(mask.size)
        rare_fraction = (hist.get(85, 0) + hist.get(170, 0)) / total
        horizontal = np.count_nonzero(mask[:, 1:] != mask[:, :-1])
        vertical = np.count_nonzero(mask[1:, :] != mask[:-1, :])
        boundary_density = (horizontal + vertical) / float(
            mask.shape[0] * (mask.shape[1] - 1)
            + (mask.shape[0] - 1) * mask.shape[1]
        )
        probs = np.asarray([hist.get(v, 0) / total for v in (0, 85, 170, 255)])
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log(probs)).sum() / math.log(4.0))
        rows.append(
            {
                "filename": path.name,
                "rare_fraction": rare_fraction,
                "boundary_density": boundary_density,
                "class_entropy": entropy,
                "labels": sorted(hist),
            }
        )

    if len(rows) != 316:
        raise RuntimeError(f"expected 316 training masks, found {len(rows)}")
    rare_rank = percentile_ranks([r["rare_fraction"] for r in rows])
    boundary_rank = percentile_ranks([r["boundary_density"] for r in rows])
    entropy_rank = percentile_ranks([r["class_entropy"] for r in rows])
    for row, rr, br, er in zip(rows, rare_rank, boundary_rank, entropy_rank):
        row["rare_rank"] = rr
        row["boundary_rank"] = br
        row["entropy_rank"] = er
        row["difficulty"] = 0.50 * rr + 0.30 * br + 0.20 * er
    return sorted(rows, key=lambda r: (-r["difficulty"], r["filename"]))


def build_entries(rows: list[dict], kind: str) -> list[dict]:
    entries = [{"source": r["filename"], "copy": 0} for r in rows]
    if kind == "mix50":
        for row in rows[:158]:
            entries.append({"source": row["filename"], "copy": 1})
    elif kind == "mix100":
        for row in rows[:79]:
            for copy_index in (1, 2, 3):
                entries.append({"source": row["filename"], "copy": copy_index})
        for row in rows[79:158]:
            entries.append({"source": row["filename"], "copy": 1})
    else:
        raise ValueError(kind)
    return entries


def destination_name(source: str, copy_index: int) -> str:
    path = Path(source)
    if copy_index == 0:
        return source
    return f"{path.stem}__deft_{copy_index}{path.suffix}"


def materialize(source_root: Path, destination_root: Path, manifest: dict) -> None:
    for leaf in ("images/train", "masks/train"):
        (destination_root / leaf).mkdir(parents=True, exist_ok=True)
    for split_leaf in ("images/val", "masks/val", "images/test"):
        dest = destination_root / split_leaf
        target = source_root / split_leaf
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(target)

    for entry in manifest["entries"]:
        name = destination_name(entry["source"], entry["copy"])
        for kind in ("images", "masks"):
            source = source_root / kind / "train" / entry["source"]
            dest = destination_root / kind / "train" / name
            if not source.exists():
                raise FileNotFoundError(source)
            if dest.exists() or dest.is_symlink():
                if dest.resolve() != source.resolve():
                    raise RuntimeError(f"conflicting destination: {dest}")
            else:
                dest.symlink_to(source)

    for kind in ("images", "masks"):
        count = len(list((destination_root / kind / "train").iterdir()))
        if count != manifest["sample_count"]:
            raise RuntimeError(f"{destination_root}: {kind} count {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--write-manifests", type=Path)
    parser.add_argument("--materialize-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--destination-root", type=Path)
    args = parser.parse_args()

    if args.write_manifests:
        rows = score_masks(args.mask_dir)
        args.write_manifests.mkdir(parents=True, exist_ok=True)
        for kind in ("mix50", "mix100"):
            entries = build_entries(rows, kind)
            payload = {
                "kind": kind,
                "scoring": "0.50*rare_rank + 0.30*boundary_rank + 0.20*entropy_rank",
                "validation_used_for_ranking": False,
                "sample_count": len(entries),
                "entries": entries,
                "scores": rows,
            }
            (args.write_manifests / f"{kind}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return

    if args.materialize_manifest:
        payload = json.loads(args.materialize_manifest.read_text())
        materialize(args.source_root, args.destination_root, payload)
        print(f"{payload['kind']}={payload['sample_count']}")
        return

    parser.error("choose --write-manifests or --materialize-manifest")


if __name__ == "__main__":
    main()
