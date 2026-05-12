#!/usr/bin/env python3
"""Validate an AnomalyGen SDG checkpoint directory and print supported anomaly types.

Fails fast (non-zero exit) if the checkpoint is missing required files.
With --step, also asserts that iter_<step:09d>.pt exists.

Usage: scripts/validate_checkpoint.py <checkpoint_dir> [--step N]
"""
import argparse
import os
import pathlib
import sys

try:
    import yaml
except ImportError:
    print("error: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint_dir",
                   help="Path to the trained AnomalyGen checkpoint directory")
    p.add_argument("--step", type=int, default=None,
                   help="Iteration step; if provided, assert iter_<step:09d>.pt exists")
    args = p.parse_args()

    cfg_path = os.path.join(args.checkpoint_dir, "ag_config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"error: {cfg_path} not found. "
              f"Checkpoint dir must contain ag_config.yaml.",
              file=sys.stderr)
        return 1

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    try:
        anomaly_types = cfg["dataloader_train"]["dataset"]["anomaly_types"]
    except (KeyError, TypeError):
        print("error: ag_config.yaml missing "
              "dataloader_train.dataset.anomaly_types",
              file=sys.stderr)
        return 1

    if not anomaly_types:
        print("error: empty anomaly_types in ag_config.yaml", file=sys.stderr)
        return 1

    if args.step is not None:
        ckpt_root = pathlib.Path(args.checkpoint_dir)
        ckpt_pt = ckpt_root / "checkpoints" / "model" / f"iter_{args.step:09d}.pt"
        if not ckpt_pt.exists():
            saved = sorted(int(p.stem.split("_")[1])
                           for p in (ckpt_root / "checkpoints" / "model").glob("iter_*.pt")
                           if p.stem.startswith("iter_"))
            print(
                f"error: checkpoint missing: {ckpt_pt}\n"
                f"  available steps: {saved or '(none found)'}\n"
                f"  pass --step <one of the above>",
                file=sys.stderr,
            )
            return 1

    print(f"OK: checkpoint at {args.checkpoint_dir}")
    print(f"supported anomaly types ({len(anomaly_types)}):")
    for t in anomaly_types:
        print(f"  {t[0]}+{t[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
