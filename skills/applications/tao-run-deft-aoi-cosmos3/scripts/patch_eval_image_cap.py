#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Lift the pinned Cosmos-RL image's 1-image-per-prompt evaluation cap.

`cosmos_rl/evaluation/base.py` hardcodes
``limit_mm_per_prompt={"video": 1, "image": 1}`` when it builds the vLLM
engine. The bare OK/NG contract mandates exactly two images per record
([AOI, golden_reference]), so evaluation fails with::

    ValueError: At most 1 image(s) may be provided in one prompt.

There is no spec key and no environment override, and the rest of that file is
already multi-image correct — only the cap is wrong.

This reads `base.py` **out of the image being used**, rewrites just that
literal, and writes the result under the run's results directory. Nothing is
vendored into the skill, so the patch tracks whatever image is pinned instead
of silently masking a newer one. When the image no longer carries the defect
the script emits nothing and exits 0, which is how the workaround retires
itself.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

CONTAINER_PATH = "/workspace/cosmos_rl_merged/cosmos_rl/evaluation/base.py"
# Tolerates spacing and key order; captures only the image cap's digits.
CAP_PATTERN = re.compile(
    r"(limit_mm_per_prompt\s*=\s*\{[^}]*?[\"']image[\"']\s*:\s*)(\d+)"
)


def read_from_image(image: str, container_path: str, *, docker: str) -> str:
    if shutil.which(docker) is None:
        raise ValueError(f"{docker} not found on PATH")
    proc = subprocess.run(
        [docker, "run", "--rm", "--entrypoint", "cat", image, container_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"could not read {container_path} from {image}: "
            f"{proc.stderr.strip() or 'unknown error'}"
        )
    if not proc.stdout.strip():
        raise ValueError(f"{container_path} in {image} is empty")
    return proc.stdout


def apply_cap(source: str, images: int) -> tuple[str, int]:
    """Return the rewritten source and the cap it previously carried."""
    matches = CAP_PATTERN.findall(source)
    if not matches:
        raise ValueError(
            "limit_mm_per_prompt image cap not found; the image has changed "
            "shape. Re-verify the defect before assuming it still applies."
        )
    if len(matches) > 1:
        raise ValueError(
            f"expected exactly one image cap, found {len(matches)}; "
            "re-verify by hand rather than rewriting all of them"
        )
    current = int(matches[0][1])
    patched = CAP_PATTERN.sub(
        lambda m: f"{m.group(1)}{images}", source, count=1
    )
    return patched, current


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Pinned Cosmos-RL image URI.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Where to write the patched base.py. Use the run's results tree. "
        "Not required with --probe.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Report cap_in_image and patch_needed without writing anything. "
        "Use before the approval gate, which forbids writes.",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=2,
        help="Required images per prompt. bare_okng needs 2.",
    )
    parser.add_argument("--container-path", default=CONTAINER_PATH)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--summary", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.images < 1:
        print("patch_eval_image_cap: --images must be >= 1", file=sys.stderr)
        return 2
    if args.output_dir is None and not args.probe:
        print(
            "patch_eval_image_cap: --output-dir is required without --probe",
            file=sys.stderr,
        )
        return 2
    try:
        source = read_from_image(
            args.image, args.container_path, docker=args.docker
        )
        patched, current = apply_cap(source, args.images)
    except (OSError, ValueError) as exc:
        print(f"patch_eval_image_cap: {exc}", file=sys.stderr)
        return 2

    summary = {
        "image": args.image,
        "container_path": args.container_path,
        "cap_in_image": current,
        "cap_required": args.images,
        "patch_needed": current < args.images,
    }
    if args.probe:
        # Read-only: preflight has to report this number before the approval
        # gate, and that gate forbids writing anything.
        summary["mount_argument"] = None
        print(
            f"patch_eval_image_cap: cap_in_image={current} "
            f"cap_required={args.images} patch_needed={summary['patch_needed']}"
        )
    elif current >= args.images:
        # The image was fixed upstream. Emit no mount so the caller stops
        # overriding a file it no longer needs to touch.
        summary["mount_argument"] = None
        print(
            f"patch_eval_image_cap: {args.image} already allows {current} "
            f"image(s) per prompt; no patch needed"
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = (args.output_dir / "base.py").resolve()
        out.write_text(patched)
        (args.output_dir / "base.py.orig").write_text(source)
        summary["patched_file"] = str(out)
        summary["mount_argument"] = f"{out}:{args.container_path}:ro"
        print(
            f"patch_eval_image_cap: raised image cap {current} -> "
            f"{args.images}; wrote {out}"
        )
        print(f"MOUNT_ARG={summary['mount_argument']}")

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
