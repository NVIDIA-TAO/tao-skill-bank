#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Make the Cosmos-RL evaluation GPU video decoder registration optional.

`cosmos_rl/evaluation/base.py`'s `BaseEvaluator.load_model()` unconditionally
registers a PyNvVideoCodec-backed GPU video decoder before loading the model —
even for a dataset (like this skill's single-image bare OK/NG contract) that
never reads a video. `PyNvVideoCodec` dlopens both the NVIDIA encode and
decode shared libraries at import time; on a host whose GPU has no hardware
video encoder (for example H200, which dropped NVENC), `libnvidia-encode.so.1`
does not exist and the import — and therefore the whole evaluate job — hard
fails, regardless of the fact that no video will ever be decoded.

There is no config key or environment override in the source for this: the
only accepted `vision.video_decoder` value is `"pynvvideocodec"`; anything
else raises `ValueError` immediately. This reads `base.py` **out of the image
being used**, classifies it the same way `patch_eval_image_cap.py` does, and
when the known unconditional-registration shape is present, rewrites it to
skip GPU decoder registration when the run sets
`TAO_SKIP_PYNV_VIDEO_DECODER=1` — the opt-in stays off by default, so a video
-carrying workload on hardware that does have NVENC/NVDEC keeps registering
the decoder exactly as before. An unrecognized shape fails closed for manual
verification, same as the image-cap patch.
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

_OLD_BLOCK = '''        decoder_name = self.vision_config.get("video_decoder", "pynvvideocodec")
        if decoder_name != "pynvvideocodec":
            raise ValueError(f"Unsupported evaluation video decoder: {decoder_name}")
        from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader

        decoder = register_pynv_video_reader(
            cache_size=int(self.vision_config.get("video_cache_size", 0)),
            video_override_map=self.vision_config.get("video_override_map"),
            strict=True,
        )
        log.info("GPU video decoder registered: %s", decoder)
'''

_NEW_BLOCK = '''        import os as _tao_os

        if _tao_os.environ.get("TAO_SKIP_PYNV_VIDEO_DECODER", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            log.info(
                "Skipping GPU video decoder registration "
                "(TAO_SKIP_PYNV_VIDEO_DECODER set; image-only evaluation)"
            )
        else:
            decoder_name = self.vision_config.get("video_decoder", "pynvvideocodec")
            if decoder_name != "pynvvideocodec":
                raise ValueError(f"Unsupported evaluation video decoder: {decoder_name}")
            from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader

            decoder = register_pynv_video_reader(
                cache_size=int(self.vision_config.get("video_cache_size", 0)),
                video_override_map=self.vision_config.get("video_override_map"),
                strict=True,
            )
            log.info("GPU video decoder registered: %s", decoder)
'''

# The opt-in marker that proves a previously-patched source already skips
# registration under the env var, so a re-run against an already-patched
# image classifies as ALREADY_SUFFICIENT rather than re-matching _OLD_BLOCK.
_OPT_IN_MARKER = "TAO_SKIP_PYNV_VIDEO_DECODER"

PATCH_REQUIRED = "patch_required"
ALREADY_SUFFICIENT = "already_sufficient"
PATTERN_ABSENT = "pattern_absent"


def read_from_image(image: str, container_path: str, *, docker: str) -> str:
    if shutil.which(docker) is None:
        raise ValueError(f"{docker} not found on PATH")
    try:
        proc = subprocess.run(
            [docker, "run", "--rm", "--entrypoint", "cat", image, container_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"timed out after {exc.timeout}s reading {container_path} from {image}; "
            "pre-pull the image or fix platform access before launch"
        ) from exc
    if proc.returncode != 0:
        raise ValueError(
            f"could not read {container_path} from {image}: "
            f"{proc.stderr.strip() or 'unknown error'}"
        )
    if not proc.stdout.strip():
        raise ValueError(f"{container_path} in {image} is empty")
    return proc.stdout


def classify(source: str) -> str:
    """Return a source-driven verdict; never guesses on an unrecognized shape."""
    if _OPT_IN_MARKER in source:
        return ALREADY_SUFFICIENT
    if source.count(_OLD_BLOCK) == 1:
        return PATCH_REQUIRED
    if "register_pynv_video_reader" in source or "video_decoder" in source:
        raise ValueError(
            "classification=unknown: the recognized unconditional GPU video "
            "decoder registration block is absent or duplicated, but the "
            "evaluator still references register_pynv_video_reader/"
            "video_decoder; verify the new source shape before evaluating"
        )
    return PATTERN_ABSENT


def apply_patch(source: str) -> str:
    if source.count(_OLD_BLOCK) != 1:
        raise ValueError(
            "expected exactly one unconditional GPU video decoder registration "
            "block; re-verify by hand rather than rewriting blindly"
        )
    return source.replace(_OLD_BLOCK, _NEW_BLOCK, 1)


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
        help="Report classification without writing anything. Use before the "
        "approval gate, which forbids writes.",
    )
    parser.add_argument("--container-path", default=CONTAINER_PATH)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--summary", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output_dir is None and not args.probe:
        print(
            "patch_eval_video_decoder: --output-dir is required without --probe",
            file=sys.stderr,
        )
        return 2
    try:
        source = read_from_image(args.image, args.container_path, docker=args.docker)
        classification = classify(source)
        patched = apply_patch(source) if classification == PATCH_REQUIRED else None
    except (OSError, ValueError) as exc:
        print(f"patch_eval_video_decoder: {exc}", file=sys.stderr)
        return 2

    summary = {
        "image": args.image,
        "container_path": args.container_path,
        "classification": classification,
        "patch_needed": classification == PATCH_REQUIRED,
        "env_flag": "TAO_SKIP_PYNV_VIDEO_DECODER=1",
    }
    if args.probe:
        summary["mount_argument"] = None
        print(
            f"patch_eval_video_decoder: classification={classification} "
            f"patch_needed={summary['patch_needed']}"
        )
    elif classification != PATCH_REQUIRED:
        summary["mount_argument"] = None
        detail = (
            "already conditioned on TAO_SKIP_PYNV_VIDEO_DECODER"
            if classification == ALREADY_SUFFICIENT
            else "contains no GPU video decoder registration block"
        )
        print(f"patch_eval_video_decoder: {args.image} {detail}; no patch needed")
    else:
        assert patched is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = (args.output_dir / "base.py").resolve()
        out.write_text(patched)
        (args.output_dir / "base.py.orig").write_text(source)
        summary["patched_file"] = str(out)
        summary["mount_argument"] = f"{out}:{args.container_path}:ro"
        print(
            "patch_eval_video_decoder: made GPU video decoder registration "
            f"opt-out via TAO_SKIP_PYNV_VIDEO_DECODER; wrote {out}"
        )
        print(f"MOUNT_ARG={summary['mount_argument']}")

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
