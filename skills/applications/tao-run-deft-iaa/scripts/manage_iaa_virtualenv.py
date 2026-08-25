#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan, install, or verify one hash-locked IAA execution virtualenv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

from virtualenv_runtime import lock_status, validate_tao_virtualenv


# TAO 7.1.0 imports NVIDIA Apex, but the matching CPython 3.12 build is not
# published as a wheel. Build the default (pure-Python) package from the exact
# public commit used by the 7.1.0 data-services image, after locked torch is in
# place. The approved IAA spec uses torch AdamW and does not request Apex CUDA
# extensions.
APEX_SOURCE = (
    "apex @ https://github.com/NVIDIA/apex/archive/"
    "6424da3b4faa6c8f062da4a48c424fff3f02d42d.tar.gz"
    "#sha256=2eb60dd79ed797c4c80e194a45927538058c22f151a992ab5c75823b4af5e725"
)

TAO_CORE_HANDLER_INIT = pathlib.Path(
    "lib/python3.12/site-packages/nvidia_tao_core/microservices/handlers/"
    "execution_handlers/__init__.py"
)
TAO_CORE_HANDLER_INIT_ORIGINAL_SHA256 = (
    "3557a28fb451c80d94bb1428288939c001c39320e042aeb3a8a44d97b75a01b9"
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_local_runtime_compatibility(target: pathlib.Path) -> pathlib.Path:
    """Make TAO Core's unused hosted handler optional in an exact known build.

    TAO Core 7.1.0 eagerly imports every execution backend.  That makes local
    Data Services imports fail when the separately distributed hosted-service
    client is absent, even though this workflow invokes only local CLIs.  Apply
    a hash-gated replacement so unknown package versions fail closed.
    """
    destination = target / TAO_CORE_HANDLER_INIT
    replacement = pathlib.Path(__file__).resolve().parent.parent / (
        "patches/tao_core_execution_handlers_init.py"
    )
    if not destination.is_file() or not replacement.is_file():
        raise ValueError("TAO Core local-runtime compatibility inputs are missing")
    original_sha = _sha256(destination)
    replacement_sha = _sha256(replacement)
    if original_sha == replacement_sha:
        return destination
    if original_sha != TAO_CORE_HANDLER_INIT_ORIGINAL_SHA256:
        raise ValueError(
            "TAO Core execution-handler module does not match the approved 7.1.0 build"
        )
    temporary = destination.with_name(destination.name + ".iaa-new")
    try:
        shutil.copyfile(replacement, temporary)
        if _sha256(temporary) != replacement_sha:
            raise ValueError("TAO Core local-runtime compatibility copy was corrupted")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256(destination) != replacement_sha:
        raise ValueError("TAO Core local-runtime compatibility promotion was corrupted")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "install", "repair", "verify"):
        command = subparsers.add_parser(verb)
        command.add_argument("--profile", required=True, choices=("pyt", "ds"))
        command.add_argument("--virtualenv", required=True, type=pathlib.Path)
        if verb == "install":
            command.add_argument("--approve-install", action="store_true")
        if verb == "repair":
            command.add_argument("--approve-repair", action="store_true")
        if verb == "verify":
            command.add_argument("--min-gpus", required=True, type=int)
            command.add_argument("--gpu-ids", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status = lock_status(args.profile)
        if args.verb == "plan":
            status["virtualenv"] = str(args.virtualenv.expanduser().resolve())
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        if args.verb == "verify":
            try:
                gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",")]
            except ValueError as exc:
                raise ValueError("--gpu-ids must be a comma-separated integer list") from exc
            resolved = validate_tao_virtualenv(
                args.virtualenv,
                profile=args.profile,
                probe_imports=True,
                minimum_gpus=args.min_gpus,
                gpu_ids=gpu_ids,
            )
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "profile": args.profile,
                        "virtualenv": str(resolved),
                    }
                )
            )
            return 0
        if args.verb == "repair":
            if not args.approve_repair:
                raise ValueError(
                    "repair requires explicit --approve-repair after the launch review"
                )
            if not status["ready_to_install"]:
                raise ValueError(str(status["blocker"]))
            target = args.virtualenv.expanduser().resolve()
            python = target / "bin" / "python"
            if (
                not (target / "pyvenv.cfg").is_file()
                or not python.is_file()
                or not os.access(python, os.X_OK)
            ):
                raise ValueError(
                    f"approved {args.profile} virtualenv is missing or invalid: {target}"
                )
            identity = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import pathlib,sys; "
                        "assert sys.version_info[:2] == (3, 12); "
                        "assert pathlib.Path(sys.prefix).resolve() == "
                        f"pathlib.Path({str(target)!r}).resolve()"
                    ),
                ],
                text=True,
                capture_output=True,
            )
            if identity.returncode != 0:
                raise ValueError(_completed_error(identity, "virtualenv identity probe"))
            lock = pathlib.Path(status["lock_file"])
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--no-deps",
                    "-r",
                    str(lock),
                ],
                check=True,
            )
            apply_local_runtime_compatibility(target)
            validate_tao_virtualenv(target, profile=args.profile, probe_imports=True)
            print(
                json.dumps(
                    {
                        "status": "REPAIRED",
                        "profile": args.profile,
                        "virtualenv": str(target),
                        "lock_sha256": status["actual_sha256"],
                    }
                )
            )
            return 0
        if not args.approve_install:
            raise ValueError("install requires explicit --approve-install after the launch review")
        if not status["ready_to_install"]:
            raise ValueError(str(status["blocker"]))
        target = args.virtualenv.expanduser().resolve()
        if target.exists():
            raise ValueError(f"refusing to overwrite existing virtualenv: {target}")
        python312 = shutil.which("python3.12")
        virtualenv = shutil.which("virtualenv")
        if python312 is None or virtualenv is None:
            raise ValueError("installation requires python3.12 and virtualenv on PATH")
        lock = pathlib.Path(status["lock_file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the final path atomically. Virtualenv console-script shebangs
        # embed this path, so constructing elsewhere and renaming would create
        # a subtly broken, non-relocatable environment.
        target.mkdir()
        installed = False
        try:
            subprocess.run([virtualenv, "--python", python312, str(target)], check=True)
            subprocess.run(
                [
                    str(target / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--no-deps",
                    "-r",
                    str(lock),
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(target / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    APEX_SOURCE,
                ],
                check=True,
            )
            apply_local_runtime_compatibility(target)
            validate_tao_virtualenv(
                target, profile=args.profile, probe_imports=True
            )
            installed = True
        finally:
            if not installed and target.exists():
                shutil.rmtree(target)
        resolved = target
        print(
            json.dumps(
                {
                    "status": "INSTALLED",
                    "profile": args.profile,
                    "virtualenv": str(resolved),
                }
            )
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"manage_iaa_virtualenv: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
