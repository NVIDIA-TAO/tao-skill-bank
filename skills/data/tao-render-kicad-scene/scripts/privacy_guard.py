#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local guard that keeps private reference data out of commits and pushes.

`build` hashes every file under the private roots into a denylist kept outside
the repository. `check` inspects staged blobs or the blobs a push would add and
fails closed on: a denylisted SHA-256 (also as a Git LFS oid), a restricted file
name, any binary or image payload not explicitly allowed, embedded base64
images, or oversized files. `install` writes pre-commit and pre-push hooks.

A hash match cannot recognise re-encoded or cropped derivatives; the binary and
name rules are what stop those. CI is a second layer, not the privacy boundary.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

NAME_PATTERNS = ("*_SolderLight.*", "SCN_*", "*contactsheet*", "*contact_sheet*", "private-data/*", "*/private-data/*")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".exr", ".npy", ".npz",
                   ".blend", ".glb", ".gltf", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".zip", ".tar",
                   ".gz", ".xz", ".7z", ".pdf"}
EMBEDDED = re.compile(rb"data:image/[a-zA-Z0-9.+-]+;base64,")
LFS = re.compile(rb"^version https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})")
MAX_BYTES = 1_000_000
ZERO = "0" * 40


def git(*args, data=None):
    return subprocess.run(["git", *args], input=data, capture_output=True, check=True).stdout


def load_denylist(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"privacy guard: denylist missing ({path}); refusing (fail closed)")
    hashes = {line.split()[0] for line in path.read_text().splitlines() if line and not line.startswith("#")}
    if not hashes:
        raise SystemExit("privacy guard: denylist is empty; refusing (fail closed)")
    return hashes


def inspect(name, blob, denylist, allow):
    """Return the reasons this blob must not leave the machine."""
    problems = []
    digest = hashlib.sha256(blob).hexdigest()
    if digest in denylist:
        problems.append("content SHA-256 is a private source file")
    pointer = LFS.match(blob)
    if pointer and pointer.group(1).decode() in denylist:
        problems.append("Git LFS oid is a private source file")
    if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(os.path.basename(name), p) for p in NAME_PATTERNS):
        problems.append("restricted file name")
    allowed = any(fnmatch.fnmatch(name, p) for p in allow)
    if not allowed:
        if Path(name).suffix.lower() in BINARY_SUFFIXES or b"\0" in blob[:8192]:
            problems.append("binary/image payload without a synthetic-provenance allow rule")
        if len(blob) > MAX_BYTES:
            problems.append(f"larger than {MAX_BYTES} bytes")
    if EMBEDDED.search(blob):
        problems.append("embedded base64 image")
    return problems


def staged_blobs():
    names = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").split(b"\0")
    for raw in filter(None, names):
        name = raw.decode("utf-8", "surrogateescape")
        yield name, git("cat-file", "blob", ":" + name)


def range_blobs(spec):
    """Blobs reachable from the new commits but not from what the remote already has."""
    listing = git("rev-list", "--objects", *spec).decode("utf-8", "surrogateescape").splitlines()
    for line in listing:
        oid, _, name = line.partition(" ")
        if name and git("cat-file", "-t", oid).strip() == b"blob":
            yield name, git("cat-file", "blob", oid)


def cmd_build(args):
    rows = []
    for root in args.root:
        root = Path(root).resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            rows.append(digest.hexdigest())
    out = Path(args.out).resolve()
    top = Path(git("rev-parse", "--show-toplevel").decode().strip()) if args.in_repo_check else None
    if top and (out == top or top in out.parents):
        raise SystemExit("Keep the denylist outside the repository")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# sha256 of private source files; hashes only\n" + "\n".join(sorted(set(rows))) + "\n")
    print(json.dumps({"denylist": str(out), "files_hashed": len(rows), "unique_sha256": len(set(rows)),
                      "roots": [str(Path(r).resolve()) for r in args.root]}))


def cmd_check(args):
    denylist = load_denylist(args.denylist)
    if args.staged:
        blobs = staged_blobs()
    elif args.pre_push:
        specs = []
        for line in sys.stdin.read().splitlines():
            _, local_sha, _, remote_sha = line.split()
            if local_sha == ZERO:
                continue  # branch deletion pushes no content
            specs.append([local_sha, "^" + remote_sha] if remote_sha != ZERO else [local_sha, "--not", "--remotes"])
        blobs = (item for spec in specs for item in range_blobs(spec))
    else:
        blobs = range_blobs(args.range.split())
    findings, count = [], 0
    for name, blob in blobs:
        count += 1
        for problem in inspect(name, blob, denylist, args.allow):
            findings.append({"path": name, "problem": problem})
    print(json.dumps({"blobs_checked": count, "denylist_entries": len(denylist), "findings": findings}, indent=2))
    if findings:
        print("privacy guard: BLOCKED", file=sys.stderr)
        sys.exit(1)


def cmd_install(args):
    hooks = Path(git("rev-parse", "--git-common-dir").decode().strip()).resolve() / "hooks"
    hooks.mkdir(exist_ok=True)
    denylist = Path(args.denylist).resolve()
    load_denylist(denylist)
    allow = "".join(f" --allow '{p}'" for p in args.allow)
    marker = "# tao-render-kicad-scene privacy guard"
    for hook, mode in (("pre-commit", "--staged"), ("pre-push", "--pre-push")):
        path = hooks / hook
        if path.exists() and marker not in path.read_text():
            raise SystemExit(f"{path} exists and is not this guard; merge it by hand")
        path.write_text(f"""#!/usr/bin/env bash
{marker}
# Fails closed: a missing script or denylist blocks the operation.
guard="$(git rev-parse --show-toplevel)/skills/data/tao-render-kicad-scene/scripts/privacy_guard.py"
[ -f "$guard" ] || {{ echo "privacy guard script missing: $guard" >&2; exit 1; }}
exec python3 "$guard" check --denylist '{denylist}' {mode}{allow}
""")
        path.chmod(0o755)
    print(json.dumps({"hooks_dir": str(hooks), "installed": ["pre-commit", "pre-push"], "denylist": str(denylist)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="hash private roots into a denylist outside the repository")
    build.add_argument("--root", action="append", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--no-in-repo-check", dest="in_repo_check", action="store_false")
    build.set_defaults(func=cmd_build)
    check = sub.add_parser("check", help="inspect staged blobs, a pre-push stdin list, or a rev range")
    check.add_argument("--denylist", required=True)
    mode = check.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--pre-push", action="store_true")
    mode.add_argument("--range", help="git rev-list arguments, e.g. 'HEAD ^origin/main'")
    check.add_argument("--allow", action="append", default=[], help="glob for reviewed synthetic binaries")
    check.set_defaults(func=cmd_check)
    install = sub.add_parser("install", help="write pre-commit and pre-push hooks")
    install.add_argument("--denylist", required=True)
    install.add_argument("--allow", action="append", default=[])
    install.set_defaults(func=cmd_install)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
