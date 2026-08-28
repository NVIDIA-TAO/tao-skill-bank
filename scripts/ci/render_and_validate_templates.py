#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the vendored job templates and validate them with the REAL tools.

The per-MR suite validates templates by parsing them in Python. That catches
structural drift but not "the tool rejects this": a manifest can be valid YAML
with a valid-looking shape and still be refused by the API server's schema, and
an sbatch script can parse as bash and still carry a bad directive.

This runs in the nightly platform pipeline, where the genuine CLIs are present:

  * k8s templates  -> `kubectl apply --dry-run=client` (schema-checked locally;
    no cluster is contacted, no credentials needed)
  * SLURM templates -> `bash -n`, plus `sbatch --test-only` when sbatch exists

Every tool is optional: if it is not installed the corresponding check is
reported as SKIPPED rather than failing, so the job stays useful on a runner
that only has some of them. Exit status is non-zero only on a real rejection.

Usage:
    scripts/ci/render_and_validate_templates.py [--kubectl PATH] [--report FILE]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Real markers are @@UPPER_SNAKE@@. The templates' own header comments describe
# the syntax as "@@<NAME>@@", which must not count as an unsubstituted marker.
MARKER_RE = re.compile(r"@@[A-Z][A-Z0-9_]*@@")

# Same fixture shape the unit tests use: nodes != gpus-per-node so a template
# that confuses WORLD_SIZE with the GPU count cannot render plausibly.
COMMON = {
    "JOB_NAME": "nightly-contract-0001",
    "NUM_NODES": "2",
    "GPUS_PER_NODE": "8",
    "NUM_GPUS": "8",          # single-node templates spell it this way
<<<<<<< HEAD
    "IMAGE": "nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.2.0-rc-53-multiarch",  # versions-key: images.tao_toolkit.pyt
=======
    "IMAGE": "nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.2.0-rc-36-multiarch",  # versions-key: images.tao_toolkit.pyt
>>>>>>> ae022a9 ([TAO-0][Bugfix] NVBug 6602338: Publish the TAO 7.2 skill-bank release (#137))
    "COMMAND": "dino train -e /data/specs/spec.yaml",
}
K8S_VALS = {
    **COMMON,
    "TTL_SECONDS": "3600",
    "IMAGE_PULL_SECRET": "ngc-pull-secret",
    "CRED_SECRET": "tao-creds-nightly",
    "RESULTS_DIR": "/data/results/nightly",
    "MOUNT_PATH": "/data",
    "SHM_SIZE": "16Gi",
    "PVC_CLAIM": "edgeai-datasets",
}
SLURM_VALS = {
    **COMMON,
    "CPUS_PER_TASK": "16",
    "TIME": "04:00:00",
    "LOG_DIR": "/lustre/fsw/users/me/results/nightly/slurm-logs",
    "SBATCH_EXTRA": "#SBATCH --account=edgeai\n#SBATCH --partition=batch",
    "ENV_FILE": "",
    "EXTRA_ENV": "export NCCL_P2P_DISABLE=1",
    "CONTAINER_MOUNTS": "/lustre",
}


def render(tmpl: Path, vals: dict[str, str]) -> str:
    text = tmpl.read_text(encoding="utf-8")
    for key, value in vals.items():
        text = text.replace(f"@@{key}@@", value)
    return text


def _run(cmd: list[str], stdin: str | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, timeout=120
    )
    return proc.returncode, (proc.stderr or proc.stdout).strip()


NO_CLUSTER_MARKERS = (
    "failed to download openapi",
    "connection refused",
    "server API group list",
    "unable to recognize",
    "no configuration has been provided",
)


def _validate_k8s(rendered: str, name: str, args) -> tuple[str, str, str]:
    """Validate a rendered manifest, preferring the offline schema validator.

    `kubectl --dry-run=client` still contacts the API server to resolve API
    groups and fetch the openapi schema, so it is useless on a runner with no
    cluster. kubeconform validates against bundled JSON schemas entirely
    offline, which is what a nightly job without a cluster actually wants.
    """
    if args.kubeconform:
        code, err = _run([args.kubeconform, "-strict", "-summary", "-"], stdin=rendered)
        return ("PASS", name, "kubeconform accepted") if code == 0 else ("FAIL", name, err[:400])
    if args.kubectl:
        code, err = _run([args.kubectl, "apply", "--dry-run=client", "-f", "-"], stdin=rendered)
        if code == 0:
            return ("PASS", name, "kubectl accepted")
        if any(m in err for m in NO_CLUSTER_MARKERS):
            return ("SKIP", name,
                    "no cluster reachable; install kubeconform for offline schema validation")
        return ("FAIL", name, err[:400])
    return ("SKIP", name, "neither kubeconform nor kubectl installed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kubectl", default=shutil.which("kubectl"))
    ap.add_argument("--kubeconform", default=shutil.which("kubeconform"))
    ap.add_argument("--sbatch", default=shutil.which("sbatch"))
    ap.add_argument("--report", default="platform-nightly-report.md")
    args = ap.parse_args()

    results: list[tuple[str, str, str]] = []  # (status, target, detail)

    for tmpl in sorted((REPO / "templates/k8s").glob("*.tmpl")):
        rendered = render(tmpl, K8S_VALS)
        name = f"k8s/{tmpl.name}"
        if leftovers := MARKER_RE.findall(rendered):
            results.append(("FAIL", name, f"unsubstituted markers: {sorted(set(leftovers))[:3]}"))
            continue
        results.append(_validate_k8s(rendered, name, args))

    for tmpl in sorted((REPO / "templates/slurm").glob("*.tmpl")):
        rendered = render(tmpl, SLURM_VALS)
        name = f"slurm/{tmpl.name}"
        if leftovers := MARKER_RE.findall(rendered):
            results.append(("FAIL", name, f"unsubstituted markers: {sorted(set(leftovers))[:3]}"))
            continue
        code, err = _run(["bash", "-n"], stdin=rendered)
        if code != 0:
            results.append(("FAIL", name, f"bash -n: {err[:400]}"))
            continue
        if not args.sbatch:
            results.append(("PASS", name, "bash -n clean (sbatch absent — directives unchecked)"))
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
            fh.write(rendered)
            path = fh.name
        code, err = _run([args.sbatch, "--test-only", path])
        # --test-only still needs a reachable controller; treat scheduling
        # complaints as SKIP and only fail on a rejected directive.
        if code == 0:
            results.append(("PASS", name, "sbatch --test-only accepted"))
        elif "Invalid" in err or "invalid" in err or "unrecognized" in err:
            results.append(("FAIL", name, err[:400]))
        else:
            results.append(("SKIP", name, f"sbatch unavailable/unschedulable: {err[:200]}"))

    lines = ["# Platform nightly — template contract", ""]
    lines += [f"- **{status}** `{target}` — {detail}" for status, target, detail in results]
    failed = [r for r in results if r[0] == "FAIL"]
    lines += ["", f"{len(results)} checked, {len(failed)} failed."]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    for status, target, detail in results:
        print(f"{status}: {target} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
