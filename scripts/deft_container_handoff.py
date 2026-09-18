#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual DEFT integration harness using existing images, never Docker builds.

Run from clean sibling checkouts:
  python scripts/deft_container_handoff.py run --skill-bank ./tao-skill-bank \
    --pytorch ./tao-pytorch --data-services ./tao-data-services --core ./tao-core \
    --pytorch-image <approved-image> --data-services-image <approved-image> \
    --suite source --output ./deft-source-checks

This is not a wired Jenkins/GitHub CI job. Source checks need network for pinned
test tools and the repositories' pinned pre-commit hook environments. Packaged
checks are offline and omit Skill Bank (which is not an installed TAO package).
Source overlays test the selected Python surfaces, not wheel packaging or
compiled-extension compatibility; packaged checks and release QA are separate.
"""

import argparse
from dataclasses import asdict, fields
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid


REPOS = ("tao-skill-bank", "tao-pytorch", "tao-data-services")
SKILL = "skills/applications/tao-run-dinov3-ssl-deft"
INPUTS = Path("/inputs")
RESULTS = Path("/results")
TEST_TOOLS = ("pytest==8.1.1", "pre-commit==4.6.2", "pylint==4.0.5",
              "pydocstyle==6.3.0", "flake8==7.3.0")


def capture(argv, cwd=None):
    """Return command output without shell interpretation."""
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def source_info(path, base_ref="origin/main"):
    """Require a clean committed snapshot and enumerate its feature diff."""
    if capture(["git", "status", "--porcelain"], path):
        raise ValueError(f"Commit or stash changes before testing: {path}")
    commit = capture(["git", "rev-parse", "HEAD"], path)
    base = capture(["git", "merge-base", base_ref, commit], path)
    files = capture(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, commit],
        path,
    )
    return {"commit": commit, "base": base, "changed_files": files.splitlines()}


def regression_commands(repo, python):
    """Mirror static hooks and select the feature's regression suites."""
    pytest = [python, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-ra"]
    if repo == "tao-skill-bank":
        return [
            ["bash", "scripts/validate-skills.sh"],
            pytest + ["scripts/tests/test_dinov3_ssl_deft_contract.py",
                      "scripts/tests/test_dinov3_container_handoff.py",
                      f"--junitxml={RESULTS / 'regression.xml'}"],
        ]
    if repo == "tao-pytorch":
        return [pytest + [
            "tests/core",
            "tests/distributed",
            "tests/ssl_unit_test/dinov3",
            "tests/ssl_unit_test/nvdinov2/test_dataloader.py",
            "tests/ssl_unit_test/nvdinov2/test_model.py",
            f"--junitxml={RESULTS / 'regression.xml'}",
        ]]
    if repo == "tao-core":
        return [pytest + [
            "nvidia_tao_core/tests/dinov3",
            "nvidia_tao_core/tests/test_common_config.py",
            "nvidia_tao_core/tests/test_downstream_dinov3_backbones.py",
            f"--junitxml={RESULTS / 'regression.xml'}",
        ]]
    return [pytest + [
        "tests/mining/test_dinov3_refinement.py",
        "tests/mining/test_dinov3_ann_audit.py",
        "tests/mining/test_dinov3_dense_store.py",
        "tests/mining/dinov3_workflow",
        "../tao-skill-bank/scripts/tests/test_dinov3_cross_repo_smoke.py",
        f"--junitxml={RESULTS / 'regression.xml'}",
    ]]


def source_pythonpath(repo, root):
    """Return candidate source roots needed by one repository's test suite."""
    paths = [root / repo]
    if repo == "tao-data-services":
        paths.append(root / "tao-pytorch")
        if (root / "tao-core").is_dir():
            paths.append(root / "tao-core")
    elif repo == "tao-pytorch":
        if (root / "tao-core").is_dir():
            paths.append(root / "tao-core")
    elif repo == "tao-core":
        paths.append(root / "tao-pytorch")
    return os.pathsep.join(str(path) for path in paths)


def source_environment(repo, root, venv):
    """Build an isolated source-test environment for one repository."""
    env = dict(os.environ)
    env.pop("TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PRE_COMMIT_HOME=str(root / "pre-commit"),
        PIP_CACHE_DIR=str(root / "pip-cache"),
        TMPDIR=str(root),
        PATH=str(venv / "bin") + os.pathsep + env["PATH"],
        TAO_DATA_SERVICES_SOURCE=str(root / "tao-data-services"),
        TAO_DEFT_REQUIRE_SCHEMA_PARITY="1",
        PYTHONPATH=source_pythonpath(repo, root),
        VALIDATE_BASE_REF="HEAD^",
        SKIP="",
    )
    if repo == "tao-data-services":
        env["TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE"] = "1"
    return env


def packaged_check(repo, require_gpu_faiss=False):
    """Check the installed image, without importing a mounted source checkout."""
    if repo == "tao-core":
        core = importlib.import_module("nvidia_tao_core.config.dinov3.default_config")
        grit = importlib.import_module("nvidia_tao_core.config.dinov3.grit_score")
        runtime = importlib.import_module("nvidia_tao_pytorch.config.dinov3.default_config")
        config = core.ExperimentConfig()
        required = {"train_manifest", "archive_cache_size", "shard_shuffle_window"}
        runtime_config = runtime.ExperimentConfig()
        mismatches = {
            "grit_schema": (
                {field.name: dict(field.metadata) for field in fields(grit.GRITScoreConfig)} !=
                {field.name: dict(field.metadata) for field in fields(runtime.GRITScoreConfig)}
            ),
            "grit_defaults": asdict(config.grit_score) != asdict(runtime_config.grit_score),
            "manifest_presence": not required.issubset(asdict(config.dataset)),
            "training_presence": not {"auto_resume", "checkpoint_keep_last_n"}.issubset(asdict(config.train)),
            "dataset_parity": asdict(config.dataset) != asdict(runtime_config.dataset),
            "train_parity": asdict(config.train) != asdict(runtime_config.train),
        }
        mismatches = {name: failed for name, failed in mismatches.items() if failed}
        if mismatches:
            raise RuntimeError(
                "Installed tao-core lacks the complete DINOv3 refinement schema: "
                + json.dumps(mismatches, sort_keys=True)
            )
        print("PACKAGED_RUNTIME_CHECK_PASSED", flush=True)
        return 0
    modules = (["nvidia_tao_ds.mining.dinov3.internal.refinement",
                "nvidia_tao_ds.mining.dinov3.workflow.cli",
                "nvidia_tao_ds.mining.dinov3.workflow.native_actions",
                "nvidia_tao_pytorch.ssl.dinov3.scripts.train",
                "nvidia_tao_pytorch.ssl.dinov3.scripts.grit_score"]
               if repo in {"tao-data-services", "tao-skill-bank"} else [
                   "nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli",
                   "nvidia_tao_pytorch.ssl.dinov3.scripts.train",
                   "nvidia_tao_pytorch.ssl.dinov3.scripts.grit_score",
                   "nvidia_tao_pytorch.ssl.dinov3.entrypoint.dinov3",
               ])
    for name in modules:
        imported = importlib.import_module(name)
        print(json.dumps({"module": name, "path": imported.__file__}), flush=True)
    if repo in {"tao-data-services", "tao-skill-bank"}:
        preflight = importlib.import_module(
            "nvidia_tao_ds.mining.dinov3.workflow.cli"
        )
        if preflight.main(["preflight", "--gpu"]) != 0:
            raise RuntimeError("Data Services DINOv3 GPU preflight failed")
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Container CUDA is unavailable")
    if require_gpu_faiss:
        import faiss
        import numpy as np
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatL2(2))
        vectors = np.array([[0, 0], [1, 1]], dtype="float32")
        index.add(vectors)
        if index.search(vectors, 1)[1][:, 0].tolist() != [0, 1]:
            raise RuntimeError("GPU FAISS neighbor check failed")
    print("PACKAGED_RUNTIME_CHECK_PASSED", flush=True)
    return 0


def inside(repo, suite, require_gpu_faiss=False):
    """Run in an ephemeral container; leave caller checkouts unchanged."""
    os.environ.pop("PYTHONPATH", None)
    if require_gpu_faiss and suite != "packaged":
        raise ValueError("--require-gpu-faiss requires --suite packaged")
    if suite == "packaged":
        return packaged_check(repo, require_gpu_faiss=require_gpu_faiss)
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="deft-ci-") as temporary:
        root = Path(temporary)
        for name in manifest["repos"]:
            destination = root / name
            destination.mkdir()
            if name == repo:
                with tarfile.open(INPUTS / f"{name}-base.tar") as archive:
                    archive.extractall(destination, filter="data")
                subprocess.run(["git", "init", "-q"], cwd=destination, check=True)
                subprocess.run(["git", "add", "--force", "-A"], cwd=destination, check=True)
                subprocess.run(["git", "-c", "user.name=Container CI", "-c", "user.email=ci@example.invalid",
                                "commit", "--no-gpg-sign", "-qm", "CI baseline snapshot"], cwd=destination, check=True)
                # Remove only files extracted into this fresh temporary directory.
                # Retain its synthetic .git; no caller checkout is writable.
                for item in destination.iterdir():
                    if item.name == ".git":
                        continue
                    if item.is_dir() and not item.is_symlink():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            with tarfile.open(INPUTS / f"{name}.tar") as archive:
                # TAO images use Python 3.12; reject escaping archive paths/links.
                archive.extractall(destination, filter="data")
        work = root / repo
        # Synthetic commits reproduce the real feature diff without copying
        # host .git, worktree pointers, hooks, or credential-bearing remotes.
        # Tar members can share timestamps and sizes while bytes differ. Drop
        # cached index stat entries so git hashes the new snapshot unconditionally.
        subprocess.run(["git", "read-tree", "--empty"], cwd=work, check=True)
        subprocess.run(["git", "add", "--force", "-A"], cwd=work, check=True)
        subprocess.run(["git", "-c", "user.name=Container CI", "-c", "user.email=ci@example.invalid",
                        "commit", "--no-gpg-sign", "-qm", "CI candidate snapshot"], cwd=work, check=True)
        observed = capture(["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD^", "HEAD"], work).splitlines()
        if observed != manifest["repos"][repo]["changed_files"]:
            raise RuntimeError("Archived snapshots do not reproduce the approved feature diff")
        venv = root / "test-tools"
        os.environ["PIP_CACHE_DIR"] = str(root / "pip-cache")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
        python = str(venv / "bin/python")
        subprocess.run([python, "-m", "pip", "install", *TEST_TOOLS], check=True)
        env = source_environment(repo, root, venv)
        commands = [[python, "-m", "pre_commit", "run", "--from-ref", "HEAD^", "--to-ref", "HEAD", "--show-diff-on-failure"]]
        commands += regression_commands(repo, python)
        results = []
        for argv in commands:
            print(json.dumps({"command": argv}), flush=True)
            results.append(subprocess.run(argv, cwd=work, env=env, check=False).returncode)
        (RESULTS / "checks.json").write_text(json.dumps({"exit_codes": results}, indent=2))
        return int(any(results))


def run(args):
    """Snapshot clean commits, pin local image IDs, and run all three checks."""
    paths = dict(zip(REPOS, (args.skill_bank, args.pytorch, args.data_services)))
    if args.core is not None:
        paths["tao-core"] = args.core
    if args.require_gpu_faiss and args.suite != "packaged":
        raise ValueError("--require-gpu-faiss requires --suite packaged")
    if "," in str(args.output.resolve()):
        raise ValueError("Docker bind paths containing commas are unsupported")
    if args.suite == "packaged":
        paths.pop("tao-skill-bank")
    metadata = {name: source_info(path.resolve(), args.base_ref) for name, path in paths.items()}
    images = {
        name: args.data_services_image
        if name in {"tao-skill-bank", "tao-data-services", "tao-core"}
        else args.pytorch_image
        for name in paths
    }
    require_gpu_faiss = args.require_gpu_faiss
    plan = {"suite": args.suite, "repos": metadata, "images": images, "gpus": args.gpus,
            "require_gpu_faiss": require_gpu_faiss}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    # Resolve to immutable IDs before execution; never pull or build implicitly.
    image_ids = {ref: capture(["docker", "image", "inspect", "--format", "{{.Id}}", ref]) for ref in set(images.values())}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    plan["image_ids"] = image_ids
    (inputs / "manifest.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(__file__, inputs / "driver.py")
    for name, path in paths.items():
        for suffix, ref in (("", "commit"), ("-base", "base")):
            with (inputs / f"{name}{suffix}.tar").open("wb") as stream:
                subprocess.run(["git", "archive", metadata[name][ref]], cwd=path, stdout=stream, check=True)
    statuses = {}
    for name in paths:
        results = output / name
        results.mkdir()
        container = "deft-ci-" + uuid.uuid4().hex
        argv = ["docker", "run", "--rm", "--pull=never", "--name", container,
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--gpus", args.gpus, "--shm-size=8g", "--entrypoint", "python",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "--mount", f"type=bind,src={inputs},dst=/inputs,readonly",
                "--mount", f"type=bind,src={results},dst=/results",
                image_ids[images[name]], "/inputs/driver.py", "inside", "--repo", name, "--suite", args.suite]
        if require_gpu_faiss:
            argv.append("--require-gpu-faiss")
        print(f"Running {name}; log: {results / 'container.log'}", flush=True)
        try:
            with (results / "container.log").open("w") as log:
                statuses[name] = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            statuses[name] = 124
        finally:
            # Clean up only this invocation's exact, freshly generated name.
            subprocess.run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=30)
        (output / "summary.json").write_text(json.dumps({**plan, "exit_codes": statuses}, indent=2))
    print(json.dumps(statuses, indent=2))
    return int(any(statuses.values()))


def main():
    """Parse public host and private container modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    host = commands.add_parser("run")
    for name in ("skill-bank", "pytorch", "data-services", "output"):
        host.add_argument("--" + name, type=Path, required=True)
    host.add_argument("--core", type=Path, help="Optional tao-core checkout for schema parity and lint")
    host.add_argument("--pytorch-image", required=True)
    host.add_argument("--data-services-image", required=True)
    host.add_argument("--base-ref", default="origin/main")
    host.add_argument("--suite", choices=("source", "packaged"), default="source")
    host.add_argument("--require-gpu-faiss", action="store_true",
                      help="Also validate GPU FAISS in packaged images when that optional backend is required")
    host.add_argument("--gpus", default="device=0")
    host.add_argument("--timeout", type=int, default=7200, help="Seconds per container")
    host.add_argument("--dry-run", action="store_true")
    child = commands.add_parser("inside")
    child.add_argument("--repo", choices=(*REPOS, "tao-core"), required=True)
    child.add_argument("--suite", choices=("source", "packaged"), required=True)
    child.add_argument("--require-gpu-faiss", action="store_true")
    args = parser.parse_args()
    return inside(args.repo, args.suite, args.require_gpu_faiss) if args.mode == "inside" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
