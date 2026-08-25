from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile


REPO = pathlib.Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "skills/platform/tao-run-on-docker/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "render_iaa_model_action", SCRIPTS / "render_iaa_model_action.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
RUNTIME = REPO / "skills/applications/tao-run-deft-iaa/scripts"
PATCHES = REPO / "skills/applications/tao-run-deft-iaa/patches"


def request(root: pathlib.Path) -> dict:
    controller = root / "skill-bank"
    runtime = controller / "skills/applications/tao-run-deft-iaa/scripts"
    patches = root / "patches"
    shutil.copytree(RUNTIME, runtime, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PATCHES, patches, ignore=shutil.ignore_patterns("__pycache__"))
    artifact = controller / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    payload = {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "platform": "docker", "name": "evaluate", "label": "baseline",
        "runtime_sha256": MODULE._python_tree_sha256(runtime / "iaa_deft"),
        "gpu_ids": [2, 5], "passed_hf_token": False, "forward_env": [],
        "workload_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        "spec_bundle": {
            "action": "deft-iaa-evaluate-0123456789abcdef",
            "network_arch": "clip", "mode": "args",
            "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
            "command": "clip", "args": ["evaluate", "-e", "/results/zs/specs/eval_config.yaml"],
            "compute_shape": {"gpus": 2, "nodes": 1},
        },
        "mounts": [
            {"source": "/tmp/results", "target": "/results", "read_only": False},
            {"source": str(patches), "target": "/patches", "read_only": True},
            {"source": "/tmp/cache", "target": "/cache", "read_only": False},
        ],
        "environment": {
            "HOME": "/tmp", "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface", "XDG_CACHE_HOME": "/cache",
        },
        "controller_snapshot": MODULE._snapshot_manifest(controller),
        "patches_snapshot": MODULE._snapshot_manifest(patches),
    }
    payload["request_sha256"] = MODULE._canonical_sha256(payload)
    return payload


def test_renderer_preserves_gpu_ids_and_identity_environment():
    with tempfile.TemporaryDirectory() as temp:
        payload = request(pathlib.Path(temp))
        argv = MODULE.render_argv(payload, "iaa-model-job")
    assert argv[argv.index("--gpus") + 1] == '"device=2,5"'
    assert "--gpus all" not in " ".join(argv)
    assert any(value.startswith("USER=") for value in argv)
    assert any(value.startswith("LOGNAME=") for value in argv)
    assert "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor" in argv


def test_renderer_fails_closed_on_gpu_and_snapshot_mutation():
    with tempfile.TemporaryDirectory() as temp:
        payload = request(pathlib.Path(temp))
        payload["gpu_ids"] = [2, 2]
        payload["request_sha256"] = MODULE._canonical_sha256(payload)
        try:
            MODULE.render_argv(payload, "iaa-model-job")
        except ValueError as exc:
            assert "gpu_ids" in str(exc)
        else:
            raise AssertionError("duplicate GPU IDs were accepted")

    with tempfile.TemporaryDirectory() as temp:
        payload = request(pathlib.Path(temp))
        pathlib.Path(payload["patches_snapshot"]["root"]).joinpath("unexpected.py").write_text("x")
        try:
            MODULE.render_argv(payload, "iaa-model-job")
        except ValueError as exc:
            assert "patches_snapshot" in str(exc)
        else:
            raise AssertionError("changed patch snapshot was accepted")
