from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[4]
SCRIPT = REPO / "skills/platform/tao-run-on-docker/scripts/render_iaa_adapter.py"
SPEC = importlib.util.spec_from_file_location("render_iaa_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
RUNTIME = REPO / "skills/applications/tao-run-deft-iaa/scripts"
PATCHES = REPO / "skills/applications/tao-run-deft-iaa/patches"


def trees(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    controller = root / "skill-bank"
    runtime = controller / "skills/applications/tao-run-deft-iaa/scripts"
    patches = root / "patches"
    shutil.copytree(RUNTIME, runtime, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PATCHES, patches, ignore=shutil.ignore_patterns("__pycache__"))
    artifact = controller / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    return controller, runtime, patches


def request(
    controller: pathlib.Path, patches: pathlib.Path,
    name: str = "gap_analysis",
) -> dict:
    runtime = controller / "skills/applications/tao-run-deft-iaa/scripts"
    runtime_sha256 = MODULE._python_tree_sha256(runtime / "iaa_deft")
    payload = {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "platform": "docker", "name": name, "label": "iter1",
        "runtime_sha256": runtime_sha256, "gpu_ids": [],
        "passed_hf_token": False, "forward_env": [],
        "workload_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "spec_bundle": {
            "network_arch": "iaa-adapter", "mode": "args",
            "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
            "command": "python3",
            "args": ["/iaa-runtime/run_iaa_compute.py", name,
                     "--results-dir", "/results", "--label", "iter1"],
            "compute_shape": {"gpus": 0, "nodes": 1},
        },
        "mounts": [
            {"source": "/tmp/results", "target": "/results", "read_only": False},
            {"source": str(patches), "target": "/patches", "read_only": True},
            {"source": str(runtime), "target": "/iaa-runtime", "read_only": True},
        ],
        "environment": {
            "HOME": "/tmp", "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface", "XDG_CACHE_HOME": "/cache",
            "IAA_COMPUTE_FRAME": "docker",
        },
        "controller_snapshot": MODULE._snapshot_manifest(controller),
        "patches_snapshot": MODULE._snapshot_manifest(patches),
    }
    payload["request_sha256"] = MODULE._canonical_sha256(payload)
    return payload


class DockerIaaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.controller, self.runtime, self.patches = trees(
            pathlib.Path(self.temporary.name)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def payload(self, name: str = "gap_analysis") -> dict:
        return request(self.controller, self.patches, name)

    def test_signed_allowlisted_adapter_omits_every_gpu_selector(self):
        payload = self.payload()
        argv = MODULE.render_argv(payload, "iaa-adapter-job")
        text = json.dumps(argv)
        self.assertNotIn("--gpus", argv)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", text)
        mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
        runtime_mount = next(value for value in mounts if "dst=/iaa-runtime" in value)
        self.assertIn("dst=/iaa-runtime", runtime_mount)
        self.assertTrue(runtime_mount.endswith(",readonly"))
        self.assertIn(f"tao-request-sha256={payload['request_sha256']}", argv)
        self.assertIn(f"tao-runtime-sha256={payload['runtime_sha256']}", argv)

    def test_visualize_finish_accepts_only_fixed_native_math_thread_caps(self):
        payload = self.payload("visualize_finish")
        payload["environment"].update(MODULE.VISUALIZE_THREAD_CAPS)
        payload["request_sha256"] = MODULE._canonical_sha256(payload)

        argv = MODULE.render_argv(payload, "iaa-visualize-finish")

        for name, value in MODULE.VISUALIZE_THREAD_CAPS.items():
            self.assertIn(f"{name}={value}", argv)
        changed = self.payload("visualize_finish")
        changed["environment"].update(MODULE.VISUALIZE_THREAD_CAPS)
        changed["environment"]["OPENBLAS_NUM_THREADS"] = "2"
        changed["request_sha256"] = MODULE._canonical_sha256(changed)
        with self.assertRaisesRegex(ValueError, "environment"):
            MODULE.render_argv(changed, "iaa-visualize-finish")

    def test_signature_unknown_action_gpu_and_runtime_mutations_fail_closed(self):
        cases = []
        unsigned = self.payload()
        unsigned["request_sha256"] = "0" * 64
        cases.append((unsigned, "signature"))
        unknown = self.payload()
        unknown["name"] = "not_allowed"
        unknown["request_sha256"] = MODULE._canonical_sha256(unknown)
        cases.append((unknown, "allowlisted"))
        gpu = self.payload()
        gpu["gpu_ids"] = [0]
        gpu["spec_bundle"]["compute_shape"]["gpus"] = 1
        gpu["request_sha256"] = MODULE._canonical_sha256(gpu)
        cases.append((gpu, "gpu_ids"))
        writable = self.payload()
        writable["mounts"][1]["read_only"] = False
        writable["request_sha256"] = MODULE._canonical_sha256(writable)
        cases.append((writable, "read-only"))
        changed = self.payload()
        changed["runtime_sha256"] = "0" * 64
        changed["request_sha256"] = MODULE._canonical_sha256(changed)
        cases.append((changed, "does not match"))
        credential = self.payload()
        credential["environment"]["AWS_SECRET_ACCESS_KEY"] = "must-not-render"
        credential["request_sha256"] = MODULE._canonical_sha256(credential)
        cases.append((credential, "environment"))
        wrong_source = self.payload()
        next(
            row for row in wrong_source["mounts"]
            if row["target"] == "/iaa-runtime"
        )["source"] = str(self.controller)
        wrong_source["request_sha256"] = MODULE._canonical_sha256(wrong_source)
        cases.append((wrong_source, "derived"))
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                MODULE.render_argv(payload, "iaa-adapter-job")

    def test_complete_controller_and_patches_snapshots_reject_tree_changes(self):
        cases = (
            ("controller", "mutate", "run_iaa_compute.py"),
            ("controller", "extra", "unexpected.py"),
            ("controller", "missing", "run_iaa_compute.py"),
            (
                "controller-root", "mutate",
                "skills/core/tao-artifacts/references/spec_bundle.schema.json",
            ),
            ("patches", "mutate", None),
            ("patches", "extra", "unexpected.patch"),
            ("patches", "missing", None),
        )
        for tree, operation, relative in cases:
            with self.subTest(tree=tree, operation=operation), tempfile.TemporaryDirectory() as temp:
                root = pathlib.Path(temp)
                controller, runtime, patches = trees(root)
                payload = request(controller, patches)
                target_root = (
                    runtime if tree == "controller"
                    else controller if tree == "controller-root"
                    else patches
                )
                files = sorted(path for path in target_root.rglob("*") if path.is_file())
                target = target_root / relative if relative else files[0]
                if operation == "mutate":
                    target.write_bytes(target.read_bytes() + b"\nmutation")
                elif operation == "extra":
                    target.write_text("extra", encoding="utf-8")
                else:
                    target.unlink()
                snapshot = "controller" if tree.startswith("controller") else "patches"
                with self.assertRaisesRegex(ValueError, f"{snapshot}_snapshot"):
                    MODULE.render_argv(payload, "iaa-adapter-job")


if __name__ == "__main__":
    unittest.main()
