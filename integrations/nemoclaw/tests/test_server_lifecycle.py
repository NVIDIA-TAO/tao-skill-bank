#!/usr/bin/env python3
"""Lifecycle tests for isolated NemoClaw result cleanup."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

INTEGRATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_DIR))

from docker_runtime import HostIdentity  # noqa: E402


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        return lambda function: function


class _FakeTransportSecuritySettings:
    def __init__(self, **kwargs):
        pass


class _FakeBaseHTTPMiddleware:
    def __init__(self, app):
        self.app = app


class _FakeJSONResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


def _load_server():
    """Load server.py without making its runtime packages test dependencies."""
    modules = {
        "mcp": types.ModuleType("mcp"),
        "mcp.server": types.ModuleType("mcp.server"),
        "mcp.server.fastmcp": types.ModuleType("mcp.server.fastmcp"),
        "mcp.server.streamable_http": types.ModuleType(
            "mcp.server.streamable_http"
        ),
        "starlette": types.ModuleType("starlette"),
        "starlette.middleware": types.ModuleType("starlette.middleware"),
        "starlette.middleware.base": types.ModuleType(
            "starlette.middleware.base"
        ),
        "starlette.responses": types.ModuleType("starlette.responses"),
        "uvicorn": types.ModuleType("uvicorn"),
    }
    modules["mcp.server.fastmcp"].FastMCP = _FakeFastMCP
    modules[
        "mcp.server.streamable_http"
    ].TransportSecuritySettings = _FakeTransportSecuritySettings
    modules[
        "starlette.middleware.base"
    ].BaseHTTPMiddleware = _FakeBaseHTTPMiddleware
    modules["starlette.responses"].JSONResponse = _FakeJSONResponse
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "nemoclaw_server_under_test", INTEGRATION_DIR / "server.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


server = _load_server()


def _completed(*args, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _inspect_payload(results: Path, token: str, state: str = "exited") -> str:
    metadata = results.stat() if results.exists() else None
    volume_name = server.workspace_volume_name(server.WORKSPACE_ROOT)
    results_path = str(results.relative_to(server.WORKSPACE_ROOT))
    return json.dumps(
        [
            {
                "Id": "container-id",
                "Config": {
                    "Image": "nvcr.io/nvidia/tao/example:1",
                    "Labels": {
                        server.MANAGED_LABEL: "true",
                        server.RESULTS_TOKEN_LABEL: token,
                        server.WORKSPACE_VOLUME_LABEL: volume_name,
                        server.RESULTS_PATH_LABEL: results_path,
                        server.RESULTS_DEVICE_LABEL: str(
                            metadata.st_dev if metadata else 1
                        ),
                        server.RESULTS_INODE_LABEL: str(
                            metadata.st_ino if metadata else 2
                        ),
                    },
                },
                "HostConfig": {"RestartPolicy": {"Name": "no"}},
                "State": {"Status": state, "ExitCode": 1},
                "Mounts": [
                    {
                        "Destination": "/results",
                        "Name": volume_name,
                        "Type": "volume",
                        "RW": True,
                    }
                ],
            }
        ]
    )


def _volume_payload(workspace: Path, *, device: str | None = None) -> str:
    volume_name = server.workspace_volume_name(workspace)
    return json.dumps(
        [
            {
                "Driver": "local",
                "Labels": {server.MANAGED_LABEL: "true"},
                "Name": volume_name,
                "Options": {
                    "device": device or str(workspace),
                    "o": "bind",
                    "type": "none",
                },
            }
        ]
    )


class ManagedLifecycleTests(unittest.TestCase):
    def setUp(self):
        identity = patch.object(
            server,
            "current_host_identity",
            return_value=HostIdentity(1000, 1000, "test-user"),
        )
        identity.start()
        self.addCleanup(identity.stop)

    def test_workspace_tools_use_no_follow_file_descriptors_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace

            written = server.tao_write("specs/train.yaml", "epochs: 1\n")
            read = server.tao_read("specs/train.yaml")
            listed = server.tao_ls("specs")

            self.assertEqual(written["bytes_written"], 10)
            self.assertEqual(read["text"], "epochs: 1\n")
            self.assertEqual(
                listed["entries"],
                [{"name": "train.yaml", "type": "file", "size": 10}],
            )

    def test_workspace_read_and_write_refuse_symlink_escape(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside_tmp,
        ):
            workspace = Path(tmp).resolve()
            outside = Path(outside_tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            secret = outside / "secret.txt"
            secret.write_text("host-secret", encoding="utf-8")
            (workspace / "read-link").symlink_to(secret)
            (workspace / "write-parent").symlink_to(
                outside, target_is_directory=True
            )

            with self.assertRaisesRegex(ValueError, "symlink"):
                server.tao_read("read-link")
            with self.assertRaisesRegex(ValueError, "symlink"):
                server.tao_write("write-parent/secret.txt", "overwritten")

            self.assertEqual(secret.read_text(encoding="utf-8"), "host-secret")

    def test_creates_and_verifies_the_fixed_workspace_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            created = _completed(stdout=server.workspace_volume_name(workspace))
            inspected = _completed(stdout=_volume_payload(workspace))
            with patch.object(
                server, "_docker", side_effect=[created, inspected]
            ) as docker:
                volume_name = server._ensure_workspace_volume()

            self.assertEqual(volume_name, server.workspace_volume_name(workspace))
            self.assertEqual(docker.call_args_list[-1].args[:2], ("volume", "inspect"))

    def test_rejects_a_workspace_volume_bound_to_another_host_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            inspected = _completed(
                stdout=_volume_payload(workspace, device="/var/run")
            )
            with patch.object(server, "_docker", return_value=inspected):
                with self.assertRaisesRegex(ValueError, "untrusted"):
                    server._inspect_workspace_volume(
                        server.workspace_volume_name(workspace)
                    )

    def test_run_refuses_a_root_bridge_before_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            with patch.object(
                server,
                "current_host_identity",
                return_value=HostIdentity(0, 0, "root"),
            ), patch.object(server, "_docker") as docker:
                with self.assertRaisesRegex(PermissionError, "as root"):
                    server.tao_run(
                        "nvcr.io/nvidia/tao/example:1", ["train"]
                    )

            docker.assert_not_called()
            self.assertFalse((workspace / "results").exists())

    def test_run_requires_a_cached_image_before_creating_results(self):
        image = "nvcr.io/nvidia/tao/example:1"
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            missing = _completed(returncode=1, stderr="No such image")
            with patch.object(
                server, "_docker", return_value=missing
            ) as docker, patch.object(
                server, "_ensure_workspace_volume"
            ) as ensure_volume:
                with self.assertRaisesRegex(RuntimeError, "Call tao_pull"):
                    server.tao_run(image, ["train"])

            docker.assert_called_once_with("image", "inspect", image)
            ensure_volume.assert_not_called()
            self.assertFalse((workspace / "results").exists())

    def test_pull_skips_an_image_that_is_already_cached(self):
        image = "nvcr.io/nvidia/tao/example:1"
        with patch.object(
            server, "_docker", return_value=_completed()
        ) as docker:
            result = server.tao_pull(image)

        self.assertEqual(
            result,
            {"image": image, "pulled": False, "status": "already present"},
        )
        docker.assert_called_once_with("image", "inspect", image)

    def test_pull_cold_image_without_a_control_plane_timeout(self):
        image = "nvcr.io/nvidia/tao/example:1"
        missing = _completed(returncode=1, stderr="No such image")
        pulled = _completed(stdout="downloaded\n")
        with patch.object(
            server, "_docker", side_effect=[missing, pulled]
        ) as docker:
            result = server.tao_pull(image)

        self.assertEqual(
            result, {"image": image, "pulled": True, "status": "pulled"}
        )
        self.assertEqual(docker.call_args_list[0].args, ("image", "inspect", image))
        self.assertEqual(docker.call_args_list[1].args, ("pull", image))
        self.assertEqual(docker.call_args_list[1].kwargs, {"timeout": None})

    def test_run_uses_unique_results_tree_and_returns_its_path(self):
        token = "a" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            with patch.object(server.uuid, "uuid4") as uuid4, patch.object(
                server, "_ensure_workspace_volume", return_value=(
                    server.workspace_volume_name(workspace)
                )
            ), patch.object(
                server, "_docker", return_value=_completed(stdout="container-id\n")
            ) as docker:
                uuid4.return_value.hex = token

                result = server.tao_run(
                    "nvcr.io/nvidia/tao/example:1", ["train"]
                )

            expected = workspace / "results" / ".tao-jobs" / token
            self.assertEqual(result["job_id"], "container-id")
            self.assertEqual(
                result["results_subdir"], f"results/.tao-jobs/{token}"
            )
            self.assertTrue(expected.is_dir())
            self.assertEqual(
                docker.call_args_list[0].args,
                ("image", "inspect", "nvcr.io/nvidia/tao/example:1"),
            )
            args = docker.call_args.args
            self.assertEqual(args[args.index("--name") + 1], f"tao-nemoclaw-{token}")
            mounts = [
                args[i + 1] for i, value in enumerate(args) if value == "--mount"
            ]
            self.assertIn(
                "type=volume,src="
                f"{server.workspace_volume_name(workspace)},dst=/results,"
                f"volume-subpath=results/.tao-jobs/{token}",
                mounts,
            )
            self.assertFalse(any(str(workspace) in arg for arg in args))

    def test_run_timeout_reconciles_the_deterministic_container_name(self):
        token = "f" * 32
        timeout = subprocess.TimeoutExpired(cmd="docker run", timeout=120)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            expected = workspace / "results" / ".tao-jobs" / token

            def docker_side_effect(*args):
                if args[:2] == ("image", "inspect"):
                    return _completed()
                if args[0] == "run":
                    raise timeout
                if args[:2] == ("inspect", f"tao-nemoclaw-{token}"):
                    return _completed(stdout=_inspect_payload(expected, token))
                self.fail(f"unexpected docker call: {args}")

            with patch.object(server.uuid, "uuid4") as uuid4, patch.object(
                server,
                "_ensure_workspace_volume",
                return_value=server.workspace_volume_name(workspace),
            ), patch.object(
                server, "_inspect_workspace_volume", return_value={}
            ), patch.object(
                server, "_docker", side_effect=docker_side_effect
            ) as docker:
                uuid4.return_value.hex = token

                result = server.tao_run(
                    "nvcr.io/nvidia/tao/example:1", ["train"]
                )

            self.assertEqual(result["job_id"], "container-id")
            self.assertTrue(result["reconciled"])
            self.assertEqual(
                docker.call_args_list[-1].args,
                ("inspect", f"tao-nemoclaw-{token}"),
            )

    def test_list_returns_only_managed_jobs_for_this_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            listed = _completed(
                stdout=(
                    "new-id\tnvcr.io/nvidia/tao/new:1\trunning\tUp 3 seconds\n"
                    "old-id\tnvcr.io/nvidia/tao/old:1\texited\tExited (0)\n"
                )
            )
            with patch.object(server, "_docker", return_value=listed) as docker:
                result = server.tao_list()

        self.assertEqual(
            result["jobs"],
            [
                {
                    "job_id": "new-id",
                    "image": "nvcr.io/nvidia/tao/new:1",
                    "state": "running",
                    "status": "Up 3 seconds",
                },
                {
                    "job_id": "old-id",
                    "image": "nvcr.io/nvidia/tao/old:1",
                    "state": "exited",
                    "status": "Exited (0)",
                },
            ],
        )
        self.assertEqual(
            docker.call_args.args,
            (
                "ps",
                "-a",
                "--filter",
                f"label={server.MANAGED_LABEL}=true",
                "--filter",
                "label="
                f"{server.WORKSPACE_VOLUME_LABEL}="
                f"{server.workspace_volume_name(workspace)}",
                "--no-trunc",
                "--format",
                "{{.ID}}\t{{.Image}}\t{{.State}}\t{{.Status}}",
            ),
        )

    def test_cleanup_deletes_only_its_results_then_removes_terminal_container(self):
        token = "b" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            results = workspace / "results" / ".tao-jobs" / token
            results.mkdir(parents=True)
            (results / "checkpoint.pth").write_bytes(b"checkpoint")
            sibling = results.parent / ("c" * 32)
            sibling.mkdir()
            outside = workspace / "important"
            outside.mkdir()
            (outside / "keep.txt").write_text("keep", encoding="utf-8")
            (results / "outside-link").symlink_to(
                outside, target_is_directory=True
            )
            inspect = _completed(stdout=_inspect_payload(results, token))
            remove = _completed(stdout="container-id\n")

            def docker_side_effect(*args):
                if args[0] == "inspect":
                    return inspect
                if args[0] == "rm":
                    self.assertFalse(results.exists())
                    return remove
                self.fail(f"unexpected docker call: {args}")

            with patch.object(
                server, "_inspect_workspace_volume", return_value={}
            ), patch.object(
                server, "_docker", side_effect=docker_side_effect
            ) as docker:
                result = server.tao_cleanup_results("container-id")

            self.assertFalse(results.exists())
            self.assertTrue(sibling.exists())
            self.assertEqual(
                (outside / "keep.txt").read_text(encoding="utf-8"), "keep"
            )
            self.assertEqual(
                result["deleted_results_subdir"], f"results/.tao-jobs/{token}"
            )
            self.assertEqual(docker.call_args_list[-1].args, ("rm", "container-id"))

    def test_cleanup_rejects_a_running_writer(self):
        token = "d" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            results = workspace / "results" / ".tao-jobs" / token
            results.mkdir(parents=True)
            inspect = _completed(stdout=_inspect_payload(results, token, "running"))

            with patch.object(server, "_docker", return_value=inspect) as docker:
                with self.assertRaisesRegex(ValueError, "stop it first"):
                    server.tao_cleanup_results("container-id")

            self.assertTrue(results.exists())
            self.assertEqual(docker.call_count, 1)

    def test_destructive_tools_reject_unmanaged_ngc_containers(self):
        server.WORKSPACE_ROOT = Path("/tmp")
        payload = json.loads(_inspect_payload(Path("/tmp/results"), "e" * 32))
        payload[0]["Config"]["Labels"] = {}
        inspect = _completed(stdout=json.dumps(payload))

        with patch.object(server, "_docker", return_value=inspect) as docker:
            with self.assertRaisesRegex(ValueError, "not launched by this TAO bridge"):
                server.tao_stop("someone-elses-container")

        self.assertEqual(docker.call_count, 1)

    def test_lifecycle_tools_reject_docker_option_injection(self):
        with patch.object(server, "_docker") as docker:
            with self.assertRaisesRegex(ValueError, "job reference"):
                server.tao_rm("--force")

        docker.assert_not_called()

    def test_failed_run_response_keeps_results_until_launch_is_reconciled(self):
        token = "1" * 32
        failed = _completed(returncode=1, stderr="daemon disconnected")
        missing = _completed(returncode=1, stderr="No such object")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            with patch.object(server.uuid, "uuid4") as uuid4, patch.object(
                server,
                "_ensure_workspace_volume",
                return_value=server.workspace_volume_name(workspace),
            ), patch.object(
                server,
                "_docker",
                side_effect=[_completed(), failed, missing],
            ):
                uuid4.return_value.hex = token
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    server.tao_run(
                        "nvcr.io/nvidia/tao/example:1", ["train"]
                    )

            results = workspace / "results" / ".tao-jobs" / token
            self.assertTrue(results.is_dir())

    def test_cleanup_failure_retains_container_metadata_for_retry(self):
        token = "2" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            results = workspace / "results" / ".tao-jobs" / token
            results.mkdir(parents=True)
            inspect = _completed(stdout=_inspect_payload(results, token))

            with patch.object(
                server, "_inspect_workspace_volume", return_value={}
            ), patch.object(
                server, "_docker", return_value=inspect
            ) as docker, patch.object(
                server,
                "remove_managed_results_tree",
                side_effect=PermissionError("denied"),
            ):
                with self.assertRaisesRegex(RuntimeError, "retained"):
                    server.tao_cleanup_results("container-id")

            self.assertTrue(results.exists())
            self.assertEqual(docker.call_count, 1)

    def test_cleanup_repairs_owner_locked_directories_without_sudo(self):
        token = "4" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            results = workspace / "results" / ".tao-jobs" / token
            locked = results / "checkpoint-dir"
            locked.mkdir(parents=True)
            (locked / "checkpoint.pth").write_bytes(b"checkpoint")
            inspect = _completed(stdout=_inspect_payload(results, token))
            locked.chmod(0)
            remove = _completed(stdout="container-id\n")

            try:
                with patch.object(
                    server, "_inspect_workspace_volume", return_value={}
                ), patch.object(
                    server, "_docker", side_effect=[inspect, remove]
                ):
                    server.tao_cleanup_results("container-id")
            finally:
                if locked.exists():
                    locked.chmod(0o700)

            self.assertFalse(results.exists())

    def test_cleanup_rejects_a_directory_swapped_after_launch(self):
        token = "3" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            server.WORKSPACE_ROOT = workspace
            results = workspace / "results" / ".tao-jobs" / token
            results.mkdir(parents=True)
            inspect = _completed(stdout=_inspect_payload(results, token))
            results.rename(results.with_name("preserved-original"))
            results.mkdir()

            with patch.object(
                server, "_inspect_workspace_volume", return_value={}
            ), patch.object(server, "_docker", return_value=inspect) as docker:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    server.tao_cleanup_results("container-id")

            self.assertTrue(results.exists())
            self.assertEqual(docker.call_count, 1)


if __name__ == "__main__":
    unittest.main()
