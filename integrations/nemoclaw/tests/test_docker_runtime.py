#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stdlib-only NemoClaw Docker launch helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docker_runtime import (  # noqa: E402
    CONTAINER_HOME,
    HostIdentity,
    JOB_NAME_PREFIX,
    JOBS_DIRNAME,
    MANAGED_LABEL,
    RESULTS_DEVICE_LABEL,
    RESULTS_INODE_LABEL,
    RESULTS_PATH_LABEL,
    RESULTS_TOKEN_LABEL,
    WORKSPACE_VOLUME_LABEL,
    build_docker_run_args,
    current_host_identity,
    ensure_workspace_directory,
    prepare_isolated_results,
    prepare_runtime_home,
    validate_managed_results_source,
)


class RuntimeHomeTests(unittest.TestCase):
    def test_prepares_writable_home_and_caches_below_results(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results = workspace / "experiment" / "results"
            results.mkdir(parents=True)

            home = prepare_runtime_home(results, workspace, identity)

            self.assertEqual(home, results / ".tao-runtime" / "home")
            self.assertTrue(os.access(home, os.W_OK | os.X_OK))
            self.assertEqual(home.stat().st_uid, os.getuid())
            for relative in (
                ".cache/huggingface",
                ".cache/torch",
                ".cache/triton",
                ".cache/torchinductor",
                ".cache/matplotlib",
            ):
                self.assertTrue((home / relative).is_dir())

    def test_rejects_runtime_symlink_that_escapes_results(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            workspace = Path(tmp)
            results = workspace / "results"
            results.mkdir()
            (results / ".tao-runtime").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                prepare_runtime_home(results, workspace, identity)

    def test_workspace_directory_walk_refuses_an_escaping_symlink(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            workspace = Path(tmp)
            (workspace / "swapped").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                ensure_workspace_directory(workspace / "swapped", workspace)


class DockerRunArgsTests(unittest.TestCase):
    def test_maps_host_identity_and_redirects_home_and_caches(self):
        identity = HostIdentity(
            uid=1234,
            gid=5678,
            username="data-scientist",
            supplementary_gids=(99, 100),
        )

        args = build_docker_run_args(
            image="nvcr.io/nvidia/tao/example:1",
            command=["train", "--spec", "/data/spec.yaml"],
            workspace_volume="tao-nemoclaw-workspace-test",
            data_subpath="data",
            results_subpath=f"results/{JOBS_DIRNAME}/{'a' * 32}",
            results_device=41,
            results_inode=42,
            gpus=2,
            shm_size="16g",
            identity=identity,
            job_token="a" * 32,
        )

        self.assertEqual(args[:2], ["run", "-d"])
        user_index = args.index("--user")
        self.assertEqual(args[user_index + 1], "1234:5678")
        group_indexes = [i for i, value in enumerate(args) if value == "--group-add"]
        self.assertEqual([args[i + 1] for i in group_indexes], ["99", "100"])
        self.assertIn(f"HOME={CONTAINER_HOME}", args)
        self.assertIn("USER=data-scientist", args)
        self.assertIn("LOGNAME=data-scientist", args)
        self.assertIn(f"XDG_CACHE_HOME={CONTAINER_HOME}/.cache", args)
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
        self.assertEqual(
            mounts,
            [
                "type=volume,src=tao-nemoclaw-workspace-test,dst=/data,"
                "volume-subpath=data",
                "type=volume,src=tao-nemoclaw-workspace-test,dst=/results,"
                f"volume-subpath=results/{JOBS_DIRNAME}/{'a' * 32}",
            ],
        )
        self.assertFalse(any("/host/workspace" in arg for arg in args))
        self.assertEqual(args[args.index("--pull") + 1], "never")
        self.assertEqual(args[args.index("--restart") + 1], "no")
        self.assertEqual(
            args[args.index("--security-opt") + 1], "no-new-privileges:true"
        )
        self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
        self.assertEqual(
            args[-4:],
            [
                "nvcr.io/nvidia/tao/example:1",
                "train",
                "--spec",
                "/data/spec.yaml",
            ],
        )

    def test_labels_and_names_an_isolated_managed_job(self):
        token = "a" * 32
        args = build_docker_run_args(
            image="nvcr.io/nvidia/tao/example:1",
            command=["train"],
            workspace_volume="tao-nemoclaw-workspace-test",
            data_subpath="data",
            results_subpath=f"results/{JOBS_DIRNAME}/{token}",
            results_device=41,
            results_inode=42,
            gpus=1,
            shm_size="8g",
            identity=HostIdentity(1234, 5678, "data-scientist"),
            job_token=token,
        )

        self.assertEqual(args[args.index("--name") + 1], f"{JOB_NAME_PREFIX}{token}")
        labels = [args[i + 1] for i, value in enumerate(args) if value == "--label"]
        self.assertEqual(
            labels,
            [
                f"{MANAGED_LABEL}=true",
                f"{RESULTS_TOKEN_LABEL}={token}",
                f"{WORKSPACE_VOLUME_LABEL}=tao-nemoclaw-workspace-test",
                f"{RESULTS_PATH_LABEL}=results/{JOBS_DIRNAME}/{token}",
                f"{RESULTS_DEVICE_LABEL}=41",
                f"{RESULTS_INODE_LABEL}=42",
            ],
        )

    def test_rejects_invalid_managed_job_token(self):
        with self.assertRaisesRegex(ValueError, "invalid managed TAO job token"):
            build_docker_run_args(
                image="nvcr.io/nvidia/tao/example:1",
                command=["train"],
                workspace_volume="tao-nemoclaw-workspace-test",
                data_subpath="data",
                results_subpath=f"results/{JOBS_DIRNAME}/{'a' * 32}",
                results_device=41,
                results_inode=42,
                gpus=1,
                shm_size="8g",
                identity=HostIdentity(1234, 5678, "data-scientist"),
                job_token="../../not-a-token",
            )

    def test_identity_fallback_is_nonempty_when_uid_has_no_passwd_entry(self):
        with patch("docker_runtime.os.getuid", return_value=4242), patch(
            "docker_runtime.os.getgid", return_value=4343
        ), patch(
            "docker_runtime.os.getgroups", return_value=[4343, 99, 100, 99]
        ), patch("docker_runtime.pwd.getpwuid", side_effect=KeyError):
            identity = current_host_identity()

        self.assertEqual(
            identity,
            HostIdentity(4242, 4343, "tao-4242", supplementary_gids=(99, 100)),
        )

    def test_does_not_pass_the_docker_socket_group_to_workloads(self):
        socket_metadata = types.SimpleNamespace(st_mode=0o140660, st_gid=987)
        with patch("docker_runtime.os.getuid", return_value=4242), patch(
            "docker_runtime.os.getgid", return_value=4343
        ), patch(
            "docker_runtime.os.getgroups", return_value=[4343, 99, 987]
        ), patch(
            "docker_runtime.os.stat", return_value=socket_metadata
        ), patch("docker_runtime.pwd.getpwuid") as getpwuid:
            getpwuid.return_value.pw_name = "host-user"
            identity = current_host_identity()

        self.assertEqual(identity.supplementary_gids, (99,))


class IsolatedResultsTests(unittest.TestCase):
    def test_prepares_and_validates_one_deletable_tree_per_job(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        token = "b" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            base = workspace / "results"

            prepared = prepare_isolated_results(base, workspace, identity, token)
            results = prepared.path

            self.assertEqual(results, base / JOBS_DIRNAME / token)
            self.assertTrue((results / ".tao-runtime" / "home").is_dir())
            self.assertEqual(
                validate_managed_results_source(
                    results,
                    workspace,
                    token,
                    expected_device=prepared.device,
                    expected_inode=prepared.inode,
                ),
                results,
            )

    def test_refuses_token_mismatch_before_cleanup(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results = prepare_isolated_results(
                workspace / "results", workspace, identity, "c" * 32
            ).path

            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_managed_results_source(results, workspace, "d" * 32)

    def test_refuses_a_different_directory_swapped_into_the_managed_path(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        token = "e" * 32
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results = prepare_isolated_results(
                workspace / "results", workspace, identity, token
            ).path
            original = results.stat()
            moved = results.with_name("saved-original")
            results.rename(moved)
            results.mkdir()

            with self.assertRaisesRegex(ValueError, "identity changed"):
                validate_managed_results_source(
                    results,
                    workspace,
                    token,
                    expected_device=original.st_dev,
                    expected_inode=original.st_ino,
                )


if __name__ == "__main__":
    unittest.main()
