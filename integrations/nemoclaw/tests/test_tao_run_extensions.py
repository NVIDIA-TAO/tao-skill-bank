#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tao_run's workdir/env/mount arguments.

These cover the rejection paths specifically. The guards are the whole point of
the feature — a mount target that escapes, or a credential value placed on the
docker command line, is a security defect rather than a usability one — and a
happy-path test proves none of it.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docker_runtime import (  # noqa: E402
    HostIdentity,
    build_docker_run_args,
    validate_container_env,
    validate_container_path,
)


IDENTITY = HostIdentity(uid=1000, gid=1000, username="tao", supplementary_gids=())
TOKEN = "0" * 32
VOLUME = "tao-nemoclaw-workspace-abc123"


def _args(**overrides):
    kwargs = dict(
        image="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        command=["visual_changenet", "train"],
        workspace_volume=VOLUME,
        data_subpath="data",
        results_subpath="results",
        results_device=1,
        results_inode=2,
        gpus=1,
        shm_size="8g",
        identity=IDENTITY,
        job_token=TOKEN,
    )
    kwargs.update(overrides)
    return build_docker_run_args(**kwargs)


class ContainerPathValidation(unittest.TestCase):
    def test_rejects_relative_and_root_and_traversal(self):
        for path in ("relative/path", "", "/", "/a/../../etc"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_container_path(path, "mount target")

    def test_rejects_embedded_control_characters(self):
        for path in ("/a\x00b", "/a\nb"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_container_path(path, "mount target")

    def test_normalizes_a_valid_path(self):
        self.assertEqual(
            validate_container_path("/workspace/./pkg", "workdir"), "/workspace/pkg"
        )


class ContainerEnvValidation(unittest.TestCase):
    def test_rejects_credential_shaped_names(self):
        for name in ("AWS_SECRET_KEY", "MY_PASSWORD", "API_TOKEN", "HF_TOKEN", "NGC_KEY"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_container_env({name: "value"})

    def test_allows_names_that_merely_contain_a_secret_substring(self):
        # TOKENIZERS_PARALLELISM is a standard HuggingFace variable; a substring
        # test rejects it, and rejecting it breaks real workloads.
        for name in ("TOKENIZERS_PARALLELISM", "MONKEY_PATCH", "KEYSTONE_DIR"):
            with self.subTest(name=name):
                self.assertEqual(validate_container_env({name: "1"}), {name: "1"})

    def test_rejects_malformed_names_and_values(self):
        with self.assertRaises(ValueError):
            validate_container_env({"9BAD": "x"})
        with self.assertRaises(ValueError):
            validate_container_env({"OK": "line\nbreak"})

    def test_builder_validates_its_own_env(self):
        # Defence in depth: server.py validates too, but the builder must not
        # trust its caller — a secret here would land in `ps`.
        with self.assertRaises(ValueError):
            _args(extra_env={"AWS_SECRET_KEY": "hunter2"})


class ExtraMounts(unittest.TestCase):
    def test_reserved_targets_and_their_subpaths_are_refused(self):
        for target in ("/data", "/results", "/workspace", "/results/.tao-runtime",
                       "/data/dataset", "/workspace/anything"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    _args(extra_mounts=[{"subdir": "aug", "target": target}])

    def test_mount_is_emitted_with_readonly_flag(self):
        line = " ".join(
            _args(
                extra_mounts=[
                    {
                        "subdir": "augmentation/anomalygen/base_checkpoints",
                        "target": "/opt/checkpoints",
                        "ro": True,
                    }
                ]
            )
        )
        self.assertIn("dst=/opt/checkpoints", line)
        self.assertIn("volume-subpath=augmentation/anomalygen/base_checkpoints", line)
        self.assertIn("readonly", line)

    def test_workdir_is_emitted(self):
        self.assertIn("--workdir", _args(workdir="/workspace/paidf-anomalygen"))

    def test_managed_mounts_are_always_present(self):
        line = " ".join(_args())
        for destination in ("dst=/data", "dst=/results", "dst=/workspace"):
            self.assertIn(destination, line)


if __name__ == "__main__":
    unittest.main()
