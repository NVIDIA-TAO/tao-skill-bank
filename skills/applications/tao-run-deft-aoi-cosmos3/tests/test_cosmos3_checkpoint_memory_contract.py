# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from nvpaw_cfw import train  # noqa: E402


class CheckpointMemoryContractTests(unittest.TestCase):
    def test_checkpoint_hook_releases_unused_cuda_cache_before_save(self) -> None:
        events: list[str] = []

        class FakeCheckpointer:
            def save(self, *args, **kwargs):
                events.append("save")
                return "saved"

        fake_dcp = types.ModuleType("cosmos_framework.checkpoint.dcp")
        fake_dcp.DistributedCheckpointer = FakeCheckpointer
        fake_checkpoint = types.ModuleType("cosmos_framework.checkpoint")
        fake_checkpoint.dcp = fake_dcp
        fake_cosmos = types.ModuleType("cosmos_framework")
        fake_cosmos.checkpoint = fake_checkpoint
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))

        modules = {
            "cosmos_framework": fake_cosmos,
            "cosmos_framework.checkpoint": fake_checkpoint,
            "cosmos_framework.checkpoint.dcp": fake_dcp,
            "torch": fake_torch,
        }
        with mock.patch.dict(sys.modules, modules, clear=False), mock.patch(
            "gc.collect", side_effect=lambda: events.append("gc")
        ):
            train.install_checkpoint_memory_release_hook()
            result = FakeCheckpointer().save("model", iteration=40)

        self.assertEqual(result, "saved")
        self.assertEqual(events, ["gc", "empty_cache", "save"])

    def test_checkpoint_hook_is_idempotent(self) -> None:
        calls = 0

        class FakeCheckpointer:
            def save(self, *args, **kwargs):
                nonlocal calls
                calls += 1

        fake_dcp = types.ModuleType("cosmos_framework.checkpoint.dcp")
        fake_dcp.DistributedCheckpointer = FakeCheckpointer
        fake_checkpoint = types.ModuleType("cosmos_framework.checkpoint")
        fake_checkpoint.dcp = fake_dcp
        fake_cosmos = types.ModuleType("cosmos_framework")
        fake_cosmos.checkpoint = fake_checkpoint
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

        modules = {
            "cosmos_framework": fake_cosmos,
            "cosmos_framework.checkpoint": fake_checkpoint,
            "cosmos_framework.checkpoint.dcp": fake_dcp,
            "torch": fake_torch,
        }
        with mock.patch.dict(sys.modules, modules, clear=False):
            train.install_checkpoint_memory_release_hook()
            first = FakeCheckpointer.save
            train.install_checkpoint_memory_release_hook()
            self.assertIs(FakeCheckpointer.save, first)
            FakeCheckpointer().save()

        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
