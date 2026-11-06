# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 1.2 paths must be basenames and resolve in the producer namespace."""
import pathlib
import sys
import tempfile
import unittest
SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
import emit_sdg_sharegpt
from anomalygen_consumer import load_generation
sys.path.insert(0, str(pathlib.Path(load_generation.__code__.co_filename).parent / "tests"))
from native_fixture import write_native

class NativePathTests(unittest.TestCase):
    def test_native_basename_uses_paired_original_not_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = write_native(root, ["PCB+bridge_00000.png"])
            records, _ = emit_sdg_sharegpt.emit_native(manifest, media_root=root, prompt="Inspect", relative=True)
            self.assertEqual(records[0]["images"], ["sdg/reconstructed_image/PCB+bridge_00000.png", "sdg/original_image/PCB+bridge_00000.png"])
    def test_missing_pair_is_not_replaced_by_original_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = write_native(root, ["PCB+bridge_00000.png"])
            (root / "sdg/original_image/PCB+bridge_00000.png").unlink()
            with self.assertRaisesRegex(ValueError, "paired"):
                emit_sdg_sharegpt.emit_native(manifest, media_root=root, prompt="Inspect", relative=False)
