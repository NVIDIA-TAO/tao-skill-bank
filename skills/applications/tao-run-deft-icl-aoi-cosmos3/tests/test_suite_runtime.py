# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constructor-only runtime fixtures; no Docker, model or data access."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from suite_runtime import Docker


class ImmutableImage(unittest.TestCase):
    def test_accepts_exact_local_image_id_and_registry_digest(self):
        digest = '16cae1489990a6a56232368d0ff62ac6a7086610a6ade749a1ed9293325409b6'
        with tempfile.TemporaryDirectory() as tmp, patch('suite_runtime.subprocess.run', side_effect=AssertionError('constructor must not launch')), patch('suite_runtime.subprocess.check_output', side_effect=AssertionError('constructor must not launch')):
            for image in ('sha256:'+digest, 'registry.example/repository@sha256:'+digest):
                with self.subTest(image=image):
                    runtime = Docker(tmp, image)
                    self.assertEqual(runtime.image, image)
                    self.assertEqual(runtime.root, Path(tmp).resolve())

    def test_rejects_mutable_tags_and_malformed_local_ids(self):
        invalid = (
            'wsp03-efficient:vllm012-hf4576-headers', 'ubuntu:latest', 'ubuntu',
            'sha256:', 'sha256:'+'a'*63, 'sha256:'+'a'*65,
            'sha256:'+'A'*64, 'sha256:'+'g'*64, 'sha256:'+'a'*64+'\n',
        )
        with tempfile.TemporaryDirectory() as tmp:
            for image in invalid:
                with self.subTest(image=image), self.assertRaisesRegex(ValueError, 'resolved immutable image required'):
                    Docker(tmp, image)


if __name__ == '__main__':
    unittest.main()
