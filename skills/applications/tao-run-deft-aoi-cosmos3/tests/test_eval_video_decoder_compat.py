#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import patch_eval_video_decoder  # noqa: E402


UNCONDITIONAL_LOAD_MODEL = '''\
class BaseEvaluator:
    def load_model(self):
        log.info("Loading model and processor...")
        start_time = time.time()

''' + patch_eval_video_decoder._OLD_BLOCK + '''
        model = CosmosFrameworkRuntime(
            self.model_config.get("model_name"),
        )
        return model
'''

FRAMEWORK_EVALUATOR = """\
from cosmos_rl.framework import checkpoints
from cosmos_rl.framework import runtime

def evaluate(config):
    return runtime.evaluate(config)
"""


class EvalVideoDecoderCompatibilityTests(unittest.TestCase):
    def test_unconditional_block_is_patch_required(self) -> None:
        with mock.patch.object(
            patch_eval_video_decoder,
            "read_from_image",
            return_value=UNCONDITIONAL_LOAD_MODEL,
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = patch_eval_video_decoder.main(
                ["--image", "nvcr.io/nvstaging/tao/cosmos-rl:test", "--probe"]
            )
        self.assertEqual(result, 0)
        self.assertIn("classification=patch_required", stdout.getvalue())
        self.assertIn("patch_needed=True", stdout.getvalue())

    def test_patch_rewrites_block_and_is_syntactically_valid(self) -> None:
        patched = patch_eval_video_decoder.apply_patch(UNCONDITIONAL_LOAD_MODEL)
        self.assertNotIn(patch_eval_video_decoder._OLD_BLOCK, patched)
        self.assertIn("TAO_SKIP_PYNV_VIDEO_DECODER", patched)
        self.assertIn("register_pynv_video_reader", patched)
        # Original block content is preserved inside the else-branch, so a
        # video-carrying workload without the opt-in env var still registers
        # the GPU decoder exactly as before.
        ast.parse(patched)

    def test_patch_writes_mount_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output_dir = root / "patch"
            summary = root / "summary.json"
            with mock.patch.object(
                patch_eval_video_decoder,
                "read_from_image",
                return_value=UNCONDITIONAL_LOAD_MODEL,
            ), contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = patch_eval_video_decoder.main(
                    [
                        "--image",
                        "nvcr.io/nvstaging/tao/cosmos-rl:test",
                        "--output-dir",
                        str(output_dir),
                        "--summary",
                        str(summary),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "base.py").exists())
            self.assertTrue((output_dir / "base.py.orig").exists())
            payload = json.loads(summary.read_text())
            self.assertEqual(payload["classification"], "patch_required")
            self.assertTrue(payload["patch_needed"])
            self.assertEqual(
                payload["mount_argument"],
                f"{(output_dir / 'base.py').resolve()}:"
                f"{patch_eval_video_decoder.CONTAINER_PATH}:ro",
            )
            self.assertIn("MOUNT_ARG=", stdout.getvalue())

    def test_already_patched_source_is_already_sufficient(self) -> None:
        patched = patch_eval_video_decoder.apply_patch(UNCONDITIONAL_LOAD_MODEL)
        with mock.patch.object(
            patch_eval_video_decoder, "read_from_image", return_value=patched
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = patch_eval_video_decoder.main(
                ["--image", "nvcr.io/nvstaging/tao/cosmos-rl:test", "--probe"]
            )
        self.assertEqual(result, 0)
        self.assertIn("classification=already_sufficient", stdout.getvalue())
        self.assertIn("patch_needed=False", stdout.getvalue())

    def test_source_without_decoder_registration_is_pattern_absent(self) -> None:
        with mock.patch.object(
            patch_eval_video_decoder,
            "read_from_image",
            return_value=FRAMEWORK_EVALUATOR,
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = patch_eval_video_decoder.main(
                ["--image", "nvcr.io/nvstaging/tao/cosmos-rl:test", "--probe"]
            )
        self.assertEqual(result, 0)
        self.assertIn("classification=pattern_absent", stdout.getvalue())
        self.assertIn("patch_needed=False", stdout.getvalue())

    def test_changed_decoder_shape_fails_with_unknown_verdict(self) -> None:
        changed = """\
        decoder_name = self.vision_config.get("video_decoder", "pynvvideocodec")
        from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader
        decoder = register_pynv_video_reader(strict=True, some_new_kwarg=True)
"""
        with self.assertRaisesRegex(
            ValueError,
            "classification=unknown.*verify the new source shape",
        ):
            patch_eval_video_decoder.classify(changed)

    def test_absent_output_dir_without_probe_errors(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = patch_eval_video_decoder.main(
                ["--image", "nvcr.io/nvstaging/tao/cosmos-rl:test"]
            )
        self.assertEqual(result, 2)
        self.assertIn("--output-dir is required", stderr.getvalue())

    def test_patches_real_captured_base_py_cleanly(self) -> None:
        # Captured verbatim via `docker run --rm --entrypoint cat <image>
        # /workspace/cosmos_rl_merged/cosmos_rl/evaluation/base.py` against
        # nvcr.io/nvstaging/tao/cosmos-rl:enhanced-hooks-custom-loggers-
        # 20260816-direct-cache-fast-v7 (the image this run resolved), so
        # this test tracks the real shape rather than a synthetic fixture.
        fixture = (
            SKILL_ROOT
            / "tests"
            / "fixtures_cosmos_rl_evaluation_base.py.txt"
        ).read_text()
        classification = patch_eval_video_decoder.classify(fixture)
        self.assertEqual(classification, patch_eval_video_decoder.PATCH_REQUIRED)
        patched = patch_eval_video_decoder.apply_patch(fixture)
        ast.parse(patched)
        self.assertIn("TAO_SKIP_PYNV_VIDEO_DECODER", patched)
        self.assertNotIn(patch_eval_video_decoder._OLD_BLOCK, patched)
        # Re-classifying the patched output must report already_sufficient,
        # not patch_required again (idempotency against a re-run).
        self.assertEqual(
            patch_eval_video_decoder.classify(patched),
            patch_eval_video_decoder.ALREADY_SUFFICIENT,
        )


if __name__ == "__main__":
    unittest.main()
