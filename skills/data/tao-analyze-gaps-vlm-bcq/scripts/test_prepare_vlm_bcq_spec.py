#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for VLM BCQ spec preparation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from prepare_vlm_bcq_spec import prepare_spec


def main() -> int:
    """Run spec preparation tests without pytest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        predictions = root / "results.json"
        videos = root / "videos"
        results = root / "gaps"
        output = root / "specs" / "vlm_bcq_spec.yaml"
        predictions.write_text("[]\n", encoding="utf-8")
        videos.mkdir()

        prepare_spec(predictions, videos, results, output)
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert payload == {
            "predictions_json": str(predictions),
            "videos_dir": str(videos),
            "results_dir": str(results),
        }

        no_media_output = root / "specs" / "vlm_bcq_no_media.yaml"
        prepare_spec(predictions, None, results, no_media_output)
        assert yaml.safe_load(no_media_output.read_text(encoding="utf-8"))["videos_dir"] == ""

    print("test_prepare_vlm_bcq_spec.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
