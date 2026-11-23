# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills ship no value that only one site can use as a default.

A dataset, a registry image, a model endpoint, or a cluster's queue names belong
to the user's site. A skill names the input, says what it must contain, and
asks; it never fills in a value that an outside user cannot reach.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_plugin_package as builder  # noqa: E402

DINO = ROOT / "skills/models/tao-train-dino"
AUTOML_INTENT = ROOT / "skills/applications/tao-run-automl/references/automl-intent-algorithms.md"

# An s3:// URI in skill text is an example only when its bucket is a placeholder.
S3_URI = re.compile(r"s3://[^\s\"'`)|,]*")
PLACEHOLDER_BUCKET = re.compile(r"s3://(?:bucket(?:/|$)|<[^>]+>|\{[^}]+\}|\$\{?\w+\}?|\.\.\.|$)")


def shipped_texts(skill_dir: Path):
    """(path, text) of every text file of a skill that the plugin package ships."""
    for path in sorted(skill_dir.rglob("*")):
        inside = path.relative_to(skill_dir).as_posix()
        if not path.is_file() or builder.left_out(skill_dir.name, inside):
            continue
        text = builder.as_text(path.read_bytes())
        if text is not None:
            yield path, text


def test_dino_offers_no_default_dataset():
    for path, text in shipped_texts(DINO):
        concrete = [uri for uri in S3_URI.findall(text) if not PLACEHOLDER_BUCKET.match(uri)]
        assert not concrete, f"{path} names a bucket instead of a placeholder: {concrete}"
        assert not re.search(r"\b\w+_PROFILE\b", text), f"{path} still names a packaged profile"
        assert "default profile" not in text, path
    for name in ("SKILL.md", "references/dino-data-specs.md"):
        text = (DINO / name).read_text(encoding="utf-8")
        assert "ships no default dataset" in text, name
        assert "look in the\nworking directory for a COCO-format dataset" in text, name
        assert "S3 path" not in text, name
    # tao-run-automl asks the model skill for a dataset; it must not expect a default there.
    assert "default profile" not in AUTOML_INTENT.read_text(encoding="utf-8")


def test_dino_tuning_note_agrees_with_the_shipped_schema():
    schema = json.loads((DINO / "schemas/train.schema.json").read_text(encoding="utf-8"))
    assert "train.optim.weight_decay" in schema["automl_default_parameters"]
    note = (DINO / "references/dino-tuning-multigpu.md").read_text(encoding="utf-8")
    assert "`train.optim.weight_decay`" in note          # still a suggested knob
    assert not re.search(r"weight_decay`? is not in", note)
    assert "`weight_decay`\nbehavior" not in (DINO / "SKILL.md").read_text(encoding="utf-8")
