# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep non-README consumers on the canonical public repository URL."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "integrations/nemoclaw/setup-tao-nemoclaw.sh",
    "skills/core/tao-launch-workflow/skill-card.md",
    "skills/data/tao-convert-dataset-format/skill-card.md",
    "skills/data/tao-route-visual-changenet-samples/skill-card.md",
)


def test_consumers_use_canonical_tao_skill_bank_repository_url():
    for relative in FILES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "NVIDIA-TAO/tao-skills-bank" not in text, relative
        assert "https://github.com/NVIDIA-TAO/tao-skill-bank" in text, relative
