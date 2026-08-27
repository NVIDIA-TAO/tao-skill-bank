# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep repository consumers on the canonical TAO Skill Bank name."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STALE_NAMES = ("tao-skills-bank", "tao-skills-external")
# PR #142 owns README's canonical checkout-name migration. Keep the exception
# explicit so this PR neither duplicates nor conflicts with that focused diff.
EXCEPTIONS = {Path("README.md")}


def test_consumers_use_canonical_tao_skill_bank_name():
    violations = []
    this_file = Path(__file__).resolve()
    for path in sorted(REPO_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.resolve() == this_file
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        relative = path.relative_to(REPO_ROOT)
        if relative in EXCEPTIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for stale in STALE_NAMES:
            if stale in text:
                violations.append(f"{relative}: contains {stale!r}")

    assert not violations, (
        "repository consumers must use the canonical 'tao-skill-bank' name:\n"
        + "\n".join(violations)
    )
