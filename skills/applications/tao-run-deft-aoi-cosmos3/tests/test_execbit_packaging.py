#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = SKILL_ROOT.parent
REPO_ROOT = SKILL_ROOT.parents[2]
DIRECT_CONTROL_SCRIPTS = {
    "deft_context.py",
    "deft_exec.py",
    "deft_python.sh",
    "finalize_run.py",
}
SCRIPT_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])scripts/(?P<name>[A-Za-z0-9_.-]+\.(?:py|sh))"
)


def _documented_direct_control_scripts() -> dict[pathlib.Path, set[pathlib.Path]]:
    """Find DEFT control programs that app docs instruct agents to execute."""
    found: dict[pathlib.Path, set[pathlib.Path]] = {}
    for skill_root in sorted(APPLICATIONS_ROOT.glob("tao-run-deft-*")):
        docs = [skill_root / "SKILL.md"]
        references = skill_root / "references"
        if references.is_dir():
            docs.extend(sorted(references.rglob("*.md")))
        for doc in docs:
            if not doc.is_file():
                continue
            names = {
                match.group("name")
                for match in SCRIPT_REFERENCE_RE.finditer(
                    doc.read_text(encoding="utf-8")
                )
                if match.group("name") in DIRECT_CONTROL_SCRIPTS
            }
            for name in names:
                found.setdefault(skill_root, set()).add(skill_root / "scripts" / name)
    return found


def _companion_source(category: str, name: str) -> pathlib.Path:
    candidates = (
        SKILL_ROOT.parent / name,
        REPO_ROOT / "skills" / category / name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"required companion skill {name!r} not found in: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


class ExecBitAndPackagingTests(unittest.TestCase):
    def test_documented_direct_control_scripts_are_executable(self) -> None:
        documented = _documented_direct_control_scripts()
        self.assertTrue(documented, "no directly invoked DEFT control scripts found")
        self.assertTrue(
            {
                "tao-run-deft-aoi",
                "tao-run-deft-aoi-cosmos3",
            }.issubset(skill.name for skill in documented),
            "both DEFT AOI variants must be covered",
        )

        missing = sorted(
            str(script)
            for scripts in documented.values()
            for script in scripts
            if not script.is_file()
        )
        self.assertFalse(missing, f"documented scripts do not exist: {missing}")

        not_executable = sorted(
            str(script)
            for scripts in documented.values()
            for script in scripts
            if not script.stat().st_mode & 0o111
        )
        self.assertFalse(
            not_executable,
            f"documented direct commands are not executable: {not_executable}",
        )

        if not (REPO_ROOT / ".git").exists():
            return
        relative = sorted(
            str(script.relative_to(REPO_ROOT))
            for scripts in documented.values()
            for script in scripts
        )
        result = subprocess.run(
            ["git", "ls-files", "-s", "--", *relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        recorded = {
            line.split("\t")[-1]: line.split()[0]
            for line in result.stdout.splitlines()
            if "\t" in line
        }
        stale = sorted(path for path in relative if recorded.get(path) != "100755")
        self.assertFalse(stale, f"executable bit is missing from git: {stale}")

    def test_cosmos3_suite_discovers_from_flattened_plugin_layout(self) -> None:
        sources = (
            ("applications", "tao-run-deft-aoi-cosmos3"),
            ("applications", "tao-run-deft-aoi"),
            ("models", "tao-finetune-cosmos-reason"),
            ("data", "tao-mine-aoi-images"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            skills_root = pathlib.Path(temporary) / ".claude" / "skills"
            skills_root.mkdir(parents=True)
            for category, name in sources:
                shutil.copytree(
                    _companion_source(category, name),
                    skills_root / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )

            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_cosmos3_*.py",
                ],
                cwd=skills_root / "tao-run-deft-aoi-cosmos3",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("OK", output)


if __name__ == "__main__":
    unittest.main()
