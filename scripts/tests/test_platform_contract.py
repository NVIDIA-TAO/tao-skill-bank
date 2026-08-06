# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform contract tests for the SDK-free platform skills.

Every regression these guard against shipped in the 7.1.0 -> main merge and was
invisible to CI, because nothing cross-checked a skill's prose against the
scripts, templates, and metadata it describes:

  * a preflight referencing a path that does not exist (the k8s SETUP_SCRIPT
    dropped its ``skills/`` segment, so preflight always failed "file not found")
  * a platform silently losing job-record integration, so its jobs became
    untrackable while every sibling platform stayed consistent (Brev)
  * frontmatter advertising narrower capability than the body implements, so a
    router never selects the platform for the case it actually supports (k8s
    said "single-pod" while documenting Indexed-Job multi-node)

These are static: no GPU, no cluster, no credentials. They run on every MR.
Live execution smokes belong in the nightly platform pipeline instead.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO / "skills/platform"

# Platforms that implement the four-verb execution contract. tao-data-io (S3
# layer) and tao-setup-nvidia-gpu-host (host preflight) launch nothing by design.
RUN_PLATFORMS = [
    "tao-run-on-docker",
    "tao-run-on-kubernetes",
    "tao-run-on-slurm",
    "tao-run-on-brev",
    "tao-run-on-virtualenv",
]

VERB_PATTERNS = {
    "submit": re.compile(r"\bsubmit\b", re.I),
    "status": re.compile(r"\bstatus\b", re.I),
    "logs": re.compile(r"\blogs\b", re.I),
    "cancel": re.compile(r"\bcancel\b|\bteardown\b", re.I),
}


def _skill_text(name: str) -> str:
    return (PLATFORM_DIR / name / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
@pytest.mark.parametrize("verb", sorted(VERB_PATTERNS))
def test_platform_documents_every_verb(platform, verb):
    """Each run platform must document all four verbs of the execution contract."""
    assert VERB_PATTERNS[verb].search(_skill_text(platform)), (
        f"{platform}/SKILL.md never mentions the '{verb}' verb; the four-verb "
        f"contract requires submit/status/logs/cancel on every run platform")


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
def test_platform_wires_the_job_record(platform):
    """Every run platform must mint and update a job record, not just describe one.

    Brev regressed to prose ("open the record") with no command, which makes its
    jobs untrackable across invocations — the capability the SDK's Job handles
    used to provide.
    """
    text = _skill_text(platform)
    assert "tao_job_record.py" in text, (
        f"{platform}/SKILL.md never invokes scripts/tao_job_record.py — jobs "
        f"launched by this platform cannot be tracked across invocations")
    assert re.search(r"tao_job_record\.py[\"']?\s+open", text), (
        f"{platform}/SKILL.md never calls `tao_job_record.py open` to mint a job id")
    assert re.search(r"tao_job_record\.py[\"']?\s+mark", text), (
        f"{platform}/SKILL.md never calls `tao_job_record.py mark` to update state")


# Shell-assigned paths that point into the bank, e.g.
#   SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/skills/platform/.../setup.sh"
BANK_PATH_RE = re.compile(
    r"\$\{(?:TAO_SKILL_BANK_PATH|TAO_SKILL_BANK_ROOT|SB|BANK)(?::-[^}]*)?\}"
    r"(/[A-Za-z0-9_./-]+\.(?:sh|py|tmpl|yaml|json|md))"
)


@pytest.mark.parametrize(
    "skill_md",
    sorted(p for p in PLATFORM_DIR.rglob("*.md") if "__pycache__" not in p.parts),
    ids=lambda p: str(p.relative_to(PLATFORM_DIR)),
)
def test_bank_relative_paths_resolve(skill_md):
    """A `$BANK/...`-anchored path in a platform skill must exist in the repo.

    Catches the dropped-path-segment class directly: the k8s preflight pointed at
    `${TAO_SKILL_BANK_ROOT}/platform/...` (missing `skills/`) and could never run.
    """
    missing = []
    for rel in BANK_PATH_RE.findall(skill_md.read_text(encoding="utf-8")):
        target = REPO / rel.lstrip("/")
        if not target.exists():
            missing.append(rel)
    assert not missing, (
        f"{skill_md.relative_to(REPO)} references bank paths that do not exist: "
        f"{sorted(set(missing))}")


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
def test_multinode_capability_matches_frontmatter(platform):
    """If a skill documents multi-node support, its description must not deny it.

    Routers select platforms from the frontmatter description; k8s shipped
    advertising "single-pod" while implementing Indexed-Job multi-node, so it was
    never selected for the case it supports.
    """
    text = _skill_text(platform)
    body = text.split("---", 2)[-1]
    frontmatter = text.split("---", 2)[1] if text.startswith("---") else ""
    desc = " ".join(
        line.strip() for line in frontmatter.splitlines()
        if not re.match(r"^\s*(name|license|metadata|tags|allowed-tools|compatibility|version):", line)
    ).lower()

    claims_multinode = bool(
        re.search(r"multi-node is (?:not )?supported|indexed job|num_nodes\s*[=:]\s*[2-9]", body, re.I)
        and not re.search(r"\*\*multi-node is not supported\*\*", body, re.I)
    )
    # A description is a denial only if it scopes itself to one node AND never
    # mentions multi-node. "single-pod for one node, Indexed Jobs for multi-node"
    # advertises both and is correct.
    denies_in_desc = bool(
        re.search(r"single-pod|single-node only", desc)
        and not re.search(r"multi[- ]node", desc)
    )
    assert not (claims_multinode and denies_in_desc), (
        f"{platform}: the body documents multi-node support but the frontmatter "
        f"description says single-pod/single-node — routers read the description, "
        f"so this platform will never be selected for multi-node work")


# Required flags of `tao_job_record.py open`, per its argparse definition. A
# documented invocation missing any of these fails at runtime with exit 2 —
# which is exactly how this test was born: a hand-written Brev example omitted
# --network-arch/--action/--storage-tier and looked entirely plausible.
JOB_RECORD_OPEN_REQUIRED = ("--platform", "--image", "--network-arch",
                            "--action", "--storage-tier")

# `\\\n` must precede `[^\n]` in the alternation: otherwise `[^\n]` consumes the
# backslash and the continuation branch can never match, truncating the capture
# at the first line of a multi-line invocation.
OPEN_INVOCATION_RE = re.compile(
    r"tao_job_record\.py[\"']?\s+open\b((?:\\\n|[^\n])*)", re.M)


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
def test_documented_job_record_open_has_required_flags(platform):
    """Every documented `tao_job_record.py open` must carry all required flags.

    Guards docs against the script's real argparse signature, so a plausible but
    incomplete example cannot ship.
    """
    text = _skill_text(platform)
    invocations = OPEN_INVOCATION_RE.findall(text)
    assert invocations, f"{platform}/SKILL.md documents no `tao_job_record.py open` call"
    for args in invocations:
        flat = args.replace("\\\n", " ")
        missing = [f for f in JOB_RECORD_OPEN_REQUIRED if f not in flat]
        assert not missing, (
            f"{platform}/SKILL.md: `tao_job_record.py open` example is missing "
            f"required flag(s) {missing} — it would exit 2 at runtime")


def test_job_record_required_flags_match_the_script():
    """Keep the list above honest against tao_job_record.py itself."""
    src = (REPO / "scripts/tao_job_record.py").read_text(encoding="utf-8")
    open_block = src[src.index('"open"'):]
    for flag in JOB_RECORD_OPEN_REQUIRED:
        assert f'"{flag}"' in open_block, (
            f"{flag} is listed as required here but no longer appears in "
            f"tao_job_record.py's `open` parser — update JOB_RECORD_OPEN_REQUIRED")


# Scripts the platform skills invoke DIRECTLY (as `"$BANK/scripts/x.py" ...`
# rather than `python3 .../x.py`) must carry the executable bit, or every
# documented submit sequence dies at step one with "permission denied".
DIRECT_EXEC_RE = re.compile(
    r'"?\$\{?(?:BANK|TAO_SKILL_BANK_PATH|TAO_SKILL_BANK_ROOT)\}?/(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))"?\s')


def test_directly_invoked_scripts_are_executable():
    """Any script a platform skill execs directly must be mode 755 in git.

    This shipped broken: tao_job_record.py and redact_secrets.py were added as
    100644 while every other script in scripts/ was 100755, so the first line of
    `submit` failed with "permission denied" on all five platforms. The git mode
    is what matters — a local chmod does not travel, so assert on the index.
    """
    referenced = set()
    for skill_md in PLATFORM_DIR.rglob("*.md"):
        if "__pycache__" in skill_md.parts:
            continue
        referenced.update(DIRECT_EXEC_RE.findall(skill_md.read_text(encoding="utf-8")))
    assert referenced, "no directly-invoked bank scripts found — has the invocation style changed?"

    modes = subprocess.run(
        ["git", "ls-files", "-s", *sorted(referenced)],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    recorded = {line.split("\t")[-1]: line.split()[0] for line in modes.splitlines() if line}

    not_executable = sorted(p for p, m in recorded.items() if m != "100755")
    missing = sorted(referenced - set(recorded))
    assert not missing, f"platform skills reference scripts absent from git: {missing}"
    assert not not_executable, (
        f"referenced directly by a platform skill but not executable in git "
        f"(chmod +x and commit the mode): {not_executable}")
