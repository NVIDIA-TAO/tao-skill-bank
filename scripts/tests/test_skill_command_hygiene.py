# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-hygiene lints for skill content (runs in CI via `pytest scripts/tests`).

Guards two defect classes found in the 7.1.0 platform-skill audit:

1. ``secret-on-argv`` — credentials must never travel as command-line
   arguments: argv is visible in the process table, in the SSH command string
   wrappers like ``brev exec`` forward, in shell history, and in CI/agent
   transcripts. ``docker login`` must use ``--password-stdin``;
   ``kubectl create secret docker-registry --docker-password=...`` is banned
   in favor of a dockerconfigjson secret fed over stdin; ``-e VAR=$VAR`` for
   secret vars is banned in favor of the pass-through form ``-e VAR``.

2. ``brev-exec-form`` — ``brev exec <instance> -- <cmd> <args...>`` silently
   misparses on brev CLI 0.6.x: ``--`` only ends flag parsing, and every
   positional except the last is treated as an instance name, so multi-token
   commands fail with ``could not look up instance "<word>"``. Remote commands
   must be a single quoted string: ``brev exec <instance> "<command>"``.

Scope: in markdown, only lines inside shell fences (```bash / ```sh / …) are
enforced — those are the commands a user copies. Prose that *describes* a banned
pattern (an error-pattern section explaining what not to do) is documentation and
is exempt automatically, with no marker needed. Every non-markdown file is
enforced in full.

That split is deliberate. The first version of this lint policed prose too, so
every teaching sentence needed a ``lint-ok`` marker; those markers were then
copy-pasted onto live commands during a merge, silencing a real defect (three of
the four Brev verbs shipped broken behind ``lint-ok: brev-exec-form``). A marker
inside a shell fence now means someone suppressed a genuine finding — treat it
as a bug, not an annotation.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("skills", "templates", "scripts", "integrations")
SCAN_SUFFIXES = {".md", ".yaml", ".yml", ".sh", ".py", ".config", ".json", ".txt"}

SECRET_VARS = "NGC_KEY|NGC_API_KEY|HF_TOKEN|BREV_API_TOKEN|WANDB_API_KEY|ACCESS_KEY|SECRET_KEY|AWS_SECRET_ACCESS_KEY"

RULES = (
    ("secret-on-argv",
     re.compile(r"docker login\b[^\n]*\s-p\s+\S"),
     "docker login -p puts the secret in argv; pipe it to --password-stdin"),
    ("secret-on-argv",
     re.compile(r"docker login\b[^\n]*--password[= ]"),
     "docker login --password puts the secret in argv; pipe it to --password-stdin"),
    ("secret-on-argv",
     re.compile(r"--docker-password[= ]"),
     "kubectl --docker-password puts the secret in argv; create a "
     "kubernetes.io/dockerconfigjson secret from stdin instead"),
    ("secret-on-argv",
     re.compile(r"-e\s+(?:%s)=\$" % SECRET_VARS),
     "docker -e VAR=$VAR puts the secret in argv; use the pass-through form -e VAR"),
    ("brev-exec-form",
     re.compile(r"brev exec\s+\S+\s+--\s"),
     'brev exec must take the remote command as ONE quoted string: '
     'brev exec <instance> "<command>" (the -- multi-token form misparses '
     "every token but the last as an instance name)"),
)


def _iter_files():
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (path.is_file() and path.suffix in SCAN_SUFFIXES
                    and "__pycache__" not in path.parts):
                yield path


SHELL_FENCE_RE = re.compile(r"^\s*```+\s*(bash|sh|shell|console|zsh)\b", re.I)
FENCE_CLOSE_RE = re.compile(r"^\s*```+\s*$")


def _executable_lines(path, text):
    """Yield (lineno, line) for lines a user would actually run.

    Markdown prose that merely *describes* a banned pattern (usually inside
    backticks, in an error-pattern or "don't do this" section) is documentation,
    not a command — policing it forced a ``lint-ok`` marker onto every teaching
    sentence, and those markers then got copy-pasted onto genuinely broken
    commands, which is how the brev-exec regression shipped. So in markdown only
    shell fences are enforced; every other file type is enforced whole.
    """
    if path.suffix != ".md":
        for lineno, line in enumerate(text.splitlines(), 1):
            yield lineno, line
        return
    in_shell_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if in_shell_fence:
            if FENCE_CLOSE_RE.match(line) or line.strip().startswith("```"):
                in_shell_fence = False
                continue
            yield lineno, line
        elif SHELL_FENCE_RE.match(line):
            in_shell_fence = True


def test_no_banned_command_patterns():
    violations = []
    for path in _iter_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in _executable_lines(path, text):
            for rule_id, pattern, why in RULES:
                if pattern.search(line) and f"lint-ok: {rule_id}" not in line:
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(
                        f"{rel}:{lineno}: [{rule_id}] {why}\n    {line.strip()}")
    assert not violations, (
        "command-hygiene violations in runnable code. These are commands a user "
        "would copy — fix them rather than suppressing. A 'lint-ok: <rule-id>' "
        "marker on a line inside a shell fence silences a REAL defect; prose "
        "outside shell fences is already exempt and needs no marker:\n"
        + "\n".join(violations))
