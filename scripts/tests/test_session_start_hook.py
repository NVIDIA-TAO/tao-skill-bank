# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the plugin's SessionStart hook."""

import atexit
import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "hooks" / "session_start.sh"
HOOKS_JSON = REPO_ROOT / "hooks" / "hooks.json"
# Claude Code replaces hook output longer than this with a short preview.
HOOK_OUTPUT_CAP = 10_000


# Everything the hook runs, apart from shell builtins.
HOOK_TOOLS = ("bash", "awk", "basename", "cat", "dirname", "grep", "head", "tail", "wc")


@functools.lru_cache(maxsize=None)
def _hook_path():
    """A PATH that holds only the tools the hook needs.

    Docker is left out on purpose. Without it the hook adds its setup hint,
    which is the longest output it can print, wherever the tests run.
    """
    bin_dir = Path(tempfile.mkdtemp(prefix="tao-hook-tools-"))
    atexit.register(shutil.rmtree, bin_dir, ignore_errors=True)
    for tool in HOOK_TOOLS:
        source = shutil.which(tool)
        assert source, f"{tool} is needed to run the hook"
        (bin_dir / tool).symlink_to(source)
    return str(bin_dir)


def _run(part, *, env=None, expect=0):
    base = {"PATH": _hook_path()}
    base.update(env or {})
    args = ["bash", str(HOOK)] + ([part] if part else [])
    result = subprocess.run(args, capture_output=True, env=base, text=True, check=False)
    assert result.returncode == expect, result.stderr
    return result.stdout


def test_each_registered_hook_command_stays_under_the_output_cap():
    """AGENTS.md alone is longer than the cap, so it is delivered in two parts."""
    for part in ("identity", "context"):
        output = _run(part, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})
        assert 0 < len(output) < HOOK_OUTPUT_CAP, f"{part}: {len(output)} characters"
    assert "Docker missing" in output  # the context part was measured at its longest


def test_the_two_parts_deliver_all_of_agents_md_exactly_once():
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    identity = _run("identity", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})
    context = _run("context", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})

    assert agents.startswith(identity)
    remainder = agents[len(identity):]
    assert remainder and remainder in context
    assert context.count(remainder) == 1


def test_split_ignores_heading_lookalikes_inside_fenced_code(tmp_path):
    """A ``## `` comment in a code block is not a place to cut the file."""
    agents = (
        "# TAO\n\n" + "identity line\n" * 40
        + "```bash\n## a shell comment, not a heading\necho hi\n```\n"
        + "## Reference\n\n" + "reference line\n" * 20
    )
    (tmp_path / "AGENTS.md").write_text(agents)
    env = {"CLAUDE_PLUGIN_ROOT": str(tmp_path)}

    identity = _run("identity", env=env)
    assert identity.endswith("echo hi\n```\n")
    assert "## Reference" in _run("context", env=env)


def test_hooks_json_registers_both_parts_with_a_quoted_plugin_root():
    registered = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    commands = [
        hook["command"]
        for group in registered["hooks"]["SessionStart"]
        for hook in group["hooks"]
    ]

    assert [c.rsplit(" ", 1)[-1] for c in commands] == ["identity", "context"]
    for command in commands:
        # An unquoted plugin root splits the command when the path has a space.
        assert command.startswith('"${CLAUDE_PLUGIN_ROOT}/hooks/session_start.sh" ')


def test_hook_finds_the_plugin_root_without_claude_variables():
    """Other harnesses run the same hook without CLAUDE_PLUGIN_ROOT."""
    identity = _run("identity")
    context = _run("context")

    assert identity.startswith("# TAO")
    assert f"TAO skill bank root: `{REPO_ROOT}`" in context
    assert "## Credentials" in context


def test_bank_path_is_persisted_once_for_the_session(tmp_path):
    env_file = tmp_path / "claude-env"
    env = {"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT), "CLAUDE_ENV_FILE": str(env_file)}
    for _ in range(3):
        context = _run("context", env=env)

    assert env_file.read_text() == f'export TAO_SKILL_BANK_PATH="{REPO_ROOT}"\n'
    assert "TAO skill bank root:" not in context
    _run("identity", env=env)
    assert env_file.read_text().count("TAO_SKILL_BANK_PATH") == 1


def test_running_the_hook_without_a_part_prints_everything():
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    everything = _run(None, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})

    assert everything.startswith(agents)
    assert "## Credentials" in everything
    assert _run("bogus", expect=2) == ""


def test_credentials_are_reported_by_name_only():
    secret = "should-never-be-printed"
    context = _run(
        "context", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT), "NGC_KEY": secret}
    )

    assert "- NGC_KEY" in context
    assert secret not in context
    assert os.environ.get("NGC_KEY") != secret
