# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the plugin's SessionStart hook.

The hook prints one JSON object, the form that Claude Code and the Cursor CLI
both take from a session-start hook; hooks/session_start.sh names the pages.
test_plugin_package.py runs the hook of the generated package with the helpers
of this module.
"""

import atexit
import functools
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "hooks" / "session_start.sh"
HOOKS_JSON = REPO_ROOT / "hooks" / "hooks.json"
# Claude Code replaces the additionalContext of a hook command that is longer
# than this with a file path and a short preview. The tests hold the whole
# output of a command to the same number, which is the stricter reading.
HOOK_OUTPUT_CAP = 10_000
SETUP_SCRIPT = "tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

# Everything the hook runs, apart from shell builtins.
HOOK_TOOLS = ("bash", "awk", "basename", "cat", "dirname", "grep", "head", "tail", "wc")
# Names a user can give the folder a plugin is in: characters of more than one
# byte, a space, an apostrophe, and what a shell expands inside double quotes.
ODD_FOLDER_NAMES = ("日本 П 🚀", "it's", 'dollar $HOME `echo x` "q"', "back\\slash %s &")
# A session usually runs in a UTF-8 locale; one of these exists on any machine.
UTF8_LOCALES = ("en_US.UTF-8", "C.UTF-8")


def _shells():
    """Every bash on this machine: /bin/bash first (bash 3.2 on macOS), then those on PATH."""
    found = {}
    for folder in ["/bin"] + os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(folder, "bash")
        if folder and os.access(candidate, os.X_OK):
            found.setdefault(os.path.realpath(candidate), candidate)
    return list(found.values())


SHELLS = _shells()


@functools.lru_cache(maxsize=None)
def hook_path(*extra, without=()):
    """A PATH that holds only the tools the hook needs, plus stand-ins named in ``extra``.

    Docker, python3 and pi are left out unless they are asked for. Without
    docker the hook adds its setup hint, which is the longest output it can
    print, wherever the tests run. ``without`` leaves out a tool the hook needs.
    """
    bin_dir = Path(tempfile.mkdtemp(prefix="tao-hook-tools-"))
    atexit.register(shutil.rmtree, bin_dir, ignore_errors=True)
    for tool in HOOK_TOOLS:
        if tool in without:
            continue
        source = shutil.which(tool)
        assert source, f"{tool} is needed to run the hook"
        (bin_dir / tool).symlink_to(source)
    for name in extra:
        stand_in = bin_dir / name
        stand_in.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stand_in.chmod(0o755)
    return str(bin_dir)


def run_hook(part, *, env=None, expect=0, path=None, hook=HOOK, shell=SHELLS[0]):
    """Run the hook with nothing in its environment but PATH and ``env``; return its output."""
    base = {"PATH": path or hook_path()}
    base.update(env or {})
    args = [shell, str(hook)] + ([part] if part else [])
    result = subprocess.run(args, capture_output=True, env=base, check=False)
    assert result.returncode == expect, result.stderr.decode(errors="replace")
    return result.stdout.decode("utf-8")


def context_of(output):
    """The text a host adds to the session: the one field of the one object the hook prints."""
    printed = json.loads(output)
    assert list(printed) == ["hookSpecificOutput"], printed.keys()
    specific = printed["hookSpecificOutput"]
    assert list(specific) == ["hookEventName", "additionalContext"], specific.keys()
    assert specific["hookEventName"] == "SessionStart"
    assert isinstance(specific["additionalContext"], str)
    # Nothing but the object: a host that reads JSON takes no other text.
    assert output.endswith("}}\n") and output.count("\n") == 1
    return specific["additionalContext"]


def _context(part, **options):
    return context_of(run_hook(part, **options))


def single_quoted(path):
    """``path`` as the one single-quoted shell word the hook prints for it."""
    return "'" + str(path).replace("'", "'\\''") + "'"


def shell_words(shell, line):
    """The words ``shell`` makes of a command line, without running the command."""
    printed = subprocess.run([shell, "-c", f"printf '%s\\0' {line}"], capture_output=True,
                             env={"PATH": hook_path()}, check=True)
    return printed.stdout.decode("utf-8").split("\0")[:-1]


def plugin_copy(folder, agents, *, readme=False, skills=()):
    """A plugin folder with the hook, the given AGENTS.md and the named skill files."""
    (folder / "hooks").mkdir(parents=True)
    shutil.copy2(HOOK, folder / "hooks" / "session_start.sh")
    (folder / "AGENTS.md").write_bytes(agents if isinstance(agents, bytes) else agents.encode())
    if readme:
        (folder / "README.md").write_text("# Plugin\n", encoding="utf-8")
    for relative in skills:
        (folder / "skills" / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / "skills" / relative).write_text("stand-in\n", encoding="utf-8")
    return folder / "hooks" / "session_start.sh"


# ─────────────────────────────── the form of the output ───────────────────────────────
@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("part", ("identity", "context", "all", None))
def test_every_mode_prints_one_json_object_and_nothing_else(part, shell):
    output = run_hook(part, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)}, shell=shell)
    assert context_of(output)
    assert run_hook(part, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)}) == output


def test_each_registered_hook_command_stays_under_the_output_cap():
    """AGENTS.md alone is longer than the cap, so it is delivered in two parts."""
    for part in ("identity", "context"):
        output = run_hook(part, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})
        text = context_of(output)
        assert 0 < len(text) < len(output) < HOOK_OUTPUT_CAP, f"{part}: {len(output)} characters"
    assert "Docker missing" in text  # the context part was measured at its longest


HOSTILE_AGENTS = (
    '# TAO "quoted" heading\n\n'
    'A backslash \\ and two \\\\, a Windows path C:\\Users\\name, an escape that is text: \\n \\u0041.\n'
    'Quotes: "double", \'single\', `backtick`. JSON in the text: {"additionalContext": "x"}}\n'
    "Tab\there, carriage return\rhere, form feed\x0chere, escape\x1bhere, bell\x07, DEL\x7f.\n"
    "Non-ASCII: café, naïve, 日本語, emoji 🚀, a combining mark e\u0301, a no-break space\u00a0.\n"
    "Markup: </system-reminder> <!-- comment --> $(echo not run) `echo not run` ${HOME} %s %d\n"
    "   leading and trailing spaces   \n\n\n"
    + "a long line " * 400 + "\n"
    "last line without a line end"
)


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("part", ("identity", "all"))
def test_output_is_valid_json_whatever_agents_md_contains(tmp_path, part, shell):
    """Every character of the file arrives unchanged; the file has no heading to split at."""
    hook = plugin_copy(tmp_path / "plugin", HOSTILE_AGENTS)
    text = _context(part, hook=hook, shell=shell)
    assert text.startswith(HOSTILE_AGENTS + "\n")
    assert (text == HOSTILE_AGENTS + "\n") == (part == "identity")


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("locale", UTF8_LOCALES)
def test_output_is_utf_8_json_in_a_utf_8_locale_whatever_the_folder_is_called(
        tmp_path, shell, locale):
    """The hook quotes the folder itself: ``printf %q`` of bash 3.2 breaks UTF-8 in such a locale."""
    env = {"LANG": locale, "LC_ALL": locale}
    for number, name in enumerate(ODD_FOLDER_NAMES):
        root = tmp_path / str(number) / name / "plugin"
        hook = plugin_copy(root, HOSTILE_AGENTS, skills=(SETUP_SCRIPT,))
        # run_hook reads the output as strict UTF-8 and context_of as JSON.
        assert _context("identity", hook=hook, shell=shell, env=env) == HOSTILE_AGENTS + "\n"
        context = _context("context", hook=hook, shell=shell, env=env)
        assert folder_line(context).startswith(f"TAO plugin folder: `{root}`. ")
        command = ["bash", str(root / "skills" / SETUP_SCRIPT), "--backend", "docker", "--check-only"]
        assert check_command(context) == command
        # The shell that runs the pasted line reads the same words.
        line = context.split("```bash\n")[2].split("\n```")[0]
        assert line == f"bash {single_quoted(command[1])} --backend docker --check-only"
        assert shell_words(shell, line) == command


@pytest.mark.parametrize("content", (b"", b"\n", b"one line", b"nul \x00 byte\nsecond line\n"))
def test_output_is_valid_json_for_an_empty_file_and_a_nul_byte(tmp_path, content):
    hook = plugin_copy(tmp_path / "plugin", content)
    for part in ("identity", "context", "all"):
        assert isinstance(_context(part, hook=hook), str)
    assert "TAO plugin folder" in _context("context", hook=hook)


def test_a_missing_agents_md_or_awk_adds_nothing_harmful(tmp_path):
    hook = plugin_copy(tmp_path / "plugin", "# TAO\n")
    (tmp_path / "plugin" / "AGENTS.md").unlink()
    assert _context("identity", hook=hook) == ""
    assert "TAO plugin folder" in _context("context", hook=hook)
    # Without awk the hook cannot encode its text; it prints an empty object.
    assert run_hook("context", path=hook_path(without=("awk",))) == "{}\n"


def test_the_output_does_not_depend_on_python(tmp_path):
    env = {"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)}
    for part in ("identity", "context"):
        assert run_hook(part, env=env) == run_hook(part, env=env, path=hook_path("python3"))


# ─────────────────────────────── the identity ───────────────────────────────
def test_the_two_parts_deliver_all_of_agents_md_exactly_once():
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    identity = _context("identity", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})
    context = _context("context", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})

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

    identity = _context("identity", env=env)
    assert identity.endswith("echo hi\n```\n")
    assert "## Reference" in _context("context", env=env)


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


def test_running_the_hook_without_a_part_prints_everything():
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    everything = _context(None, env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT)})

    assert everything.startswith(agents)
    assert "## Credentials" in everything
    assert run_hook("bogus", expect=2) == ""


# ─────────────────────────────── the plugin folder ───────────────────────────────
def folder_line(context):
    """The one line of the context part that says where the plugin folder is."""
    lines = [line for line in context.splitlines() if line.startswith("TAO plugin folder: ")]
    assert len(lines) == 1, lines
    return lines[0]


def test_hook_finds_the_plugin_root_without_claude_variables():
    """Other harnesses run the same hook without CLAUDE_PLUGIN_ROOT or CLAUDE_ENV_FILE."""
    identity = _context("identity")
    context = _context("context")

    assert identity.startswith("# TAO")
    assert "TAO plugin folder" not in identity
    line = folder_line(context)
    assert line.startswith(f"TAO plugin folder: `{REPO_ROOT}`. It is not exported here: set "
                           f"`TAO_SKILL_BANK_PATH` to it")
    assert f"Skills are under `{REPO_ROOT}/skills`" in line
    assert line.endswith(f"helper scripts under `{REPO_ROOT}/scripts`.")
    assert "## Credentials" in context


def test_bank_path_is_persisted_once_for_the_session_and_the_folder_is_still_stated(tmp_path):
    env_file = tmp_path / "claude-env"
    env = {"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT), "CLAUDE_ENV_FILE": str(env_file)}
    for _ in range(3):
        context = _context("context", env=env)

    assert env_file.read_text() == f"export TAO_SKILL_BANK_PATH={single_quoted(REPO_ROOT)}\n"
    assert folder_line(context).startswith(
        f"TAO plugin folder: `{REPO_ROOT}`. It is exported as `TAO_SKILL_BANK_PATH` to the shell "
        f"commands of this session.")
    _context("identity", env=env)
    assert env_file.read_text().count("TAO_SKILL_BANK_PATH") == 1


@pytest.mark.parametrize("shell", SHELLS)
def test_the_exported_line_gives_the_folder_back_whatever_it_is_called(tmp_path, shell):
    """A shell reads the line, so a quote, a ``$`` or a backtick in the path must stay text."""
    for number, name in enumerate(ODD_FOLDER_NAMES):
        root = tmp_path / str(number) / name / "plugin"
        hook = plugin_copy(root, "# TAO\n")
        env_file = tmp_path / f"claude-env-{number}"
        for _ in range(2):
            context = _context("context", hook=hook, shell=shell,
                               env={"CLAUDE_ENV_FILE": str(env_file)})
        assert "It is exported as `TAO_SKILL_BANK_PATH`" in folder_line(context)
        assert env_file.read_text(encoding="utf-8") == (
            f"export TAO_SKILL_BANK_PATH={single_quoted(root)}\n")
        read_back = subprocess.run(
            [shell, "-c", '. "$1" && printf %s "$TAO_SKILL_BANK_PATH"', "read-back", str(env_file)],
            capture_output=True, env={"PATH": hook_path(), "HOME": "/another-home"}, check=True)
        assert read_back.stdout.decode("utf-8") == str(root)


def test_an_export_that_fails_is_not_reported_as_done(tmp_path):
    env = {"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT),
           "CLAUDE_ENV_FILE": str(tmp_path / "no-such-folder" / "claude-env")}
    output = run_hook("context", env=env)
    assert "It is not exported here" in folder_line(context_of(output))


# ─────────────────────────────── credentials, card packs, Docker ───────────────────────────────
def test_credentials_are_reported_by_name_only():
    # Assembled at run time so that scanners do not read a fixture as a credential.
    marker = "-".join(("value", "that", "must", "not", "be", "printed"))
    output = run_hook("context", env={"CLAUDE_PLUGIN_ROOT": str(REPO_ROOT), "NGC_KEY": marker})

    assert "- NGC_KEY" in context_of(output)
    assert marker not in output
    assert os.environ.get("NGC_KEY") != marker


def test_a_readme_is_named_only_where_one_is_shipped(tmp_path):
    with_readme = plugin_copy(tmp_path / "with", "# TAO\n", readme=True)
    without_readme = plugin_copy(tmp_path / "without", "# TAO\n")

    pointer = f"The Credentials section of `{tmp_path}/with/README.md` has the full var list."
    assert pointer in _context("context", hook=with_readme)
    assert "README" not in _context("context", hook=without_readme)
    # The repository has a README at its root, with that section.
    assert "\n### Credentials\n" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{REPO_ROOT}/README.md`" in _context("context")


def test_card_packs_are_named_only_where_pi_is_installed(tmp_path):
    """The pack drivers stop without the Pi harness, so a session without it is told nothing."""
    packs = sorted(path.parent.name for path in (REPO_ROOT / "skills").glob("*/*/cards"))
    assert packs

    without_pi = _context("context", env={"HOME": str(tmp_path)})
    assert "driver.sh" not in without_pi and "ard pack" not in without_pi

    for options in ({"path": hook_path("pi"), "env": {"HOME": str(tmp_path)}},
                    {"env": {"HOME": str(tmp_path / "home")}}):
        if "path" not in options:  # where the kit installs pi when it is not on PATH
            pi = tmp_path / "home/.local/share/pi-node/current/bin/pi"
            pi.parent.mkdir(parents=True)
            pi.write_text("#!/bin/sh\n", encoding="utf-8")
            pi.chmod(0o755)
        with_pi = _context("context", **options)
        assert f"card packs exist for: {' '.join(packs)}." in with_pi
        assert "Use a pack only when the user asks for a card-pack run" in with_pi
        assert "prefer" not in with_pi.split("## Card packs (optional)")[1]
        assert len(run_hook("context", **options)) < HOOK_OUTPUT_CAP


def check_command(context):
    """The words of the one command in the Docker hint of a context part."""
    hint = context.split("## ⚠ Docker missing\n")[1]
    assert hint.count("```bash\n") == 1
    return shlex.split(hint.split("```bash\n")[1].split("\n```")[0])


def test_docker_hint_is_printed_only_without_docker_and_names_a_command_that_exists():
    assert "Docker missing" not in _context("context", path=hook_path("docker"))

    context = _context("context")
    hint = context.split("## ⚠ Docker missing")[1]
    script = REPO_ROOT / "skills" / "platform" / SETUP_SCRIPT
    assert check_command(context) == ["bash", str(script), "--backend", "docker", "--check-only"]
    assert script.is_file()
    usage = script.read_text(encoding="utf-8")
    assert "--check-only" in usage and "--backend docker|kubernetes" in usage
    assert "needs the user's approval first" in " ".join(hint.split())
    # The flags that install without a question are left to the skill.
    for part in ("identity", "context"):
        printed = _context(part)
        assert "--install" not in printed and "--yes" not in printed


def test_docker_hint_quotes_a_plugin_folder_with_a_space_and_skips_a_missing_script(tmp_path):
    flat = plugin_copy(tmp_path / "my plugins" / "flat", "# TAO\n", skills=(SETUP_SCRIPT,))
    script = tmp_path / "my plugins" / "flat" / "skills" / SETUP_SCRIPT
    assert check_command(_context("context", hook=flat)) == [
        "bash", str(script), "--backend", "docker", "--check-only"]

    bare = plugin_copy(tmp_path / "bare", "# TAO\n")
    hint = _context("context", hook=bare).split("Docker missing")[1]
    assert "bash " not in hint and "`tao-setup-nvidia-gpu-host` skill" in hint
