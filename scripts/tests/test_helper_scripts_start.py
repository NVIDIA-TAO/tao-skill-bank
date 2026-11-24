# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every helper script the plugin ships starts cleanly, whatever Python runs it.

An agent starts these scripts from the user's project, often as
``python3 -I <script>`` (isolated mode, which leaves the folder of the script off
the import path), and on a Python that may lack PyYAML. A helper then prints its
help or one line that names the missing package, never a traceback, and leaves
no ``__pycache__`` in the plugin folder. test_plugin_package.py runs the same
checks on the generated package.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_plugin_package as builder  # noqa: E402

# The helper scripts of the plugin: what the package ships from scripts/.
HELPERS = sorted(
    Path(path).name for path in builder.SUPPORT_FILES
    if path.startswith("scripts/") and path.endswith(".py")
)
# The helpers that read YAML. No other helper may need PyYAML to start.
NEEDS_PYYAML = {
    "check_tao_launch_preflight.py", "list_automl_support.py", "list_tao_capabilities.py",
    "list_tao_models.py", "resolve_tao_image.py", "resolve_tao_model.py",
    "resolve_versions_key.py",
}
# Modules that other helpers import; they have no command line and print nothing.
MODULES_ONLY = {"skill_layers.py", "tao_slurm_ssh.py"}
PYYAML_MISSING_EXIT = 4
INSTALL_COMMAND = "python3 -m pip install pyyaml"
# What the line adds when Python runs without the user site ("python3 -I" or
# "python3 -s"), where a PyYAML that pip installed for the user is hidden.
NO_USER_SITE_HINT = "start the script without that option first"
# The Python of the operating system. On macOS it is older than the plugin asks
# for and, in isolated mode, has no PyYAML.
SYSTEM_PYTHON = "/usr/bin/python3"
# Runs a script the way "python3 -I <script>" does, on a Python that has no PyYAML.
WITHOUT_PYYAML = (
    "import runpy, sys; sys.modules['yaml'] = None; sys.argv = sys.argv[1:]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)


def _environment():
    """The caller's environment without the variables that change how Python starts."""
    return {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}


def start(python, script, cwd, *, isolated=True, bytecode=False, without_pyyaml=False,
          user_base=None):
    """Start ``script --help`` from another folder and return the finished process.

    ``user_base`` is a folder for Python to keep the user site in; isolated mode ignores it.
    """
    command = [python] + (["-I"] if isolated else []) + ([] if bytecode else ["-B"])
    command += ["-c", WITHOUT_PYYAML] if without_pyyaml else []
    environment = _environment()
    if user_base is not None:
        environment["PYTHONUSERBASE"] = str(user_base)
    return subprocess.run(
        command + [str(script), "--help"], cwd=cwd, env=environment, capture_output=True,
        text=True, check=False,
    )


def one_line_about_pyyaml(result):
    """Whether a run ended with the one line that says PyYAML is missing, and nothing else."""
    lines = result.stderr.splitlines()
    return (result.returncode == PYYAML_MISSING_EXIT and result.stdout == "" and len(lines) == 1
            and "PyYAML" in lines[0] and INSTALL_COMMAND in lines[0])


def check_started(result, name):
    """A started helper printed its help or the one line about PyYAML, never a traceback."""
    shown = f"{name}: exit {result.returncode}\n{result.stdout}{result.stderr}"
    assert "Traceback" not in result.stdout + result.stderr, shown
    if result.returncode == 0:
        assert result.stderr == "", shown
        assert name in MODULES_ONLY or result.stdout.startswith("usage: " + name), shown
    else:
        assert name in NEEDS_PYYAML and one_line_about_pyyaml(result), shown


def has_pyyaml(python):
    """Whether ``python`` finds PyYAML in isolated mode; None when it cannot be run."""
    try:
        probe = subprocess.run(
            [python, "-I", "-c", "import sys; assert sys.version_info >= (3, 9); import yaml"],
            env=_environment(), capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if "AssertionError" in probe.stderr or (probe.returncode and "yaml" not in probe.stderr):
        return None
    return probe.returncode == 0


def check_helpers_start(scripts, cwd):
    """Start every helper in the folder ``scripts`` as ``python3 -I -B <script> --help``.

    Three times: on the Python of the tests, on the same Python with PyYAML
    hidden, and on the Python of the operating system when there is one.
    """
    pythons = [(python, has_pyyaml(python)) for python in (sys.executable, SYSTEM_PYTHON)]
    pythons = [(python, found) for python, found in pythons if found is not None]
    for name in HELPERS:
        for python, finds_pyyaml in pythons:
            result = start(python, scripts / name, cwd)
            check_started(result, name)
            assert (result.returncode == 0) == (finds_pyyaml or name not in NEEDS_PYYAML), name
        hidden = start(sys.executable, scripts / name, cwd, without_pyyaml=True)
        check_started(hidden, name)
        assert (hidden.returncode == 0) == (name not in NEEDS_PYYAML), name


def check_no_bytecode_is_written(scripts, cwd):
    """Start every helper without -B, isolated and not, and look for what it left behind."""
    for name in HELPERS:
        for isolated in (True, False):
            check_started(start(sys.executable, scripts / name, cwd, isolated=isolated,
                                bytecode=True), name)
    left = sorted(path.relative_to(scripts).as_posix() for path in scripts.rglob("*")
                  if path.name == "__pycache__" or path.suffix == ".pyc")
    assert left == [], f"running the helpers left bytecode in the plugin folder: {left}"


@pytest.fixture
def scripts(tmp_path):
    """A copy of the shipped helper scripts, so that a failing test leaves the checkout clean."""
    folder = tmp_path / "plugin" / "scripts"
    folder.mkdir(parents=True)
    for name in HELPERS:
        shutil.copy2(REPO_ROOT / "scripts" / name, folder / name)
    return folder


def test_the_helpers_that_need_pyyaml_are_the_ones_listed():
    assert NEEDS_PYYAML < set(HELPERS) and MODULES_ONLY < set(HELPERS)
    importing = {
        name for name in HELPERS
        if "import yaml" in (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
    }
    # list_automl_support.py reads YAML through list_tao_models.py.
    assert importing == NEEDS_PYYAML - {"list_automl_support.py"}


def test_every_helper_prints_help_or_one_line_about_pyyaml(scripts, tmp_path):
    check_helpers_start(scripts, tmp_path)


def test_a_missing_pyyaml_is_one_line_with_the_install_command(scripts, tmp_path):
    for name in sorted(NEEDS_PYYAML):
        for isolated in (True, False):
            result = start(sys.executable, scripts / name, tmp_path, isolated=isolated,
                           without_pyyaml=True)
            assert one_line_about_pyyaml(result), f"{name}: {result.stdout}{result.stderr}"
            # The line says which Python it is and what that Python cannot do. It does
            # not say "not installed": under -I a PyYAML in the user site is installed.
            assert f"this Python ({sys.executable}) cannot import PyYAML" in result.stderr, name
            assert "not installed" not in result.stderr, name
            assert (NO_USER_SITE_HINT in result.stderr) == isolated, name


def test_a_pyyaml_in_the_user_site_is_named_where_python_leaves_that_site_out(scripts, tmp_path):
    """``python3 -I`` hides what ``pip install`` put in the user site; the line says so.

    The install command changes nothing there, and pip reports the package as
    already installed. Run on the Python of the operating system, with a
    stand-in for PyYAML in a user site of its own.
    """
    if has_pyyaml(SYSTEM_PYTHON) is not False:
        pytest.skip(f"needs a {SYSTEM_PYTHON} that has no PyYAML in isolated mode")
    user_base = tmp_path / "user-base"
    where = "import site; print(site.ENABLE_USER_SITE, site.getusersitepackages())"
    probe = subprocess.run(
        [SYSTEM_PYTHON, "-c", where], env={**_environment(), "PYTHONUSERBASE": str(user_base)},
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    enabled, _, user_site = probe.partition(" ")
    if enabled != "True" or not user_site.startswith(str(user_base)):
        pytest.skip(f"{SYSTEM_PYTHON} has no user site of its own here: {probe}")
    (Path(user_site) / "yaml").mkdir(parents=True)
    (Path(user_site) / "yaml" / "__init__.py").write_text("", encoding="utf-8")

    for name in sorted(NEEDS_PYYAML):
        # As pip leaves it: installed for the user, and found by a plain start.
        found = start(SYSTEM_PYTHON, scripts / name, tmp_path, isolated=False, user_base=user_base)
        assert found.returncode == 0 and found.stdout.startswith("usage: " + name), found.stderr
        hidden = start(SYSTEM_PYTHON, scripts / name, tmp_path, user_base=user_base)
        assert one_line_about_pyyaml(hidden), f"{name}: {hidden.stdout}{hidden.stderr}"
        assert NO_USER_SITE_HINT in hidden.stderr and "not installed" not in hidden.stderr, name


def test_running_a_helper_leaves_no_bytecode_beside_it(scripts, tmp_path):
    check_no_bytecode_is_written(scripts, tmp_path)
