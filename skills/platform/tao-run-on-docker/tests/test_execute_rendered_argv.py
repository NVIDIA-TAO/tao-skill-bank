from __future__ import annotations

import importlib.util
import pathlib


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/execute_rendered_argv.py"
SPEC = importlib.util.spec_from_file_location("execute_rendered_argv", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_executes_exact_argv_and_verifies_returned_container():
    calls = []
    cid = "a" * 64

    def runner(argv, **_kwargs):
        calls.append(argv)
        return Result(stdout=cid + "\n") if len(calls) == 1 else Result(stdout="/job-name\n")

    payload = {
        "backend_name": "job-name",
        "argv": ["docker", "run", "-d", "--name", "job-name", "image", "command"],
    }
    assert MODULE.execute(payload, runner=runner)["backend_ref"] == cid
    assert calls[0] == payload["argv"]
    assert calls[1][-1] == cid


def test_rejects_shell_or_mismatched_name_before_execution():
    for payload in (
        {"backend_name": "job", "argv": ["sh", "-c", "docker run -d"]},
        {"backend_name": "job", "argv": ["docker", "run", "-d", "--name", "other", "image"]},
    ):
        try:
            MODULE.execute(payload, runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe rendered payload was accepted")
