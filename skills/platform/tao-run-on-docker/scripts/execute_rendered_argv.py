#!/usr/bin/env python3
"""Execute one validated rendered Docker argv without shell reconstruction."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")


def execute(payload: Any, *, runner=subprocess.run) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("rendered submit payload must be an object")
    name = payload.get("backend_name")
    argv = payload.get("argv")
    if not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None:
        raise ValueError("rendered submit payload has an invalid backend_name")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or "\x00" in item for item in argv)
        or argv[:3] != ["docker", "run", "-d"]
    ):
        raise ValueError("rendered submit payload is not a detached Docker argv")
    if argv.count("--name") != 1:
        raise ValueError("rendered backend_name does not match Docker --name")
    name_index = argv.index("--name")
    if name_index + 1 >= len(argv) or argv[name_index + 1] != name:
        raise ValueError("rendered backend_name does not match Docker --name")

    launched = runner(argv, capture_output=True, text=True)
    if launched.returncode != 0:
        detail = (launched.stderr or launched.stdout).strip().splitlines()
        raise ValueError(
            "Docker submit failed" + (f": {detail[-1]}" if detail else "")
        )
    container_id = launched.stdout.strip()
    if CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("Docker submit did not return one full container ID")
    inspected = runner(
        ["docker", "inspect", "--format", "{{.Name}}", container_id],
        capture_output=True,
        text=True,
    )
    if inspected.returncode != 0 or inspected.stdout.strip() != f"/{name}":
        raise ValueError("returned container ID does not own the rendered backend_name")
    return {"backend_name": name, "backend_ref": container_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.submit.read_text())
        result = execute(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"execute_rendered_argv: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
