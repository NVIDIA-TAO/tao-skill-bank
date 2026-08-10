#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create scripts/mission_control/.venv and install requirements.txt into it.
# Works even when python3-venv is not installed via apt.
#
# Usage: bootstrap_venv.sh [PYTHON] [--sync]
#   PYTHON  base interpreter to use (default: output of deft_python.sh)
#   --sync  reinstall requirements.txt into an existing venv

set -eu

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VENV="$HERE/.venv"
REQS="$HERE/requirements.txt"

PYTHON=
SYNC=0
for arg in "$@"; do
  case "$arg" in
    --sync) SYNC=1 ;;
    "") ;;
    *) PYTHON="$arg" ;;
  esac
done
[ -n "$PYTHON" ] || PYTHON="$("$HERE/../deft_python.sh")"

# Usable means the deps actually import, not just that a directory exists.
# Debian's python3-venv failure leaves bin/python behind with no pip, and
# bin/python is a symlink to the base interpreter either way — so testing the
# interpreter, or even pip, can pass against a venv that cannot serve.
# pyvenv.cfg proves it is a venv; the import proves the install finished.
VENV_PROBE='import fastapi, uvicorn, sklearn, PIL, yaml, pandas'
venv_ok() {
  [ -f "$VENV/pyvenv.cfg" ] && [ -x "$VENV/bin/python" ] \
    && "$VENV/bin/python" -c "$VENV_PROBE" >/dev/null 2>&1
}

if venv_ok; then
  [ "$SYNC" -eq 1 ] || exit 0
  "$VENV/bin/python" -m pip install -q -r "$REQS"
  exit 0
fi

# Air-gap: never run a package manager, not even as a fallback. A missing venv
# is a hard stop naming the fix, like Pre-Flight's staged-asset checks.
if [ "${AIR_GAPPED:-0}" = 1 ]; then
  echo "FATAL: air-gap mode — no usable venv at $VENV, and no installer will run." >&2
  echo "  Pre-stage it on a networked host with this script, then copy the tree." >&2
  echo "  See references/air-gap.md." >&2
  exit 2
fi

# A half-built tree from an earlier attempt would poison every strategy below.
rm -rf "$VENV"

first_err=

# Strategy 1: standard venv (needs python3-venv on Debian/Ubuntu). Its failure
# names the exact apt package, so keep the message for the final report.
if err=$("$PYTHON" -m venv "$VENV" 2>&1); then
  "$VENV/bin/python" -m pip install -q -r "$REQS"
  exit 0
fi
first_err=$err
rm -rf "$VENV"

# Strategy 2: system virtualenv binary — no python3-venv, no install needed.
if command -v virtualenv >/dev/null 2>&1 \
   && virtualenv --python="$PYTHON" "$VENV" >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install -q -r "$REQS"
  exit 0
fi
rm -rf "$VENV"

# Strategy 3: venv --without-pip, then seed pip from the base interpreter.
if "$PYTHON" -m venv --without-pip "$VENV" >/dev/null 2>&1; then
  BASE_PIP="$("$PYTHON" -c "import pip, os; print(os.path.dirname(pip.__file__))" 2>/dev/null || true)"
  if [ -n "$BASE_PIP" ]; then
    SITE="$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")"
    cp -r "$BASE_PIP" "$SITE/"
    if "$VENV/bin/python" -m pip install -q -r "$REQS" >/dev/null 2>&1; then
      exit 0
    fi
  fi
fi
rm -rf "$VENV"

echo "mission_control/bootstrap_venv: could not create a venv using any strategy" >&2
echo "  tried: python -m venv, system virtualenv, venv --without-pip" >&2
[ -n "$first_err" ] && printf '  python -m venv said:\n%s\n' "$first_err" >&2
echo "  quickest fix: sudo apt-get install python3-venv" >&2
echo "  or:           pip install virtualenv" >&2
exit 1
