# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit new-toolset entrypoint; the frozen six-arm suite is unchanged."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['prepare-references', 'prepare-maxwell', 'run-maxwell'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    bank = Path(__file__).resolve().parents[4]
    scripts = bank / 'skills/data/tao-in-context-learning/scripts'
    if a.operation == 'prepare-references': command = [str(scripts / 'reference_policy.py')]
    else: command = [str(scripts / 'maxwell_profile.py'), 'prepare' if a.operation == 'prepare-maxwell' else 'run']
    raise SystemExit(subprocess.call([sys.executable, *command, '--config', a.config]))


if __name__ == '__main__': main()
