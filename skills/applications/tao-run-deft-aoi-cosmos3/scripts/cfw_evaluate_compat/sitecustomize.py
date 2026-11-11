# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Expose the isolated Evaluate helper as a cosmos_rl namespace extension."""

from pathlib import Path

import cosmos_rl

_compat_package_root = str(Path(__file__).resolve().parent / "cosmos_rl")
if _compat_package_root not in cosmos_rl.__path__:
    cosmos_rl.__path__.append(_compat_package_root)
