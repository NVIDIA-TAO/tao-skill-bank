#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegate to the DINOv3 controller packaged in TAO Data Services.

Run this compatibility shim inside the allocated DS runtime. The host needs
only its platform launcher; this script never installs packages or starts Docker.
"""

import runpy


def main():
    """Forward command-line arguments to the installed DS workflow CLI."""
    try:
        runpy.run_module("nvidia_tao_ds.mining.dinov3.workflow", run_name="__main__")
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("nvidia_tao_ds"):
            raise SystemExit(
                "Run this command inside a TAO Data Services image containing "
                "DINOv3 SSL DEFT; no host TAO installation is required."
            ) from error
        raise


if __name__ == "__main__":
    main()
