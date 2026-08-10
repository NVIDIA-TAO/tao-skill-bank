# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RCA chat agent for a TAO 7.0.1 DEFT AOI run.

A thin narrator (agent.py) over deterministic, run-grounded tools (tools.py):
run overview, data breakdown, failure listing/slicing, per-defect margins,
SigLIP coverage census, and image viewing. All evidence comes from the loaded
run's own artifacts via the app's RunIndex — no external metadata store.
"""
