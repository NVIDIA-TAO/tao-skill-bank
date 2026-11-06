<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoints, exact recipes, and geometry

Default public example: the published PCB-Nano AnomalyGen 1.2 adapter and its
training recipe. User checkpoints are first-class inputs: provide checkpoint or
run directory, exact training recipe, and the matching dataset/class contract.
Public Metal-Nano and Glass-Nano examples and other user datasets share this API; the skill does not
embed private checkpoint or dataset lookup rules.

Native 1.2 adapters contain `_anomalygen_geometry`, including crop mode, input
size ladder, shift, model size and sorted anomaly types. Keep it intact; native
inference resolves class order and geometry. Crop-ratio distribution also comes
from the exact training recipe. Do not fabricate stamps, remap weights manually,
or override geometry to make an incompatible artifact load.

An unstamped 1.1 Cosmos3 texture adapter can use native legacy inference with its
original recipe; native code upgrades the recipe in memory and remaps class order.
This is generation compatibility, not training resume or weight equivalence.
Verify the actual adapter before claiming compatibility; a stamped public 1.2
smoke or synthetic loader fixture does not prove a user's old adapter works.
1.0 Predict2 checkpoints are incompatible and require retraining.

Native `best_checkpoint.txt` is authoritative. Its selection excludes scored
iterations below 7500 when the planned run exceeds 7500 and eligible later
checkpoints exist; short/early runs keep available iterations. NN improvement
validation follows this native eligibility rule and requires improvement over
iteration zero; `--min-nn-improvement` remains configurable. Never silently use
latest because best is absent. The saved run recipe and selected adapter are
published together with hashes. Base Nano models use native DCP
`checkpoint.json` and `model/{.metadata,*.distcp}`; a Transformers directory is
not directly a DCP base.
