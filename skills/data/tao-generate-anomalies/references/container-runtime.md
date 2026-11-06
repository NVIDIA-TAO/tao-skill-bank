<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pinned container and platform lifecycle

Resolve `images.metropolis_sdg.anomalygen` from repository `versions.yaml`; pin
its resolved image digest in the job receipt. The 1.2 image root is
`/workspace/cdf-anomalygen`. Test CUDA/imports using the actual pinned image,
not the host Python environment or a newer source checkout.

The selected platform skill owns `create → run → wait → remove`. Keep the job
ID, container ID, PID/state, stdout/stderr and bounded wait intervals. A native
training subprocess must remain in a live container until complete; do not
remove the container immediately after `train_control.sh start`. Cancellation
first stops native work; cleanup happens after logs/status/artifacts are saved.

Mount shared skill scripts read-only, user data/models read-only where possible,
complete checkpoint assets at `/workspace/cdf-anomalygen/checkpoints`, outputs
writable, and training runtime results persistently at
`/workspace/cdf-anomalygen/results`. A specific model subtree is insufficient
when selected stages need tokenizer, VAE, DINOv2, SAM2, or guardrail assets.

Use `scripts/skill_utility/download_required_weights.py --help` in the image and
its native required-weights mapping for the selected stages. Generation needs
Nano DCP base, Wan VAE, Qwen tokenizer JSON/TXT assets and the two guardrail
repositories. Evaluation needs its native backbones. AMP/AMG have additional
route-specific models. Global `preflight_env_ckpt.sh` checks all optional models;
distinguish intentionally unneeded assets from missing selected-stage assets.
Do not download Edge/Gemma solely to make a global check green.

Credentials remain in ephemeral environment/credential mounts; do not log or
commit them. Native conversion may invoke HF CLI even with local source weights
and may create a tokenizer checkpoint beside source assets. Stage writable
metadata/tokenizer inputs and a download cache; preserve the user's original
checkpoint read-only. Record failed attempts and conversion provenance.

For offline Docker (`--network=none`), ensure the advertised container hostname
resolves to loopback inside the container. A reproducible single-node setting is
`--hostname localhost`; otherwise `torchrun --standalone` can remain stuck in
rendezvous before workers/model initialization. This is a launch failure, not a
model or guardrail failure. Preserve its receipt and verify that no old workers
remain before starting a fresh named attempt.
