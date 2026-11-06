<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native evaluation and per-sample search

The default generation route evaluates with native `evaluate.py`, then invokes
`quality_refine.py run --num_search_run 3 --score nn`. It preserves native
per-sample parameters, geometry restrictions and generated/blocked accounting.
Initially blocked testcases are excluded from search, retaining their preset
indices and original blocked receipt; selected winners are joined back with that
receipt and verified against the complete original testcase set.
The selected bucket is `searched/`; retain original and round receipts.

Native `filter.py` supports opt-in drop-ratio filtering; `--drop_ratio` is
required and has no default. There is no automatic threshold or replacement
loop in this skill. A requested filter policy must record its parameters,
retained/dropped counts and selected output. Never describe dropping rows as
passing the original requested count. Legacy threshold 0.4/top-up behavior is
not an equivalence claim for 1.2 metrics.

`--skip-quality` is an explicitly recorded smoke option. An all-blocked run
skips quality and COCO export with training_ready=false, while keeping native
blocked records. Other failures remain errors and retain state/logs.
