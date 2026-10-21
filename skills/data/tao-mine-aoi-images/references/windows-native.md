<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Windows mining

The pinned data-services image is Linux-only. On a native Windows host, run
`scripts/native_mine_aoi.py` instead: it reproduces the three-step contract
(embed targets, embed source pool, top-K cosine) with
`google/siglip-base-patch16-224` through transformers, and emits the same
parquet schemas, so `filter_mined_by_cosine.py` and `filter_mined_history.py`
consume its output unchanged. Parity was verified by the cosine recompute in
`filter_mined_by_cosine.py` reproducing the miner's scores on a live DEFT run
(see the DEFT skill's `references/windows-route.md`).

It needs the Cosmos Windows venv (torch + transformers + pyarrow, or duckdb on ARM64 where pyarrow has no wheel), one GPU,
and network access for the one-time encoder download. `filter_mined_history.py`
previously failed on Windows with `EBADF` — os.fsync on a read-only handle,
legal on POSIX, not on win32 — and is fixed in this bank.

For per-class mining recipes (separate NG/OK target passes with their own
top-K, e.g. NG top-15 / OK top-3), pass `--filter-label OK|NG` to restrict the
source pool to one label before embedding; run the miner once per class and
merge the outputs. The pool is single-image-tolerant: only `images[0]` is
embedded, so records without a golden reference mine unchanged.
