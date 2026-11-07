# WSP numeric editing engineering boundary

This code supports a separately approved experiment. It does not launch a
scientific editor automatically. Keep manifests, models, data, credentials,
results and approval receipts outside the skill-bank checkout.

## Entry points

- `scripts/wsp03_data.py prepare|validate`: component-preserving development,
  holdout, exploration and probe views. Bind source and exclusion SHA256 values.
- `scripts/wsp03_runtime.py probe`: exactly two base-model, batch-one, 64-query
  generation passes (zero-shot and ICL), under the host-wide GPU lease. Supply
  the frozen image URI, immutable model directory, exact request JSON files,
  PNG-only input view, new output directory and shared lease path. The CLI
  records platform jobs, tokens, timing, memory and throughput estimates. It
  requires passwordless permission to set and restore GPU compute mode.
- `IsolatedKernel`: persistent CPU numeric process with read-only source/model
  mounts, no network, no GPU, no provider credentials, read-only root and a
  private mailbox. The actual child records its isolation checks. Every launch
  uses the existing platform job-record and four-verb lifecycle.
- `Service`: trusted dispatch binding development membership, reference
  exclusion, candidate registration, native worker results and durable ledger.
  It accepts only an explicit matching scientific approval receipt. Trusted
  phase transitions and evaluator adapters are not editor-facing methods.

The request renderer, frozen strategy registry, editor/provider relay and
trusted scorer are experiment-specific adapters. Bind and review those adapters
before claiming a complete scientific execution gate. Do not expose a spec or
context pack containing withheld IDs or historical predictions to the editor.
The engineering probe alone never loads gold or computes accuracy.

## Numeric tools

The CPU kernel exposes `load_tensor`, `save_edit`, `reset_candidate`, and
`commit_candidate` through a persistent restricted numeric Python namespace.
Real tensor values, CPU linear algebra (including closed-form least squares),
and replay manifests are available. Gradients, optimizers, imports, arbitrary
files, subprocesses and networking are outside the exposed capabilities. The
AST checks are capability reduction, **not the OS sandbox**.

Edits retain shapes and dtype, reject nonfinite and BF16-zero deltas, whole
parameter rescaling and protected embedding/output/final-normalization tensors.
No new architecture parameters may be introduced. Candidate manifests record
actual changed elements and bitwise before/after tensor hashes; replay validates
those hashes. Strict reasoning edits and algebraic closed-form edits have
separate method labels. A real experiment must additionally bind the frozen
maximum-five numerical configurations per semantic edit family.

`wsp03_worker.py` separates generation from teacher-forced reads. Generation
rejects hook and target fields. Reads support hidden/MLP tensors, diagnostic
activation replacement, canonical target log-probabilities, and attention rows
reconstructed in FP32 from the same pass's normalized Q/K, RoPE and mask. This
is labelled diagnostic reconstruction, not SDPA's native probability output.
Reads retain at most 256 MiB per request batch. Completed first answers remain
immutable; unresolved attempts block automatic replay.

## State and accounting

SQLite transactions retain admitted request identities, counters, native token
usage, complete candidate rankings, shortlist and freeze across restart. Soft
editor stop does not disable trusted finalist selection or evaluation. The
frozen allocation is configurable; proposed allocation changes cannot silently
replace an existing ledger contract. Probe debit is bound once. Generation
exploration reserves measured time for up to three full-development finalists.

GPU allocation seconds include lease idle and model loading. Process memory is
sampled every 100 ms with gaps and unknown-memory coverage retained. Missing
kernel-active timing is reported as unknown. CUDA allocated/reserved peaks,
worker RSS, cgroup peak and input/output token counts remain distinct measures.

## CPU regression checks

Run `python -m unittest discover -s tests -p 'test_wsp03*.py'` inside the pinned
framework image with `--runtime runc`, no GPU, no network and read-only source.
The tests use synthetic tensors only. Real GPU throughput is a separately
recorded engineering probe, never a unit test.
