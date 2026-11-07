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
  exact PNG-only input view, frozen ordered probe-ID hash, new output directory and shared lease path. The CLI
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

The experiment request renderer, concrete retrieval strategy executor, editor/provider relay and
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
separate method labels. The atomic editing interface registers at most five immutable numerical
configurations per semantic family; renamed families with identical declared
mechanism/tensors/lineage are rejected. Scientific review must still verify that
declared lineage describes the actual hypothesis and numeric Python cells.

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

## Generic atomic interface

`scripts/wsp03_contract.py --output <private-context>/tool-contract.json` emits
operation parameters and an explicit capability/non-capability inventory.
`wsp03_ops.py` implements tensor reads, paired reference attribution, direction,
rank-one, regularized key-value edits, head scaling, replay, revert and HF export.
The MEMIT-style operation takes desired output **deltas** and development-only
preservation covariance; it uses a diagonal Woodbury or full linear solve with
the frozen ridge rule, without gradients or an optimization loop.

The CPU finiteness gate is `scripts/wsp03_scan.py --base /model --output
/results/finiteness.json`. It scans bounded vectorized chunks, rejects empty or
unindexed shards, and does not use a GPU. `wsp03_access_probe.py` records hashed
path denial receipts without reading protected contents. Its canary policy is
private input, not a committed host-specific list.

Actual editor-shell, numeric-kernel and forward-worker-shaped CPU processes
must each pass path-denial proofs. These proofs do not by themselves validate
provider token accounting or real model activation behavior. The optional
saved-generation-prefix read adapter remains unavailable until its identity
validation is implemented and independently reviewed.
