---
name: tao-run-deft-icl-aoi-cosmos3
description: Run Cosmos3 AOI DEFT with adaptive in-context learning, including the frozen six-arm THT suite, a separately approved AnomalyGen 1.2 single-image generator addendum, and compatible paired PCB OK/NG workflows. Uses native Cosmos Framework LoRA, external strict evaluation, isolated Codex classification, SigLIP retrieval, durable recovery and evidence export. Use for THT ICL/DEFT comparisons or Cosmos3 AOI plus ICL, not generic model training.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Linux, Python 3.11 or later, Docker with NVIDIA GPU support, pinned model and data-service images, and external frozen experiment inputs.
tags: [computer-vision, aoi, deft, in-context-learning, anomaly-generation]
allowed-tools: Read, Bash, Write
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
---

# Cosmos3 AOI with ICL

This is a separate application copied from `tao-run-deft-aoi-cosmos3`.
Keep the original application and shared model/platform implementation intact.
Model-first routing resolves Cosmos3 to the existing model skill and selects
`cosmos-framework` for the validated native Nano conversion/training path.
Read its backend contract, the [application metadata](references/skill_info.yaml),
and the Docker platform skill before execution.

## Route the task explicitly

- THT: `mode: tht_single`, one image, eight original class names, exactly one
  JSON label. Use `scripts/suite.py` and the protocol below. No golden image;
  A1/A2 are views of a joint. No multilabel mode.
- Generator addendum: explicit separate `generator-addendum-v1` arm G in
  `tht_single` mode. Use `scripts/generator_addendum.py` and the
  [native generator contract](references/generator-addendum.md). It preserves
  same-D classifier controls and consumes external prepared source manifests.
- Paired PCB: retain default `bare_okng`, two ordered images, exact OK/NG
  target and the copied [paired workflow](references/paired-workflow.md).
  Its binary Benchmark-gate behavior does not apply to the THT suite.
- General ICL policy/toolset work delegates to `tao-in-context-learning`.
  Do not substitute a fixed grid or preselect twelve candidates.

## THT execution contract

The THT suite supports the authorized Brev host's local Docker backend. It
consumes the platform's submit/status/logs/cancel verbs through shared job
records. Every attempt gets a record and results directory before launch.
No implicit image pulls, asset downloads, AnomalyGen checks, or SDG generation.
Preparation/conversion and smoke evidence must exist before formal preflight.

1. Copy [task.example.json](references/task.example.json) outside the checkout
   and fill absolute paths, immutable digests, measured budgets and evidence.
   Keep credentials at their approved private path; never read or copy contents.
2. Run `python scripts/suite.py preflight --task TASK.json --output PREFLIGHT.json`.
   Review effective parameters, origins, hashes, resource estimates, frozen
   rules, access-isolation evidence and native token-cap semantics. This is a
   read-only validation; it performs no inference/training/agent launches.
3. Obtain one approval of that concrete preflight, unless already explicitly
   granted for this exact formal run. Save an external approval JSON containing
   `preflight_sha256`, `approved_by`, and `approved_at`. M3 smoke authorization
   is not formal approval. Never manufacture approval.
4. Run `python scripts/suite.py run --preflight PREFLIGHT.json --approval APPROVAL.json`.
   One supervisor owns the SQLite journal and runs all dependencies; no prompts
   for next policy, checkpoint, stage or iteration. Resume with the same command.
5. Inspect `python scripts/suite.py status --results-dir RUN`. Native jobs are
   polled from Docker; the journal records progress and immutable evidence.
   `export --results-dir RUN` runs Brev-side analysis without launching models.

## Frozen protocol

A is fresh Astra+Codex zero-shot per query, generic tools only. B is base Nano;
C base+ICL; F full Mining LoRA; D independent base+DEFT LoRA; E exactly the
selected D checkpoint+ICL. The dependency is F epoch selection -> D -> E.

Preserve Mining3642, Proxy74, Benchmark271, source eligibility/exclusions and
two declared original Mining/Benchmark joint overlaps. Never resplit or repair
source metadata. Query manifests have only identity/image/hash. Joint identity
comes from a filename projection validated against all Mining metadata; no
private source rows enter retrieval. ICL excludes query identity, bytes and
normalized physical joint. References and training remain Mining-only.

Use the hash-pinned external evaluator/definition; do not copy internal source
or implement a replacement scorer. B/D/F use the exact externally supplied
Maxwell prompt hash, RGB/NEAREST padded256 PNG, greedy4096 output cap. C/E keep
that query profile. Persist actual processor/template, rendered prompts,
reference IDs/order and raw answers before trusted grading. INVALID and
length-limited final answers are completed and never repaired or retried.

F is one continuous native run, saves/scores every complete epoch on full Proxy,
min5/max20, patience3 counted from epoch1; strict MCC improvement, earliest ties.
D has five rounds, each e* epochs, first from base and later from its own prior
native checkpoint. Model/optimizer/scheduler/trainer state continue; dataloader
cursor resets only for a new accumulated dataset. Within-round recovery restores
its cursor. D saves/exports only the terminal round checkpoint. Training engine,
LoRA and export stay model-owned; the application owns the dataset adapter and
bounded checkpoint/Proxy decision callback.

D's explicit THT adaptation is `tht-error-neighbors-v1`: descriptive multiclass
confusion analysis, every wrong/INVALID Proxy query as a target, four native
cosine neighbors, monotonically accumulated unique Mining records. With no
errors all Proxy queries tie as targets. This replaces the paired binary routing
contract; it is not an image-based causal diagnosis. Every round records RCCA,
routing, targets, provenance and exposures. SDG is `skipped: user-requested`,
zero images, even when errors exist. Paired opt-in SDG remains available.

C/E independently propose one adaptive policy at a time, at most12 complete
Proxy trials, equal frozen wall/agent-token budgets. Failed attempts consume
budgets. Full identity coverage is required. Within0.01 of best MCC choose fewer
images, fewer calls, shorter prompt, then trial ID. Native retrieval caches are
keyed by embeddings, allowed IDs, query, metric, capacity and runtime, so prompt
and presentation changes reuse results. Shared retrieval cost is attributed once.

Freeze all six settings together before Benchmark, including model fingerprints,
policies, prompts, preprocessing, seed and statistics. Then one final answer per
271 queries per arm, 1626 total. Missing answers leave the suite incomplete.

## Isolation, costs and recovery

A runs in a fresh read-only generic container: current query, eight classes,
JSON/budgets, generic Python/image tools, static Codex helpers and private auth
mount. No host home/source/skills, Docker socket, Mining, gold, prior query or
results. Run the access-negative probe from that runtime. Policy agents see the
ICL data skill and Proxy summaries; classifiers never receive gold.

Codex0.166 native rollout-budget enforcement uses input/output weights1. A
completed response and already-issued tools can overshoot the threshold; no
subsequent model response is admitted. Preserve actual usage from per-response
session events, including failed turns. No final answer at wall/token exhaustion
means incomplete, no free retry. The native feature is experimental and must
pass the real cap smoke on the exact installed version before preflight.

At most two attempts per stage, infrastructure failures only. Retry wall time
starts with the first attempt; C/E and A token counters persist. Preserve raw
answers across retry; never rerun completed wrong/INVALID outputs. Native DCP
recovery retains optimizer/scheduler/trainer/data cursor and immutable prior
epoch evidence. Deterministic program/resource failures stop with evidence.

`report_data.json` uses `tht-report/1`: six arms including incomplete ones,
external metrics/per-class confusion, historical descriptive Fail/Pass mapping,
paired joint bootstrap(seed42,2000 percentile resamples,95%), learning curves,
exposures, actual stage/job costs, Benchmark-only latency/tokens, shared/setup
costs, tool traces, autonomy and recovery. Dollars and unavailable measurements
are null. WSP is a separate experiment. Export is evidence awaiting review,
not a published report or a claim that incomplete arms succeeded.

Continuing-session token accounting uses `(stage_start, stage_end]` deltas and
the last snapshot at or before start as its baseline. Adjacent stages cannot
share a boundary event. Preserve source/turn identity when removing replays;
equal token counts are not a duplicate identity. Preserve the explicit session
identity and birth timestamp; a reset needs a new session boundary. A missing
baseline remains unknown, not a new full-session stage total.

Run the Brev exporter to produce `astra_trace_analysis.json`: count actual
tool calls across all attempts, report coverage against271 planned queries,
success/failure and elapsed times paired from Docker item.started/item.completed
timestamps. Missing timestamps remain unknown. Inspect its selected actual
arguments/results and source logs on Brev to describe observable operations and
supported reusable patterns. Record evidence paths for every finding. Do not
infer hidden reasoning or invent a fixed number of findings. Local renderers
consume these aggregates and traces without scientific recomputation.

## Validation

Run `THT_EVALUATOR_DIR=/frozen/evaluator python -m pytest tests/` plus the ICL
skill tests. M3 evidence must establish native training, both resume modes,
export, actual1/2/k-image inference at the allowed maximum, exact loss masks,
paired compatibility, real embedding/mining, tokenizer parity, isolated generic
tools, live native token enforcement and external scorer parity. Synthetic tests
cover decisions/recovery/statistics; they do not establish model capacity.
