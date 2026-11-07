# Fixed-checkpoint ICL addendum

Use this entrypoint only for separately authorized `icl-addendum-v1` arm H.
The six-arm and generator entrypoints retain their original behavior. H never
calls either entrypoint, trains a model, generates images or exports weights.
Private task inputs and all experiment artifacts stay outside the checkout.

## Inputs and preflight

Start with [icl-task.example.json](icl-task.example.json). File bindings contain
absolute `path`, byte size `bytes` and SHA256 `sha256`. Bind the frozen E task,
its original approved preflight, the first E proposal context, and the exact G
Proxy selection plus its frozen model manifest. The checkpoint field must equal
that sealed selection; there is no checkpoint search or fallback. Verify every
model file and the native export's original-base lineage. Comparator roots are
only names at preflight: historical Benchmark data stays unread until export.

The baseline context contributes only `baseline_metrics`, the original B Proxy74
aggregate. E's policies and trials never seed H. All scientific/runtime settings
come from E's frozen task and preflight, including Codex0.166 bytes, native
rollout token semantics, original prompt, processor, seed42, greedy4096,
RGB padded256 NEAREST PNG and images. The current reviewed Suite recovery and
cost-accounting implementation is retained; the adaptive search, policy agent,
retrieval and classifier are reused.

The read-only preflight validates package membership, checksums, model lineage,
role isolation, images, resources and current environment without launching any
model or agent. It seals code, the exact input task, native dependencies,
models, source manifests and environment. The source selection and model manifest
must also match the original G journal's frozen checkpoint and artifact metadata;
this read-only check never opens Benchmark metrics or predictions. Host runtime
binding retains the invoked virtualenv executable, prefix/base prefix, dependency
versions, module origins and installed source/native file hashes. Credentials are checked for presence
only and are never read or hashed. Record the controlling orchestration runtime
separately from the frozen policy runtime in the environment manifest.

H uses a new private journal and a cold retrieval cache. Mining remains the
original3642 real single-label rows; Proxy74 and Benchmark271 remain unchanged.
Identity, original bytes and normalized physical joint exclusions execute before
retrieval. No generated references or historical cache reuse is accepted. The
fresh embedding/mining cost is included in H's measured allocation timeline.

## Run, resume and observe

From the application directory, run these on the execution host:

```bash
python scripts/icl_addendum.py preflight --task /private/h-task.json --output /private/h-preflight.json
python scripts/icl_addendum.py run --preflight /private/h-preflight.json --approval /private/h-approval.json
python scripts/icl_addendum.py status --results-dir /runs/icl-addendum-h
python scripts/icl_addendum.py export --results-dir /runs/icl-addendum-h
```

Approval contains `preflight_sha256`, `approved_by` and `approved_at` and binds
this exact effective preflight. Approval is an explicit technical gate, never
inferred from preparation evidence. The run/export lock excludes another writer.
Status reads the journal and returns actual attempt handles for native backend
observation. The same run command resumes; use `--resume-origin autonomous-recovery`
only for an actual autonomous restart. Artifact/code/environment drift blocks
resume. Deadlines, usage, native handles and completed answers persist; no budget
reset or repair of a wrong, INVALID or length-limited completed answer is allowed.
Infrastructure-only recovery remains bounded to two attempts per stage. H first
reconciles any already-issued proposal against its original native handle and
deadline, even if the next-trial admission reserve expired during a restart.
It records actual usage and terminal state without launching another proposal;
unknown usage or no final at budget exhaustion stops honestly. This is execution
accounting around the unchanged E search, not a new scientific decision rule.
Previously issued native inference/retrieval work is likewise reconciled before
admission, with original deadlines, costs and raw answers retained. A completed
job discovered after a supervisor outage is admissible only if native FinishedAt
is within its original deadline. Completed Proxy trial summaries are recovered
with the exact E fields; over-deadline work cannot become a completed candidate.
Freeze rejects every pending/running attempt. Deterministic program failures
remain terminal on later resumes.

The unchanged E search proposes one adaptive trial at a time, maximum12 complete
trials;21600s wall;1000000 cumulative policy-agent tokens;1400s admission reserve;
180s proposal wall;maximum4 references;one classification call/query. The agent
may stop. Highest MCC wins; within0.01 prefer fewer images, calls, prompt
characters, then trial ID. A full frozen H policy/model manifest is required
before the sole `H/benchmark/final` candidate. Final completion requires all271
identities and immutable answers. No training, generation, baseline rerun or
postfreeze search is admitted.

## Trusted export and limits

Only after H final evaluation completes does export open read-only G/E/F
journals and their sealed final metrics. The original external evaluator and
paired joint-percentile bootstrap are reused:255 groups,seed42,2000resamples,
95%CI. H−G is primary; H−E/H−F are secondary. Report MCC,macro-F1,balanced
accuracy,per-class results/confusion and raw prediction provenance. Synthetic
CPU fixtures cover this boundary; development must not execute export on real
historical Benchmark results.

Exports are immutable `exports/report-*.json` snapshots with a local
`latest-report.json` pointer. They preserve policy/inference usage, attempts,
native launch commands and GPU allocation intervals. Setup/controller usage
comes from separate control-role exports; unknown costs and USD stay null.
Inherited G training cost is disclosed separately from incremental H cost.
Generator training overlap with Benchmark remains unknown; the original two
Mining/Benchmark joint overlaps remain disclosed. These are engineering
comparisons, not proof of a leakage-free causal gain. Do not claim completion
for missing answers or turn a failed/incomplete experiment into a result.
