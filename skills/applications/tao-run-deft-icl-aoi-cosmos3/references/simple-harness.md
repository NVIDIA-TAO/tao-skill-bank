# Simple harness: Cosmos owns the tool loop

Use this route when the user asks for multi-turn Cosmos tool use, the
grid + tools + guidance flow, or an inference-only experiment without DEFT.
It does not call `suite.py`, training, generation or an Astra policy agent.
Resolve the model as usual; the native local backend loads the configured
Cosmos3-Nano checkpoint with `Qwen3VLForConditionalGeneration`.

## Division of work

| Component | Responsibility |
|---|---|
| Orchestrating agent | Prepare inputs, approve/freeze the run, submit and monitor it. |
| Cosmos | Read the query/grid, choose a tool and arguments, inspect returned images, answer when ready. |
| Deterministic runner | Validate/execute requested tools, return images, enforce round limits, save traces. |
| Independent evaluator | Join final predictions with gold after the run; no gold enters the classifier. |

Each query starts a fresh conversation with a fixed Mining atlas of 32 labeled
images (8 classes × 4), the query, native task/tool descriptions and guidance.
The image count can grow after Cosmos requests tools:

- `search_similar_examples`: C-RADIO similarity identifies two candidate
  classes per page and returns four examples for each.
- `get_class_examples`: returns more examples for a model-selected class/page.

Guidance recommends search, narrow to one or two classes, then inspect more
examples. It does not force that sequence: Cosmos may start with either tool,
page further, or answer immediately. One tool call per turn, up to ten tool
rounds, then one final-only model turn. A first non-tool answer ends the query;
INVALID stays INVALID. Do not inject Astra judgments or replace a model call
with a preselected reference sequence.

## Native implementation and scope

Read the data skill's [native contract](../../../data/tao-in-context-learning/references/simple-harness.md).
This adapter calls Maxwell's pinned `tht_ablation.Mode('A_full')`,
`tht_ablation_runtime.run_case` and `CosmosBackend` directly. It retains native
prompts, tools, image rendering and C-RADIO caches. The private source and data
are independently supplied; none is copied into this public skill bank.

Default profile: `simple-harness-grid32-tools-guidance-native4096`, greedy,
seed 0, 4,096 output tokens per turn. Optional explicitly requested
`simple-harness-grid32-tools-guidance-cap128` changes only that cap to 128.
Report them separately. Both use the screenshot-style ablation executor with
zero final-format retries, not historical R97's formatting-repair loop.
The adapter additionally blocks initial-atlas/query joint or image overlaps.
Retaining the native loop does not establish checkpoint or score parity.

This first native profile is THT-specific. It requires the original frozen
atlas, eight original labels and C-RADIO caches. A different atlas, encoder or
dataset requires a separately registered profile; do not silently substitute
SigLIP under the native name. The existing single-call SigLIP ICL path remains
available and unchanged.

## Prepare, freeze, run

1. Copy [the config template](simple-harness.example.json) outside the bank.
   Supply immutable source/model/cache inventories and a narrow dataset view:
   original Mining labels plus label-free query identities/images. See the
   data contract for each inventory format and required hashes. Do not mount
   Benchmark gold, historical predictions, reports, credentials or host home
   into the classifier. The local Cosmos backend requires no API credentials.
2. Run CPU preparation in the same container path layout intended for the
   model job:

   ```bash
   python scripts/simple_harness.py preflight --config /input/config.json \
     --output /results/preflight.json
   ```

   This validates inputs and renders the atlas, but does not load the model
   or establish GPU capacity. Run a separate development/Proxy smoke before
   a formal experiment; it must show actual Cosmos tool calls and returned
   images in later requests. Do not use a forced-tool fixture as evidence of
   model agency. Check direct-answer support and the round-limit path with
   CPU tests. Freeze model, source, atlas, cache, prompt and generation hashes
   before Benchmark. Use fresh output directories for smoke and formal runs.
3. Record the user's approval of that exact run as JSON with
   `preflight_sha256`, `approved_by`, `approved_at`. Existing explicit approval
   may apply; do not invent approval or ask again between tool turns.
4. Through the existing Docker platform `submit` verb, **open the job record
   before launch**, bind the results directory and use narrow read-only
   inputs plus writable results. Pass its ID to this in-container worker:

   ```bash
   python scripts/simple_harness.py run --preflight /results/preflight.json \
     --approval /input/approval.json --job-id "$JOB_ID"
   ```

   `--job-id` records platform provenance; it is not a substitute for mount
   isolation. Preserve compute-frame paths between preparation and execution.
   Monitor/cancel with the platform's `status` / `logs` / `cancel` verbs.
5. `python scripts/simple_harness.py status --results-dir /results/profile`
   reports artifacts, not live Docker state. Resume the same native job/run
   with unchanged bindings. Completed valid/INVALID cases are retained; an
   interrupted ambiguous case stops for reconciliation rather than free retry.
6. Score once after freeze with the existing independent evaluator. Export
   MCC, macro-F1, coverage, INVALID, tool counts/failures, model turns, tokens,
   per-query/process wall and peak GPU memory. Keep kernel time/currency null
   if unmeasured. Record reused-Benchmark/post-hoc status; never reuse a prior
   0.727 or historical 0.731 as this branch's new result.

## Progressive exploration extension

For an explicit request to search again and explore unresolved alternatives,
choose `simple-harness-grid32-explore-cap128-v1`. Copy the same config template,
set that profile and set `protocol.max_new_tokens` to 128. Use a fresh output
directory. The adapter appends the versioned TAO exploration guidance to the
native prompt and records its SHA256; the original profiles remain unchanged.

Cosmos checks whether retrieved appearances distinguish the plausible classes.
When evidence conflicts or there is no convincing match, the guidance asks it
to search an unseen page or inspect an unseen class-example page. Cosmos still
chooses every call and may stop for a clear case. The runner does not force a
second search, select candidate classes or repair a final answer. Search pages
0–3 expose different candidate-class pairs, not rewritten search queries.

Start with a separately recorded Proxy evaluation. Use
`tao-in-context-learning/scripts/simple_harness_audit.py` on completed results.
Separate multiple successful search pages/new reference IDs from duplicates
and invalid-argument retries. Compare accuracy and costs with the cap128
baseline using the existing independent evaluator. A larger call count alone
is not success. Do not claim actual exploration until traces demonstrate it.

Keep the frozen historical runs and their scores intact. This extension is
not an exact reproduction. Freeze a selected policy before any separately
authorized Benchmark evaluation; a Proxy smoke does not authorize that run.

```text
Add progressive evidence gathering to the Cosmos Simple harness.
Use simple-harness-grid32-explore-cap128-v1 with base Nano, the frozen Mining
grid and C-RADIO tools. Let Cosmos decide when unresolved alternatives need
another search page or more class examples; keep the ten-tool-round limit.
Skip DEFT, SFT and AnomalyGen. Run Proxy only and report actual new search
pages, new references, tool errors, classification metrics and resource costs.
Do not run Benchmark or replace an existing measured score.
```

## Suggested native reproduction launch prompt

```text
Run an inference-only Simple harness experiment in my workspace.
Start from base Cosmos3-Nano; skip DEFT, SFT and AnomalyGen.
Use simple-harness-grid32-tools-guidance-native4096 and the pinned native
Maxwell source, fixed 8-class × 4-image Mining grid and C-RADIO caches.
Let Cosmos choose example/search tools and continue the conversation itself.
Use Proxy only for the technical smoke. Freeze all settings before Benchmark,
evaluate each Benchmark query once, and report metrics, traces and costs.
Use the existing platform job records and GPU lease. Do not modify old runs.
```
