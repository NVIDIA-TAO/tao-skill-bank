# Separate Weight-Space Programming preparation

`scripts/wsp.py` is an additive, bounded feasibility boundary. It does not read,
resume, tune, or register an arm in `suite.py`. Preparation is not a WSP result.
The trusted preparer/evaluator and the future fresh editor have different
capabilities. No task-specific weight change or expected gain is prescribed.

The research motivation is direct, traceable weight editing without a training
loop. The [authors' article](https://rikiyat.github.io/blog/weight-space-programming/)
and [Experiment 8 contract](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
describe a different model, task, tool environment and budget. This Nano/THT
engineering experiment is not a reproduction or evidence of expected accuracy.

## CPU preparation and later approval

Run from this application directory with an existing Python environment. These
commands prepare images/manifests and inspect checkpoint **headers**, never real
tensor payloads, models, agents, GPUs, or full checkpoint copies:

```bash
python scripts/wsp.py prepare-views \
  --public /approved/public-package \
  --exclusions /private/wsp/exclusions.identity-only.json \
  --output /private/wsp/preparation/views
python scripts/wsp.py preflight \
  --model /approved/prepared-Nano \
  --views /private/wsp/preparation/views \
  --direct-profile /approved/direct-profile.json \
  --evaluator /approved/frozen-evaluator \
  --run-dir /private/wsp/scientific-run \
  --output /private/wsp/preparation/preflight.json
python scripts/wsp.py verify-preparation \
  --preflight /private/wsp/preparation/preflight.json
```

The exclusion input must be exactly two objects with only `object_name` and
`normalized_scan_id`. A trusted owner projects those fields from the declared
overlap manifest. Never supply Benchmark rows, labels or private source records.
Only the eligible `mining/samples.jsonl` and its images are read by the preparer.

Grouping uses `object_name` plus `scan_id.removeprefix('SCN_')`, verified against
the existing label-free filename grammar on every Mining row. Duplicate image
hashes join their joint groups into a connected component; excluding a joint
excludes its component. Rank eligible components by SHA256 of canonical
`{"seed":42,"joint_component":[object_name,scan_id]}`. Take the first16 for
development and the next16 for holdout, retaining every view of each selected
joint. Fail if either draw exceeds32 images; never redraw to improve coverage.
This is an unstratified small subset. Unselected eligible Mining stays unused.
Save class counts privately, including absent classes, and all membership/view
hashes. Main split membership and files are unchanged.

`preflight.json` is a concrete proposal with code/image hashes, declared base
fingerprint, header catalogue, split hashes, frozen native profile, tool prompt,
budgets, origins and unresolved gates. Its checkpoint payload hashes are
**declared preparation provenance**, not a new full-byte verification. Full
verification, independent copying and finiteness scans happen only after real
execution approval. No approved patch is prepared in advance.

## Proposed engineering envelope

| Resource | Limit |
|---|---:|
| Fresh editor | gpt-6-astra, xhigh; one isolated Codex exec session |
| Editor observed-usage threshold | 200,000 raw input+output tokens; cache included once; polling and interrupted work can overrun |
| Editor wall | 2,400 seconds |
| Entire experiment wall | 5,400 seconds, including initial verification/copy |
| Candidate attempts, including failures | 3 |
| Changed tensors / candidate | 4 |
| Changed elements / candidate | 65,536 |
| Cumulative attempted edit elements | 196,608 |
| Tensor value inspections | 64 requests; 1,024 elements each; 65,536 total |
| Development forwards | 4 total: unchanged base once and up to3 candidates once each |
| Holdout forwards | 2 total: unchanged base once and frozen edited candidate once |
| Images / request | Full fixed role view, at most32 |
| Total images | `4 * Ndevelopment + 2 * Nholdout`, sealed to actual prepared counts |
| Generated tokens | Image ceiling multiplied by4,096; reserve worst case before launch |
| One native forward request | 600 seconds including model load |
| Allocated GPU wall | 3,600 seconds reserved across at most6 requests |
| Retained storage | 80 GiB, including base, all candidates, failures and outputs |
| Editor CPU / memory | 2 CPU / 4 GiB; no GPU |
| Future native forward container | 4 CPU / 64 GiB host RAM; only released GPU0 |
| Editor writable task/results/codex/tmp | 256 / 16 / 64 / 64 MiB tmpfs |
| Editor raw log | 64 MiB; exceeding it aborts, with truncation disclosed |
| Capability requests / input bytes | 160 total / 4 MiB per request |

These values are engineering limits derived from the permitted M3 mechanical
handoff: one-image generation2.622s, peak allocated17.72GB, and prepared model
files about16.34GiB. The image arithmetic is sealed to the actual split; at the
maximum192 images it is503.424s generation alone. Six600s reservations leave
margin for startup and long outputs. Four full model copies require about65.4GiB.
Those tiny smoke measurements do not establish corpus throughput, worst-case
4096-token latency or scientific capability. No price was supplied: dollars
remain null. No retries, reclaimed reservations, candidate overwrite or cap
reset is allowed. A fresh run requires a separate review, not recovery by reset.

The proposed agent budget is an **observed raw usage threshold**, not a strict
spending ceiling or an API request admission guarantee. Count cumulative raw
`input_tokens + output_tokens` from fresh session birth, with cached input
included once. No reset, subtraction or invented total is allowed.

Diagnostic01 disproved the prior native cache-inclusive claim: completed usage
was13,880/27,904/42,046/56,306/70,684; response5 completed after response4 crossed
50k. There was no native exhaustion marker. The old process-group stop killed
capture before its final record; interrupted usage remains unknown. Preserve
that failed attempt unchanged. Native rollout configuration now remains only
secondary protection; it cannot prove raw stopping.

The WSP-only `wsp_capture.py` supervisor uses a100ms polling sleep and records
actual observation gaps. A gap over1s is an incomplete runtime failure. On the
first observed crossing it records raw event prefix/hash/usage/timestamp, sends
TERM to its separately grouped child and owned descendants, and notifies the
host through a close-only Unix control endpoint. The host closes future WSP or
fixture admissions in its own ledger, independently of an in-flight forward.
The acknowledgement has a1s deadline. Already-admitted work stays charged under
its original bounds. Request admission at the provider is not observed.

The supervisor survives external TERM/INT and host stop-file requests. It owns
child/process-group cleanup, including adopted orphan helpers, permits3s TERM
grace, then uses KILL and bounded2s reap waits. Accepted termination must finish
within7s and all cleanup must be established. Actual signal path, child return
code, supervisor return code and container return code are separate fields;
Python `-15` is never equated to shell/container241. Final raw capture follows
reap. The wrapper remains alive until trusted tar-stream harvest is complete.
Archives stay unextracted; `docker cp` is unsuitable for these writable tmpfs.

`wsp_collect.collect_native_evidence` is invoked by both real runners and
replayed by the proof validator. Evidence v2 binds the final raw capture v2,
source identity/coverage, supervised observation/termination, host closure ack,
invocation/runtime/collector hashes, final archives and independent exit codes.
Normal scientific completion requires child exit0, one native completed turn
with final usage matching raw, usage below the observed threshold, complete
local capture and a valid frozen candidate. That normal path retains trusted
holdout. No freeze stays incomplete. Any observed-token, external, hard-kill or
other stop is scientifically incomplete, even if the child gracefully exits0
or a candidate was frozen. Missing provider usage for interrupted work stays
**unknown** despite a complete local capture. Report measured completed usage,
measured overrun and any extra completed snapshots separately; never manufacture
provider totals or an implicit freeze.

Diagnostic02 is a new immutable proposal:50k observed/native target,180s observed
wall including probe/cleanup,100k observed emergency,10 host fixture admissions,
128MiB retained evidence,2CPU/4GiB and no GPU/model context/real data/weights.
The command/runtime/policy binding includes the supervisor and control channel.
The live acceptance requires cached multiple-response raw crossing, recorded
first observed stop, host closure and bounded known termination/cleanup. Extra
completed usage is charged and may fit the emergency/wall bounds; unknown or
excessive termination and cleanup failure cannot pass. Interrupted provider
usage remains unknown and this proof never establishes a strict total spend cap.
The [diagnostic driver](wsp-diagnostic-driver.md) seals the exact command and
manifest; preparation creates neither approval nor a diagnostic run directory.

The new `wsp-live-observed-usage-proof/4` is explicitly distinct from native
proof v3 and records `case: cached_multiresponse_observed_stop`. A Boolean-only
claim is insufficient. The validator replays raw source evidence and host
lifecycle hashes. Separate exact-manifest live approval, independent live review,
scientific approval and GPU release remain required. No live diagnostic or
scientific run is authorized by this engineering repair.

### Factual read-only context

`/model_context` contains architecture configuration, exact header-derived tensor
names/dtypes/shapes, tokenizer vocabulary-to-ID mapping and added/special tokens,
the chat template, processor configuration, base generation defaults and exact
frozen inference prompt/preprocessing/generation/seed. It contains no preferred
edit, class bias, hypothesis or external examples. Tokenizer merge/normalizer
implementation is omitted: this is descriptive context, not executable code.

`wsp_context.py` reads only explicit same-base JSON/text files, verifies their
existing provenance hashes and projects allowlisted inert fields. Host paths,
credentials and executable configuration hooks are excluded; removed field
names are recorded. Vocabulary strings and the approved chat template remain
literal data. The manifest binds original source hashes, derived file hashes,
header catalogue, base fingerprint and frozen profile. Preflight and runtime
reverify both the projection and sources. The entire bundle mounts read-only;
no full checkpoint or host model directory enters the editor. Access gives no
permission to change architecture, tokenizer, template, prompt, preprocessing
or generation. CPU tests and the real sandbox probe check readability,
immutability and the absence of weight/holdout/host additions.

## Weight tools and scientific invariants

The editor invokes `python3 /tools/wsp_client.py` with one JSON object on stdin.
It can page tensor metadata, read bounded flat indices, propose a candidate,
evaluate a complete development view, inspect remaining budgets and freeze once.
There is no holdout, arbitrary filesystem path, shell, model-loader, optimizer,
gradient, arbitrary forward input or generation-configuration RPC.

An editor-chosen patch is `wsp-patch/1`: original base inventory fingerprint,
testable hypothesis, and a list of exact tensor names, unique flat indices and
replacement floating values. Its JSON is the authoritative replayable program.
The editor also submits its actual Python construction source, retained by hash
as provenance; trusted code never executes that source. The fresh generic editor
may use Python to reason and construct a patch, but fitting/training loops and
gradient estimation are forbidden in its task. No torch/autograd/optimizer or
model payload is mounted into that process. This does not claim to prove from
arbitrary Python source that no conceivable fitting algorithm was written;
the actual trace must also be reviewed against that rule.

The service supports finite F64/F32/F16/BF16 assignments, including BF16
round-to-nearest/ties-to-even, without importing a tensor library. It rejects
zero changes after casting, nonfinite values/deltas/norms, duplicate indices or
tensors, wrong base lineage, out-of-bounds indices and over-budget edits. It
copies from the immutable base independently for every attempt, never a previous
candidate. A streaming audit proves that only the declared bytes changed, then
records exact tensor/element counts, L1/L2/L-infinity deltas, actual assignments,
patch/source hashes and the full result inventory. Initial execution scans all
base tensors for finiteness; each candidate scans every value in edited tensors.
Untouched bytes remain proved identical. Base and candidate copies have distinct
inodes; the shared prepared checkpoint is never writable.

A structurally valid patch is not yet model-loadability evidence. A complete
native development call establishes that it loads and returns the expected
identity coverage. The existing M3 `inference_worker.py` owns model execution:
prepared Qwen3-VL Nano in the pinned Cosmos Framework image, BF16/SDPA,
seed42, exact hash-pinned Maxwell THT prompt, RGB/NEAREST padded256 PNG,
greedy4096 generation. The external hash-pinned evaluator alone parses/scores.
Raw answers are retained before grading. INVALID and length-capped answers are
completed observations, never repaired or retried. Actual processor/template,
library, seed and generation profiles must agree between calls.

Freeze requires complete unchanged-base and chosen-candidate development calls,
a fresh candidate fingerprint verification, and a development-only rationale.
The single freeze is committed in SQLite before its JSON export and immediately
closes all editor capabilities. Only after the editor exits, its complete final
usage and normal-exit evidence pass the declared observed-usage policy and its runtime is stopped may trusted code run one base/edited
holdout comparison. A failed holdout is terminal. Its feedback never reopens the
editor or selects a second patch. No main Proxy/Benchmark feedback is consumed.

## Actual isolation and evidence

The CPU/editor sandbox uses the cached Ubuntu image, explicit `runc`, a nonroot
UID, no GPU devices, network none, private IPC, default private PID namespace,
read-only root, all capabilities dropped, no-new-privileges, and cgroup limits.
It receives only the development directory, generic interpreter/image libraries,
static Codex helpers, the factual read-only model context, a private auth-file mount when approved, the generic prompt/
tool contract and exact Unix socket files. It never receives the host home,
worktree, full environment, plugins/skills, other sessions, Docker socket, full
Mining, holdout, main results or full model files. Weight access is through the
bounded service, not a host checkpoint mount.

For a future editor, API traffic may traverse a separate Unix CONNECT relay
restricted to the frozen `chatgpt.com:443` and `api.openai.com:443` authorities.
Public-address validation pins the resolved IP; arbitrary hosts, private IPs,
other ports and ordinary HTTP are rejected. This relay's real API compatibility
is unvalidated during CPU preparation. Auth contents are never read/exported.
Negative filesystem/network/GPU checks run inside the actual sandbox before the
editor starts. A generic subprocess sandbox test additionally checks that the
wrapper stays alive to harvest bounded tmpfs after child exit.

The broker journals actual requests, results, failures and costs outside editor
mounts in transactional SQLite with a hash-chained event stream. Reservations
precede work and survive interrupted calls. The editor writes only bounded
tmpfs; its raw top-level CLI/tool events are streamed to trusted storage and its
code/results are preserved as **unextracted tar bytes** before stopping the
container. This prevents archive path/symlink extraction on the trusted host.
The protocol does not claim resistance to a compromised host/kernel or malicious
same-UID trusted operator. Native/agent runtime failures remain auditable and
are not negative scientific results if no edit/comparison was attempted.

## Private artifact/export contract

All real data, models, proposals, approvals, results and logs stay outside Git.
`wsp-export.schema.json` defines the report envelope. The external evaluator's
metric objects remain unchanged; missing measurements/prices stay null.

| Artifact | Contents / authoritative owner |
|---|---|
| preparation `views/split.json` | source/exclusion SHA, seed/grouping, IDs and view fingerprints, private class counts |
| `views/development`, `views/holdout` | preprocessed images, queries, gold and source/group provenance; only development is mounted for editor |
| `preflight.json`, approval/release/proof evidence | exact immutable run proposal and separate authorization evidence |
| `initialization*.json`, `base_manifest.json` | copy/verification attempt, full independent base fingerprint |
| `job-records/`, `jobs/`, `editor-jobs/` | record-before-launch IDs, exact argv, native status/logs, allocation intervals |
| `candidates/cNNNN/{patch.json,source.py,validation.json,model/}` | all actual patches, source, counts/norms/hashes and retained candidate bytes |
| candidate `failure.json` | rejected/failed attempt, retained partial copy if any |
| `forwards/fNNNN/`, native job output | exact requests, raw outputs, actual profiles, trusted scores, failure/partial usage |
| `freeze.json` | the one chosen model/patch/base/split/profile hash and development rationale |
| `editor_usage_events.json`, `editor_events.timestamped.log` | fresh actual token-only snapshots and observable tool/code/output trajectory |
| `editor_task.tar`, `editor_results.tar` | bounded raw archives; never auto-extract on trusted host |
| `holdout_comparison.private.json` | one base/edited comparison, raw outputs and frozen evaluator metrics |
| `ledger.private.json`, `report_data.private.json` | all reservations, measured complete/failed costs, unknown coverage, metrics/INVALID/per-class behavior, paired gains/regressions and limitations |

Cost events distinguish worst-case admission reservations from actual complete
input/output tokens and Docker allocated-GPU wall. A failed forward retains
observed completed-image usage and an unknown partial-response remainder, never
a fabricated zero. Missing GPU intervals stay unknown. The eventual exporter
runs on the execution host; no local scientific recomputation is implied.

## Required later commands and gates

Run CPU tests and the optional restricted synthetic container probe:

```bash
THT_EVALUATOR_DIR=/approved/frozen-evaluator python -m pytest -q tests/test_wsp.py tests/test_wsp_context.py tests/test_wsp_native.py
python tests/run_wsp_cpu_probe.py --output /private/wsp/new-cpu-probe
```

Before a real run, root must review the code/proposal, obtain primary GPU release,
validate the real API/runtime and cached-response token gate, and supply three
separate private records. Never manufacture them from this example:

- Approval: `preflight_sha256`, `approved_by`, `approved_at`,
  `approved_for_real_weights: true`, `actual_usage_proof_sha256`.
- Release: same proposal/reviewer/time fields, `primary_gpu_released: true`,
  exact `host`, `gpu_ids: ["0"]`.
- Live usage proof: `wsp-live-observed-usage-proof/4`,
  `runtime: isolated_codex_exec`, `synthetic: false`,
  `case: cached_multiresponse_observed_stop`, exact shared `runtime_identity`,
  `policy_sha256`, `scientific_threshold_tokens: 200000`,
  `diagnostic_threshold_tokens: 50000`, `threshold_rationale`, exact
  `diagnostic_differences`, `evidence_directory`, `lifecycle_source_hashes`, `native_evidence_sha256`
  (canonical JSON digest) and `observed_actual_usage`.

The proof validator replays the actual collector read-only from all saved
sources and requires the regenerated evidence to equal the saved document and
reviewed hash. It verifies the invocation's command, prompt hash, fixture bytes,
limits and explicit differences, plus multiple actual snapshots, nonzero cached
input, first observed raw crossing and bounded supervised termination/host
closure. Opaque success Booleans are not evidence. Source authenticity and live
API compatibility remain subjects of independent review; the parser does not
fabricate provider admission or interrupted usage.

No proof/approval/release is created by CPU tests or preparation. The scientific
runner separately checks current runtime/binary/helper identity, consumes the
reviewed live proof and produces its own native evidence before trusted holdout.
Do not run the following command during preparation:

```bash
python scripts/wsp.py run \
  --preflight /private/wsp/preparation/preflight.json \
  --approval /private/wsp/review/approval.json \
  --gpu-release /private/wsp/review/gpu-release.json \
  --usage-proof /private/wsp/review/live-actual-usage-proof.json \
  --codex-binary /approved/static/codex \
  --auth-file /approved/private/auth.json
```

The command refuses an existing run directory, altered code/views/base,
unreviewed usage proof, absent release, occupied GPU or missing cached image.
There are no automatic editor restarts. An interrupted run retains its journal
and artifacts for review; replay a patch into a separately approved destination
only when reproducing a frozen result, never to reset this trial's caps.

Remaining real-runtime checks include full checkpoint hashes/finiteness/copy
independence, selected-patch loadability, equal native profiles, actual GPU/wall
timeouts, live API relay behavior, fresh editor context/tool provenance, and
actual cached-response observed stopping and trusted final-usage collection. CPU tensor/output tests establish
none of the model's scientific performance or those unexecuted runtime claims.
