# Efficient development operations

These are atomic Python functions/CLIs, not an admission service or supervisor.
Use only development mounts. Native editors choose and hash their own exact
preregistration **before their own diagnostics**. No historical class dominance
is supplied. R3-ext2 starts at pristine base; R4-ext starts at original R4's H000
patch plus exactly frozen P09/reference state. Six-hour clocks start at each new
track's LIGHT_START; token_target is null.

## Isolated interpreters

Build references/wsp03_efficient.Dockerfile. It inherits the frozen HF image
unchanged and copies the official vLLM Python runtime into a separate prefix:

- /workspace/.venv/bin/python: original HF Python3.13.14, torch2.10.0+cu130,
  transformers4.57.6; use for captures, edits, HF fallback, parity reference and
  FULL2917.
- /opt/wsp-vllm/bin/python: Python3.12.12, vLLM0.12.0, torch2.9.0+cu129,
  transformers4.57.3; use for fast vLLM generation.

Do not alter PYTHONHOME or LD_LIBRARY_PATH, install over either environment, or
keep models resident across sequential subprocesses. Native editors need no
Docker socket. Source/image imports and synthetic fixtures establish engineering
readiness only; actual checkpoint support and paired64 agreement require GPU.

Official support evidence:
https://docs.vllm.ai/en/v0.12.0/models/supported_models/ lists
Qwen3VLForConditionalGeneration, and
https://docs.vllm.ai/en/stable/deployment/docker/ documents vllm/vllm-openai.
The actual original config declares that architecture. This is architecture
support, not evidence of exact Cosmos checkpoint parity.

## Request and pass config

PYTHONPATH points at the application scripts. Generation consumes the existing
wsp03_worker request JSON: id, messages, references, optional policy_sha256.
Prompt text, references, class order, model, images and dev membership are inputs,
not hidden constants. Recipe.render/Recipe.request_images remain available for
frozen P00/P09 request construction. Images must be the original RGB padded256
PNGs under a narrow /input mount; do not change preprocessing.

For native light launches, the owner may set development_input to a pre-staged,
development-only PNG directory. wsp03_light.prepare mounts it read-only at
/input alongside /development. Name each original PNG with
wsp03_bindings.image_name(query_id); preserve the original bytes and use the
same mapping for queries and references. The native editor uses this prepared
view directly and does not need permission to create /input.

A fast pass config (JSON, placeholders require actual values):

~~~json
{
  "model": "/model",
  "model_files": {"config.json": "SHA256", "ALL_ORIGINAL_FILES": "SHA256"},
  "candidate": "/work/patches/C01",
  "exported_model": "/work/patches/exports/C01",
  "candidate_id": "C01",
  "requests": "/work/C01-S1-requests.json",
  "development_ids": "/context/development-ids.json",
  "image_root": "/input",
  "classes": ["EXACT_CLASS_ORDER_FROM_EVALUATOR"],
  "evaluator_directory": "/context/evaluator",
  "split": "explore",
  "stage": "S1",
  "batch_size": 32,
  "backend": "vllm",
  "max_new_tokens": 4096,
  "max_images": 5,
  "max_model_len": 16384,
  "output": "/work/C01-S1-fast",
  "track_started_at": 1234567890,
  "wall_hours": 6,
  "token_target": null
}
~~~

model_files is the full filename->SHA256 inventory, not the example placeholders.
Omit candidate/exported_model for pristine base. HF uses candidate replacements;
vLLM requires an HF export from Operations.export and verifies its replay-patch
manifest/inventory. Export once on CPU, preserving original base. Never load an
edited export and replay its patch again. backend=auto selects an installed,
registered vLLM architecture; absence is reported as availability, not lack of
checkpoint support. A runtime failure is preserved, never silently retried with
different settings. backend=hf uses left padding and batch >=32.

~~~sh
/opt/wsp-vllm/bin/python /bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/wsp03_fast.py generate /work/fast.json
~~~

Stages are PARITY64=64, S1=64, S2=512. S2 additionally requires s1_receipt,
candidate_id match and advance=true. Once at least32 new answers exist,
union(INVALID,finish_reason=length)>=80% abandons the pass. This is a union, so
overlap counts once. Every completed batch is retained; receipt.json reports
ABANDONED_DEGENERATE and retry_allowed=false. Existing output directories reject
reuse. A six-hour wall receipt includes model setup and all completed batches.
A batch may finish after another request has stopped; EOS tail padding is not
counted as output or truncation. No fast FULL2917 operation exists.

## Paired screen, parity and full HF

Use identical ordered request JSON for baseline and candidate. The root probe
uses a fixed deterministic64 dev subset; editors freeze their own selection
rules before diagnostics. No correctness diagnostics flow back from the probe.

~~~python
from wsp03_fast import paired_screen, nominate, parity64
s1 = paired_screen("S1", baseline_rows, candidate_rows, labels_for_these64,
                   frozen_evaluator, development_ids=all_dev_ids,
                   candidate_id="C01", baseline_id="original-R4-P09")
# s1 includes MCC delta, invalid/truncation rate, and mean output tokens.
# Only delta_mcc > 0 AND invalid_rate < .20 advances.
s2 = paired_screen("S2", baseline512, candidate512, labels_for_these512,
                   frozen_evaluator, development_ids=all_dev_ids,
                   candidate_id="C01", baseline_id="original-R4-P09", s1=s1)
jobs = nominate(all_complete_s2_receipts)
# <=3 nominees: MCC desc, macro-F1 desc, INVALID asc, changed BF16 elements asc, ID asc.
# Freeze that ranking before diagnostics.
~~~

CLI screen config: stage, baseline/candidate raw JSONL files, baseline_receipt/
candidate_receipt pointing to their COMPLETE pass receipts, development_labels
for exactly the paired subset, development_ids, classes, evaluator_directory,
candidate_id, baseline_id, changed_elements, optional s1_receipt, output receipt path.
CLI nominate config: s2_receipts (paths), output. Full-dev nomination does not
reuse fast predictions.

CLI parity config: fast/frozen raw JSONL paths, fast_profile/frozen_profile,
base_inventory_sha256, development_ids, classes, evaluator_directory, output.
It requires exactly64 paired IDs, identical request/rendered prompt hashes,
candidate manifest, chat template, cap and greedy settings. The reference must
be the original batch1 wsp03-worker/1 profile. >=95% means at least61/64 matches.
Lower agreement is explicitly coarse_screen_only. **Neither outcome authorizes
fast answers in the frozen HF final/full-dev scorer.**

Run unchanged HF in a dev-only subprocess:

~~~sh
/workspace/.venv/bin/python /bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/wsp03_dev_hf.py /work/hf-pass.json
~~~

hf-pass.json uses model/model_files/candidate/candidate_id/requests/
development_ids/classes/evaluator_directory/split/stage/output/
track_started_at from above. No generation overrides are exposed.
FULL2917 additionally requires nominations pointing to the top<=3 receipt.
PARITY64 requires no nomination. S2 requires s1_receipt.
The wrapper watches the original worker's append-only answers, applies the same
dev-only degeneration rule, and terminates its own child if necessary. It never
changes final HF code, cap, seed, attention, padding or retry behavior.
Outputs are receipt.json and worker/{raw_answers.jsonl,runtime_profile.json,...}.
A wall/degeneration stop preserves partial output and never restarts the pass.

## Contrast and edit configs

Capture config for wsp03_direction.py capture:

~~~json
{
  "model": "/model",
  "parent_patch": "/frozen-r4/patch",
  "development_ids": "/context/development-ids.json",
  "image_root": "/input",
  "preregistration": "/work/PREREGISTRATION.json",
  "preregistration_sha256": "HASH_FROZEN_BEFORE_OWN_DIAGNOSTICS",
  "modules": ["model.language_model.layers.18"],
  "position": "label_end",
  "pairs": [{
    "request": {"id": "dev-id", "messages": [], "references": []},
    "undesired_text": "{\"labels\":[\"CLASS_DIAGNOSED_ON_DEV\"]}",
    "desired_text": "{\"labels\":[\"CORRECT_DEV_CLASS\"]}"
  }],
  "output": "/work/contrast01"
}
~~~

Use actual module names from model.named_modules. Omit parent_patch for R3-ext2.
The /frozen-r4 example is generic; the current native launcher uses the legacy
/frozen-r3 mount name for the sanitized original R4/H000 patch.
R4-ext applies original H000 before every contrast forward. label_end captures
the residual after the actual final class token (target_end is also available).
The pre-label token cannot distinguish different continuations sharing a
prefix. Pairs are same-image/same-prompt teacher forcing with gradients disabled.
Directions are mean(undesired-desired), normalized per layer. Captures preserve
continuation tokens/positions and hashes. No generic dominant class is assumed.

Edit config for wsp03_direction.py edit:

~~~json
{
  "track": "R4-ext",
  "parent": "R4",
  "model": "/model",
  "patches": "/work/patches",
  "parent_patch": "/frozen-r4/patch",
  "parent_patch_manifest_sha256": "CANONICAL_JSON_DIGEST_OF_PARENT_MANIFEST",
  "parent_freeze_sha256": "ORIGINAL_R4_FREEZE_FILE_HASH",
  "reference_state_sha256": "FIXED_P09_STATE_FILE_HASH",
  "profile": "P09",
  "preregistration": "/work/PREREGISTRATION.json",
  "preregistration_sha256": "HASH",
  "directions": "/work/contrast01/directions.safetensors",
  "direction_mapping": {
    "model.language_model.layers.18.mlp.down_proj.weight": "model.language_model.layers.18"
  },
  "layers": ["model.language_model.layers.18.mlp.down_proj.weight"],
  "writer_mode": "absolute",
  "k": 16,
  "alpha": 1,
  "control": "direction",
  "candidate_id": "C01",
  "hypothesis": "Preregistered development-derived obstruction removal"
}
~~~

All values here are schema examples, **not** a prescribed scientific search.
R3-ext2 uses track=R3-ext2,parent=base,parent_patch=null and
parent_patch_manifest_sha256=null. Every candidate starts from that immutable
parent, never from the previous dose. Writer score is u.T@W; modes absolute,
positive or negative must be frozen. Removal is
W = W_parent - alpha * outer(u, mask_k * (u.T@W_parent)).
The committed replacement patch remains relative to pristine base and retains
every original R4 parent tensor, including edits outside down_proj.

control=random_same_norm also needs seed and random_mask_policy (own or target).
The random update matches the target float32 delta Frobenius norm, and records
actual BF16-rounded norm separately. control=single_layer requires one layer.
No gradients, optimizer or training are involved.

~~~python
from wsp03_kernel import TensorStore
from wsp03_ops import Operations
store = TensorStore("/model", "/work/patches")
export = Operations(store).export("C01")
# HF export + replay-patch + export.json; no changed final generation settings.
~~~

## Six-arm final gate

wsp03_light_final.py requires the existing R3/R4 freezes, r4_c_freeze and
r3_ext/r3_ext2/r4_ext records. Each extension record has freeze, preregistration
(pinned path/sha256 assets), export (directory containing patch),
patch_manifest_sha256 (manifest FILE hash), optional export_directory_name.
editor_containers must list R3,R4,R3-ext,R3-ext2,R4-ext and every original/recovery
container; all must be exited/dead. R4-C is frozen under the R4 editor.

R3-ext2 FREEZE additionally records parent="base".
R4-ext FREEZE records parent="R4", parent_freeze_sha256 of original R4,
profile="P09", and reference_state_sha256 of cfg.reference_state.
The original three statuses must be FROZEN. Extensions may instead finish with
immutable NO_VALID_PATCH, their preregistration hash, a nonempty reason and
candidate/stage counts. Their stopped editors still count toward the all-six
terminal gate. These arms receive explicit unavailable inference/statistics
receipts; no base substitution, aliases, scores or bootstrap draws are created.
extension_statistics is a pinned JSON asset
mapping the three extension arm names to their preregistered comparison pairs;
R4-ext minus R4 is required. This plan is verified before protected inputs.

Original results.json/statistics.json retain their original three-arm contracts.
Each extension has separate results-{arm.lower()}.json and
statistics-{arm.lower()}.json. All generation remains the original frozen HF
worker. Never invoke final evaluation during development.


## Owner paired64 probe

The owner-only wsp03_probe.py prepare CONFIG stages development inputs and
verified CPU exports. It makes no GPU calls. A later, explicit
wsp03_probe.py run CONFIG executes the two runtimes sequentially and writes
backend-probe.json with status PASS, COARSE_ONLY, or INCOMPLETE. It reports
support, parsed agreement, output quality and throughput, never dev accuracy.
The probe clock is separate from subsequent new-track clocks.

Owner config fields: output (new private directory), image (derived immutable
image ID), model, model_files, evaluator_directory, classes, recipe
{path,sha256}, reference_state {path,sha256}, backend="vllm", gpu=0,
max_model_len=16384, cases. Each case has name, profile (P00/P09), candidate_id,
candidate (host patch directory), candidate_manifest_sha256 (canonical digest).
Use the final R3-ext selection for P00 and original R4/H000 for P09.
The helper selects eight examples per class from fixed512 explore, ordered by
sha256 of the fixed seed/purpose/id tuple; both cases and backends use the same64.
Incomplete or abandoned passes remain INCOMPLETE and are not retried.

The owner probe helper's direct Docker processes write receipts as root.
Run the owner-side prepare/run orchestration in an appropriate root context, or
ensure its existing process can read those root-owned receipts. This concerns
host orchestration; native editor tools execute inside their own container.
Retain completed outputs and read them in the proper context instead of
regenerating answers to work around host file permissions.
