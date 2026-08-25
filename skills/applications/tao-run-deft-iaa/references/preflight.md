# Pre-flight and Initialization

Use this reference for a new run or to resolve an existing run. Discovery is
read-only. Installation, pulls, probes that start containers, extraction, and
run creation occur only after approval.

## Contents

- [Initial intake](#initial-intake)
- [Read-only discovery](#read-only-discovery)
- [Defaults](#defaults)
- [Approval summary](#approval-summary)
- [Approved initialization](#approved-initialization)
- [Resume](#resume)

## Initial intake

IAA supports Docker (a local daemon or an approved remote `DOCKER_HOST`),
SLURM, Kubernetes, Brev, and virtualenv as compute platforms. Airflow is an
optional IAA-only orchestrator over any of them. If the user did not select a
compute platform, include that single choice with the other missing intake;
never default to Docker. Ask about Airflow only when the user requested it.
When Docker is selected, resolve whether its compute frame
is local or remote before approval and persist that choice; never infer local
filesystem semantics merely because the platform name is `docker`. Before full
preflight, do only bounded, lightweight discovery that can reduce user
questions:

1. Resolve an explicitly named workspace or the conventional `~/workspace`
   candidate without creating it.
2. Resolve the archive root using this exact precedence and scope:

   - If the user supplied both archive file paths, require the exact names
     `images_raw.tar` and `meta.tar.gz`, a shared non-symlink parent, and two
     non-symlink regular files. Validate only that explicit pair's parent
     location during intake. If it is invalid, report why and ask for a
     corrected pair; do not search for alternatives.
   - If the user supplied an archive directory, treat it as the sole approved
     search root. Inspect that root and directories at candidate depth `1` or
     `2` beneath it. Do not search outside it.
   - Otherwise use, in order, `~/iaa` at candidate depth `0..2`, the resolved
     workspace root at depth `0` only, then `<workspace>/iaa`,
     `<workspace>/input`, and `<workspace>/inputs` at depth `0..2`. Normalize
     absolute paths and deduplicate overlapping search-root/depth pairs and
     discovered candidate directories.
   - A candidate is a non-symlink directory whose direct children include both
     `images_raw.tar` and `meta.tar.gz` as non-symlink regular files. Use a
     non-symlink-following, depth-limited directory enumeration inside each
     search root, then check the two exact child names; do not run a filename
     search over `$HOME`. Missing roots are reported as checked/missing.
     Permission errors and unsafe symlinks are reported and never cause the
     search to widen.
   - Do not add the current repository, development checkouts, or
     `<workspace>/data` as implicit search roots. If the user explicitly names
     one of those locations, its bounded subtree is user-authorized and may be
     checked.
   - If no candidate exists in that bounded set, ask for the archive root. Do
     not broaden the search automatically.

   Candidate depth counts directories relative to the search root: the root is
   depth `0`, its direct directory children are depth `1`, and their directory
   children are depth `2`. Do not follow a symlink supplied as a search root or
   encountered below one. This mirrors AOI's useful discovery pattern: recurse
   only below a semantically selected subtree, with an explicit IAA depth cap.

   The archive root is an input location. It is distinct from `DATASET_ROOT`,
   which is the approved extraction/rebuild destination under workspace data.
   The run consumes one archive pair from one root; multiple roots are
   alternatives, not multiple required datasets.

   Classify a candidate from direct-child metadata only:

   - `archive-only`: the required pair is present and extracted-dataset markers
     such as `images_raw/`, `images/`, `captions/`, `rebuild.py`, or
     `train_pairs.json` are absent. Auto-select it only when it is the sole
     candidate.
   - `archive-with-extracted-data`: the pair is present alongside one or more
     extracted-dataset markers. Describe it as a mixed/development location
     and require an explicit user choice unless the user supplied it.

   Record file sizes, but do not open, hash, compare, or validate either large
   archive yet.
3. Enumerate only `<workspace>/results/run_*/deft_state.json`. For each small
   state file, read only enough JSON to identify `workflow`, `schema_version`,
   `results_dir`, `current_iteration`, and `max_iterations`. Present it as a
   resume option only when `workflow` is exactly `tao-run-deft-iaa`. Ignore and
   briefly label malformed, unidentified, AOI, or other-workflow states; do not
   offer them as IAA runs. Do not run the full audit yet.
4. Report discovery provenance before asking for missing intake. Use absolute
   paths and this compact shape, omitting sections that do not apply:

   ```text
   IAA DEFT intake (read-only)
     workspace: <absolute path> (source=<user | conventional ~/workspace>)
     archive lookup:
       search root: <absolute path> (reason=<user-supplied | conventional ~/iaa | workspace root | workspace child>; candidate_depth=<0 | 0..2>; status=<searched | missing | inaccessible | unsafe-symlink>)
       broad home/repository search: not performed
       candidate: <absolute archive root> (found_under=<search root>; depth=<0..2>; type=<archive-only | archive-with-extracted-data>)
         images: <absolute path>/images_raw.tar (<size>)
         metadata: <absolute path>/meta.tar.gz (<size>)
       selection: <selected sole archive-only root | user choice required | none found>
     resume lookup:
       checked: <workspace>/results/run_*/deft_state.json
       IAA candidates: <none | concise list with iteration/budget>
       ignored: <none | concise path and non-IAA/unreadable reason>
   ```

   If more than one valid archive root remains, say explicitly that they are
   alternatives and ask the user to select one. Never label a list of directory
   names as "dataset archives"; name the archive root and the two files under
   it. Paths found this way come from the current filesystem and must be labeled
   `source=discovery` with the search-root reason, scope, and candidate depth,
   not implied to come from session memory or packaged defaults.
   Never resolve ambiguity by search order, directory name, modification time,
   or archive size.

If `max_iterations` or a time budget is absent, stop here and ask one
consolidated question for that value plus any ambiguous archive/run selection.
State the defaults below and explain that full read-only discovery and the one
approval summary come next. Do not ask whether the user wants a KPI target,
authenticated Hugging Face access, or other overrides; their documented
defaults apply unless the prompt already provides an override.

## Read-only discovery

Run this section only after required intake is resolved.

1. Resolve absolute paths for `WORKSPACE`, the two archives, and the intended
   dataset root. IAA workflow code and templates are bundled under
   `SKILL_ROOT`; never ask the user for an implementation checkout.
   `RESULTS_DIR` must be a child of `WORKSPACE`; `DATASET_ROOT` must be nested
   below a workspace data directory (for example
   `$WORKSPACE/data/iaa_v31_tao_ft`), not directly below `WORKSPACE`. Neither
   may contain the other, and approved paths may not traverse symlinks.
2. Resolve either one identity-filtered IAA run directory from initial intake,
   or a new path such as `<workspace>/results/run_<UTC timestamp>`. Never select
   among multiple IAA runs by guessing; summarize the candidates or ask once.
   Never send a state from another workflow to the IAA audit as a resume
   candidate.
3. Verify `images_raw.tar` and `meta.tar.gz` are regular, non-empty, readable
   archives. Check them to end without printing their member lists:

   ```bash
   set -e
   tar -tf "$IMAGES_ARCHIVE" >/dev/null
   tar -tzf "$METADATA_ARCHIVE" >/dev/null
   ```

   Record the presence of adjacent `SHA256SUMS`. Its absence is a warning, not
   a blocker; verification by the skill's bundled dataset rebuild remains
   mandatory.
4. Read the selected compute-platform skill and run its read-only access
   preflight. When Airflow orchestration was requested, additionally read
   `airflow-execution.md` and run `airflow_action.py preflight`; do not look for
   or create a global Airflow platform skill. For Docker, SLURM, Kubernetes,
   and Brev, also run the shared checker with
   the exact platform, pinned image, and approved GPU requirements. Do not pass
   local archive paths to a remote platform check; validate their staged
   compute paths after the approved staging step. Do not use
   `--skip-platform-access` for launch readiness. The shared checker's Docker
   image-specific CUDA framework probes are side-effecting, so defer those
   invocations until approval.
   Virtualenv uses its platform skill's venv/import/CLI checks instead of the
   container-oriented shared checker. Platform-native checks are:

   ```bash
   python3 "$BANK/scripts/check_tao_launch_preflight.py" \
     --platform "$PLATFORM" \
     --container-image nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt \
     --gpu-min-count "$NUM_GPUS"
   ```

   Docker inspects the selected local or `DOCKER_HOST` daemon, images, GPU, and
   paths from that daemon's compute frame; SLURM checks SSH, scheduler, Lustre,
   Pyxis/Enroot, and scheduler-declared allocation shape; Kubernetes checks
   context, namespace/PVC, and allocatable GPUs; Brev checks its CLI/API and
   target-instance reachability. Remote Docker/GPU compatibility is not
   inferred from the launcher's inventory: the approved compute-frame probe in
   step 2 verifies it. Virtualenv checks the venv interpreter, CUDA, and that
   `clip`, `embedding`, and `tmm` resolve under `<venv>/bin`. Optional Airflow
   orchestration separately validates TLS/authentication, the exact unpaused
   versioned DAG contract, shared evidence storage, one coordinator pool, and
   selected-backend access from the worker. Missing
   system/native prerequisites are blockers, not reasons to choose another
   platform. Image acquisition and CUDA jobs remain planned actions until
   approval.
5. Check only whether credentials required by the selected platform and model
   exist in the current process environment. When the user approves an env
   file under the repository credential contract, source it only in the same
   shell as the consuming check or command; never print, grep, copy, or inspect
   its contents and never echo a value. This can include `NGC_KEY`, `HF_TOKEN`,
   `BREV_API_TOKEN`, SLURM connection variables, Kubernetes context variables,
   Airflow API authentication variables, or tier-C storage variables. `NGC_KEY` is required only if the selected
   platform must acquire an image. The default SigLIP2 model is public, so
   `HF_TOKEN` is optional unless authenticated access is required.
   Include every pinned prebuilt component and serving image from
   `sdg_config.yaml` in the selected platform's availability plan; follow
   `local-sdg.md` for per-role VRAM, ports, disk, endpoint ownership, and API
   compatibility. Missing images are planned acquisitions. Customers never
   build workflow component images. Do not acquire images or run a CUDA probe
   before approval.
   Do not inspect credential-file metadata when neither variable is required.
   If the user explicitly asks for a permissions check, `stat` only that named
   file and warn about group/other readability.
6. Record the complete platform-visible GPU ID list as `VISIBLE_GPU_IDS`, then
   resolve the approved subset or allocation as `GPU_IDS`/`NUM_GPUS`.
   SigLIP2-so400m training commonly needs
   roughly 30–45 GB free per selected GPU at the bundled batch size. Treat this
   as a planning estimate, not a capability guarantee. Surface occupied GPUs;
   do not silently reshape `gpu_ids`.
   Managed generation is the default. Also require explicit non-empty GPU lists for image
   edit, VLM, and LLM. Account for aggregate VRAM when roles share a GPU. For
   external endpoints, proceed only when the user explicitly asked to reuse
   endpoints and supplied all three URLs. Record that request with
   `--reuse-external-endpoints`, validate the URLs, and do not claim, inspect,
   or control their GPU allocation. Never probe ports to infer or suggest reuse.
7. Resolve all run values and their sources. Validate the metric contract
   vocabulary against `references/metric-contract.md`. Do not create a config
   to discover defaults; read the bundled templates.
8. For resume, use the bundled IAA runtime with the prior workspace venv. The
   audit verifies the bundled runtime hash recorded when the run was created:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RESULTS_DIR"
   ```

   Report the audit status and next action. Do not change immutable values in
   an existing run. If the dependency environment or matching skill version is
   unavailable, report the resume audit as blocked during discovery. Restore
   the same skill version/runtime rather than asking for a separate source
   checkout, and do not launch GPU work before the audit.

Ask one consolidated question for unresolved required inputs. `max_iterations`
is required unless a time budget was supplied. Estimate conservatively:
baseline setup and pool embedding can take tens of minutes, and one 8-GPU
iteration is commonly about 30 minutes at the bundled 10k/1-epoch settings;
hardware, pool size, and accumulated data can change this substantially.

## Defaults

| Field | Default |
|---|---|
| metric | `Rank-1`, query type `medium`, operator `>=`, no target |
| training epochs | `1` per iteration |
| GPU shape | selected `num_gpus=1`, `gpu_ids=0`; visible IDs recorded from host preflight |
| mining | budget `10000`, top-N `25`, cosine distance |
| history selection | enabled, replay fraction `0.20` |
| continual behavior | dataset `true`, model `true` |
| visualization | contact sheets `true`, embedding plot `true` |
| Hugging Face token forwarding | disabled; enable only when the approved model/environment requires it |
| PyTorch image | `nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt` | <!-- versions-key: images.tao_toolkit.pyt -->
| data-services image | `nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services` | <!-- versions-key: images.tao_toolkit.data_services -->
| endpoint mode | managed platform-local services; external reuse only when explicitly requested with three user-supplied URLs |
| generation nodes | `1`; distributed platforms may approve more independent eight-GPU workers |
| generation models | pinned role defaults from `sdg_config.yaml` |
| generation budget | `1000` source people per iteration |
| verification attempts | `2` per source; approved range `1..5` |
| generated caption policy | `all` (`easy`, `medium`, and `hard`) |
| monitoring | attached, poll every `5 minutes` |

An ungated run evaluates every allowed iteration and completes with
`max_iterations`. Never invent a KPI target.

## Approval summary

Show this one compact summary and stop for explicit approval. It is the sole
approval boundary for the planned run; do not split platform, image, setup, or
launch confirmation into separate prompts.

```text
IAA DEFT pre-flight

Workflow
  skill: tao-run-deft-iaa 0.6.0 (source=SKILL.md frontmatter)
  compute platform: <docker | slurm | kubernetes | brev | virtualenv> (source=<user | resume state>)
  orchestrator: <direct | airflow> (source=<user | default | resume state>)
  Docker endpoint: <local daemon | remote DOCKER_HOST | n/a> (source=<user | environment | resume state>)
  workspace: <absolute path> (source=<user | default>)
  monitoring: attached=true; interval=5 minutes (source=default)

Run
  results: <absolute path> (new | resume; source=<user | discovery | default>)
  stop: metric=<value> (source=<user | template | default>);
        query=<value> (source=<user | template | default>);
        operator=<value> (source=<user | template | default>);
        target=<value | no target> (source=<user | default>);
        max_iterations=<N> (source=<user | derived from approved time budget>)
  train: epochs=<N> (source=<user | template | default>);
         visible_gpus=<count>, visible_gpu_ids=<list> (source=host preflight);
         num_gpus=<N> (source=<user | default>);
         gpu_ids=<selected list> (source=<user | default>)
  mining: budget=<N> (source=<user | template | default>);
          topn=<N> (source=<user | template | default>);
          metric=<name> (source=<user | template | default>);
          history=<bool> (source=<user | template | default>);
          replay=<f> (source=<user | template | default>)
  continual: dataset=<bool> (source=<user | template | default>);
             model=<bool> (source=<user | template | default>)
  visualization: sheets=<bool> (source=<user | template | default>);
                 embeddings=<bool> (source=<user | template | default>)

Inputs
  archive root: <absolute path> (type=<archive-only | archive-with-extracted-data>;
                source=<user | discovery: checked-root reason>)
  images archive: <absolute path and size> (source=<user | discovery: checked-root reason>)
  metadata archive: <absolute path and size> (source=<user | discovery: checked-root reason>)
  SHA256SUMS: <absolute path | absent> (source=<user | discovery>)
  dataset root: <absolute intended path> (source=<user | default>)
  IAA runtime: bundled with skill (source=fixed by workflow);
               integrity=<verified | mismatch>
  credentials: <selected platform variables with set/not-needed/missing only>;
               NGC_KEY=<set | not needed | missing>;
               HF_TOKEN=<set | optional/unset | missing>
  token forwarding: requires_hf_token=<bool> (source=<user | default>)
  PyTorch image: nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt  # versions-key: images.tao_toolkit.pyt
                 (source=versions.yaml; status=<available | acquire after approval>)
  data-services image: nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services  # versions-key: images.tao_toolkit.data_services
                       (source=versions.yaml; status=<available | acquire after approval>)
  TAO SigLIP cache: google/siglip2-so400m-patch16-256;
                    integrity=<verified manifest SHA | acquire/stage after approval>
  virtualenv profiles: pyt=<absolute path | n/a>; ds=<absolute path | n/a>;
                       ABI/packages/entrypoints/imports/pip/CUDA=<pass | fail | n/a>
  control environment: <absolute path>; distinct from execution profiles=<true | false>
  storage/staging: tier=<A | B | C>; compute targets=<resolved platform paths>
  GPUs: <selected IDs or platform allocation> (source=<user | default>);
        memory=<free/total or platform inventory>
  Airflow: base_url=<credential-free origin | n/a>; dag=<id | n/a>;
           contract=<tao-deft-iaa-action-v1 | n/a>; paused=<false | n/a>;
           storage_scope=<shared evidence scope | n/a>; coordinator_pool=<summary | n/a>;
           backend_consumer=<exact script/digest | n/a>

Generation
  execution frame: selected platform for every workload (source=fixed by workflow);
  generation nodes: <up to N independent 8-GPU workers for distributed platforms;
                     1 for Docker/virtualenv>;
  generation slots: <8*N maximum distributed | explicit local image-edit GPU count>;
  endpoint mode: <managed | external> (source=<user | default>)
  image edit: <model@revision>; endpoint=<port | URL>; gpu_ids=<list | user-managed>
  VLM: <model@revision>; endpoint=<port | URL>; gpu_ids=<list | user-managed>
  LLM: <model@revision>; endpoint=<port | URL>; gpu_ids=<list | user-managed>
  budget: <sources/iteration>; verification attempts=<1..5>;
          caption policy=<all | easy | medium | hard>
  component images: <pinned prebuilt images with local/pull status>
  serving images: <pinned images with local/pull status>
  lifecycle: reuse matching run-owned containers; preserve on failure;
             never mutate user-managed endpoints; cleanup only when explicit

Planned writes/actions
  <platform image/venv preparation>; image-specific CUDA framework/CLI probes;
  <runtime venv install if needed>;
  config/state creation; archive extraction/rebuild; approved endpoint startup
  or validation; generated-data mutation; staged action submits; baseline and
  at most N iterations

Estimate: <baseline + bounded-loop estimate and assumptions>
```

Label every configurable parameter source, including defaulted parameters; do
not limit source labels to overrides. Use `user`, `template`, `default`,
`discovery: <checked-root reason>`, `fixed by workflow`, `versions.yaml`, or
`derived from approved time budget` as applicable. If the user changes a row
after approval, show only the changed rows and wait for approval again.

## Approved initialization

For a new run, perform the following in order.

Before dataset extraction or any submitted action, validate the exact TAO
SigLIP cache on every compute frame that will stage actions:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" cache-preflight \
  --cache-dir "$WORKSPACE/cache"
```

The helper hashes the same bounded cache subset later signed into each action.
If it is absent, acquire the public
`google/siglip2-so400m-patch16-256` snapshot at revision
`e8708ab72d125807e45b36fb7d4e0aacbb59f379` after approval, or copy that
exact verified cache from an approved shared cache, then rerun this check.
For SLURM, Brev, and Kubernetes this must pass on the staged compute frame—not
only on the controller—before dataset setup begins. Never defer discovery to
`pool_embed` or `target_embed`.

1. Follow the selected platform skill's approved image/runtime acquisition.
   Docker and Brev acquire the pinned images through Docker; SLURM converts and
   caches both images as SQSH before allocating GPUs; Kubernetes makes both
   images pullable by the namespace. Virtualenv uses two immutable execution
   profiles: `pyt` for `clip` train/evaluate and `ds` for `embedding`/`tmm`.
   They may resolve to the same directory only when that one environment
   independently satisfies both contracts. The checked-in
   `virtualenv-runtime-manifest.json` binds CPython 3.12, Linux x86_64/glibc,
   exact TAO/PyTorch distributions, console-script ownership, imports, and
   CUDA 13.0. Inspect acquisition readiness without mutation:

   ```bash
   python3 "$SKILL_ROOT/scripts/manage_iaa_virtualenv.py" plan \
     --profile pyt --virtualenv "$IAA_PYT_VIRTUALENV"
   python3 "$SKILL_ROOT/scripts/manage_iaa_virtualenv.py" plan \
     --profile ds --virtualenv "$IAA_DS_VIRTUALENV"
   ```

   A supplied prebuilt profile is verified directly against the manifest's
   exact ABI, resolved distribution/version set, console-script metadata,
   imports, dependency consistency, CUDA build, and real GPU behavior. That
   read-only verification does not depend on an installation lock and never
   mutates the supplied environment. Creating a new profile is different: it
   is allowed only from the same complete, reviewed, hash-locked transitive
   requirements file. The packaged combined lock selects one HTTPS artifact
   and SHA-256 for every transitive dependency for CPython 3.12/Linux x86_64;
   its three source-only public dependencies are content-hashed and built by
   pip at the approved final path. Both profiles may use the resulting shared
   environment because verification independently enforces each profile's
   distribution, entrypoint, import, CUDA, and dependency contract. Do not
   install unpinned packages, infer hashes, use a source checkout as runtime,
   or weaken the acquisition boundary. Never expose credentials or silently
   substitute a platform.

2. Run an image-specific CUDA framework smoke using the exact approved resource
   shape. GPU enumeration or `nvidia-smi` inside a container is not sufficient:
   it can succeed when the image's PyTorch/CUDA build cannot initialize against
   the host driver. Bind-mount and execute
   `scripts/check_iaa_cuda_runtime.py` inside every pinned runtime that the
   approved run will use. Require `clip` for the PyTorch image and require both
   `embedding` and `tmm` for the data-services image:

   ```bash
   python3 /probe/check_iaa_cuda_runtime.py \
     --min-gpus "$NUM_GPUS" --require-cli clip
   python3 /probe/check_iaa_cuda_runtime.py \
     --min-gpus "$NUM_GPUS" --require-cli embedding --require-cli tmm
   ```

   The selected platform consumer owns the surrounding Docker, `srun`, pod, or
   Brev command and must allocate the approved GPUs to the probe exactly as it
   will to the action. A bounded verification that uses only one image may
   probe only that image; a normal DEFT loop must probe both. For virtualenv,
   perform the complete profile verification, including the same real probe:

   For Docker-backed platforms, use `scripts/run_iaa_cuda_gate.py` to invoke
   the probe. It first tests the pinned image normally. Only when the output
   contains both the NVIDIA driver-insufficient diagnostic and the failed
   PyTorch CUDA availability check may it verify and retry with the image's
   `/usr/local/cuda/compat/lib.real` bundle. A missing bundle, an unrelated CUDA
   error, or a failed compatibility retry is terminal. Write each passing
   receipt as `config/cuda-runtime-<image-kind>.json`; the signed action producer
   binds the receipt to the exact image and GPU IDs and forwards only the
   allowlisted compatibility loader path when required. The receipt records the
   selected mode/path, never inherited environment or credential values.

   ```bash
   python3 "$SKILL_ROOT/scripts/manage_iaa_virtualenv.py" verify \
     --profile pyt --virtualenv "$IAA_PYT_VIRTUALENV" \
     --min-gpus "$NUM_GPUS" --gpu-ids "$GPU_IDS"
   python3 "$SKILL_ROOT/scripts/manage_iaa_virtualenv.py" verify \
     --profile ds --virtualenv "$IAA_DS_VIRTUALENV" \
     --min-gpus "$NUM_GPUS" --gpu-ids "$GPU_IDS"
   ```

   The verifier rejects a fake executable unless pinned distribution metadata
   owns the exact console script, and checks the Python/platform ABI, package
   versions, action imports, `pip check`, PyTorch CUDA build, and then tensor
   allocation. Virtualenv verification binds the approved host selection via
   `CUDA_VISIBLE_DEVICES`, then the probe must allocate
   and synchronize a CUDA tensor on every requested visible device. Any
   framework initialization failure, missing TAO CLI entrypoint, insufficient
   visible GPU count, or unsupported architecture is a hard stop before state,
   data staging, or action submission.

   If verification of an already approved profile fails only because a
   packaged action import is absent or has the wrong locked version, do not run
   an ad-hoc `pip install`. After the launch review authorizes package mutation,
   synchronize that exact profile to the manifest-bound combined lock and
   verify it again:

   ```bash
   python3 "$SKILL_ROOT/scripts/manage_iaa_virtualenv.py" repair \
     --profile ds --virtualenv "$IAA_DS_VIRTUALENV" --approve-repair
   ```

   Use the failing profile (`pyt` or `ds`). The repair preserves the existing
   environment on failure, installs only hash-bound artifacts from the
   packaged lock with dependency resolution disabled, and passes only after
   the full profile import, metadata, `pip check`, and initialization probes.
   Do not use it for an ABI, CUDA, GPU-count, unsupported-architecture, unknown
   environment, or unrelated runtime failure.
3. Reuse a complete **control** workspace venv if the bundled-runtime import
   passes. This is separate from the two TAO execution profiles above and may
   use CPU PyTorch because it runs host-side analysis, not TAO actions. Otherwise
   create it and install only the bundled runtime's third-party dependencies:

   ```bash
   python3 -m venv "$WORKSPACE/.venv"
   "$WORKSPACE/.venv/bin/pip" install \
     pandas numpy matplotlib pyarrow pillow pyyaml scikit-learn
   "$WORKSPACE/.venv/bin/pip" install torch \
     --index-url https://download.pytorch.org/whl/cpu
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     -c 'import iaa_deft; print("bundled IAA runtime: OK")'
   ```

   Never install into the system interpreter. If package installation was not
   in the approved actions, obtain approval first.
   The control interpreter used by `run_deft_action.py` must also import
   `jsonschema`; install that small helper in an approved non-system
   environment if no existing Python provides it.
4. Materialize an immutable run config and `approval.json` from the bundled
   templates. Pass every approved override explicitly; this command writes
   only `${RESULTS_DIR}/config/` and refuses an initialized run:

   ```bash
   PREP_OPTIONAL_ARGS=()
   PLATFORM_ARGS=(--platform "$PLATFORM")
   if [ "${ORCHESTRATOR:-direct}" = airflow ]; then
     PLATFORM_ARGS+=(--orchestrator airflow)
   fi
   if [ "$PLATFORM" = docker ] && [ "${DOCKER_REMOTE:-false}" = true ]; then
     PLATFORM_ARGS+=(--docker-remote)
   fi
   if [ "$PLATFORM" = virtualenv ]; then
     PLATFORM_ARGS+=(
       --pyt-virtualenv "$IAA_PYT_VIRTUALENV"
       --ds-virtualenv "$IAA_DS_VIRTUALENV"
     )
   fi
   if [ -n "${CHECKSUMS_FILE:-}" ]; then
     PREP_OPTIONAL_ARGS+=(--checksums-file "$CHECKSUMS_FILE")
   fi
   if [ -n "${METRIC_TARGET:-}" ]; then
     PREP_OPTIONAL_ARGS+=(--metric-target "$METRIC_TARGET")
   fi
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     PREP_OPTIONAL_ARGS+=(--requires-hf-token)
   fi
   SDG_ENDPOINT_ARGS=(--sdg-endpoint-mode "$SDG_ENDPOINT_MODE")
   if [ "$SDG_ENDPOINT_MODE" = managed ]; then
     SDG_ENDPOINT_ARGS+=(
       --image-edit-gpu-ids "$IMAGE_EDIT_GPU_IDS"
       --vlm-gpu-ids "$VLM_GPU_IDS" --llm-gpu-ids "$LLM_GPU_IDS" \
       --image-edit-port "$IMAGE_EDIT_PORT" \
       --vlm-port "$VLM_PORT" --llm-port "$LLM_PORT"
     )
   else
     SDG_ENDPOINT_ARGS+=(
       --reuse-external-endpoints
       --image-edit-url "$IMAGE_EDIT_URL"
       --vlm-url "$VLM_URL" --llm-url "$LLM_URL"
     )
   fi

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/prepare_deft_config.py" \
       --workspace "$WORKSPACE" --results-dir "$RESULTS_DIR" \
       --dataset-root "$DATASET_ROOT" \
       --images-archive "$IMAGES_ARCHIVE" \
       --metadata-archive "$METADATA_ARCHIVE" \
       "${PLATFORM_ARGS[@]}" \
       "${PREP_OPTIONAL_ARGS[@]}" \
       "${SDG_ENDPOINT_ARGS[@]}" \
       --generation-nodes "$GENERATION_NODES" \
       --sdg-max-samples "$SDG_MAX_SAMPLES" \
       --sdg-verification-attempts "$SDG_VERIFICATION_ATTEMPTS" \
       --sdg-caption-policy "$SDG_CAPTION_POLICY" \
       --max-iterations "$MAX_ITERATIONS" \
       --training-epochs "$TRAINING_EPOCHS" \
       --num-gpus "$NUM_GPUS" --gpu-ids "$GPU_IDS" \
       --visible-gpu-ids "$VISIBLE_GPU_IDS" \
       --mining-topn "$MINING_TOPN" --knn-metric "$KNN_METRIC" \
       --target-query-count "$TARGET_QUERY_COUNT" \
       --history-aware "$HISTORY_AWARE" --replay-fraction "$REPLAY_FRACTION" \
       --continual-dataset "$CONTINUAL_DATASET" \
       --continual-model "$CONTINUAL_MODEL" \
       --visualize "$VISUALIZE" \
       --visualize-embeddings "$VISUALIZE_EMBEDDINGS" \
       --metric-name "$METRIC_NAME" --metric-query-type "$QUERY_TYPE" \
       --metric-op "$METRIC_OP"
   ```

5. Before state initialization, run the deterministic endpoint plan against
   the just-materialized configuration. This is a mandatory read-only gate for
   managed endpoints. For Airflow-orchestrated Docker/virtualenv, run it in the
   Airflow compute-worker frame; the packaged local service uses that same
   host. It catches
   exact port ownership, GPU/VRAM, runtime, image-presence, and cache-capacity
   conflicts while the proposed run is still safe to abandon. Do not initialize
   state and then change ports or another immutable endpoint value to recover.
   For an operator-managed remote platform, run the corresponding plan in that
   platform's compute frame; never treat the controller host as evidence.

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" plan \
       --config "$RESULTS_DIR/config/sdg_config.yaml" \
       --run-id "$(basename "$RESULTS_DIR")" \
       --platform "$PLATFORM"
   ```

   The command must exit zero before continuing. Report an occupied port as a
   conflict, not as an endpoint-reuse opportunity; never stop, replace, or
   inspect the application behind a foreign listener. With the user's approval,
   choose a different explicit port set, materialize a new proposed run, and
   repeat this gate. Write `--output "$RESULTS_DIR/endpoints/plan.json"` only
   after approval when a durable plan receipt is useful.

6. Initialize state once. Omit checksum, target, and token flags when they are
   not approved:

   ```bash
   INIT_OPTIONAL_ARGS=()
   PLATFORM_ARGS=(--platform "$PLATFORM")
   if [ "${ORCHESTRATOR:-direct}" = airflow ]; then
     PLATFORM_ARGS+=(--orchestrator airflow)
   fi
   if [ "$PLATFORM" = docker ] && [ "${DOCKER_REMOTE:-false}" = true ]; then
     PLATFORM_ARGS+=(--docker-remote)
   fi
   if [ "$PLATFORM" = virtualenv ]; then
     PLATFORM_ARGS+=(
       --pyt-virtualenv "$IAA_PYT_VIRTUALENV"
       --ds-virtualenv "$IAA_DS_VIRTUALENV"
     )
   fi
   if [ -n "${CHECKSUMS_FILE:-}" ]; then
     INIT_OPTIONAL_ARGS+=(--checksums-file "$CHECKSUMS_FILE")
   fi
   if [ -n "${METRIC_TARGET:-}" ]; then
     INIT_OPTIONAL_ARGS+=(--metric-target "$METRIC_TARGET")
   fi
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     INIT_OPTIONAL_ARGS+=(--requires-hf-token)
   fi

   IAA_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt  # versions-key: images.tao_toolkit.pyt
   IAA_DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services  # versions-key: images.tao_toolkit.data_services

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/init_deft_state.py" \
       --results-dir "$RESULTS_DIR" --workspace "$WORKSPACE" \
       --dataset-root "$DATASET_ROOT" \
       --images-archive "$IMAGES_ARCHIVE" \
       --metadata-archive "$METADATA_ARCHIVE" \
       "${INIT_OPTIONAL_ARGS[@]}" \
       --max-iterations "$MAX_ITERATIONS" \
       --metric-name "$METRIC_NAME" --metric-query-type "$QUERY_TYPE" \
       --metric-op "$METRIC_OP" \
       "${PLATFORM_ARGS[@]}" \
       --pyt-image "$IAA_PYT_IMAGE" \
       --ds-image "$IAA_DS_IMAGE" \
       --deft-config "$RESULTS_DIR/config/deft_config.yaml" \
       --sdg-config "$RESULTS_DIR/config/sdg_config.yaml" \
       --tao-spec "$RESULTS_DIR/config/tao_spec.yaml"
   ```

6. Run the audit. Its only legal first action for a new run is
   `baseline/dataset_setup`. If initialization or audit fails, do not
   reinitialize the same directory; diagnose the invalid input and start a new
   run directory.

## Resume

For an existing run, do not repeat image/config/state setup merely because a
new shell lacks variables. `deft_python.sh` takes the workspace explicitly and
the action producer reads platform, workspace, images, mounts, and config paths
from state. Run the audit, read its one stage reference, and follow `next_action`.
If the user's requested values differ from immutable state, explain the
difference and start a separately approved run rather than mutating history.
