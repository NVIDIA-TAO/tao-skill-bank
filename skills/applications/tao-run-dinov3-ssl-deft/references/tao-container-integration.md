# TAO component execution

The DINOv3 SSL DEFT controller is packaged in Data Services, not Skill Bank.
Use the existing DS image, which includes TAO PyTorch/Core and the data-stack
dependencies. No new Dockerfile, host TAO/Conda installation, nested Docker,
or startup package patch is part of this deployment contract.

## Minimal single-container profile

The platform's native four-verb launcher allocates one DS container with the
reviewed GPUs, mounted input data and durable output directory. Inside it run:

```bash
python -m nvidia_tao_ds.mining.dinov3.workflow preflight --gpu
python -m nvidia_tao_ds.mining.dinov3.workflow init --recipe grit-score --output run.yaml
# Fill paths, resources and explicitly reviewed scoring backend, then:
python -m nvidia_tao_ds.mining.dinov3.workflow validate run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow plan run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow run run.yaml
```

Set `execution.backend: local` and omit `container_images` for this profile.
The controller's durable process runner invokes DS mining and PyTorch scoring/
training as subprocesses in the same already-allocated runtime. It does not
schedule resources or start containers. Resource requests must fit the inherited
allocation exactly for GPU-consuming stages; multi-node runner requests are
rejected. Packaged recipes start with one GPU and explicit `torch_exact` CUDA
scoring, avoiding an additional GPU FAISS dependency. Increase allocations only
with matching stage resource settings. The controller and its
packaged recipes do not need Skill Bank mounted or installed.

Installed TAO packages do not imply installed DEFT support. The approved
7.2.0-rc-36 DS image contains DINOv3 training but predates the DEFT additions.
Use a normal DS release containing the reviewed DS/PyTorch/Core changes; fail
preflight on missing modules. Source overlays are development QA only.
The rc-36 DS runtime also fails mixed-precision convolution and backward probes
on the tested RTX A6000/580.173.02 host, even without DEFT source. Do not treat
installed TAO packages or CUDA tensor allocation as proof of training compatibility.
Until the image/runtime combination passes preflight, use the separate-image
profile below with the validated PyTorch image for scoring and training.
GPU FAISS availability is a separate scoring-backend check, not established by
the CUDA convolution forward/backward preflight. Select `torch_exact` explicitly where appropriate;
never silently substitute a backend.

## External or separate-image profile

For distributed or separate-image jobs, preserve the existing external runner:

The DS controller may be CPU-only. Run its `workflow preflight` without
`--gpu` to check the installed modules; it still needs matching PyTorch sources
for implementation sealing. Run GPU preflight/probes through the selected
platform in each actual GPU compute image, including scoring-backend checks.
Do not require DS-controller convolution when only the PyTorch leaves use GPUs.

```yaml
execution:
  backend: external
  runner_command: ["/path/to/approved-platform-runner"]
  container_images:
    pytorch: tao_toolkit.pyt
    data_services: tao_toolkit.data_services
  capabilities:
    containers: true
```

The keys resolve from `versions.yaml` before config locking and appear in the
approval plan. Explicit image references are accepted as reviewed overrides.
Symbolic keys require `execution.skill_bank` or `TAO_SKILL_BANK_PATH`; explicit
image references need no bank. Pin deployment
images by digest after image preflight; tags alone cannot detect registry retags.

Scoring and training use PyTorch; selection, retrieval, and materialization use
Data Services. `actions.<name>.container_image` overrides that action's image.
Audited search supports independent `candidate.container_image` and
`rerank.container_image`. Custom evaluators must declare their own image.
The original base-checkpoint-each-round policy is unchanged.

Each StageRequest carries the resolved image in
`execution_contract.container_image`. The platform consumer must run the command
inside that image with the approved mounts/environment/resources and return
`container_image` in its terminal status, derived from its backend inspection
and recorded image mapping, not blindly echoed from the request. Completion with
a missing/different image is rejected. LocalRunner rejects container requests.
LocalRunner is only for processes in its current allocation, never container requests.

Slurm maps this to Pyxis/Enroot through the existing platform skill. Import the
reviewed images to cached Lustre squashfs files on CPU before GPU allocation.
Record the registry-image-to-squashfs identity; report the approved image identity
in status. Preserve node-level TAO rendezvous, one launcher per node, full-gang
retry/attempt IDs, shared paths, and the declared per-stage GPU allocation.
Docker uses the same component commands through the Docker platform consumer.
This workflow does not ship or replace a platform runner.

Controller-side implementation sealing currently needs readable approved source
files through its declared Python search paths. Those files are orchestration
inputs, not authorization to mount/override model packages in the containers.
The leaf's own entrypoint and implementation digests must match the locked
source. A container without the reviewed branch implementation fails preflight;
do not mask a release mismatch by modifying its installed package at startup.

For explicit `faiss_exact` scoring, declare `gpu_faiss: true` and test GPU FAISS
in the actual scoring image. If unavailable, reject that configuration; never
silently fall back to CPU FAISS. The default `torch_exact` recipe does not need
FAISS. The container CI handoff checks that optional dependency only when passed
`--require-gpu-faiss`; use that flag for releases supporting that backend.
Ordinary DINOv3 training continues through the packaged TAO
entrypoint with the existing experiment spec and result/checkpoint conventions.
