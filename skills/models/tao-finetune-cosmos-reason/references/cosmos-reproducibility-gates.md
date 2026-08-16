# Cosmos clean-source reproducibility gates

Use this ownership map when a fresh session or a new image appears to need a
runtime patch. The gate passes only when the behavior is present in the named
repository, covered by its test, packaged by the clean action image, and
selected by the generated plan. A campaign script or container-layer edit is
not an implementation.

| Behavior | Owning repository and proof |
|---|---|
| A100 Qwen3-VL trainable PatchEmbed fallback | Framework: `cosmos_framework/model/generator/qwen3_vl_compat.py` and its test. Native RL: `cosmos_rl/policy/model/hf_models/patch.py` and `tests/test_qwen3_vl_patch_embed_compat.py`. The Framework model invokes its own compatibility function; do not depend on TAO Core microservice import side effects for training. |
| Dataset-neutral conversation and task-aware adapters | Framework: `cosmos_framework/data/generator/local_datasets/tao_vl_reason.py` plus TAO video recipe tests. Native RL: packaged `tao_sft_example.py` and `tao_vl_reason_daft_sft_example.py`; never stage a modified hook. |
| Dense/PEFT semantic parity | Framework LoRA utilities/config tests and native RL `LoraConfig`/`tests/test_lora.py`; the shared planner blocks rank, alpha, dropout, target, bias, RS-LoRA, modules-to-save, precision, or trainable-scope mismatches. |
| Dataset-neutral evaluation resolution | Skill: `scripts/evaluation_workflow.py` and its tests. It verifies the sealed training plan, inherits exact validation/prompt/preprocessing/model/LoRA/resource fields, classifies only genuinely missing values as user intake, and leaves Framework export as an automated action. The raw template contains no dataset prompt, path, task profile, checkpoint, or text-metric assumption. |
| Exact Framework DCP export | Framework: `cosmos_framework/scripts/export_vlm_dcp.py` and its test. Integration: Framework action wrapper. Skill: `framework_checkpoint_action.py`; never copy or rename shards manually. |
| Token-weighted global train/validation metrics | Framework loss statistics and TAO status callback tests. Native RL SFT token metrics and lifecycle status tests. Console-line or rank-local averages are invalid. |
| Terminal status and child exit propagation | Both native status implementations, controller-owned lifecycle tests, integration action lifecycle tests, and `cosmos_workflow.py` metadata finalization. Scheduler `COMPLETED` alone is provisional. |
| Cosmos-RL video runtime profiles | Native RL software: `cosmos_rl/utils/system_pyav_video_reader.py`, spawned-worker and codec-resolution tests, plus the integration Dockerfile's official-wheel SHA gate. Fast conversation path: `cosmos_rl/utils/pynv_video_reader.py` proves device RGBP through DLPack, exact-batch/scanned-retry behavior, explicit native-session capacity, and rank-local processed-output LRU; `SFTDataset.__getitems__` proves order-preserving `TAO_SFT_BATCH_THREADS`. The generated plan must resolve exactly one profile, include every cache/worker/transfer value, prove the installed Qwen spawned-worker helper forwards `TAO_PYNV_DECODER_CACHE_SIZE`, require persistent workers for nonzero worker counts, and never label the PyNv profile as the software-codec route. |
| Cosmos Framework video runtime profile | Framework: the dataset-neutral video SFT recipe uses CUDA TorchCodec with first-decode actual-device attestation, a rank-local decoded-frame LRU populated on demand, and single-flight reuse; the canonical dataflow loader provides bounded order-preserving in-process preprocessing threads. The generated plan derives cache capacity from inspected unique media, selects four process threads and one decoder thread by default, uses no DataLoader subprocess or prewarm, and fingerprints the resolved profile. |
| Conversation processor cache | Native RL prewarm/provenance code and cache tests. Train and validation keys are separate dataset+model+processor fingerprints and require complete manifests. |
| Task-aware direct decoding and override artifact | Integration builder/validator modules and tests. The schema-v2 manifest pairs each annotation with its media root, scans macroblock limits, forces complete validation coverage, carries explicit diagnosed train sources, and proves the clean image's `ACTIONS_COMMIT`. |
| Worker-zero data loading | Both planners omit/null prefetch when workers are zero. The RL fast profile defaults to worker=1/prefetch=2; the software profile defaults to worker=0 with prefetch omitted. Explicit recipe values win. |
| Task-aware optimizer/update parity | Shared planner tests prove hybrid expansion before update calculation, fused AdamW for native RL, a constant schedule floor of 1.0, and Framework shard=`gpus_per_node`, replica=`nodes`. |
| Synchronous multi-node checkpoints | Framework SFT config exposes synchronous DCP and the planner rejects asynchronous multi-node checkpoints. Native RL uses synchronous save mode for the same contract. |
| Clean image and SQSH provenance | Integration Dockerfiles and Makefile build the two action images from explicit clean commits and embed `/opt/tao/image-provenance.json`; the skill verifies commits/trees and forbids host source mounts, startup patches, and reused SQSH after source changes. |
| Non-root Pyxis runtime | SLURM plans disable host-home mounting, create an isolated per-job container home, preserve explicit mounts, and run packaged imports as the invoking user. |
| Deterministic SLURM recovery | The TAO job-record id is the exact job name. Ambiguous submission is reconciled through `squeue` and `sacct` before retry; real retries use a new `--retry-of` record and validated node exclusions. Cosmos emits `--no-requeue`. |

TAO Core's Qwen3-VL microservice handler is a separate inference-service
surface. A fork-only TAO Core fix is not a valid integration submodule target:
merge it into the configured authoritative TAO Core remote first, then update
the submodule and rebuild. Framework and native RL training must continue to
use their repository-owned fallbacks regardless.

After any owning-source change, rerun the repository unit tests, rebuild the
action image and SQSH from clean commits, prove packaged imports/provenance, run
the affected smoke through validation/checkpoint/export/evaluation, and then
rerun every affected full cell. Infrastructure retries do not justify a source
change unless the evidence identifies a code defect.
