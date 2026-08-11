# Cosmos SLURM guardrails

When `tao-finetune-cosmos-reason` resolves a backend, read that backend
contract before rendering the SLURM command. Cosmos jobs require a prebuilt,
compute-node-readable `.sqsh`; convert the selected image before the GPU
allocation and do not substitute a direct registry reference in the training
job. Use the model planner's explicit post-review `materialize` verb to write
generated TOMLs and any merged/smoke manifests atomically on user-supplied
shared storage, then verify their checksums before rendering or submitting the
job. Planning and preflight must remain read-only and must not require the
shared filesystem to be mounted on the SSH launch host. Stage the Framework
status bridge in the repository-derived image, not as an ad hoc source patch.

- Cosmos Framework: one Pyxis task/container per node; inside each task set
  `NODE_RANK=$SLURM_PROCID` and launch native torchrun with node count,
  GPUs-per-node, master address/port. Set Framework HSDP shard degree to GPUs
  per node and replicate degree to nodes. Use `--no-container-mount-home`,
  `/workspace/.venv/bin/python`, `ulimit -n 65536`, and disable asynchronous DCP
  for multi-node shared-SLURM runs.
- Cosmos-RL: single-node uses its normal CLI. Validated policy-only multi-node
  SFT starts the controller only on node zero and one policy replica on every
  node; its spec uses global GPU count as shard size and one replicate. Enable
  CUDA video driver capability and the image's PyNvVideoCodec path rather than
  falling back to CPU decoding.

For both backends, preserve the real `srun`/torchrun/policy exit code through
cleanup and any requeue footer. A zero exit from a later shell command must not
mask a failed training process. Treat SLURM `COMPLETED` as provisional until
the Cosmos structured status contains terminal `SUCCESS`; then extract train
and epoch validation metrics with the model skill's packaged helper.
