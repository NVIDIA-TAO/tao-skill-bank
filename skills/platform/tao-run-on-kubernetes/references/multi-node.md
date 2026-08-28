# Multi-node training on Kubernetes

Moved verbatim from SKILL.md (size cap); the contract is unchanged.

Set `num_nodes > 1` (see the [Multi-node (nodes > 1)](#multi-node-nodes--1) verb
steps above) to run distributed training across N pods. Rendering
`templates/k8s/indexed-job.yaml.tmpl` provisions:

1. A **headless Service** named after the Job (selector: `job-name=<job-name>`, `clusterIP: None`, `publishNotReadyAddresses: true` so pods can rendezvous before they're all Ready).
2. An **Indexed Job** with `parallelism = completions = num_nodes`, `completionMode: Indexed`. Each pod gets `JOB_COMPLETION_INDEX` injected by k8s automatically (= the node rank).
3. A **command wrapper** that exports the rendezvous env vars before invoking the user command. Two naming conventions are exported simultaneously:

   | Env var | Value | Read by |
   |---|---|---|
   | `WORLD_SIZE` | `num_nodes` | TAO PyTorch container's `nvidia_tao_pytorch/core/entrypoint.py` (uses this to mean *node count*, even though PyTorch's own convention is *total processes*) |
   | `NUM_GPU_PER_NODE` | `gpu_count` | TAO PyTorch container's entrypoint |
   | `NNODES` | `num_nodes` | `torchrun` and PyTorch-standard rendezvous |
   | `NPROC_PER_NODE` | `gpu_count` | `torchrun` |
   | `NODE_RANK` | `$JOB_COMPLETION_INDEX` | both |
   | `MASTER_ADDR` | `<job-name>-0.<job-name>` (pod-0's DNS) | both |
   | `MASTER_PORT` | `29500` | both (TAO's default) |

   Both naming conventions are set so TAO entrypoints (`dino train`, etc.) and raw `torchrun` commands work without modification.

For a TAO entrypoint, the container reads `spec.train.num_nodes` and the wired
env vars — e.g. `dino train -e /tmp/spec.yaml` with `gpu_count=8`, `num_nodes=4`
(4 × 8 = 32 GPUs total).

For raw `torchrun`-based commands (non-TAO containers), the wrapper invokes:

```bash
torchrun --nnodes=$NNODES --nproc-per-node=$NPROC_PER_NODE --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT train.py
```

The capacity check sums across nodes: `gpu_count × num_nodes` ≤ cluster's allocatable `nvidia.com/gpu`.

### Cluster requirements for multi-node

- **k8s 1.28+** is required for stable pod hostnames in Indexed Jobs (the `PodIndexLabel` feature). On older clusters the `MASTER_ADDR=<job>-0.<svc>` DNS lookup fails. Verify with `kubectl version`.
- **Pod-to-pod networking** must be open on port 29500 (PyTorch default; configurable via `MASTER_PORT` env var). Most CNIs (Calico, Cilium, AWS VPC CNI) allow this by default; restrictive NetworkPolicies must be relaxed.
- **NCCL** in the container talks GPU-to-GPU; if the cluster has multi-NIC nodes or RDMA, set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` in the container `env` of the rendered manifest.

### Reference reading

- Kubernetes Indexed Job: <https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode>
- Indexed Job for batch ML: <https://kubernetes.io/blog/2022/06/01/indexed-jobs-mpi/>
- PyTorch distributed (env-var rendezvous): <https://pytorch.org/docs/stable/elastic/run.html>
- NCCL networking tuning (NCCL_SOCKET_IFNAME, NCCL_IB_HCA): <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>

### Kubernetes operator alternatives

For more sophisticated topologies (gang scheduling, PyTorch elastic / fault-tolerant training, MPI / Horovod, RDMA setup), reach for an operator instead of plain Indexed Job:

- **MPI Operator** — <https://github.com/kubeflow/mpi-operator> — for MPI / Horovod workloads.
- **Kubeflow Training Operator** (`PyTorchJob`, `TFJob`) — <https://www.kubeflow.org/docs/components/training/> — for elastic PyTorch training with built-in restart logic.
- **Volcano** — <https://volcano.sh/> — gang scheduling, queues, fair-share. Useful in shared multi-tenant clusters.
- **Kueue** — <https://kueue.sigs.k8s.io/> — quota / queue layer on top of any of the above.

This skill's Indexed Job path is intentionally simple and dependency-free; if you need elastic restart or gang scheduling, layer one of these on top and submit jobs through the operator's CRD instead.
