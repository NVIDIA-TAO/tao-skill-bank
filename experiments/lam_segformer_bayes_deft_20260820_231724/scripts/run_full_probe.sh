#!/usr/bin/env bash
set -euo pipefail

probe_root=/lustre/fsw/portfolios/edgeai/users/rarunachalam/lam_segformer_bayes_deft_20260820_231724/controller
export TAO_NODE_COUNT=1
export TAO_GPUS_PER_NODE=8
export TAO_NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29537

timeout 150s torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr=127.0.0.1 \
  --master-port=29537 \
  "$probe_root/nccl_allreduce_probe.py"

timeout 420s python3 "$probe_root/model_load_probe.py"
