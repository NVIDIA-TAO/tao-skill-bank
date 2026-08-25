#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TAO CLIP owns its single-node multi-GPU launcher. A one-task srun correctly
# owns the allocation, but Lightning otherwise mistakes the inherited task
# topology for externally launched DDP ranks and refuses devices > 1.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$1" != "clip" ] || [ "$2" != "train" ]; then
  echo "run_clip_train_slurm: expected exact 'clip train' argv" >&2
  exit 64
fi

unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NODEID
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset NODE_RANK
unset MASTER_ADDR
unset MASTER_PORT
unset NUM_GPU_PER_NODE

exec "$@"
