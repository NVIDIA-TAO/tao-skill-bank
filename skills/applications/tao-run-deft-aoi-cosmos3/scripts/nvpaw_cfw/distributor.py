"""Batch-aligned deterministic NVPAW sharding with resumable worker positions."""

from __future__ import annotations

import os
import random
from typing import Any, Iterator

try:
    import torch
except ImportError:  # Host-side contract tests retain a deterministic fallback.
    torch = None  # type: ignore[assignment]

try:
    from cosmos_framework.data.generator.dataflow.base import DataDistributor
except ImportError:  # The packaged class binds to the real base in the image.
    class DataDistributor:  # type: ignore[no-redef]
        pass


def rank_epoch_indices(
    dataset_size: int, *, seed: int, epoch: int, rank: int, world_size: int
) -> list[int]:
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid DP coordinate {rank}/{world_size}")
    if torch is not None:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        indices = torch.randperm(dataset_size, generator=generator).tolist()
    else:
        indices = list(range(dataset_size))
        random.Random(int(seed) + int(epoch)).shuffle(indices)
    return indices[rank::world_size]


def full_batches_per_rank(
    dataset_size: int, world_size: int, micro_batch_size: int
) -> int:
    """Return the common batch count that every DP rank can safely emit."""

    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if world_size <= 0 or micro_batch_size <= 0:
        raise ValueError("world_size and micro_batch_size must be positive")
    return (dataset_size // world_size) // micro_batch_size


class NVPAWMapDistributor(DataDistributor):
    """CosmosDataLoader distributor protocol using whole micro-batches."""

    def __init__(
        self,
        dataset: Any,
        seed: int = 1993,
        micro_batch_size: int = 4,
        name: str = "nvpaw_train",
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        self._dataset = dataset
        self._seed = int(seed)
        self._micro_batch_size = int(micro_batch_size)
        self._name = name

    def __len__(self) -> int:
        return len(self._dataset)

    def stream(
        self,
        dp_rank: int,
        dp_world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterator[Any]:
        if not 0 <= worker_id < num_workers:
            raise ValueError(f"invalid worker coordinate {worker_id}/{num_workers}")
        full_batches = full_batches_per_rank(
            len(self._dataset), dp_world_size, self._micro_batch_size
        )
        if full_batches < num_workers:
            required = dp_world_size * num_workers * self._micro_batch_size
            raise ValueError(
                "NVPAW dataset is too small for one batch per data-loader worker; "
                f"requires at least {required} rows, found {len(self._dataset)}"
            )
        prefix = f"COSMOS_DL_STATE_{self._name}_" if self._name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0))
        resume_position = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1))
        epoch = resume_epoch
        while True:
            rank_indices = rank_epoch_indices(
                len(self._dataset),
                seed=self._seed,
                epoch=epoch,
                rank=dp_rank,
                world_size=dp_world_size,
            )
            worker_position = 0
            for batch_id in range(worker_id, full_batches, num_workers):
                start = batch_id * self._micro_batch_size
                for offset in range(self._micro_batch_size):
                    position = worker_position
                    worker_position += 1
                    if epoch == resume_epoch and position <= resume_position:
                        continue
                    item = self._dataset[rank_indices[start + offset]]
                    if isinstance(item, dict):
                        yield {
                            "_dp_epoch": epoch,
                            "_dp_stream_pos": position,
                            "_nvpaw_epoch": epoch,
                            **item,
                        }
                    else:
                        yield item
            epoch += 1
