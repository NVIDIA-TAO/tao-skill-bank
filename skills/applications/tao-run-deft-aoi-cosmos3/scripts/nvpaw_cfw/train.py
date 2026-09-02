"""Entrypoint that registers NVPAW, then delegates to Framework training."""

from __future__ import annotations

import functools
import importlib
import runpy


def install_registration_hook() -> None:
    from cosmos_framework.configs.base.reasoner import config as base_vlm_config

    original_make_config = base_vlm_config.make_config

    def make_config_then_register():
        config = original_make_config()
        from . import processor
        from . import experiment

        importlib.reload(processor)
        importlib.reload(experiment).register_experiment()
        return config

    base_vlm_config.make_config = make_config_then_register


def install_checkpoint_memory_release_hook() -> None:
    """Release unused CUDA cache immediately before synchronous DCP saves.

    Framework saves at the optimizer-step boundary before its final batch
    locals leave scope.  NCCL checkpoint planning allocates an additional
    device buffer, so cached allocator segments can otherwise turn a stable
    training micro-batch into a save-only OOM.  This hook releases only
    unused cache; live model, optimizer, and batch tensors are unchanged.
    """

    import gc

    import torch
    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

    if getattr(DistributedCheckpointer, "_nvpaw_memory_release_installed", False):
        return
    original_save = DistributedCheckpointer.save

    @functools.wraps(original_save)
    def save_after_memory_release(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        return original_save(self, *args, **kwargs)

    DistributedCheckpointer.save = save_after_memory_release
    DistributedCheckpointer._nvpaw_memory_release_installed = True


def main() -> None:
    install_registration_hook()
    install_checkpoint_memory_release_hook()
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
