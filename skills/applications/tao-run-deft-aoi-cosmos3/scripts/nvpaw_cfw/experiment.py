"""Register the reviewed NVPAW full-parameter Cosmos Framework experiment."""

from __future__ import annotations

import copy

from .dataset import NVPAWJsonlDataset
from .distributor import NVPAWMapDistributor
from .processor import runtime_processor_class


EXPERIMENT_NAME = "nvpaw_omni_vlm_sft"


def register_experiment() -> None:
    from hydra.core.config_store import ConfigStore
    from cosmos_framework.callbacks.cosmos_dataloader_state import (
        CosmosDataLoaderStateCallback,
    )
    from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator
    from cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano import (
        videophy2_sft_nano,
    )
    from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, SimpleBatcher
    from cosmos_framework.data.vfm.processors import build_processor
    from cosmos_framework.utils.lazy_config import LazyCall as L
    from cosmos_framework.utils.vlm.constant import IGNORE_INDEX

    Processor = runtime_processor_class()

    def dataset() -> dict:
        return L(NVPAWJsonlDataset)(
            jsonl_path="${oc.env:NVPAW_TRAIN_JSONL}",
            media_root="${oc.env:NVPAW_MEDIA_ROOT}",
            index_path="${oc.env:NVPAW_INDEX_PATH}",
            expected_rows=1,
            expected_sha256="0" * 64,
            expected_image_items=1,
        )

    experiment = copy.deepcopy(videophy2_sft_nano)
    experiment.job.name = EXPERIMENT_NAME
    experiment.trainer.max_iter = 500
    experiment.trainer.logging_iter = 1
    experiment.trainer.run_validation = False
    experiment.trainer.run_validation_on_start = False
    experiment.trainer.grad_accum_iter = 16
    experiment.trainer.seed = 0
    experiment.trainer.callbacks = dict(
        log_tensor_shape=dict(num_log=2),
        dataloader_state=L(CosmosDataLoaderStateCallback)(),
    )
    experiment.model.config.freeze.freeze_vision_encoder = True
    experiment.model.config.freeze.freeze_mm_projector = False
    experiment.model.config.freeze.freeze_llm = False
    experiment.model.config.policy.model_max_length = 4096
    experiment.model.config.parallelism.data_parallel_shard_degree = 8
    experiment.model.config.parallelism.data_parallel_replicate_degree = 1
    experiment.model.config.parallelism.context_parallel_shard_degree = 1
    experiment.model.config.parallelism.cfg_parallel_shard_degree = 1
    experiment.optimizer.optimizer_type = "AdamW"
    experiment.optimizer.lr = 1.0e-6
    experiment.optimizer.weight_decay = 0.05
    experiment.optimizer.betas = [0.9, 0.999]
    experiment.optimizer.fused = True
    experiment.optimizer.keys_to_select = []
    experiment.optimizer.lr_multipliers = {"merger": 20.0}
    experiment.scheduler.warm_up_steps = [5]
    experiment.scheduler.cycle_lengths = [500]
    experiment.scheduler.f_start = [0.05]
    experiment.scheduler.f_max = [1.0]
    experiment.scheduler.f_min = [0.1]
    experiment.checkpoint.save_iter = 100
    experiment.checkpoint.dcp_async_mode_enabled = False
    experiment.dataloader_train = L(CosmosDataLoader)(
        distributor=L(NVPAWMapDistributor)(
            dataset=dataset(), seed=1993, micro_batch_size=4, name=""
        ),
        processor=L(Processor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            resample_dataset=dataset(),
            ignore_index=IGNORE_INDEX,
            model_max_length=4096,
            max_retries=50,
            resample_seed=271828,
        ),
        batcher=L(SimpleBatcher)(batch_size=4, drop_last=True),
        collator=L(VLMCollator)(),
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    experiment.dataloader_val = None
    experiment.upload_reproducible_setup = False
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=EXPERIMENT_NAME,
        node=experiment,
    )
