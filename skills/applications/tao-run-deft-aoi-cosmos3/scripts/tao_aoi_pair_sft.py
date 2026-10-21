# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""AOI OK/NG SFT recipe for Cosmos3 (one- or two-image records).

DEFT AOI records are bare ShareGPT with an ``images`` list of one or two
paths — the inspection capture, optionally followed by its golden reference
(single-image classification datasets such as wafer defect have no goldens) —
and a single-word OK/NG assistant answer. This experiment reuses the shared
VLM pipeline end to end; only the record adapter (all record images attached
to the first user turn, in record order) is new.

Env contract (mirrors the WTS_* convention):
  AOI_TRAIN_ANNOTATION / AOI_TRAIN_MEDIA / AOI_TRAIN_LIMIT
  AOI_VAL_ANNOTATION   / AOI_VAL_MEDIA   / AOI_VAL_LIMIT
  AOI_SYSTEM_PROMPT    (default "")
  AOI_MAX_PIXELS / AOI_MIN_PIXELS  (per-image bounds; best-effort on the
                                    processor's image size config)
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from hydra.core.config_store import ConfigStore
from PIL import Image
from torch.utils.data import Dataset

from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.configs.base.reasoner.experiment.wts_vlm import wts_vlm
from cosmos_framework.data.generator.dataflow import CosmosDataLoader, MapDistributor, PoolPackingBatcher
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX


class AoiPairConversationDataset(Dataset):
    """Map-style loader for one- or two-image ShareGPT records (DEFT AOI bare OK/NG)."""

    def __init__(
        self,
        annotation_path: str,
        media_path: str,
        limit: int | str | None = None,
    ) -> None:
        self.annotation_path = os.path.abspath(os.path.expanduser(annotation_path))
        self.media_path = os.path.abspath(os.path.expanduser(media_path))
        if limit in ("", None):
            parsed_limit = None
        else:
            parsed_limit = int(limit)
            if parsed_limit < 1:
                parsed_limit = None

        with open(self.annotation_path, encoding="utf-8") as annotation_file:
            records = json.load(annotation_file)
        if not isinstance(records, list):
            raise TypeError(f"AOI annotations must be a JSON array, got {type(records).__name__}")
        self.records = records[:parsed_limit] if parsed_limit is not None else records
        if not self.records:
            raise ValueError(f"AOI annotation file contains no usable records: {self.annotation_path}")

        for index, record in enumerate(self.records):
            images = record.get("images") if isinstance(record, dict) else None
            if not isinstance(images, list) or len(images) not in (1, 2) or not all(isinstance(p, str) for p in images):
                raise ValueError(
                    f"AOI record {index} must carry an `images` list of one or two string paths "
                    f"(inspection capture[, golden reference])"
                )
            conversations = record.get("conversations") or record.get("messages")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"AOI record {index} must contain at least two conversation turns")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        resolved: list[str] = []
        for path in record["images"]:
            if not os.path.isabs(path):
                path = os.path.join(self.media_path, path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"AOI media does not exist: {path}")
            resolved.append(path)
        record["images"] = resolved
        return record


class AoiPairSFTProcessor(VLMProcessor):
    """Attach every record image to the first user turn, in record order."""

    def __init__(
        self,
        processor: Any,
        ignore_index: int = IGNORE_INDEX,
        max_pixels: int | str | None = None,
        min_pixels: int | str | None = None,
        system_prompt: str = "",
    ) -> None:
        super().__init__(processor=processor, ignore_index=ignore_index)
        self.system_prompt = system_prompt
        # Best-effort pixel bounds on the underlying image processor. Both the
        # min/max_pixels attribute form and the size-dict form exist across
        # processor versions; set whichever is present, never invent one.
        hf_processor = getattr(processor, "processor", processor)
        image_processor = getattr(hf_processor, "image_processor", None)
        for name, value in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
            if value in (None, "", 0, "0") or image_processor is None:
                continue
            parsed = int(value)
            if hasattr(image_processor, name):
                setattr(image_processor, name, parsed)
            size = getattr(image_processor, "size", None)
            if isinstance(size, dict):
                size["longest_edge" if name == "max_pixels" else "shortest_edge"] = parsed

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        conversations = item.get("conversations", [])
        images = [Image.open(path).convert("RGB") for path in item["images"]]
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        images_inserted = False
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"].replace("<image>", "").strip()
            if role == "user" and not images_inserted:
                content: Any = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": text})
                images_inserted = True
            else:
                content = text
            messages.append({"role": role, "content": content})
        return messages


def _aoi_pair_dataloader(*, annotation_env: str, media_env: str, limit_env: str, shuffle: bool) -> LazyDict:
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(AoiPairConversationDataset)(
                annotation_path=f"${{oc.env:{annotation_env}}}",
                media_path=f"${{oc.env:{media_env}}}",
                limit=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed="${oc.env:TAO_DATALOADER_SEED,42}",
            name="train" if shuffle else "val",
        ),
        processor=L(AoiPairSFTProcessor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
            max_pixels="${oc.env:AOI_MAX_PIXELS,''}",
            min_pixels="${oc.env:AOI_MIN_PIXELS,''}",
            system_prompt="${oc.env:AOI_SYSTEM_PROMPT,''}",
        ),
        batcher=L(PoolPackingBatcher)(
            max_tokens="${data_setting.max_tokens}",
            pool_size=16 if shuffle else 1,
            max_batch_size=1,
            long_threshold=6400,
        ),
        collator=L(VLMCollator)(),
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
    )


# Same trainer/model/optimizer scaffold as the validated WTS recipe; only the
# data path and the sequence budget differ (two small crops, one-word answers).
aoi_pair_conversation = deepcopy(wts_vlm)
aoi_pair_conversation["job"]["group"] = "aoi_sft"
aoi_pair_conversation["model"]["config"]["policy"]["model_max_length"] = 8192
aoi_pair_conversation["data_setting"]["max_tokens"] = 8192
aoi_pair_conversation["dataloader_train"] = _aoi_pair_dataloader(
    annotation_env="AOI_TRAIN_ANNOTATION", media_env="AOI_TRAIN_MEDIA",
    limit_env="AOI_TRAIN_LIMIT", shuffle=True,
)
aoi_pair_conversation["dataloader_val"] = _aoi_pair_dataloader(
    annotation_env="AOI_VAL_ANNOTATION", media_env="AOI_VAL_MEDIA",
    limit_env="AOI_VAL_LIMIT", shuffle=False,
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name="aoi_pair_conversation", node=aoi_pair_conversation
)
