"""Multi-image Qwen3-VL processing with deterministic over-context replacement."""

from __future__ import annotations

import hashlib
import io
from typing import Any

try:
    from PIL import Image
    from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import (
        VideoPhy2Processor,
    )
    from cosmos_framework.utils import log
    from cosmos_framework.utils.vlm.constant import PROCESSOR_KEYS_TO_ADD
except ImportError:  # Host-side tests exercise only the deterministic selector.
    Image = None  # type: ignore[assignment]
    log = None
    PROCESSOR_KEYS_TO_ADD: tuple[str, ...] = ()

    class VideoPhy2Processor:  # type: ignore[no-redef]
        pass


class _OverContextRow(ValueError):
    pass


def deterministic_replacement_index(
    *,
    dataset_size: int,
    base_index: int,
    attempt: int,
    epoch: int,
    seed: int,
    used: set[int],
) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if attempt == 0:
        return base_index
    if len(used) >= dataset_size:
        raise ValueError("all dataset rows have already been attempted")
    payload = f"{seed}\0{epoch}\0{base_index}\0{attempt}".encode()
    candidate = (
        int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        % dataset_size
    )
    while candidate in used:
        candidate = (candidate + 1) % dataset_size
    return candidate


class NVPAWResamplingProcessor(VideoPhy2Processor):
    """Preserve ordered image parts and supervise assistant tokens only."""

    _ALIGNMENT_ERROR = "truncation would break vision token alignment"

    def __init__(
        self,
        processor: Any,
        resample_dataset: Any,
        ignore_index: int,
        model_max_length: int = 4096,
        max_retries: int = 50,
        resample_seed: int = 271828,
    ) -> None:
        super().__init__(processor=processor, ignore_index=ignore_index)
        if max_retries < 2:
            raise ValueError("max_retries must be at least 2")
        self._resample_dataset = resample_dataset
        self._model_max_length = int(model_max_length)
        self._max_retries = int(max_retries)
        self._resample_seed = int(resample_seed)
        self._replacement_count = 0

    def _tokenize(self, item: dict[str, Any]) -> dict[str, Any]:
        if Image is None:
            raise RuntimeError("NVPAW processor requires the Cosmos Framework image")
        conversation = item.get("texts")
        if not isinstance(conversation, list):
            raise TypeError("NVPAW sample texts must be a list")
        media_bytes = item.get("media") or {}
        decoded: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        for message in conversation:
            if not isinstance(message, dict):
                raise TypeError("NVPAW message must be an object")
            content = message.get("content")
            if not isinstance(content, list):
                messages.append(dict(message))
                continue
            rewritten: list[Any] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    rewritten.append(part)
                    continue
                key = part.get("image")
                if not isinstance(key, str) or key not in media_bytes:
                    raise KeyError(f"NVPAW image key is unavailable: {key!r}")
                if key not in decoded:
                    decoded[key] = Image.open(io.BytesIO(media_bytes[key])).convert("RGB")
                image_part = dict(part)
                image_part["image"] = decoded[key]
                rewritten.append(image_part)
            messages.append({**message, "content": rewritten})
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": token_mask,
            "pad_token_id": self._pad_token_id,
            "ignore_index": self._ignore_index,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result

    def _process_once(self, item: dict[str, Any]) -> dict[str, Any]:
        result = self._tokenize(item)
        if int(result["input_ids"].shape[-1]) > self._model_max_length:
            raise _OverContextRow("tokenized row exceeds model_max_length")
        if int(result["token_mask"].sum().item()) == 0:
            raise ValueError("NVPAW row has no supervised assistant tokens")
        return result

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        base_index = int(item["_nvpaw_source_index"])
        epoch = int(item.get("_nvpaw_epoch", 0))
        used: set[int] = set()
        last_error: Exception | None = None
        selected_item = item
        selected_index = base_index
        for attempt in range(self._max_retries):
            selected_index = deterministic_replacement_index(
                dataset_size=len(self._resample_dataset),
                base_index=base_index,
                attempt=attempt,
                epoch=epoch,
                seed=self._resample_seed,
                used=used,
            )
            used.add(selected_index)
            if attempt:
                selected_item = self._resample_dataset[selected_index]
                selected_item["_nvpaw_epoch"] = epoch
            try:
                result = self._process_once(selected_item)
                break
            except _OverContextRow as exc:
                last_error = exc
            except ValueError as exc:
                if self._ALIGNMENT_ERROR not in str(exc):
                    raise
                last_error = exc
        else:
            raise RuntimeError(
                "no deterministic context-safe NVPAW replacement was found"
            ) from last_error
        if selected_index != base_index:
            self._replacement_count += 1
            if self._replacement_count <= 5 and log is not None:
                log.warning(
                    "NVPAW over-context replacement: "
                    f"source_index={base_index}, replacement_index={selected_index}, "
                    f"epoch={epoch}"
                )
        return result


def runtime_processor_class() -> type[NVPAWResamplingProcessor]:
    """Return the module-level, spawn-pickleable runtime processor class."""

    if Image is None:
        raise ImportError("Cosmos Framework and Pillow are required in the runtime image")
    return NVPAWResamplingProcessor
