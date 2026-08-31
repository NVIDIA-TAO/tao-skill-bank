#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run Cosmos Framework inference directly over canonical NVPAW JSONL.

The runtime intentionally uses the Framework Transformers shim, including its
attention implementation, rather than the generic evaluation dataset loader.
That keeps the canonical JSONL boundary and preserves ordered multi-image
messages and their per-image pixel limits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Iterator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 runtime fallback
    import tomli as tomllib


MEDIA_TYPES = {"image", "image_url", "video", "video_url"}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item["text"])
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def iter_source_rows(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Stream validated rows without materializing a JSON array."""

    with path.open(encoding="utf-8") as stream:
        yielded = 0
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            row_id = row.get("id")
            task_type = row.get("task_type")
            messages = row.get("messages")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if not isinstance(task_type, str) or not task_type:
                raise ValueError(
                    f"{path}:{line_number}: task_type must be a non-empty string"
                )
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: messages must be a non-empty list")
            yielded += 1
            yield row
        if not yielded:
            raise ValueError(f"{path}: JSONL is empty")


def _resolve_media_item(
    item: dict[str, Any], media_root: pathlib.Path, row_id: str
) -> dict[str, Any]:
    rewritten = copy.deepcopy(item)
    media_type = rewritten.get("type")
    if media_type not in MEDIA_TYPES:
        return rewritten
    key = "image" if media_type == "image" else "video" if media_type == "video" else media_type
    value = rewritten.get(key)
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row_id!r} has an invalid {media_type} item")
    if "://" in value:
        raise ValueError(f"row {row_id!r} uses unsupported remote media: {value}")
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = media_root / path
    resolved = path.resolve(strict=True)
    if media_type.endswith("_url"):
        rewritten[key] = {"url": str(resolved)}
    else:
        rewritten[key] = str(resolved)
    return rewritten


def prepare_prompt_messages(
    row: dict[str, Any], media_root: pathlib.Path
) -> list[dict[str, Any]]:
    """Remove gold turns and resolve media while retaining message/part order."""

    row_id = str(row.get("id", ""))
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"row {row_id!r} has no messages")
    prompts: list[dict[str, Any]] = []
    media_items = 0
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"row {row_id!r} contains a non-object message")
        if message.get("role") == "assistant":
            continue
        content = message.get("content")
        rewritten = copy.deepcopy(message)
        if isinstance(content, list):
            rewritten_content: list[Any] = []
            for item in content:
                if not isinstance(item, dict):
                    rewritten_content.append(copy.deepcopy(item))
                    continue
                rewritten_item = _resolve_media_item(item, media_root, row_id)
                if rewritten_item.get("type") in MEDIA_TYPES:
                    media_items += 1
                rewritten_content.append(rewritten_item)
            rewritten["content"] = rewritten_content
        prompts.append(rewritten)
    if not prompts:
        raise ValueError(f"row {row_id!r} has no prompt messages")
    if not media_items:
        raise ValueError(f"row {row_id!r} has no image or video message item")
    return prompts


def normalized_result(
    row: dict[str, Any], prediction: str, *, require_ground_truth: bool = True
) -> dict[str, Any]:
    """Build the exact application prediction row without changing source data."""

    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("source row requires messages")
    prompts: list[dict[str, Any]] = []
    ground_truth: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("source row contains a non-object message")
        if message.get("role") == "assistant":
            text = _content_text(message.get("content")).strip()
            if text:
                ground_truth.append(text)
        else:
            prompts.append(copy.deepcopy(message))
    row_id = row.get("id")
    task_type = row.get("task_type")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("source row requires a non-empty id")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"source row {row_id!r} requires task_type")
    if require_ground_truth and not ground_truth:
        raise ValueError(f"source row {row_id!r} has no assistant ground truth")
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError(f"Framework result for {row_id!r} is empty")
    return {
        "id": row_id,
        "task_type": task_type,
        "message": prompts,
        "GT": "\n".join(ground_truth),
        "raw_prediction": prediction.strip(),
    }


def _register_cosmos_attention(name: str) -> None:
    if name != "cosmos":
        return
    import torch
    from transformers import AttentionInterface

    from cosmos_framework.model.attention import attention as cosmos_attention
    from cosmos_framework.model.attention.masks import CausalType

    def hf_attention_cosmos_inference(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        if dropout != 0.0:
            raise NotImplementedError("Cosmos inference requires dropout=0")
        if attention_mask is not None:
            raise NotImplementedError(
                "Cosmos inference requires an unpadded single-example batch"
            )
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        original_dtype = q.dtype
        cast = original_dtype not in (torch.float16, torch.bfloat16)
        if cast:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        causal_type = None
        if bool(getattr(module, "is_causal", False)):
            causal_type = (
                CausalType.DontCare
                if q.shape[1] == k.shape[1]
                else CausalType.BottomRight
            )
        output = cosmos_attention(
            q,
            k,
            v,
            is_causal=bool(getattr(module, "is_causal", False)),
            causal_type=causal_type,
            scale=scaling,
        )
        if cast:
            output = output.to(original_dtype)
        return output, None

    AttentionInterface.register("cosmos", hf_attention_cosmos_inference)


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_model(
    model_path: pathlib.Path,
    *,
    dtype: str,
    device_map: str,
    attn_implementation: str,
) -> tuple[Any, Any]:
    _register_cosmos_attention(attn_implementation)
    from transformers import AutoProcessor
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Cosmos3ForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=_torch_dtype(dtype),
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def _model_input_device(model: Any, requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _generate(
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    input_device: str,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model_input_device(model, input_device))
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = None
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
    input_length = int(inputs["input_ids"].shape[1])
    return processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _atomic_predictions(
    output: pathlib.Path,
    rows: Iterator[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    processor: Any,
    model: Any,
    max_new_tokens: int,
    input_device: str,
    require_ground_truth: bool,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                messages = prepare_prompt_messages(row, media_root)
                prediction = _generate(
                    processor,
                    model,
                    messages,
                    max_new_tokens=max_new_tokens,
                    input_device=input_device,
                )
                stream.write(
                    json.dumps(
                        normalized_result(
                            row,
                            prediction,
                            require_ground_truth=require_ground_truth,
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if not count:
            raise ValueError("no predictions were generated")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def _load_evaluate_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    dataset = config.get("dataset")
    generation = config.get("generation")
    model = config.get("model")
    if not all(isinstance(value, dict) for value in (dataset, generation, model)):
        raise ValueError("evaluation TOML requires dataset, generation, and model tables")
    return config


def _inference_row(args: argparse.Namespace) -> dict[str, Any]:
    media_type = args.media_type
    if media_type not in {"image", "video"}:
        raise ValueError("inference media_type must be image or video")
    return {
        "id": args.id,
        "task_type": args.task_type,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: args.media},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("evaluate", "inference"), required=True)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--media", type=str)
    parser.add_argument("--media-type", choices=("image", "video"), default="image")
    parser.add_argument("--prompt")
    parser.add_argument("--id", default="inference-0")
    parser.add_argument("--task-type", default="Inference")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--input-device", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "evaluate":
            if args.config is None:
                raise ValueError("evaluate requires --config")
            config = _load_evaluate_config(args.config.expanduser().resolve(strict=True))
            dataset = config["dataset"]
            generation = config["generation"]
            model_config = config["model"]
            source = pathlib.Path(dataset.get("annotation_path", "")).expanduser().resolve(strict=True)
            media_root = pathlib.Path(dataset.get("media_dir", "")).expanduser().resolve(strict=True)
            rows: Iterator[dict[str, Any]] = iter_source_rows(source)
            max_new_tokens = args.max_new_tokens or int(generation.get("max_tokens", 1024))
            dtype = args.dtype or str(model_config.get("dtype", "bfloat16"))
            attention = args.attn_implementation or str(
                model_config.get("attn_implementation", "cosmos")
            )
        else:
            if not args.media or not args.prompt:
                raise ValueError("inference requires --media and --prompt")
            media = pathlib.Path(args.media).expanduser().resolve(strict=True)
            media_root = media.parent
            rows = iter((_inference_row(argparse.Namespace(**{**vars(args), "media": media.name})),))
            max_new_tokens = args.max_new_tokens or 1024
            dtype = args.dtype or "bfloat16"
            attention = args.attn_implementation or "cosmos"
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        model_path = args.model_path.expanduser().resolve(strict=True)
        if not model_path.is_dir():
            raise NotADirectoryError(f"model path is not a directory: {model_path}")
        if args.validate_only:
            selected = list(rows)
            for row in selected:
                prepare_prompt_messages(row, media_root)
            print(
                json.dumps(
                    {
                        "action": args.action,
                        "backend": "cosmos-framework",
                        "model_path": str(model_path),
                        "selected_rows": len(selected),
                    },
                    sort_keys=True,
                )
            )
            return 0
        processor, model = _load_model(
            model_path,
            dtype=dtype,
            device_map=args.device_map,
            attn_implementation=attention,
        )
        count = _atomic_predictions(
            args.output_jsonl.expanduser().resolve(),
            rows,
            media_root=media_root,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
            input_device=args.input_device,
            require_ground_truth=args.action == "evaluate",
        )
        print(json.dumps({"predictions": count, "output": str(args.output_jsonl.resolve())}))
        return 0
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"cfw_jsonl_runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
