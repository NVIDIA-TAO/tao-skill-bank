#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ask a local vision-language model questions about a private reference image.

Runs entirely on the host: the model is loaded from a local directory with
network access to the Hugging Face hub switched off, and only the answers
(text) and timings are written. No pixels, crops or encodings are stored or
printed. Run it inside an environment that has torch and transformers, for
example a container started with `--network none`; this script does not
install anything.

The questions file is a JSON list of
`{"id", "prompt", "crop_px": [x0, y0, x1, y1]?, "upscale": 1?, "max_new_tokens": 400?}`.
A crop is taken in memory from the original image and never saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="local directory with weights and processor files")
    parser.add_argument("--image", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-image-pixels", type=int, default=2560 * 1440,
                        help="upper bound on pixels given to the model per question")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    questions = json.loads(Path(args.questions).read_text())
    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    load_seconds = time.time() - started

    source = Image.open(args.image).convert("RGB")
    answers = []
    for question in questions:
        image = source
        if question.get("crop_px"):
            image = image.crop(tuple(int(v) for v in question["crop_px"]))
        scale = float(question.get("upscale", 1))
        if scale != 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        if image.width * image.height > args.max_image_pixels:
            shrink = (args.max_image_pixels / (image.width * image.height)) ** 0.5
            image = image.resize((int(image.width * shrink), int(image.height * shrink)), Image.LANCZOS)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": question["prompt"]}]}]
        tick = time.time()
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=int(question.get("max_new_tokens", 400)), do_sample=False)
        text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        answers.append({"id": question["id"], "prompt": question["prompt"], "crop_px": question.get("crop_px"),
                        "model_input_size": [image.width, image.height], "input_tokens": int(inputs["input_ids"].shape[1]),
                        "answer": text, "seconds": round(time.time() - tick, 2)})

    config = Path(args.model_dir) / "config.json"
    result = {
        "reader": "local vision-language model via transformers; answers are model output, not ground truth",
        "model_dir": str(Path(args.model_dir).resolve()), "model_config_sha256": sha256(config),
        "torch": torch.__version__, "device": str(model.device), "greedy_decoding": True,
        "image": Path(args.image).name, "image_sha256": sha256(args.image), "image_size": list(source.size),
        "load_seconds": round(load_seconds, 2), "total_seconds": round(time.time() - started, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0,
        "answers": answers,
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": args.out_json, "questions": len(answers), "total_seconds": result["total_seconds"]}))


if __name__ == "__main__":
    main()
