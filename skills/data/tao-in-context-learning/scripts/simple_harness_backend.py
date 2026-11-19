# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local prepared Cosmos3-Nano HF backend; no credentials, HTTP, or source checkout."""
import time


class CosmosBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('Cosmos inference requires a CUDA GPU in the approved runtime')
        self.torch = torch
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config['model']['path'], local_files_only=True)
        torch.manual_seed(config['protocol']['seed'])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config['model']['path'], torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda').eval()

    def generate(self, messages, tools):
        from PIL import Image
        torch = self.torch
        images = []
        try:
            for message in messages:
                if isinstance(message.get('content'), list):
                    for part in message['content']:
                        if part['type'] == 'image':
                            with Image.open(part['image']) as image:
                                images.append(image.convert('RGB'))
            text = self.processor.apply_chat_template(
                messages, tools=tools or None, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors='pt').to('cuda')
            size = inputs['input_ids'].shape[1]
            max_tokens = self.config['protocol']['max_new_tokens']
            context_limit = getattr(self.model.config.text_config, 'max_position_embeddings', None)
            if context_limit and size + max_tokens > context_limit:
                raise ValueError('Context limit exceeded; freeze a smaller grid/tool-image budget in a new run')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            completion = generated[0, size:]
            raw = self.processor.decode(completion, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended = bool(len(completion) and int(completion[-1]) in eos)
            return dict(raw=raw, input_tokens=size, output_tokens=len(completion), seconds=seconds,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        finish_reason='stop' if ended or len(completion) < max_tokens else 'length')
        finally:
            for image in images:
                image.close()
