# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic, dev-only batched generation and paired staged screening.

The frozen final HF worker is deliberately not imported as a backend. This
module never accepts protected splits, never retries a pass, and never supplies
fast answers to FULL2917. Use in a narrow development-only container.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import time

from wsp03_worker import apply_candidate, digest, validate_requests
from wsp03_bindings import file_hash
from wsp03_io import json_write

SIZES = {'S1': 64, 'S2': 512, 'PARITY64': 64}
TERMINAL = {'COMPLETE', 'ABANDONED_DEGENERATE', 'ABANDONED_WALL', 'FAILED'}


def read_rows(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.suffix == '.json' else [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique(rows):
    byid = {row['id']: row for row in rows}
    if len(byid) != len(rows):
        raise ValueError('duplicate identities')
    return byid


def require_dev(ids, development_ids):
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(development_ids):
        raise ValueError('unique nonempty development identities required')


def evaluator_for(cfg):
    # THT's actual immutable parser and score implementation, not a replacement.
    from tht_contract import external_evaluator
    evaluator = external_evaluator(cfg['evaluator_directory'])
    if list(evaluator.CLASSES) != cfg['classes']:
        raise ValueError('configured classes differ from frozen evaluator')
    return evaluator


def summary(rows, evaluator, gold=None):
    unique(rows)
    predictions = [evaluator.parse(row['raw_answer']) for row in rows]
    invalid = [p not in evaluator.CLASSES for p in predictions]
    length = [row['finish_reason'] == 'length' for row in rows]
    n = len(rows)
    result = {
        'n': n, 'invalid_count': sum(invalid), 'truncation_count': sum(length),
        'invalid_rate': sum(invalid)/n if n else None,
        'truncation_rate': sum(length)/n if n else None,
        'degenerate_union_count': sum(a or b for a, b in zip(invalid, length)),
        'degenerate_union_rate': sum(a or b for a, b in zip(invalid, length))/n if n else None,
        'mean_output_tokens': sum(row['inference_output_tokens'] for row in rows)/n if n else None,
    }
    if gold is not None:
        truth = unique(gold)
        if set(truth) != {row['id'] for row in rows}:
            raise ValueError('exact paired label coverage required')
        scored = []
        for row, prediction in zip(rows, predictions):
            labels = truth[row['id']]['labels']
            if len(labels) != 1 or labels[0] not in evaluator.CLASSES:
                raise ValueError('one configured development class required')
            scored.append({'id': row['id'], 'truth': labels[0], 'prediction': prediction})
        result['metrics'] = evaluator.score(scored)
    return result


def degenerate(rows, evaluator):
    value = summary(rows, evaluator)
    return len(rows) >= 32 and value['degenerate_union_count'] * 5 >= len(rows) * 4


def trim_completion(ids, eos_ids, cap):
    """Remove batch tail padding only after the first EOS, retaining that EOS."""
    values = list(ids)
    eos = set(eos_ids)
    for index, token in enumerate(values):
        if token in eos:
            return values[:index+1], 'stop'
    return values, 'length' if len(values) >= cap else 'stop'


def select_backend(cfg):
    config = json.loads((Path(cfg['model'])/'config.json').read_text())
    archs = config.get('architectures', [])
    try:
        version = importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
        version = None
    reason = 'vllm not installed in this isolated environment'
    supported = False
    if version:
        from vllm import ModelRegistry
        supported = any(name in ModelRegistry.get_supported_archs() for name in archs)
        reason = 'installed vllm registry supports model architecture' if supported else 'architecture absent from installed vllm registry'
    choice = cfg.get('backend', 'auto')
    if choice not in ('auto', 'hf', 'vllm'):
        raise ValueError('backend must be auto, hf or vllm')
    if choice == 'vllm' and not supported:
        raise ValueError(reason)
    if choice == 'hf':
        chosen, reason = 'hf', cfg.get('backend_reason', 'explicit isolated HF batch fallback')
    else:
        chosen = 'vllm' if supported else 'hf'
    if chosen == 'vllm' and cfg.get('candidate') and not cfg.get('exported_model'):
        raise ValueError('vllm candidate requires a verified HF export; no silent base substitution')
    return {'backend': chosen, 'reason': reason, 'architectures': archs,
            'vllm_version': version, 'real_checkpoint_support': 'pending actual generation/parity'}


def verify_model(cfg):
    """Pin the immutable base inventory; verify optional exported patch replay."""
    base = Path(cfg['model']).resolve(strict=True)
    files = cfg['model_files']
    actual = {p.name for p in base.iterdir() if p.is_file()}
    if not files or set(files) != actual:
        raise ValueError('complete original model file inventory required')
    for name, expected in files.items():
        if Path(name).name != name or file_hash(base/name) != expected:
            raise ValueError('original model inventory changed')
    candidate_hash = None
    if cfg.get('candidate'):
        candidate_hash = digest(json.loads((Path(cfg['candidate'])/'manifest.json').read_text()))
    if cfg.get('exported_model'):
        export = Path(cfg['exported_model']).resolve(strict=True)
        replay = export/'replay-patch'/'manifest.json'
        if candidate_hash is None or digest(json.loads(replay.read_text())) != candidate_hash:
            raise ValueError('vllm export must bind the exact candidate patch')
        receipt = json.loads((export/'export.json').read_text())
        if not receipt.get('replay_equivalence_verified'):
            raise ValueError('export replay equivalence receipt required')
        for entry in receipt['files']:
            target = (export/entry['file']).resolve(strict=True)
            if not target.is_relative_to(export) or file_hash(target) != entry['sha256']:
                raise ValueError('export inventory changed')
        # The export receipt must cover every original file; edited tensors are
        # checked by Operations.export against base+patch before receipt creation.
        if not set(files) <= {entry['file'] for entry in receipt['files']}:
            raise ValueError('incomplete exported model inventory')
    return {'base_inventory_sha256': digest(files), 'candidate_manifest_sha256': candidate_hash}


def render_batch(processor, requests, image_root):
    from PIL import Image
    root = Path(image_root).resolve(strict=True)
    texts, image_groups, paths = [], [], []
    try:
        for request in requests:
            messages, images, image_paths = [], [], []
            for message in request['messages']:
                if message['role'] not in ('user', 'assistant'):
                    raise ValueError('unsupported message role')
                parts = []
                for part in message['content']:
                    if part['type'] == 'text':
                        parts.append(dict(part))
                    elif part['type'] == 'image':
                        path = Path(part['image']).resolve(strict=True)
                        if not path.is_relative_to(root):
                            raise ValueError('image outside narrow development input root')
                        with Image.open(path) as source:
                            image = source.convert('RGB')
                        if image.size != (256, 256):
                            image.close()
                            raise ValueError('expected frozen RGB padded256 input')
                        images.append(image)
                        image_paths.append(str(path))
                        parts.append({'type': 'image', 'image': str(path)})
                    else:
                        raise ValueError('unsupported input part')
                messages.append({'role': message['role'], 'content': parts})
            image_groups.append(images)
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            paths.append(image_paths)
        return texts, image_groups, paths
    except BaseException:
        for group in image_groups:
            for image in group:
                image.close()
        raise


class HFBatch:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.torch, self.cfg = torch, cfg
        self.processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg['model'], torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True).eval()
        self.model.requires_grad_(False)
        if cfg.get('candidate'):
            apply_candidate(self.model, cfg['candidate'])
        self.model.to('cuda')
        self.eos = self.model.generation_config.eos_token_id
        self.eos = self.eos if isinstance(self.eos, list) else [self.eos]

    def generate(self, requests):
        texts, groups, paths = render_batch(self.processor, requests, self.cfg['image_root'])
        try:
            inputs = self.processor(text=texts, images=[image for group in groups for image in group],
                                    return_tensors='pt', padding=True).to('cuda')
            width = inputs['input_ids'].shape[1]
            with self.torch.inference_mode():
                tokens = self.model.generate(**inputs, do_sample=False,
                    max_new_tokens=self.cfg.get('max_new_tokens', 4096))
            self.torch.cuda.synchronize()
            result = []
            for index, request in enumerate(requests):
                completion, finish = trim_completion(tokens[index, width:].cpu().tolist(),
                    self.eos, self.cfg.get('max_new_tokens', 4096))
                mask = inputs['attention_mask'][index].bool()
                result.append({
                    'id': request['id'], 'raw_answer': self.processor.tokenizer.decode(completion, skip_special_tokens=True),
                    'completion_token_ids': completion, 'inference_output_tokens': len(completion),
                    'finish_reason': finish, 'inference_input_tokens': int(mask.sum()),
                    'prompt_token_ids': inputs['input_ids'][index][mask].cpu().tolist(),
                    'rendered_prompt': texts[index], 'image_order': paths[index]})
            return result
        finally:
            for group in groups:
                for image in group:
                    image.close()


class VLLMBatch:
    def __init__(self, cfg):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.sampling = SamplingParams(temperature=0, max_tokens=cfg.get('max_new_tokens', 4096), seed=42)
        self.model = LLM(model=cfg.get('exported_model', cfg['model']),
                         tokenizer=cfg['model'], dtype='bfloat16', seed=42,
                         trust_remote_code=False, tensor_parallel_size=1,
                         max_num_seqs=cfg.get('batch_size', 32),
                         limit_mm_per_prompt={'image': cfg.get('max_images', 5)},
                         gpu_memory_utilization=cfg.get('gpu_memory_utilization', .85),
                         **({'max_model_len': cfg['max_model_len']} if cfg.get('max_model_len') else {}))

    def generate(self, requests):
        texts, groups, paths = render_batch(self.processor, requests, self.cfg['image_root'])
        try:
            prompts = [{'prompt': text, 'multi_modal_data': {'image': images}}
                       for text, images in zip(texts, groups)]
            outputs = self.model.generate(prompts, self.sampling, use_tqdm=False)
            if len(outputs) != len(requests):
                raise ValueError('vllm returned incomplete batch')
            return [{
                'id': request['id'], 'raw_answer': output.outputs[0].text,
                'completion_token_ids': list(output.outputs[0].token_ids),
                'inference_output_tokens': len(output.outputs[0].token_ids),
                'finish_reason': output.outputs[0].finish_reason,
                'inference_input_tokens': len(output.prompt_token_ids),
                'prompt_token_ids': list(output.prompt_token_ids),
                'rendered_prompt': texts[index], 'image_order': paths[index]}
                for index, (request, output) in enumerate(zip(requests, outputs))]
        finally:
            for group in groups:
                for image in group:
                    image.close()


def generation(cfg, *, backend_factory=None, evaluator=None):
    """One atomic development pass. Existing directories cannot be resumed."""
    if cfg.get('split') not in ('development', 'explore'):
        raise ValueError('fast backend is development-only')
    stage = cfg['stage']
    if stage not in SIZES:
        raise ValueError('fast backend only permits S1/S2/PARITY64; FULL2917 requires frozen HF')
    batch_size = cfg.get('batch_size', 32)
    if type(batch_size) is not int or batch_size < 32:
        raise ValueError('batch size must be at least 32')
    if cfg.get('wall_hours', 6) != 6 or cfg.get('token_target') is not None:
        raise ValueError('new tracks use six-hour wall and unlimited editor tokens')
    requests = read_rows(cfg['requests'])
    validate_requests(requests, 'generation', 'development')
    ids = [row['id'] for row in requests]
    require_dev(ids, read_rows(cfg['development_ids']))
    if len(ids) != SIZES[stage]:
        raise ValueError('exact preregistered stage size required')
    if stage == 'S2':
        promotion = json.loads(Path(cfg['s1_receipt']).read_text())
        if not promotion.get('advance') or promotion.get('stage') != 'S1' or promotion.get('candidate_id') != cfg['candidate_id']:
            raise ValueError('S2 requires this candidate S1 promotion receipt')
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = float(cfg['track_started_at']) + 6*3600
    profile = {'schema': 'wsp03-fast-runtime/1', 'stage': stage, 'split': cfg['split'],
               'candidate_id': cfg['candidate_id'], 'batch_size': batch_size, 'padding_side': 'left',
               'max_new_tokens': cfg.get('max_new_tokens', 4096), 'do_sample': False, 'seed': 42,
               'requests_sha256': digest(requests), 'config_sha256': digest(cfg),
               'track_started_at': cfg['track_started_at'], 'wall_hours': 6, 'token_target': None,
               'development_only': True, 'final_hf_eligible': False, 'started_at': started}
    rows, status, error = [], 'FAILED', None
    json_write(output/'attempt.json', profile)
    previous_handler = None
    try:
        if time.time() >= deadline:
            status = 'ABANDONED_WALL'
            raise TimeoutError("six-hour track wall already exhausted")
        evaluator = evaluator or evaluator_for(cfg)
        if backend_factory is None:
            profile.update(verify_model(cfg))
            profile.update(select_backend(cfg))
            backend_factory = HFBatch if profile['backend'] == 'hf' else VLLMBatch
        else:
            profile.update(cfg.get('fixture_identity', {}), backend='synthetic')
        if stage == 'S2' and promotion.get('candidate_binding'):
            bound = promotion['candidate_binding']
            for key in ('candidate_manifest_sha256', 'base_inventory_sha256', 'max_new_tokens', 'do_sample', 'backend'):
                if bound.get(key) != profile.get(key):
                    raise ValueError('S1 promoted candidate/runtime changed: '+key)
        json_write(output/'runtime_profile.json', profile)
        # One wall deadline includes model setup and all batches. Prior completed
        # batches stay durable if the native backend times out in an active batch.
        def timeout(*_):
            raise TimeoutError('six-hour track wall exhausted')
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.time()))
        backend = backend_factory(cfg)
        profile['chat_template_sha256'] = hashlib.sha256(str(backend.processor.chat_template).encode()).hexdigest()
        json_write(output/'runtime_profile.json', profile)
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset:offset+batch_size]
            tick = time.monotonic()
            produced = backend.generate(batch)
            if [row['id'] for row in produced] != [row['id'] for row in batch]:
                raise ValueError('batch output identity/order mismatch')
            wall = time.monotonic()-tick
            with (output/'raw_answers.jsonl').open('a') as stream:
                for request, row in zip(batch, produced):
                    row.update(request_sha256=digest(request), references=request.get('references', []),
                               rendered_prompt_sha256=hashlib.sha256(row['rendered_prompt'].encode()).hexdigest(),
                               batch_wall_seconds=wall, batch_n=len(batch), completed=True)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    stream.flush()
                    rows.append(row)
            if degenerate(rows, evaluator):
                status = 'ABANDONED_DEGENERATE'
                break
        else:
            status = 'COMPLETE'
    except TimeoutError as caught:
        status, error = 'ABANDONED_WALL', repr(caught)
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        receipt = dict(profile, status=status, n=len(rows), planned_n=len(requests),
                       error=error, ended_at=time.time(), wall_seconds=time.time()-started,
                       retry_allowed=False, partial_results_preserved=True,
                       summary=summary(rows, evaluator) if evaluator else None)
        json_write(output/'receipt.json', receipt)
    return receipt



def complete_pass(receipt_path, raw_path, stage, candidate_id):
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get('status') != 'COMPLETE' or receipt.get('n') != SIZES[stage]
            or receipt.get('stage') != stage or receipt.get('candidate_id') != candidate_id):
        raise ValueError('only this candidate complete stage receipt may be screened')
    rows = read_rows(raw_path)
    if len(rows) != receipt['n']:
        raise ValueError('receipt/raw count mismatch')
    return rows, receipt


def generation_binding(receipt):
    return {key: receipt.get(key) for key in (
        'base_inventory_sha256', 'candidate_manifest_sha256', 'max_new_tokens',
        'do_sample', 'chat_template_sha256', 'backend')}


def paired_screen(stage, baseline, candidate, gold, evaluator, *, development_ids, candidate_id,
                  baseline_id, s1=None, changed_elements=0):
    require_dev([row['id'] for row in candidate], development_ids)
    left, right = unique(baseline), unique(candidate)
    if stage not in ('S1', 'S2') or len(right) != SIZES[stage] or left.keys() != right.keys():
        raise ValueError('complete paired exact stage coverage required')
    if any(left[key]['request_sha256'] != right[key]['request_sha256'] for key in left):
        raise ValueError('baseline and candidate must use identical requests')
    if stage == 'S2' and (not s1 or not s1.get('advance') or s1.get('candidate_id') != candidate_id
                          or s1.get('baseline_id') != baseline_id):
        raise ValueError('S2 requires matching S1 promotion')
    b, c = summary(baseline, evaluator, gold), summary(candidate, evaluator, gold)
    delta = c['metrics']['mcc']-b['metrics']['mcc']
    if not math.isfinite(delta):
        raise ValueError('nonfinite MCC')
    return {'schema': 'wsp03-screen/1', 'stage': stage, 'candidate_id': candidate_id,
            'baseline_id': baseline_id, 'ids_sha256': digest(sorted(right)), 'changed_elements': changed_elements,
            'baseline': b, 'candidate': c, 'delta_mcc': delta,
            'advance': delta > 0 and c['invalid_rate'] < .20,
            'full_dev_backend': 'original_frozen_hf', 'fast_answers_final_hf_eligible': False}


def nominate(s2_receipts, *, tie_break=('mcc', 'macro_f1', 'invalid', 'changed_elements', 'candidate_id')):
    """Rank complete promoted S2 candidates; return no more than three HF jobs."""
    if tie_break != ('mcc', 'macro_f1', 'invalid', 'changed_elements', 'candidate_id'):
        raise ValueError('declare a supported immutable ranking before diagnostics')
    rows = list(s2_receipts)
    ids = [row['candidate_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate screening receipt')
    if any(row.get('stage') != 'S2' or row['candidate']['n'] != 512 for row in rows):
        raise ValueError('full-dev nomination requires complete S2 receipts')
    if len({(row['baseline_id'], row['ids_sha256']) for row in rows}) > 1:
        raise ValueError('S2 rankings must share baseline and paired development subset')
    rows.sort(key=lambda row: (-row['candidate']['metrics']['mcc'], -row['candidate']['metrics']['macro_f1'],
                               row['candidate']['invalid_count'], row['changed_elements'], row['candidate_id']))
    return {'schema': 'wsp03-full-dev-nominations/1', 'ranking': list(tie_break),
            'nominees': [row['candidate_id'] for row in rows[:3]],
            'candidate_bindings': {row['candidate_id']: row.get('candidate_binding') for row in rows[:3]}, 'n': 2917,
            'required_backend': 'original_frozen_hf', 'reuse_fast_predictions': False}


def parity64(fast, frozen, evaluator, *, development_ids, fast_profile, frozen_profile,
             base_inventory_sha256):
    left, right = unique(fast), unique(frozen)
    require_dev(list(left), development_ids)
    if len(left) != 64 or left.keys() != right.keys():
        raise ValueError('exact same paired64 development identities required')
    if frozen_profile.get('schema') != 'wsp03-worker/1' or frozen_profile.get('batch_size') != 1:
        raise ValueError('reference must come from original frozen HF worker')
    if (frozen_profile.get('mode') != 'generation' or frozen_profile.get('split') not in ('development', 'explore')
            or frozen_profile.get('max_new_tokens') != 4096 or frozen_profile.get('do_sample') is not False):
        raise ValueError('frozen HF generation contract changed')
    for key in ('candidate_manifest_sha256', 'chat_template_sha256', 'max_new_tokens', 'do_sample'):
        if fast_profile.get(key) != frozen_profile.get(key):
            raise ValueError('paired backend binding differs: '+key)
    if fast_profile.get('base_inventory_sha256') != base_inventory_sha256:
        raise ValueError('frozen reference base inventory differs')
    if any(left[key]['request_sha256'] != right[key]['request_sha256'] or
           left[key]['rendered_prompt_sha256'] != right[key]['rendered_prompt_sha256'] for key in left):
        raise ValueError('paired64 request/prompt identity differs')
    agree = sum(evaluator.parse(left[key]['raw_answer']) == evaluator.parse(right[key]['raw_answer']) for key in left)
    return {'schema': 'wsp03-fast-parity/1', 'n': 64, 'agreements': agree,
            'agreement': agree/64, 'threshold': .95, 'passed': agree/64 >= .95,
            'use': 'development_screen' if agree/64 >= .95 else 'coarse_screen_only',
            'candidate_manifest_sha256': fast_profile['candidate_manifest_sha256'],
            'base_inventory_sha256': base_inventory_sha256,
            'ids_sha256': digest(sorted(left)), 'full_dev_backend': 'original_frozen_hf',
            'reuse_fast_predictions_for_hf': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'screen', 'nominate', 'parity'))
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.action == 'generate':
        result = generation(cfg)
    elif args.action == 'nominate':
        result = nominate([json.loads(Path(path).read_text()) for path in cfg['s2_receipts']])
        json_write(cfg['output'], result)
    else:
        evaluator = evaluator_for(cfg)
        dev = read_rows(cfg['development_ids'])
        if args.action == 'screen':
            baseline, baseline_receipt = complete_pass(cfg['baseline_receipt'], cfg['baseline'], cfg['stage'], cfg['baseline_id'])
            candidate, candidate_receipt = complete_pass(cfg['candidate_receipt'], cfg['candidate'], cfg['stage'], cfg['candidate_id'])
            result = paired_screen(cfg['stage'], baseline, candidate,
                read_rows(cfg['development_labels']), evaluator, development_ids=dev,
                candidate_id=cfg['candidate_id'], baseline_id=cfg['baseline_id'], changed_elements=cfg['changed_elements'],
                s1=json.loads(Path(cfg['s1_receipt']).read_text()) if cfg.get('s1_receipt') else None)
            result['baseline_binding'] = generation_binding(baseline_receipt)
            result['candidate_binding'] = generation_binding(candidate_receipt)
        else:
            result = parity64(read_rows(cfg['fast']), read_rows(cfg['frozen']), evaluator,
                development_ids=dev, fast_profile=json.loads(Path(cfg['fast_profile']).read_text()),
                frozen_profile=json.loads(Path(cfg['frozen_profile']).read_text()),
                base_inventory_sha256=cfg['base_inventory_sha256'])
        json_write(cfg['output'], result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
