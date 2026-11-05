#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only factual model context, projected from hash-pinned JSON/text only.

Never opens tensor files or executes a tokenizer/template/configuration hook.
The supplied header catalogue is already part of the trusted preflight.
"""
from pathlib import Path
import re

from tht_contract import json_hash, sha256
from suite_runtime import json_write
from wsp_weights import regular_file, strict_json

MAX_SOURCE_BYTES = 24 * 1024**2
ARCH_KEYS = set('architectures dtype torch_dtype model_type transformers_version text_config vision_config '
                'tie_word_embeddings image_token_id video_token_id vision_start_token_id vision_end_token_id '
                'attention_bias attention_dropout bos_token_id eos_token_id pad_token_id head_dim hidden_act '
                'hidden_size initializer_range intermediate_size max_position_embeddings num_attention_heads '
                'num_hidden_layers num_key_value_heads rms_norm_eps rope_scaling rope_theta use_cache vocab_size '
                'mrope_interleaved mrope_section rope_type type factor original_max_position_embeddings '
                'deepstack_visual_indexes depth in_channels num_heads num_position_embeddings out_hidden_size '
                'patch_size spatial_merge_size temporal_patch_size'.split())
TOKEN_KEYS = set('add_bos_token add_prefix_space added_tokens_decoder additional_special_tokens bos_token '
                 'clean_up_tokenization_spaces eos_token errors extra_special_tokens model_max_length pad_token '
                 'processor_class split_special_tokens tokenizer_class unk_token content lstrip rstrip normalized '
                 'single_word special id'.split())
PROCESSOR_KEYS = set('crop_size height width data_format default_to_square disable_grouping do_center_crop '
                     'do_convert_rgb do_normalize do_pad do_rescale do_resize image_mean image_processor_type '
                     'image_std input_data_format max_pixels merge_size min_pixels pad_size patch_size '
                     'processor_class resample rescale_factor return_tensors size shortest_edge longest_edge '
                     'temporal_patch_size'.split())
GENERATION_KEYS = set('bos_token_id do_sample eos_token_id pad_token_id temperature top_k top_p transformers_version'.split())
PRIVATE_PATH = re.compile(r'(/(?:home|workspace|root|tmp|mnt|opt|var|Users)/|[A-Za-z]:\\|https?://)')


def project(value, allowed, removed, path=''):
    """Allowlisted inert configuration values; record removed field names only."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            location = path + '/' + key
            if key not in allowed and not key.isdecimal():
                removed.append(location)
                continue
            if isinstance(child, str) and PRIVATE_PATH.search(child):
                removed.append(location)
                continue
            result[key] = project(child, allowed, removed, location)
        return result
    if isinstance(value, list):
        if any(isinstance(v, str) and PRIVATE_PATH.search(v) for v in value):
            raise ValueError('host path in projected configuration list')
        return [project(v, allowed, removed, path + '/*') for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('non-JSON context')


def context_documents(task):
    """Only these explicit metadata files may be read, all from the same base."""
    root = Path(task['model'])
    declared = {f['name']: f for f in task['declared_base']['files']}
    sources, removed = {}, {}

    def read(name, keys=None):
        expected = declared[name]
        path = regular_file(root, name)
        if path.stat().st_size != expected['size'] or path.stat().st_size > MAX_SOURCE_BYTES or sha256(path) != expected['sha256']:
            raise ValueError('context base source changed: ' + name)
        sources[name] = {'sha256': expected['sha256'], 'bytes': expected['size']}
        data = path.read_bytes()
        if name.endswith('.jinja'):
            text = data.decode('utf-8')
            if PRIVATE_PATH.search(text):
                raise ValueError('host path in chat template')
            return text
        result = strict_json(data)
        if keys is not None:
            removed[name] = []
            result = project(result, keys, removed[name])
        return result

    architecture = read('config.json', ARCH_KEYS)
    tokenizer = read('tokenizer.json')
    vocab = tokenizer['model']['vocab']
    if not isinstance(vocab, dict) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in vocab.items()):
        raise ValueError('tokenizer vocabulary must map literal token strings to nonnegative IDs')
    if len(set(vocab.values())) != len(vocab):
        raise ValueError('duplicate vocabulary ID')
    # Token strings are data, so '/' and code-shaped vocabulary entries are
    # retained verbatim. No path or code is interpreted by this projection.
    added = []
    for row in tokenizer.get('added_tokens', []):
        if set(row) - {'id', 'content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special'}:
            raise ValueError('unexpected added-token fields')
        if type(row.get('id')) is not int or not isinstance(row.get('content'), str):
            raise ValueError('invalid added token')
        added.append(row)
    removed['tokenizer.json'] = sorted('/' + k for k in tokenizer if k not in ('model', 'added_tokens'))
    removed['tokenizer.json'] += sorted('/model/' + k for k in tokenizer['model'] if k not in ('type', 'vocab'))
    documents = {
        'architecture.json': architecture,
        'tokenizer.json': {'model_type': tokenizer['model']['type'], 'vocabulary': vocab, 'added_tokens': added,
                           'configuration': read('tokenizer_config.json', TOKEN_KEYS),
                           'special_tokens': read('special_tokens_map.json', TOKEN_KEYS)},
        'chat_template.jinja': read('chat_template.jinja'),
        'inference.json': {'frozen_profile': task['inference'], 'seed': task['seed'],
                           'base_generation_defaults': read('generation_config.json', GENERATION_KEYS),
                           'processor_configuration': read('preprocessor_config.json', PROCESSOR_KEYS),
                           'inference_worker_sha256': task['code_hashes']['inference_worker.py'],
                           'precedence': 'Frozen runtime generation settings override base defaults; context is descriptive only.'},
        'tensor_names.json': {
            'description': 'Exact dotted module/parameter names from the base safetensors headers. Numeric path segments are module indices; final segments are parameter names. Shapes/dtypes are header declarations, not measured values. No tensor or edit is recommended.',
            'tensors': {name: {k: meta[k] for k in ('dtype', 'shape', 'elements')} for name, meta in sorted(task['header_catalog']['tensors'].items())}},
    }
    manifest = {'schema': 'wsp-model-context/1', 'base_fingerprint': task['declared_base']['fingerprint'],
                'sources': sources, 'header_catalog_sha256': json_hash(task['header_catalog']),
                'inference_profile_sha256': json_hash(task['inference']), 'seed': task['seed'],
                'projection': {'removed_fields': removed,
                               'policy': 'Inert allowlisted JSON/text only; no host paths, credentials, executable hooks, examples or holdout. Vocabulary strings and approved chat template are literal data. Tokenizer merges/normalizers are omitted; this is context, not an executable tokenizer.'},
                'permissions': 'read-only context; architecture/tokenizer/prompt/preprocessing/generation remain frozen'}
    return documents, manifest


def build_context(task, output):
    output = Path(output)
    documents, manifest = context_documents(task)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in documents.items():
        if name.endswith('.json'):
            json_write(output / name, content)
        else:
            (output / name).write_text(content)
    manifest['files'] = {name: sha256(output / name) for name in sorted(documents)}
    json_write(output / 'manifest.json', manifest)
    for path in output.iterdir():
        path.chmod(0o444)
    return {'directory': str(output.resolve()), 'manifest_sha256': sha256(output / 'manifest.json'),
            'bundle_sha256': json_hash(manifest), 'files': manifest['files'],
            'base_fingerprint': manifest['base_fingerprint']}


def verify_context(task, *, verify_sources=True):
    binding = task['model_context']
    root = Path(binding['directory'])
    manifest_path = regular_file(root, 'manifest.json')
    manifest = strict_json(manifest_path.read_bytes())
    if (sha256(manifest_path) != binding['manifest_sha256'] or json_hash(manifest) != binding['bundle_sha256'] or
            manifest['base_fingerprint'] != task['declared_base']['fingerprint'] or manifest['files'] != binding['files']):
        raise ValueError('context lineage/hash mismatch')
    if {p.name for p in root.iterdir()} != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('unexpected model-context file')
    for name, digest in manifest['files'].items():
        if sha256(regular_file(root, name)) != digest:
            raise ValueError('model context changed')
    if verify_sources:
        documents, expected = context_documents(task)
        if {k: v for k, v in manifest.items() if k != 'files'} != expected:
            raise ValueError('model-context source lineage changed')
        for name, content in documents.items():
            actual = (root / name).read_text()
            if (strict_json(actual) if name.endswith('.json') else actual) != content:
                raise ValueError('model-context projection changed')
    return root
