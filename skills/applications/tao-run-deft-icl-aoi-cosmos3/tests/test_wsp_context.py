# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic metadata only: no real checkpoint or model/tokenizer loading."""
import copy
import json
from pathlib import Path

import pytest

from test_wsp import json_hash, sha256
from wsp_context import build_context, context_documents, verify_context
from wsp_runtime import sandbox_mounts


def context_task(tmp_path):
    base = tmp_path / 'metadata-source'
    base.mkdir()
    files = {
        'config.json': {'model_type': 'fixture', 'hidden_size': 2, '_name_or_path': '/home/secret',
                        'auto_map': {'AutoModel': 'malicious.py'}, 'token': 'secret'},
        'tokenizer.json': {'model': {'type': 'BPE', 'vocab': {'a': 0, '/': 1}, 'merges': []},
                           'added_tokens': [{'id': 2, 'content': '<end>', 'special': True}], 'decoder': {'type': 'fixture'}},
        'tokenizer_config.json': {'eos_token': '<end>', 'auto_map': 'hook.py', 'name_or_path': '/home/secret'},
        'special_tokens_map.json': {'eos_token': '<end>'},
        'generation_config.json': {'eos_token_id': 2, 'do_sample': True},
        'preprocessor_config.json': {'do_convert_rgb': True, 'size': {'height': 256, 'width': 256}, 'device': '/dev/hidden'},
        'chat_template.jinja': "{{ messages[0]['content'] }}",
    }
    for name, data in files.items():
        (base / name).write_text(data if isinstance(data, str) else json.dumps(data))
    return {'model': str(base), 'declared_base': {'fingerprint': 'a' * 64,
            'files': [{'name': name, 'size': (base / name).stat().st_size, 'sha256': sha256(base / name)} for name in files]},
            'header_catalog': {'tensors': {'model.layers.0.weight': {'dtype': 'F32', 'shape': [2], 'elements': 2}}},
            'inference': {'prompt': 'Frozen synthetic prompt', 'do_sample': False, 'max_new_tokens': 4096,
                          'preprocess': 'RGB/ImageOps.pad/256x256/NEAREST/PNG', 'dtype': 'bfloat16', 'attention': 'sdpa'},
            'seed': 42, 'code_hashes': {'inference_worker.py': 'b' * 64}}


def test_context_exact_projection_lineage_and_no_payload_reads(tmp_path, monkeypatch):
    task = context_task(tmp_path)
    inference = copy.deepcopy(task['inference'])
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert not str(path).endswith('.safetensors')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    task['model_context'] = build_context(task, tmp_path / 'context')
    root = verify_context(task)
    joined = '\n'.join(p.read_text() for p in root.iterdir())
    assert '/home/secret' not in joined and 'hook.py' not in joined and 'malicious.py' not in joined
    assert json.loads((root / 'tokenizer.json').read_text())['vocabulary'] == {'a': 0, '/': 1}
    assert json.loads((root / 'inference.json').read_text())['frozen_profile'] == inference == task['inference']
    assert json.loads((root / 'manifest.json').read_text())['projection']['removed_fields']['config.json']
    mounts = sandbox_mounts(tmp_path, tmp_path / 'contract', tmp_path / 'socket', model_context=root)
    assert (root, '/model_context', True) in mounts
    assert not any(target in ('/model', '/holdout', '/home/ubuntu/workspace') for _, target, _ in mounts)


@pytest.mark.parametrize('change', ['source', 'projection', 'extra', 'symlink', 'profile', 'base'])
def test_context_tamper_rejected(tmp_path, change):
    task = context_task(tmp_path)
    task['model_context'] = build_context(task, tmp_path / 'context')
    root = Path(task['model_context']['directory'])
    if change == 'source': (Path(task['model']) / 'config.json').write_text('{}')
    elif change == 'projection':
        p = root / 'architecture.json'; p.chmod(0o644); p.write_text('{}')
    elif change == 'extra': (root / 'weights.txt').write_text('forbidden')
    elif change == 'symlink':
        p = root / 'architecture.json'; p.unlink(); p.symlink_to(Path(task['model']) / 'config.json')
    elif change == 'profile': task['inference']['prompt'] = 'changed'
    elif change == 'base': task['declared_base']['fingerprint'] = 'f' * 64
    with pytest.raises(ValueError): verify_context(task)


def test_context_rejects_path_in_template_and_invalid_vocabulary(tmp_path):
    task = context_task(tmp_path)
    p = Path(task['model']) / 'chat_template.jinja'
    p.write_text('/home/ubuntu/private')
    for row in task['declared_base']['files']:
        if row['name'] == p.name: row.update(size=p.stat().st_size, sha256=sha256(p))
    with pytest.raises(ValueError, match='host path'): context_documents(task)


def test_chat_template_literal_newline_is_not_a_windows_drive(tmp_path):
    task = context_task(tmp_path)
    p = Path(task['model']) / 'chat_template.jinja'
    for text, valid in [('XML tags:\\n<tools>', True), ('C:\\private\\file', False)]:
        p.write_text(text)
        for row in task['declared_base']['files']:
            if row['name'] == p.name: row.update(size=p.stat().st_size, sha256=sha256(p))
        if valid:
            assert context_documents(task)[0]['chat_template.jinja'] == text
        else:
            with pytest.raises(ValueError, match='host path'): context_documents(task)
