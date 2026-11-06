# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic regression for complete WSP base assets, without model execution."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import wsp_prepare
import wsp_runtime
from tht_contract import sha256
from test_wsp import write_tensors


def fixture_model(tmp_path):
    model = tmp_path / 'model'
    write_tensors(model)
    (model / 'tokenizer.json').write_text(json.dumps({'model': {'merges': [['a', 'b'], ['ab', 'c']]}}))
    declared = {'files': [{'name': p.name, 'size': p.stat().st_size, 'sha256': sha256(p)}
                          for p in sorted(model.iterdir())]}
    (model / 'tao_conversion_provenance.json').write_text(json.dumps({'prepared': declared}))
    (model / 'merges.txt').write_text('#version: 0.2\na b\nab c\n')
    return model, declared


def test_sidecar_is_bound_to_pinned_tokenizer_without_weight_reads(tmp_path, monkeypatch):
    model, declared = fixture_model(tmp_path)
    real_hash = wsp_prepare.sha256

    def nonweight_hash(path):
        assert Path(path).suffix != '.safetensors'
        return real_hash(path)

    monkeypatch.setattr(wsp_prepare, 'sha256', nonweight_hash)
    extra = wsp_prepare.supplemental_assets(model, declared)
    assert extra[0]['sha256'] == sha256(model / 'merges.txt')
    assert extra[0]['derived_from']['sha256'] == sha256(model / 'tokenizer.json')
    assert all(row['name'] != 'merges.txt' for row in declared['files'])


def test_original_manifest_without_sidecar_remains_supported(tmp_path):
    model, declared = fixture_model(tmp_path)
    (model / 'merges.txt').unlink()
    assert wsp_prepare.supplemental_assets(model, declared) == []


@pytest.mark.parametrize('mutation', ['unknown_asset', 'missing_asset', 'changed_merges',
                                     'reordered_merges', 'symlink', 'changed_tokenizer', 'unknown_header'])
def test_incomplete_or_unbound_assets_fail_before_approval(tmp_path, mutation):
    model, declared = fixture_model(tmp_path)
    if mutation == 'unknown_asset':
        (model / 'extra.json').write_text('{}')
    elif mutation == 'missing_asset':
        (model / 'model.safetensors').unlink()
    elif mutation == 'changed_merges':
        (model / 'merges.txt').write_text('a b\nx y\n')
    elif mutation == 'reordered_merges':
        (model / 'merges.txt').write_text('ab c\na b\n')
    elif mutation == 'symlink':
        (model / 'merges.txt').unlink()
        (model / 'merges.txt').symlink_to(model / 'tokenizer.json')
    elif mutation == 'changed_tokenizer':
        (model / 'tokenizer.json').write_text('{}')
    else:
        (model / 'merges.txt').write_text('#version: 0.3\na b\nab c\n')
    with pytest.raises(ValueError):
        wsp_prepare.supplemental_assets(model, declared)


def test_approved_initializer_copies_and_verifies_supplemental_asset(tmp_path, monkeypatch):
    model, declared = fixture_model(tmp_path)
    task = {'model': str(model), 'run_dir': str(tmp_path / 'run'), 'declared_base': declared,
            'additional_base_assets': wsp_prepare.supplemental_assets(model, declared),
            'conversion_provenance_sha256': sha256(model / 'tao_conversion_provenance.json'),
            'budgets': {'storage_bytes': 10_000_000}}
    monkeypatch.setattr(wsp_runtime, 'verify_views', lambda _: None)
    root, manifest = wsp_runtime.initialize_run(task, {})
    assert sha256(root / 'base/merges.txt') == sha256(model / 'merges.txt')
    assert (root / 'base/merges.txt').stat().st_ino != (model / 'merges.txt').stat().st_ino
    assert {r['name'] for r in manifest['files']} == {p.name for p in model.iterdir()}


def test_changed_sidecar_after_approval_is_rejected(tmp_path, monkeypatch):
    model, declared = fixture_model(tmp_path)
    task = {'model': str(model), 'run_dir': str(tmp_path / 'run'), 'declared_base': declared,
            'additional_base_assets': wsp_prepare.supplemental_assets(model, declared),
            'conversion_provenance_sha256': sha256(model / 'tao_conversion_provenance.json'),
            'budgets': {'storage_bytes': 10_000_000}}
    (model / 'merges.txt').write_text('a b\nab x\n')
    monkeypatch.setattr(wsp_runtime, 'verify_views', lambda _: None)
    with pytest.raises(ValueError, match='base content fingerprint mismatch'):
        wsp_runtime.initialize_run(task, {})
    assert not (tmp_path / 'run/base').exists()
