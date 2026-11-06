# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial synthetic fixtures for generated training-data ancestry."""
import copy
import csv
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from tht_contract import CLASSES, sha256
from suite_runtime import json_write


def api():
    module = importlib.import_module('generator_contract')
    assert hasattr(module, 'verify_prepared'), 'source ancestry contract is not implemented'
    return module


def image(path, color, mode='RGB'):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (8, 8), color).save(path)
    return path


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.package = self.root / 'package'; self.public = self.root / 'public'
        self.mapping = {f'Widget+defect_{i}': c for i, c in enumerate(CLASSES[1:])}
        self.sources = []; self.mining = []
        for i, label in enumerate(CLASSES):
            clean = i == 0; kind = 'clean' if clean else 'donor_reference'
            category = 'clean_image' if clean else f'anomaly_image/defect_{i-1}'
            maskcat = 'cad_mask' if clean else f'mask/defect_{i-1}'
            source = image(self.package / 'dataset/Widget' / category / f'item{i}.png', (i * 20, 20, 40))
            mask = image(self.package / 'dataset/Widget' / maskcat / f'item{i}_mask.png', 255, 'L')
            if clean:
                mask = mask.rename(mask.with_name('item0.png'))
            raw = self.public / 'mining/images' / source.name; raw.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, raw)
            ident = {'id': f'item{i}', 'image': str(raw.relative_to(self.public)),
                     'image_sha256': sha256(raw), 'labels': [label], 'object_name': f'joint{i}', 'scan_id': 'scan'}
            self.mining.append(ident)
            assets = []
            for path, role in [(source, 'clean_image' if clean else 'donor_image'), (mask, 'cad_mask' if clean else 'donor_mask')]:
                assets.append({'prepared_path': str(path.relative_to(self.package)), 'prepared_sha256': sha256(path),
                               'original_sha256': sha256(path), 'bytes': path.stat().st_size, 'asset_role': role,
                               'archive_member': 'original/' + path.name, 'transform': 'byte_copy_and_filename_normalization'})
            self.sources.append({'kind': kind, 'source_stem': source.stem, 'category': category.split('/')[-1],
                                 'matched_identity': {'source_id': ident['id'], 'role': 'mining', 'labels': [label],
                                     'joint': [f'joint{i}', 'scan'], 'image': ident['image'], 'image_sha256': sha256(raw)},
                                 'assets': assets, 'classifier_training_admission': False,
                                 'pixel_proof': {'equals_frozen_mining': True, 'mode': 'RGB', 'width': 8, 'height': 8,
                                     'rgb_sha256': hashlib.sha256(Image.open(source).convert('RGB').tobytes()).hexdigest()}})
        lines(self.public / 'mining/samples.jsonl', self.mining)
        self.reseal()
        self.prep = self.root / 'prepare'; shutil.copytree(self.package / 'dataset', self.prep / 'dataset-view')
        pairs = []; cases = []; summary = []; self.tmp = self.root / 'native-tmp'
        for i, key in enumerate(self.mapping):
            clean = self.prep / 'dataset-view/Widget/clean_image/item0.png'
            donor = self.prep / f'dataset-view/Widget/mask/defect_{i}/item{i+1}_mask.png'
            processed = image(self.tmp / str(i) / (donor.stem + '_largest.png'), 255, 'L')
            placed = image(self.prep / 'inference/item0' / key / (processed.stem + '__seed0.png'), 255, 'L')
            pairs.append({'clean_image': str(clean), 'defect_type': key, 'submask': str(donor), 'n_seeds': 1,
                          'cad_mask': str(self.prep / 'dataset-view/Widget/cad_mask/item0.png'),
                          'cad_mask_label': str(self.prep / 'dataset-view/semantic_segmentation_labels.json')})
            cases.append({'image_filename': str(clean), 'mask_filename': str(placed), 'anomaly_type': key,
                          'guidance': 3.0, 'num_steps': 35, 'crop_ratio': 2.0, 'num_generated_images': 1,
                          'poisson_blend': False, 'iteration_generation_max_instance': 5})
            summary.append({'name': 'item0', 'defect_type': key, 'status': 'OK', 'seeds': {'0': {'status': 'OK'}}})
        json_write(self.prep / 'inference/amp_samples.json', pairs)
        json_write(self.prep / 'inference/summary.json', summary)
        lines(self.prep / 'inference-testcase.jsonl', cases)
        self.cases = cases

    def reseal(self):
        json_write(self.package / 'class_mapping.json', self.mapping)
        json_write(self.package / 'audit/protected_joints.json', [['evaluation', 'scan']])
        self.package.joinpath('audit/exclude_stems.txt').write_text('protected\n')
        json_write(self.package / 'dataset/semantic_segmentation_labels.json', {'pad': [255, 255, 255]})
        lines(self.package / 'audit/admitted_sources.jsonl', self.sources)
        files = [{'path': str(p.relative_to(self.package)), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted(self.package.rglob('*')) if p.is_file() and p.name != 'inventory.json']
        json_write(self.package / 'inventory.json', files)

    def generation(self, blocked=False):
        sys.path.insert(0, str(SCRIPTS.parents[2] / 'data/tao-generate-anomalies/scripts'))
        from anomalygen_contract import verify_generation
        g = self.root / 'generation'; bucket = g / 'sdg'; bucket.mkdir(parents=True)
        shutil.copytree(self.package/'dataset', g/'dataset-view')
        lines(g / 'testcase.jsonl', self.cases)
        header = ['index', 'output_filename', 'image_filename', 'mask_filename', 'anomaly_type']
        with (bucket / 'texture_ft_generation_result.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=header); writer.writeheader()
            if not blocked:
                for i, case in enumerate(self.cases):
                    name = f'output{i}.png'; image(bucket / 'reconstructed_image' / name, (130 + i, 90, 60))
                    image(bucket / 'original_image' / name, (0, 20, 40)); image(bucket / 'original_mask' / name, 255, 'L')
                    writer.writerow({'index': i, 'output_filename': name, **{k: case[k] for k in header[2:]}})
        with (bucket / 'guardrail_blocked.csv').open('w') as f:
            header = ['index', 'output_idx', 'image_filename', 'mask_filename', 'anomaly_type', 'guardrail']
            writer = csv.DictWriter(f, fieldnames=header); writer.writeheader()
            if blocked:
                for i, case in enumerate(self.cases): writer.writerow({'index': i, 'output_idx': -1, 'guardrail': 'text', **{k: case[k] for k in header[2:-1]}})
        report = verify_generation(g / 'testcase.jsonl', bucket)
        manifest = {'status': 'COMPLETE', 'guardrail_requested': True, 'native_timing': {'guardrail_enabled': True},
                    'testcase': str(g / 'testcase.jsonl'), 'testcase_sha256': sha256(g / 'testcase.jsonl'),
                    'selected_output': str(bucket), 'generation': report,
                    'quality': {'skipped': True, 'search_rounds': 0, 'score': 'nn'}}
        json_write(g / 'manifest.json', manifest)
        return g / 'manifest.json'


class AncestryTests(unittest.TestCase):
    def test_source_policy_and_immutable_assets(self):
        c = api()
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); c.verify_prepared(f.package, f.public)
            p = f.package / f.sources[1]['assets'][1]['prepared_path']; p.write_bytes(b'changed')
            with self.assertRaises(ValueError): c.verify_prepared(f.package, f.public)
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.sources[1]['matched_identity']['joint'] = ['evaluation', 'scan']; f.reseal()
            with self.assertRaises(ValueError): c.verify_prepared(f.package, f.public)
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.sources[1]['matched_identity']['labels'] = ['Good']; f.reseal()
            with self.assertRaises(ValueError): c.verify_prepared(f.package, f.public)

    def test_amp_binding_carries_original_donor_and_transformed_mask(self):
        c = api()
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); binding = c.bind_amp(f.package, f.prep, {k: 1 for k in f.mapping}, f.tmp)
            self.assertEqual(len(binding['cases']), 7)
            self.assertIn('donor_source', binding['cases'][0])
            c.verify_ancestry(binding, f.package)
            path = Path(binding['cases'][0]['transformed_mask']['path']); path.write_bytes(b'changed')
            with self.assertRaises(ValueError): c.verify_ancestry(binding, f.package)

    def test_substituted_pair_donor_or_cad_is_rejected(self):
        c = api()
        for key in ('submask', 'cad_mask'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp); path = f.prep / 'inference/amp_samples.json'; pairs = json.loads(path.read_text())
                replacement = image(f.root / 'protected.png', 255, 'L'); pairs[0][key] = str(replacement); json_write(path, pairs)
                with self.assertRaises(ValueError): c.bind_amp(f.package, f.prep, {k: 1 for k in f.mapping}, f.tmp)

    def test_generated_and_all_blocked_admission_revalidate_cumulative_bytes(self):
        c = api()
        for blocked in (False, True):
            with self.subTest(blocked=blocked), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp); b = c.bind_amp(f.package, f.prep, {k: 1 for k in f.mapping}, f.tmp)
                bp = f.root / 'ancestry.json'; json_write(bp, b); manifest = f.generation(blocked)
                result = c.admit_generation(manifest, bp, f.package, 1, expected_search_rounds=0)
                self.assertEqual(len(result['records']), 0 if blocked else 7)
                if blocked:
                    default = c.admit_generation(manifest, bp, f.package, 1)
                    self.assertEqual(default['records'], [])
                else:
                    with self.assertRaises(ValueError): c.admit_generation(manifest, bp, f.package, 1)
                if not blocked:
                    rec = result['records'][0]; image(Path(rec['image']), (1, 1, 1))
                    with self.assertRaises(ValueError): c.validate_admission(result, f.package)



class TrainingViewTests(unittest.TestCase):
    def test_combined_view_deduplicates_and_revalidates_including_duplicates(self):
        c = api(); self.assertTrue(hasattr(c, 'combined_training_view'), 'combined training admission is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); b = c.bind_amp(f.package, f.prep, {k: 1 for k in f.mapping}, f.tmp)
            bp = f.root / 'ancestry.json'; json_write(bp, b); manifest = f.generation()
            a = c.admit_generation(manifest, bp, f.package, 1, expected_search_rounds=0)
            result = c.combined_training_view(f.public, ['item0'], 'prompt', [a], f.package, f.root / 'input')
            self.assertEqual((result['real_count'], result['synthetic_count'], result['train_records']), (1, 7, 8))
            rows = json.loads((f.root / 'input/train.json').read_text())
            self.assertEqual(len(rows[1]['images']), 1)
            self.assertEqual(json.loads(rows[1]['conversations'][1]['value']), {'labels': [CLASSES[1]]})
            self.assertEqual(c.combined_training_view(f.public, ['item0'], 'prompt', [a], f.package, f.root / 'input'), result)
            image(Path(a['records'][0]['image']), (0, 20, 40))
            with self.assertRaises(ValueError): c.combined_training_view(f.public, ['item0'], 'prompt', [a], f.package, f.root / 'new')

    def test_content_collision_with_different_training_target_stops(self):
        c = api(); self.assertTrue(hasattr(c, 'deduplicate_records'), 'content deduplication is not implemented')
        real = {'id': 'real', 'label': 'Good', 'pixel_key': 'same', 'source': 'real'}
        duplicate = {'id': 'synthetic', 'label': 'Good', 'pixel_key': 'same', 'source': 'synthetic', 'round': 1, 'index': 0}
        kept, skipped = c.deduplicate_records([duplicate, real])
        self.assertEqual([r['id'] for r in kept], ['real']); self.assertEqual(len(skipped), 1)
        with self.assertRaises(ValueError): c.deduplicate_records([real, {**duplicate, 'label': 'Bridge'}])

if __name__ == '__main__': unittest.main()
