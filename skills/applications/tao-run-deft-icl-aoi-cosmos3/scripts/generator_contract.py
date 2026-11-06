# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External generator policy, immutable ancestry and single-image admission."""
from __future__ import annotations

from collections import Counter

from tht_contract import CLASSES


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or len(mapping) != len(CLASSES) - 1:
        raise ValueError('one external native-type mapping per defect class is required')
    if set(mapping.values()) != set(CLASSES) - {'Good'}:
        raise ValueError('generator mapping must cover each defect class exactly once')
    if any(not isinstance(k, str) or len(k.split('+')) != 2 or any(not p or '/' in p or '\\' in p or p in ('.', '..') for p in k.split('+')) for k in mapping):
        raise ValueError('native anomaly keys must be safe texture+defect names')
    return mapping


def validate_cases(rows, counts):
    if not counts or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('every known native type needs a nonnegative integer count')
    if set(r['anomaly_type'] for r in rows) - counts.keys() or any(sum(r['anomaly_type'] == k for r in rows) != n for k,n in counts.items()):
        raise ValueError('realized per-class testcase counts differ from the frozen vector')
    expected = {'num_generated_images': 1, 'guidance': 3.0, 'num_steps': 35,
                'crop_ratio': 2.0, 'poisson_blend': False}
    for row in rows:
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('native testcase generation settings differ from policy')
        if row.get('iteration_generation_max_instance', 5) not in (1, 5):
            raise ValueError('unexpected native instance iteration limit')

import hashlib
import json
import struct
import sys
from pathlib import Path
from PIL import Image, ImageOps
from tht_contract import json_hash, load_jsonl, sha256

_SHARED = (Path(__file__).resolve().parents[2] / 'tao-generate-anomalies/scripts',
           Path(__file__).resolve().parents[3] / 'data/tao-generate-anomalies/scripts')
sys.path.insert(0, str(next((p for p in _SHARED if p.is_dir()), _SHARED[-1])))
from anomalygen_contract import load_exclusions, _excluded, verify_generation
from anomalygen_consumer import load_generation, validate_lineage


def within(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path == root:
        raise ValueError('artifact escapes its declared root')
    return path


def file_binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def verify_file(binding):
    path = Path(binding['path'])
    if not path.is_file() or path.stat().st_size != binding['bytes'] or sha256(path) != binding['sha256']:
        raise ValueError('sealed input or lineage artifact changed: ' + str(path))
    return path


def pixel_identity(path, training=False):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if training:
            image = ImageOps.pad(image, (256, 256), method=Image.Resampling.NEAREST)
        pixels = image.tobytes()
        return {'width': image.width, 'height': image.height,
                'rgb_sha256': hashlib.sha256(pixels).hexdigest(),
                'content_sha256': hashlib.sha256(struct.pack('>II', *image.size) + pixels).hexdigest()}


def verify_prepared(package, public=None):
    """Verify external copied assets and their declared eligible source identities."""
    package = Path(package).resolve(strict=True)
    inventory = json.loads((package / 'inventory.json').read_text())
    names = [r['path'] for r in inventory]
    if len(names) != len(set(names)):
        raise ValueError('duplicate package inventory path')
    for row in inventory:
        p = within(package, row['path'])
        if p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            raise ValueError('prepared inventory changed: ' + row['path'])
    if {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()} != set(names) | {'inventory.json'}:
        raise ValueError('prepared package has uninventoried files')
    mapping = validate_mapping(json.loads((package / 'class_mapping.json').read_text()))
    protected = {tuple(x) for x in json.loads((package / 'audit/protected_joints.json').read_text())}
    excluded = load_exclusions(package / 'audit/exclude_stems.txt')
    sources = load_jsonl(package / 'audit/admitted_sources.jsonl')
    assets = {}; counts = Counter(); native_images = set()
    mining = {r['source_id']: r for r in load_jsonl(Path(public) / 'mining/samples.jsonl')} if public else None
    for source in sources:
        identity = source['matched_identity']; kind = source['kind']
        if kind not in ('clean', 'donor_reference') or identity['role'] != 'mining':
            raise ValueError('generator sources must be explicit eligible Mining identities')
        if tuple(identity['joint']) in protected or any(_excluded(p, excluded) for p in (identity['source_id'], source['source_stem'])):
            raise ValueError('protected original source ancestry')
        if source.get('classifier_training_admission') is not False:
            raise ValueError('original generator references cannot be added as classifier examples')
        expected_roles = {'clean_image', 'cad_mask'} if kind == 'clean' else {'donor_image', 'donor_mask'}
        if {a['asset_role'] for a in source['assets']} != expected_roles or len(source['assets']) != 2:
            raise ValueError('incomplete or duplicate source image/mask binding')
        source_assets = {}
        for asset in source['assets']:
            relative = asset['prepared_path']; path = within(package, relative)
            if relative in assets or _excluded(path, excluded) or _excluded(asset['archive_member'], excluded):
                raise ValueError('duplicate or excluded original/prepared asset')
            if not relative.startswith('dataset/') or sha256(path) != asset['prepared_sha256'] or asset['original_sha256'] != asset['prepared_sha256']:
                raise ValueError('prepared asset differs from its original copied bytes')
            if path.stat().st_size != asset['bytes']:
                raise ValueError('prepared asset size changed')
            assets[relative] = {'source': source, 'asset': asset}; source_assets[asset['asset_role']] = path
            native_images.add(relative)
        image = source_assets['clean_image' if kind == 'clean' else 'donor_image']
        mask = source_assets['cad_mask' if kind == 'clean' else 'donor_mask']
        key = image.parts[-3] + '+' + image.parts[-2] if kind != 'clean' else None
        # Native layout: texture/anomaly_image/defect/image. Texture is four components from the end.
        if kind != 'clean': key = image.parts[-4] + '+' + image.parts[-2]
        label = 'Good' if kind == 'clean' else mapping.get(key)
        if label is None or identity['labels'] != [label]:
            raise ValueError('source class differs from the explicit native class mapping')
        proof = source['pixel_proof']; actual = pixel_identity(image)
        if proof.get('equals_frozen_mining') is not True or proof.get('mode') != 'RGB' or any(proof.get(k) != actual[k] for k in ('width', 'height', 'rgb_sha256')):
            raise ValueError('prepared source pixel proof changed')
        with Image.open(mask) as im:
            if im.size != (actual['width'], actual['height']):
                raise ValueError('source image/mask dimensions differ')
        if mining is not None:
            original = mining.get(identity['source_id'])
            if original is None or original['labels'] != identity['labels'] or original['image_sha256'] != identity['image_sha256']:
                raise ValueError('source identity is not in the frozen Mining pool')
            if [original['object_name'], original['scan_id'].removeprefix('SCN_')] != identity['joint']:
                raise ValueError('source physical joint differs from frozen Mining')
            original_path = within(public, original['image'])
            if sha256(original_path) != identity['image_sha256'] or pixel_identity(original_path) != actual:
                raise ValueError('generator source differs from frozen Mining pixels')
        counts[kind] += 1
    disk_images = {str(p.relative_to(package)) for p in (package / 'dataset').rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') and p.is_file()}
    if disk_images != native_images or not counts['clean'] or not counts['donor_reference']:
        raise ValueError('native generator input files lack complete source admission')
    return {'package': str(package), 'inventory': file_binding(package / 'inventory.json'),
            'mapping': mapping, 'assets': assets, 'source_counts': dict(counts)}


def _largest_pixels(path):
    """Native threshold>127, 8-connected largest component, earliest tie."""
    with Image.open(path) as im:
        im = im.convert('L'); width, height = im.size; pixels = list(im.tobytes())
    remaining = {i for i, p in enumerate(pixels) if p > 127}; best = []
    while remaining:
        seed = min(remaining); remaining.remove(seed); component = [seed]; todo = [seed]
        while todo:
            i = todo.pop(); x, y = i % width, i // width
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and ny * width + nx in remaining:
                        j = ny * width + nx; remaining.remove(j); component.append(j); todo.append(j)
        if len(component) > len(best): best = component
    result = bytearray(width * height)
    for i in best: result[i] = 255
    return (width, height), bytes(result)


def bind_amp(package, preparation, counts, native_tmp):
    prepared = verify_prepared(package); package = Path(package).resolve(); preparation = Path(preparation).resolve()
    pair_path = preparation / 'inference/amp_samples.json'; summary_path = preparation / 'inference/summary.json'
    testcase = preparation / 'inference-testcase.jsonl'; pairs = json.loads(pair_path.read_text())
    summaries = json.loads(summary_path.read_text()); rows = load_jsonl(testcase); validate_cases(rows, counts)
    if set(counts) != set(prepared['mapping']) or len(pairs) != len(summaries):
        raise ValueError('AMP pairs/summary/class map mismatch')
    view = preparation / 'dataset-view'; assets = prepared['assets']; by_output = {}
    def original(path, role):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(view): raise ValueError('AMP source outside the frozen copied view')
        relative = 'dataset/' + path.relative_to(view).as_posix()
        item = assets.get(relative)
        if item is None or item['asset']['asset_role'] != role or sha256(path) != item['asset']['prepared_sha256']:
            raise ValueError('AMP source does not match approved original ancestry')
        return item['source'], file_binding(path)
    for ordinal, (pair, summary) in enumerate(zip(pairs, summaries)):
        clean, clean_file = original(pair['clean_image'], 'clean_image')
        donor, donor_file = original(pair['submask'], 'donor_mask')
        cad, cad_file = original(pair['cad_mask'], 'cad_mask')
        if clean != cad or donor['matched_identity']['labels'] != [prepared['mapping'].get(pair['defect_type'])]:
            raise ValueError('AMP clean/CAD or donor/target mismatch')
        if summary['name'] != Path(pair['clean_image']).stem or summary['defect_type'] != pair['defect_type']:
            raise ValueError('AMP summary is not bound to its pair ordinal')
        labels = Path(pair['cad_mask_label']).resolve(strict=True)
        if labels != (view / 'semantic_segmentation_labels.json') or sha256(labels) != sha256(package / 'dataset/semantic_segmentation_labels.json'):
            raise ValueError('CAD semantic palette changed')
        candidates = sorted(Path(native_tmp).rglob(Path(pair['submask']).stem + '_largest.png'))
        if not candidates: raise ValueError('native transformed donor mask was not retained')
        expected_size, expected_pixels = _largest_pixels(pair['submask'])
        for transformed in candidates:
            with Image.open(transformed) as im:
                if im.size != expected_size or im.convert('L').tobytes() != expected_pixels:
                    raise ValueError('transformed donor mask differs from native largest-component rule')
        transformed = candidates[0]
        for seed in range(pair['n_seeds']):
            if str(seed) not in summary.get('seeds', {}) or summary['seeds'][str(seed)].get('status') == 'FAILED':
                continue
            placed = preparation / 'inference' / Path(pair['clean_image']).stem / pair['defect_type'] / (transformed.stem + f'__seed{seed}.png')
            if not placed.is_file(): raise ValueError('authoritative AMP pair output missing')
            key = (str(Path(pair['clean_image']).resolve()), str(placed.resolve()), pair['defect_type'])
            if key in by_output: raise ValueError('ambiguous AMP pair-to-testcase binding')
            by_output[key] = {'pair_ordinal': ordinal, 'pair': pair, 'seed': seed, 'placement_status': summary['seeds'][str(seed)],
                              'clean_source': clean, 'donor_source': donor,
                              'copied_clean': clean_file, 'copied_donor_mask': donor_file,
                              'cad_mask': cad_file, 'cad_labels': file_binding(labels),
                              'transformed_mask': file_binding(transformed), 'placed_mask': file_binding(placed)}
    cases = []
    for index, row in enumerate(rows):
        key = (str(Path(row['image_filename']).resolve()), str(Path(row['mask_filename']).resolve()), row['anomaly_type'])
        if key not in by_output: raise ValueError('testcase lacks unambiguous original donor ancestry')
        cases.append({'index': row.get('index', index), 'testcase': row, **by_output.pop(key)})
    if by_output: raise ValueError('AMP produced unaccounted placements')
    extra_files = []
    pairing_path = preparation / 'pairing.json'
    if pairing_path.exists():
        pairing = json.loads(pairing_path.read_text())
        order_path=preparation/'testcase-order.json'
        order=json.loads(order_path.read_text())
        for key in ('original','filtered','final'): verify_file(order[key])
        native_rows=load_jsonl(order['filtered']['path'])
        permutation=order['permutation']
        if sorted(r['native_position'] for r in permutation)!=list(range(len(rows))) or sorted(r['slot'] for r in permutation)!=list(range(len(rows))):
            raise ValueError('native testcase permutation is not complete and unique')
        expected={r['slot']:{**native_rows[r['native_position']],'index':r['slot']} for r in permutation}
        if rows!=[expected[i] for i in sorted(expected)]: raise ValueError('final testcase differs from explicit native slot permutation')
        extra_files = [file_binding(pairing_path), file_binding(order_path), order['original'], order['filtered'], *pairing['native_artifacts']]
        for artifact in extra_files: verify_file(artifact)
        resolved = pairing['resolved']
        if len(resolved) != len(cases) or len({r['clean']['id'] for r in resolved}) != len(resolved):
            raise ValueError('native clean work-item coverage is not unique and complete')
        requested = {r['slot']:r for r in pairing['requested']}
        assigned = {(r['anomaly_type'],r['clean']['id']):r for r in resolved}
        for case in cases:
            identity = case['clean_source']['matched_identity']['source_id']
            key = (case['testcase']['anomaly_type'], identity)
            if key not in assigned: raise ValueError('AMP clean differs from immutable requested work item')
            actual = assigned.pop(key); original = requested[actual['slot']]
            if case['index']!=actual['slot']: raise ValueError('final testcase index differs from immutable work slot')
            if original['clean'] is not None and original != actual:
                raise ValueError('native AMP changed an anchored clean work item')
            if original['clean'] is None and (original['anchor_id'] is not None or actual['anchor_id'] is not None):
                raise ValueError('invalid native anchor fallback')
            case['work_item'] = actual
        if assigned: raise ValueError('unaccounted native work items')
    return {'schema': 'generator-ancestry/1', 'preparation': str(preparation), 'native_tmp': str(Path(native_tmp).resolve()), 'package_inventory': prepared['inventory'],
            'counts': counts, 'files': [file_binding(pair_path), file_binding(summary_path), file_binding(testcase), *extra_files], 'cases': cases}


def verify_ancestry(binding, package):
    prepared = verify_prepared(package)
    if binding['package_inventory'] != prepared['inventory']:
        raise ValueError('prepared source inventory changed')
    for item in binding['files']: verify_file(item)
    sources = load_jsonl(Path(package) / 'audit/admitted_sources.jsonl')
    source_digests = {json_hash(s) for s in sources}
    for case in binding['cases']:
        if json_hash(case['clean_source']) not in source_digests or json_hash(case['donor_source']) not in source_digests:
            raise ValueError('bound original source or donor is not approved')
        if case['donor_source']['matched_identity']['labels'] != [prepared['mapping'].get(case['testcase']['anomaly_type'])]:
            raise ValueError('bound donor target class changed')
        for key in ('copied_clean', 'copied_donor_mask', 'cad_mask', 'cad_labels', 'transformed_mask', 'placed_mask'):
            verify_file(case[key])
    validate_cases([x['testcase'] for x in binding['cases']], binding['counts'])
    rebuilt = bind_amp(package, binding['preparation'], binding['counts'], binding['native_tmp'])
    if rebuilt != binding: raise ValueError('native pair ancestry or target binding changed')
    return prepared


def verify_quality_selection(native, manifest_root, search_rounds):
    """Verify native best-NN selection, retaining original/earliest ties."""
    import csv, math
    if not search_rounds: return []
    directories=[manifest_root/'sdg',*[manifest_root/'rounds'/f'round_{n}' for n in range(1,search_rounds+1)]]
    candidates={}
    for ordinal,directory in enumerate(directories):
        data=json.loads((directory/'kpi.json').read_text());scores={}
        for name,block in data.items():
            if name=='Average' or not isinstance(block,dict): continue
            for row in block.get('per_sample',[]):
                try: value=float(row.get('nn_score'))
                except (ValueError,TypeError): value=float('nan')
                scores[Path(row['path']).name]=value if math.isfinite(value) else None
        with (directory/'texture_ft_generation_result.csv').open() as stream:
            for row in csv.DictReader(stream):
                name=Path(row['output_filename']).name
                candidate={'row':row,'source':str(directory),'score':scores.get(name),'ordinal':ordinal}
                incumbent=candidates.get(name)
                if incumbent is None or (candidate['score'] is not None and (incumbent['score'] is None or candidate['score']>incumbent['score'])):
                    candidates[name]=candidate
    selected=Path(native['selected_output']);result=[]
    with (selected/'texture_ft_generation_result.csv').open() as stream:
        for row in csv.DictReader(stream):
            name=Path(row['output_filename']).name;best=candidates.pop(name,None)
            if best is None or any(row[k]!=best['row'][k] for k in ('index','anomaly_type')):
                raise ValueError('selected quality result lacks the same logical candidate')
            for folder in ('reconstructed_image','original_image','original_mask'):
                if sha256(selected/folder/name)!=sha256(Path(best['source'])/folder/name):
                    raise ValueError('quality selection is not the highest finite NN/earliest tie candidate')
            result.append({'index':int(row['index']),'output_filename':name,'selected_bucket':best['source'],
                           'nn_score':best['score'],'tie_priority':best['ordinal']})
    if candidates: raise ValueError('quality selection dropped logical generated identities')
    return result


def admit_generation(manifest, ancestry, package, round_number, expected_search_rounds=3):
    manifest, ancestry = Path(manifest), Path(ancestry)
    binding = json.loads(ancestry.read_text()); prepared = verify_ancestry(binding, package)
    excluded = Path(package) / 'audit/exclude_stems.txt'
    report, rows = load_generation(manifest, excluded)
    native = json.loads(manifest.read_text()); cases = load_jsonl(native['testcase'])
    if cases != [r['testcase'] for r in binding['cases']]:
        raise ValueError('generation testcase differs from pre-generation ancestry')
    expected_rounds = expected_search_rounds if report['generated'] else 0
    quality = native['quality']
    if quality['search_rounds'] != expected_rounds or quality['score'] != 'nn' or bool(quality['skipped']) != (expected_rounds == 0):
        raise ValueError('native quality path differs from frozen policy')
    buckets = [verify_generation(native['testcase'], manifest.parent / 'sdg', excluded)]
    import csv, random
    with (manifest.parent/'sdg/texture_ft_generation_result.csv').open() as stream:
        eligible = {int(r['index']) for r in csv.DictReader(stream)}
    search_rows = [{**r,'index':r.get('index',i)} for i,r in enumerate(cases) if r.get('index',i) in eligible]
    artifacts = [file_binding(manifest), file_binding(ancestry), file_binding(native['testcase'])]
    for number in range(1, expected_rounds + 1):
        bucket = manifest.parent / 'rounds' / f'round_{number}'
        result = verify_generation(bucket / 'testcase.jsonl', bucket, excluded)
        actual_rows=load_jsonl(bucket/'testcase.jsonl')
        if len(actual_rows)!=len(search_rows): raise ValueError('refinement identity count changed')
        for actual,base in zip(actual_rows,search_rows):
            stable={k:v for k,v in actual.items() if k not in ('guidance','crop_ratio')}
            if stable!={k:v for k,v in base.items() if k not in ('guidance','crop_ratio')}:
                raise ValueError('native refinement changed pairing/identity/generation controls')
            if not 1.0<=actual['guidance']<=10.0 or not 1.5<=actual['crop_ratio']<=2.5:
                raise ValueError('native refinement outside frozen trained parameter ranges')
            if number==1:
                rng=random.Random((number*100003+int(base['index'])) & 0x7fffffff)
                if actual['guidance']!=round(rng.uniform(1,10),2) or actual['crop_ratio']!=round(rng.uniform(1.5,2.5),2):
                    raise ValueError('first native search round differs from per-sample uniform fallback')
        if result['testcases'] != report['generated']:
            raise ValueError('refinement attempted blocked or missing initial identities')
        timing = json.loads((bucket / 'timing_summary.json').read_text())
        if timing.get('guardrail_enabled') is not True: raise ValueError('refinement guardrail disabled')
        artifacts += [file_binding(bucket / 'testcase.jsonl'), file_binding(bucket / 'kpi.json'), file_binding(bucket / 'timing_summary.json')]
        buckets.append(result)
    for result in buckets:
        bucket = Path(result['output_dir'])
        for p in sorted(bucket.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.png', '.json', '.jsonl', '.csv'): artifacts.append(file_binding(p))
    records = []
    for row in rows:
        validate_lineage(row['lineage'], exclude_stems=excluded)
        index = int(row['index']); case = next(x for x in binding['cases'] if x['index'] == index)
        label = prepared['mapping'][row['anomaly_type']]
        if case['testcase']['anomaly_type'] != row['anomaly_type']: raise ValueError('generated target differs from approved testcase')
        for key in ('generated_image', 'paired_original', 'paired_mask'): artifacts.append(file_binding(row[key]))
        records.append({'id': f'synthetic-{round_number:03d}-{index:06d}', 'round': round_number, 'index': index,
                        'label': label, 'image': row['generated_image'], 'image_sha256': sha256(row['generated_image']),
                        'training_pixels': pixel_identity(row['generated_image'], training=True), 'lineage': row['lineage']})
    for p in sorted(Path(native['selected_output']).rglob('*')):
        if p.is_file(): artifacts.append(file_binding(p))
    selection=verify_quality_selection(native,manifest.parent,expected_rounds)
    unique = {a['path']: a for a in artifacts}
    return {'schema': 'generator-admission/1', 'round': round_number,
            'manifest': str(manifest.resolve()), 'ancestry': str(ancestry.resolve()),
            'expected_search_rounds': expected_search_rounds, 'package_inventory': prepared['inventory'],
            'records': records, 'artifacts': list(unique.values()), 'logical_counts': report,
            'quality_selection':selection,
            'physical_buckets': buckets, 'physical_requested': sum(r['testcases'] for r in buckets),
            'physical_generated': sum(r['generated'] for r in buckets),
            'physical_guardrail_blocked': sum(r['guardrail_blocked'] for r in buckets),
            'label_semantics': 'intended generator target, not independently human-verified visual truth'}


def validate_admission(admission, package):
    for item in admission['artifacts']: verify_file(item)
    refreshed = admit_generation(admission['manifest'], admission['ancestry'], package, admission['round'], admission['expected_search_rounds'])
    if refreshed != admission: raise ValueError('cumulative synthetic admission changed')
    return admission


def deduplicate_records(records):
    """Real first, then earliest generated round/index; conflicts always stop."""
    ordered = sorted(records, key=lambda r: (r['source'] != 'real', r.get('round', 0), r.get('index', 0), r['id']))
    kept = []; duplicates = []; by_content = {}; ids = set()
    for row in ordered:
        if row['id'] in ids: raise ValueError('duplicate training record identity')
        ids.add(row['id']); previous = by_content.get(row['pixel_key'])
        if previous:
            if previous['label'] != row['label']:
                raise ValueError('identical training pixels have conflicting labels')
            duplicates.append({'id': row['id'], 'retained_id': previous['id'], 'pixel_key': row['pixel_key'],
                               'source': row['source'], 'label': row['label']})
        else:
            by_content[row['pixel_key']] = row; kept.append(row)
    return kept, duplicates


def immutable_json(path, value):
    from suite_runtime import json_write
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('immutable derived artifact changed: ' + str(path))
    else:
        json_write(path, value)


def combined_training_view(public, selected_ids, prompt, admissions, package, output):
    import io
    from tht_contract import validate_sharegpt
    public, output = Path(public), Path(output); output.mkdir(parents=True, exist_ok=True)
    pool = {r['id']: r for r in load_jsonl(public / 'mining/samples.jsonl')}
    if len(set(selected_ids)) != len(selected_ids) or not selected_ids or set(selected_ids) - pool.keys():
        raise ValueError('real training selection must contain unique existing Mining identities')
    records = []; rounds = set()
    for identity in sorted(selected_ids):
        row = pool[identity]; path = within(public, row['image'])
        if sha256(path) != row['image_sha256']: raise ValueError('real Mining image changed')
        records.append({'id': identity, 'source': 'real', 'label': row['labels'][0], 'image': str(path),
                        'image_sha256': row['image_sha256'], 'pixel_key': pixel_identity(path, training=True)['content_sha256']})
    for admission in admissions:
        if admission['round'] in rounds: raise ValueError('duplicate generation round admission')
        rounds.add(admission['round']); validate_admission(admission, package)
        for row in admission['records']:
            records.append({**row, 'source': 'synthetic', 'pixel_key': row['training_pixels']['content_sha256']})
    retained, duplicates = deduplicate_records(records)
    annotations = []; provenance = []
    for index, row in enumerate(retained):
        with Image.open(row['image']) as image:
            image = ImageOps.pad(image.convert('RGB'), (256, 256), method=Image.Resampling.NEAREST)
            encoded = io.BytesIO(); image.save(encoded, format='PNG'); data = encoded.getvalue()
        target = output / f'image-{index:05d}.png'
        if target.exists():
            if target.read_bytes() != data: raise ValueError('prepared training image changed')
        else:
            target.write_bytes(data)
        annotations.append({'id': row['id'], 'images': ['/input/' + target.name],
                            'conversations': [{'from': 'human', 'value': prompt},
                                              {'from': 'gpt', 'value': json.dumps({'labels': [row['label']]}, separators=(',', ':'))}]})
        provenance.append({**row, 'preprocessed': file_binding(target)})
    validate_sharegpt(annotations, 'tht_single')
    if {p.name for p in output.glob('image-*.png')} != {Path(r['images'][0]).name for r in annotations}:
        raise ValueError('unaccounted training-view image')
    summary = {'train_records': len(retained), 'selected_real_count': len(selected_ids),
               'real_count': sum(r['source'] == 'real' for r in retained),
               'synthetic_count': sum(r['source'] == 'synthetic' for r in retained),
               'generated_before_dedup': sum(len(a['records']) for a in admissions),
               'duplicates': duplicates, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
               'admission_sha256': [json_hash(a) for a in admissions], 'all_source_records': records}
    immutable_json(output / 'train.json', annotations)
    immutable_json(output / 'provenance.json', provenance)
    immutable_json(output / 'composition.json', summary)
    return summary
