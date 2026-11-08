# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU-only membership, scoring and freeze-binding audit; no Benchmark."""
from pathlib import Path
import os
import json,hashlib
from tht_contract import external_evaluator
from wsp03_worker import digest

ROOT=Path(os.environ['WSP05_RUN_ROOT']).expanduser().resolve()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def run():
    work=ROOT/'work';plan=read(work/'PLAN.json');freeze=read(work/'FREEZE.json')
    assert freeze['plan_sha256']==sha(work/'PLAN.json')
    original=read(ROOT/'context/manifest.json')
    assert all(sha(ROOT/'context'/name)==h for name,h in original.items())
    bank=ROOT.parent/'bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts'
    for name in ['wsp04_data.py','wsp03_worker.py','wsp03_kernel.py']:
        assert sha(work/'source/vendor'/name)==sha(bank/name)
    data=ROOT.parent/'inputs/development';rows={r['id']:r for r in lines(data/'source_rows.jsonl')}
    mining=read(data/'mining-ids.json');proxy=read(data/'proxy-ids.json');sample=read(work/'mining-sample.json')
    assert len(mining)==3642 and len(proxy)==74 and len(set(sample))==len(sample)<=256
    assert set(sample)<=set(mining) and not set(sample)&set(proxy)
    evaluator=external_evaluator(ROOT/'context/evaluator')
    candidates={c['id']:{'group':g['id'],'origin':g['origin'],'settings':c} for g in plan['groups'] for c in g['settings']}
    assert len(candidates)==len({c['id'] for g in plan['groups'] for c in g['settings']})<=plan['candidate_rules']['maximum_new_patches']
    for g in plan['groups']:assert len(g['settings'])<=g['maximum_settings']
    expected_ids=set(candidates)|{'BASE_G','BASE_C'};passes={};patches={}
    for p in sorted((work/'runs').glob('*/receipt.json')):
        value=read(p);cid=value['candidate'];split=value['split'];origin=value['origin']
        assert cid in expected_ids and split in ['mining','proxy']
        assert cid.startswith('BASE_') or split=='proxy'
        ordered=sample if split=='mining' else proxy
        requests=read(work/f'requests/{origin}-{split}.json')
        assert [r['id'] for r in requests]==ordered
        assert all(len(r['references'])==(origin=='C') for r in requests)
        raw=lines(p.parent/'raw_answers.jsonl')
        assert sha(p.parent/'raw_answers.jsonl')==value['raw_sha256']
        assert [r['id'] for r in raw]==ordered[:len(raw)]
        assert len(raw)==len({r['id'] for r in raw})==value['n']
        for r,q in zip(raw,requests):
            assert r['request_sha256']==digest(q)
            assert r['truth']==rows[r['id']]['labels'][0]
            assert r['prediction']==evaluator.parse(r['raw_answer'])
            assert r['inference_output_tokens']==len(r['completion_token_ids'])<=128
            assert r['finish_reason']==('length' if r['inference_output_tokens']>=128 else 'stop')
        assert evaluator.score(raw)==value['metrics']
        rt=value['runtime']
        assert all(rt[k]==v for k,v in {'dtype':'bfloat16','attention':'sdpa','batch_size':1,'seed':42,
            'max_new_tokens':128,'do_sample':False,'eval':True,'requires_grad':False,'inference_mode':True}.items())
        check=min(64,len(ordered));bad=sum(r['prediction']=='INVALID' or r['finish_reason']=='length' for r in raw[:check])
        if value['status']=='COMPLETE':assert len(raw)==len(ordered) and bad/check<.8
        elif value['status']=='DEGENERATE_EARLY_STOP':assert len(raw)==check and bad/check>=.8
        else:raise ValueError('unrecognized pass state')
        attempts=lines(p.parent/'attempts.jsonl')
        assert [r['id'] for r in attempts]==[r['id'] for r in raw]
        passes[p.parent.name]={'n':len(raw),'status':value['status'],'metrics':value['metrics'],
            'receipt_sha256':sha(p),'raw_sha256':value['raw_sha256'],'runtime_source_sha256':rt['source_sha256']}
    for directory in sorted((work/'patches').iterdir()):
        if not directory.is_dir():continue
        assert directory.name in candidates
        manifest=read(directory/'manifest.json');spec=candidates[directory.name]
        assert manifest['configuration']==spec['settings'] and manifest['group']==spec['group']
        assert manifest['origin']==spec['origin'] and manifest['pristine_parent_reset'] is True
        assert manifest['fit_sample_sha256']==sha(work/'mining-sample.json')
        assert manifest['parent']==('G' if spec['origin']=='G' else 'B')
        assert manifest['valid'] and manifest['changed_elements']>0
        assert sum(i['changed_elements'] for i in manifest['tensors'])==manifest['changed_elements']
        for item in manifest['tensors']:assert sha(directory/item['file'])==item['file_sha256']
        replay_path=directory/'hf-replay.json'
        if replay_path.exists():
            replay=read(replay_path);assert replay['status']=='PASS' and replay['manifest_sha256']==sha(directory/'manifest.json')
            assert replay['model_parameters_frozen'] and replay['checked_before_cuda']
        patches[directory.name]={'manifest_sha256':sha(directory/'manifest.json'),'changed_elements':manifest['changed_elements'],
                                'fresh_parent_HF_replay':replay_path.exists()}
    captures={}
    for p in sorted((work/'diagnosis').glob('*-capture/receipt.json')):
        value=read(p);records=lines(p.parent/'receipts.jsonl');attempts=lines(p.parent/'attempts.jsonl')
        assert value['n']==len(records)==len(sample) and value['class_answer_tokens']==0
        assert [r['id'] for r in records]==sample and [r['id'] for r in attempts]==sample
        assert all(r['forward_calls']==1 and r['class_answer_tokens']==0 for r in records)
        assert value['sample_sha256']==sha(work/'mining-sample.json')
        assert read(p.parent/'partial.json')['all_hooks_removed']
        captures[p.parent.name]={'n':len(records),'class_answer_tokens':0,'receipt_sha256':sha(p)}
    result={'status':'PASS','plan_sha256':sha(work/'PLAN.json'),'freeze_sha256':sha(work/'FREEZE.json'),
            'sample_count':len(sample),'mining_count':3642,'proxy_count':74,'passes':passes,'patches':patches,'captures':captures,
            'scope':'independent complete membership/parser/score/manifest and actual fresh-parent replay evidence; source method reviewed separately',
            'benchmark_data_read':False,'new_model_calls':0}
    out=ROOT/'setup/development-audit.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'PASS','passes':len(passes),'patches':len(patches),'captures':len(captures),'sha256':sha(out)}))
if __name__=='__main__':run()
