# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose bounded new-development baseline evidence without sealed probe quality."""
import json
from pathlib import Path
from wsp03_bindings import file_hash
from wsp03_io import json_write

def publish_baselines(service,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);summaries={};assets={}
    for payload, in service.ledger.db.execute("SELECT payload FROM requests WHERE kind='baseline' AND state='complete'"):
        value=json.loads(payload);profile=value['profile'];raw=value['raw'];score=json.loads((Path(value['output_directory'])/'trusted-score.json').read_text())
        tokens=sorted(r['inference_output_tokens'] for r in raw);byid={r['id']:r for r in raw};cases=[]
        for prediction in score['predictions']:
            if prediction['truth']==prediction['prediction']:continue
            answer=byid[prediction['id']]
            cases.append({'id':prediction['id'],'truth':prediction['truth'],'prediction':prediction['prediction'],'finish_reason':answer.get('finish_reason'),'output_tokens':answer['inference_output_tokens'],'raw_prefix':answer['raw_answer'][:512],'raw_prefix_truncated':len(answer['raw_answer'])>512})
        case_path=output/(profile+'-wrong-cases.jsonl')
        case_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in cases));case_path.chmod(0o600)
        name='phase0-'+profile+'-wrong-cases.jsonl';assets[name]={'path':str(case_path),'sha256':file_hash(case_path)}
        summaries[profile]={'metrics':score['metrics'],'wrong_cases':len(cases),'wrong_case_asset':name,'length_limit_count':sum(r.get('finish_reason')=='length' for r in raw),'output_tokens':{'min':min(tokens),'max':max(tokens),'mean':sum(tokens)/len(tokens),'p95':tokens[min(len(tokens)-1,int(.95*len(tokens)))]},'evidence_scope':'new frozen512 development pass; no probe/holdout/Benchmark quality'}
    if set(summaries)!={'P00','P01','P02','P03','B04'}:raise ValueError('all five completed baseline summaries required')
    path=output/'phase0-baselines.json';json_write(path,summaries);assets[path.name]={'path':str(path),'sha256':file_hash(path)}
    service.result_assets=dict(getattr(service,'result_assets',{}),**assets)
    with service.ledger.tx():service.ledger.set('baseline_evidence',{'summary_asset':path.name,'profiles':{key:{'metrics':value['metrics'],'wrong_case_asset':value['wrong_case_asset'],'length_limit_count':value['length_limit_count']} for key,value in summaries.items()}})
    return {'baseline_summary_asset':path.name,'wrong_case_assets':[value['wrong_case_asset'] for value in summaries.values()],'available_via':'context.read','scope':'new development only'}
