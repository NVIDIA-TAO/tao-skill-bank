# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic boundary/replay tests, run on Brev CPU only."""
from pathlib import Path
import json,tempfile
import torch
from safetensors.torch import save_file
import oracle
from partition_private import split_groups
from wsp05_replay import apply_candidate
from wsp03_kernel import digest_tensor

def run():
    passed=[]
    def rejects(label,fn):
        try:fn()
        except (ValueError,FileNotFoundError,KeyError):passed.append(label);return
        raise AssertionError(label+' failed closed check')
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);work=root/'work';work.mkdir();(work/'source').mkdir()
        (work/'PLAN.json').write_text('{"groups":[{"max_settings":3}]}')
        request={'id':'Q01','parent':'B','profile':'single','patch_relative':None,'hypothesis':'Synthetic baseline hypothesis',
                 'expected':'Synthetic zero difference expectation','plan_relative':'PLAN.json','plan_sha256':oracle.sha(work/'PLAN.json'),
                 'source_relative':'source','gpu_released':True}
        oracle.validate_request(request,work);passed.append('valid baseline request')
        rejects('no subset API',lambda:oracle.validate_request(dict(request,subset=[0]),work))
        rejects('must prerecord hypothesis',lambda:oracle.validate_request(dict(request,hypothesis=''),work))
        rejects('plan hash checked',lambda:oracle.validate_request(dict(request,plan_sha256='0'*64),work))
        rejects('absolute path rejected',lambda:oracle.validate_request(dict(request,plan_relative='/tmp/plan'),work))
        rejects('traversal rejected',lambda:oracle.validate_request(dict(request,plan_relative='../PLAN.json'),work))
        (work/'alias').symlink_to(work/'source',target_is_directory=True)
        rejects('symlink rejected',lambda:oracle.validate_request(dict(request,source_relative='alias'),work))
        rejects('GPU release required',lambda:oracle.validate_request(dict(request,gpu_released=False),work))
        ledger=root/'ledger';ledger.mkdir()
        for n in range(20):oracle.admit(ledger,{'id':'q'+str(n),'hypothesis':'h','expected':'e','configuration_sha256':str(n)})
        rejects('21st query denied',lambda:oracle.admit(ledger,{'id':'q21','hypothesis':'h','expected':'e','configuration_sha256':'21'}))
        assert len(oracle.ledger_entries(ledger))==20;passed.append('failed/used admissions persist')
        identifies={'q'+str(n):['s'+str(n//2),'o'] for n in range(18)}
        first=split_groups(identifies);second=split_groups(dict(reversed(list(identifies.items()))))
        assert first==second and not set(first['A'])&set(first['B'])
        assert not {tuple(identifies[q]) for q in first['A']}&{tuple(identifies[q]) for q in first['B']}
        passed.append('deterministic group split without leakage')
        model=torch.nn.Module();model.lm_head=torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16)
        original=model.lm_head.weight.detach().clone();edited=original*2
        patch=root/'patch';patch.mkdir();save_file({'value':edited},str(patch/'value.safetensors'))
        m={'valid':True,'changed_elements':int((original!=edited).sum()),'tensors':[{'name':'lm_head.weight','file':'value.safetensors',
           'before_sha256':digest_tensor(original),'after_sha256':digest_tensor(edited),'changed_elements':int((original!=edited).sum())}]}
        (patch/'manifest.json').write_text(json.dumps(m));apply_candidate(model,patch)
        assert torch.equal(model.lm_head.weight,edited);passed.append('authorized head whole scaling replay')
        rejects('parent hash prevents double application',lambda:apply_candidate(model,patch))
        with torch.inference_mode():model.lm_head.weight.copy_(original)
        changed=edited.clone();changed[0,0]=float('nan');save_file({'value':changed},str(patch/'value.safetensors'))
        rejects('nonfinite replacement rejected',lambda:apply_candidate(model,patch))
        metrics={k:0 for k in oracle.METRIC_FIELDS};oracle.aggregate({'metrics':metrics,'predictions':[{'id':'synthetic-secret'}]})
        rejects('unexpected aggregate field rejected',lambda:oracle.aggregate({'metrics':dict(metrics,ids=['synthetic-secret'])}))
        passed.append('raw predictions excluded from public schema')
    return {'status':'PASS','synthetic_checks':passed,'gpu_used':False,'withheld_data_accessed':False}
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run();oracle.write(args.output,result);print(json.dumps(result))
