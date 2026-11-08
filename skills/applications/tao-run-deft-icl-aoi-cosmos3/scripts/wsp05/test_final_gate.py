# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic freeze gate tests; contains no real Benchmark data or model calls."""
from pathlib import Path
import json,tempfile
import final_evaluate as final
import oracle

def run():
    passed=[]
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    def rejects(label,fn):
        try:fn()
        except (ValueError,FileNotFoundError):passed.append(label);return
        raise AssertionError(label)
    original_state=final.terminal_state;original_subprocess=final.subprocess.check_output
    final.terminal_state=lambda c:{'container':c,'state':'exited','exit_code':0}
    final.subprocess.check_output=lambda *a,**k:''
    try:
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'run';work=root/'work';private=root/'private'
            write(work/'PLAN.json',{'groups':[{'max_settings':1}]})
            freeze={'schema':'wsp05-autonomous-freeze/1','status':'FROZEN','plan_sha256':oracle.sha(work/'PLAN.json'),
                    'finalists':[{'name':'edited','query_id':'Q02','baseline_query_id':'Q01','reason':'synthetic'}]}
            write(work/'FREEZE.json',freeze);write(root/'runtime/run.json',{'job':{'container_id':'editor'}})
            write(root/'setup/development-audit.json',{'status':'PASS','freeze_sha256':oracle.sha(work/'FREEZE.json')})
            write(root.parent/'W5/runtime/run.json',{'job':{'container_id':'old-editor'}})
            write(private/'oracle-terminal.json',{'queries_used':2})
            base={'parent':'G','profile':'single','model_files':{'x':'hash'},'patch_files':None,'max_new_tokens':128,
                  'backend':'fixed','worker_sha256':'w','replay_sha256':'r'}
            for q,patch in [('Q01',None),('Q02',{'value':'hash'})]:
                query=private/'queries'/q;write(query/'snapshot/configuration.json',dict(base,patch_files=patch))
                write(query/'response.json',{'status':'COMPLETE','configuration_sha256':oracle.sha(query/'snapshot/configuration.json')})
            gate=final.freeze_gate(root);assert len(gate['pairs'])==1;passed.append('valid stopped freeze without any B assets')
            write(root/'setup/development-audit.json',{'status':'FAIL','freeze_sha256':oracle.sha(work/'FREEZE.json')})
            rejects('failed independent audit blocks B',lambda:final.freeze_gate(root))
            write(root/'setup/development-audit.json',{'status':'PASS','freeze_sha256':oracle.sha(work/'FREEZE.json')})
            write(work/'FREEZE.json',dict(freeze,finalists=[dict(freeze['finalists'][0],query_id='../B')]))
            rejects('query path escape rejected',lambda:final.freeze_gate(root));write(work/'FREEZE.json',freeze)
            write(work/'FREEZE.json',dict(freeze,plan_sha256='changed'))
            rejects('changed plan blocks B',lambda:final.freeze_gate(root));write(work/'FREEZE.json',freeze)
            write(work/'FREEZE.json',dict(freeze,finalists=freeze['finalists']*5))
            rejects('more than four finalists blocks B',lambda:final.freeze_gate(root));write(work/'FREEZE.json',freeze)
            write(work/'FREEZE.json',dict(freeze,finalists=freeze['finalists']+[dict(freeze['finalists'][0],name='duplicate')]))
            rejects('duplicate finalist configuration rejected',lambda:final.freeze_gate(root));write(work/'FREEZE.json',freeze)
            final.terminal_state=lambda c:(_ for _ in ()).throw(ValueError('running'))
            rejects('running editor blocks B',lambda:final.freeze_gate(root))
            final.terminal_state=lambda c:{'container':c,'state':'exited','exit_code':0}
            (private/'oracle-terminal.json').rename(private/'oracle-not-terminal.json')
            rejects('active oracle blocks B',lambda:final.freeze_gate(root))
            (private/'oracle-not-terminal.json').rename(private/'oracle-terminal.json')
            path=private/'queries/Q01/snapshot/configuration.json';write(path,dict(base,profile='icl'))
            write(path.parent.parent/'response.json',{'status':'COMPLETE','configuration_sha256':oracle.sha(path)})
            rejects('unmatched ICL baseline blocks B',lambda:final.freeze_gate(root))
            write(path,base);write(path.parent.parent/'response.json',{'status':'COMPLETE','configuration_sha256':oracle.sha(path)})
            (private/'admissions.jsonl').write_text('\n'.join(json.dumps({'id':str(n)}) for n in range(21)))
            rejects('over-budget development blocks B',lambda:final.freeze_gate(root))
    finally:
        final.terminal_state=original_state;final.subprocess.check_output=original_subprocess
    return {'status':'PASS','checks':passed,'withheld_assets_present':False,'model_calls':0}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=run();oracle.write(a.output,result);print(json.dumps(result))
