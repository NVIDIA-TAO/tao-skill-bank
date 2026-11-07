# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted final-only historical prediction attachment and native CPU statistics."""
import json,time
from pathlib import Path
from wsp03_bindings import file_hash,pinned
from wsp03_io import json_write
from wsp03_runtime import APP
from wsp03_artifacts import child_json

def stage_evaluator(source,destination,group):
    """Expose only the two exact frozen source files to the dedicated scorer UID."""
    import subprocess
    from tht_contract import EXTERNAL_HASHES
    source=Path(source);destination=Path(destination);destination.mkdir(mode=0o750)
    subprocess.run(['sudo','-n','chgrp',str(group),str(destination)],check=True)
    for name,expected in EXTERNAL_HASHES.items():
        data=pinned(source/name,expected).read_bytes();target=destination/name
        target.write_bytes(data);target.chmod(0o440)
        subprocess.run(['sudo','-n','chgrp',str(group),str(target)],check=True)
    return destination

def final_statistics(r3,r4,results,config,output):
    from contextlib import nullcontext
    runtime=r4.worker.runtime;resources=getattr(runtime,'resources',None)
    with resources.reserve(64*1024**2,'trusted final statistics inputs/results',runtime.resource_account) if resources else nullcontext():
        return compute_final_statistics(r3,r4,results,config,output)

def compute_final_statistics(r3,r4,results,config,output):
    if any(s.ledger.status()['phase']!='frozen' for s in (r3,r4)) or set(results)!={arm+':'+split for arm in ('R3','R4','R4-C') for split in ('holdout','benchmark')}:raise ValueError('all final frozen arms required before historical scoring access')
    if any(not r.get('complete') for r in results.values()):raise ValueError('incomplete final pass')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);payload={'all_final_passes_complete':True,'both_freezes':{name:s.ledger.get('selected') for name,s in [('R3',r3),('R4',r4)]},'splits':{}}
    for split in ('holdout','benchmark'):
        arms={};aliases={}
        for arm in ('R3','R4','R4-C'):
            result=results[arm+':'+split];score=Path(result['output_directory'])/'trusted-score.json';arms[arm]=json.loads(score.read_text())
            if result.get('alias_of'):aliases[arm]=result['alias_of']
            elif result.get('reused_kind'):aliases[arm]=result['source_track']
        if split=='benchmark':
            for arm in ('B','C'):
                asset=config['historical_predictions'][arm];path=pinned(**asset);arms[arm]=json.loads(path.read_text())
        recipe=r3.request_factory;identities={qid:{'object_name':recipe.rows[qid]['object_name'],'scan_id':recipe.rows[qid]['scan_id']} for qid in recipe.members[split]}
        payload['splits'][split]={'arms':arms,'identities':identities,'aliases':aliases}
    inp=output/'input.json';json_write(inp,payload,mode=0o640)
    import subprocess
    runtime=r4.worker.runtime;subprocess.run(['sudo','-n','chgrp',str(runtime.gid),str(inp)],check=True)
    evaluator_view=stage_evaluator(config['evaluator_directory'],output/'evaluator',runtime.gid)
    bank=APP.parents[2];script='/bank/'+str(APP.relative_to(bank))+'/scripts/wsp03_statistics.py'
    job=runtime.submit('wsp03-final-statistics',[script,'--input','/input.json','--output','/results/statistics.json','--evaluator','/evaluator'],gpu=False,
        mounts=[(bank,'/bank',True),(inp,'/input.json',True),(evaluator_view,'/evaluator',True)],extra_args=['--runtime','runc','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=256m','--memory','4g','--cpus','4'])
    deadline=time.time()+min(r3.ledger.remaining_joint_seconds(),r4.ledger.remaining_joint_seconds())
    if runtime.wait(job['job_id'],deadline)['state']!='COMPLETE':raise RuntimeError('final statistics did not finish inside joint budget')
    value=child_json(job['results_dir'],'statistics.json',owner=runtime.uid,max_bytes=8*1024**2)
    json_write(output/'statistics.json',value);json_write(output/'receipt.json',{'job':job,'input_sha256':file_hash(inp),'historical_prediction_assets':config['historical_predictions'],'model_calls':0,'editor_access':False});return value
