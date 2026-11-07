# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen post-evaluation paired statistics; never an editor/model capability."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from wsp03_io import json_write

METRICS=('mcc','macro_f1','balanced_accuracy')
PAIRS=(('R3','C'),('R4','C'),('R3','B'),('R4','B'),('R4','R3'),('R4-C','C'),('R4','R4-C'),('R4-C','R3'))

def percentile(values,q):
    ordered=sorted(values);position=(len(ordered)-1)*q;lo=int(position);hi=min(len(ordered)-1,lo+1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)

def paired(arms,identities,evaluator,*,expected_groups,replicates=2000,seed=42,confidence=.95,aliases=None,pairs=PAIRS):
    if (replicates,seed,confidence)!=(2000,42,.95):raise ValueError('frozen paired-bootstrap protocol changed')
    predictions={name:{r['id']:r for r in value['predictions']} for name,value in arms.items()}
    if not predictions:raise ValueError('complete arms required')
    ids=set(next(iter(predictions.values())))
    if any(len(predictions[name])!=len(arms[name]['predictions']) or set(rows)!=ids for name,rows in predictions.items()) or set(identities)!=ids:raise ValueError('paired identity coverage mismatch')
    first=next(iter(predictions.values()))
    if any(any(rows[q]['truth']!=first[q]['truth'] for q in ids) for rows in predictions.values()):raise ValueError('paired gold labels differ')
    groups={}
    for q in sorted(ids):
        row=identities[q];groups.setdefault((row['object_name'],row['scan_id'].removeprefix('SCN_')),[]).append(q)
    if len(groups)!=expected_groups:raise ValueError('frozen joint grouping count changed')
    scores={name:evaluator.score(list(rows.values())) for name,rows in predictions.items()}
    for name,score in scores.items():
        for metric in METRICS:
            if abs(score[metric]-arms[name]['metrics'][metric])>1e-12:raise ValueError('reported frozen scorer metric differs')
    keys=sorted(groups);rng=random.Random(seed);draws={name:{metric:[] for metric in METRICS} for name in arms}
    for _ in range(replicates):
        sample=[q for _ in keys for q in groups[rng.choice(keys)]]
        for name,rows in predictions.items():
            score=evaluator.score([rows[q] for q in sample])
            for metric in METRICS:draws[name][metric].append(score[metric])
    differences={};alpha=(1-confidence)/2
    for a,b in pairs:
        if a not in arms or b not in arms:continue
        differences[a+'__minus__'+b]={'left_arm':a,'right_arm':b,**{metric:{'estimate':scores[a][metric]-scores[b][metric],'lower':percentile([x-y for x,y in zip(draws[a][metric],draws[b][metric])],alpha),'upper':percentile([x-y for x,y in zip(draws[a][metric],draws[b][metric])],1-alpha)} for metric in METRICS}}
    return {'method':'paired_joint_percentile','replicates':replicates,'seed':seed,'confidence':confidence,'group_count':len(keys),'query_count':len(ids),'metrics':{name:value['metrics'] for name,value in arms.items()},'pairs':differences,'derived_aliases':aliases or {},'invalid':'retained exactly as frozen evaluator','multiplicity':'exploratory comparisons are not selectively promoted to primary claims'}

def run(input_file,output,evaluator_directory):
    from tht_contract import external_evaluator
    payload=json.loads(Path(input_file).read_text())
    if payload.get('all_final_passes_complete') is not True or set(payload.get('both_freezes',{}))!={'R3','R4'}:raise ValueError('final statistics require complete frozen evaluation')
    evaluator=external_evaluator(evaluator_directory);results={}
    for split,value in payload['splits'].items():
        results[split]=paired(value['arms'],value['identities'],evaluator,expected_groups=255 if split=='benchmark' else 600,aliases=value['aliases'])
    benchmark=results['benchmark']['pairs'];results['primary_outcomes']={'R3_noninferiority_minus_C':benchmark['R3__minus__C']['mcc']['lower']>=-.03,'R4_superiority_minus_C':benchmark['R4__minus__C']['mcc']['lower']>0,'R3_point_ge_C':benchmark['R3__minus__C']['mcc']['estimate']>=0}
    json_write(output,results,mode=0o640);return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--evaluator',required=True);args=parser.parse_args();run(args.input,args.output,args.evaluator)
