# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit honest generic operation capabilities for an experiment context pack."""
import argparse
import json
from pathlib import Path
from wsp03_io import json_write
from wsp03_worker import digest

OPERATIONS={
 'status':{'executor':'trusted_metadata','required':[],'optional':[]},
 'data.list':{'executor':'trusted_dev_metadata','required':['split'],'optional':['offset','limit']},
 'context.read':{'executor':'trusted_context_allowlist','required':[],'optional':['name','offset','limit']},
 'reference.candidates':{'executor':'trusted_dev_metadata','required':['query_id'],'optional':[]},
 'help':{'executor':'trusted_metadata','required':[],'optional':['operation']},
 'python':{'executor':'isolated_cpu_kernel','required':['code'],'optional':[]},
 'tensor.read':{'executor':'isolated_cpu_kernel','required':['name'],'optional':['candidate','rows','columns']},
 'activation.read':{'executor':'dev_teacher_forced_worker','required':['records'],'optional':['candidate','split','profile']},
 'activation.patch':{'executor':'dev_teacher_forced_worker','required':['records'],'optional':['candidate','split','profile']},
 'reference.attribute':{'executor':'isolated_cpu_kernel','required':['records'],'optional':[]},
 'edit.family':{'executor':'isolated_cpu_kernel','required':['id','operation','mechanism','tensors','lineage','configurations'],'optional':[]},
 'edit.direction':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','direction'],'optional':[]},
 'edit.rank1':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','left','right'],'optional':[]},
 'edit.memit':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','keys','delta_values','covariance','covariance_source'],'optional':[]},
 'edit.head_scale':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','axis','indices'],'optional':[]},
 'edit.commit':{'executor':'isolated_cpu_kernel','required':['id','hypothesis','method'],'optional':[]},
 'edit.replay':{'executor':'isolated_cpu_kernel','required':['candidate'],'optional':[]},
 'edit.revert':{'executor':'isolated_cpu_kernel','required':[],'optional':['parent']},
 'eval.dev':{'executor':'trusted_generation_and_grader','required':['candidate','ids'],'optional':['references','split','profile']},
 'export.checkpoint':{'executor':'isolated_cpu_kernel','required':['candidate'],'optional':[]}}


OPERATION_HELP={
 'activation.read':{'records':'1..32 {query_id,references:[dev IDs],capture:{points:[{module,kind:input|output,positions:[pre_label|answer_boundary|integer]}],attention:[{module,rows:[pre_label],heads:[integer]}]}}; candidate defaults base; canonical correct label is supplied by trusted renderer','numeric_access':'python load_capture(request_id,record_id,key) loads retained safetensors; returned metadata contains available keys','phase1':'Use empty capture for paired baseZS/reference canonical-class logprob attribution; reference.candidates gives frozen top8+2+2 eligible references'},
 'activation.patch':{'records':'Same dev read record shape; capture.patches:[{module,kind:input|output,positions:[pre_label],mode:replace|add,source:{request_id,record_id,key,scale:1},head:{count,dimension,index}}]. head is optional and valid only at attention output-projection input; source must be a completed charged read on the same query/class. Inline values are also accepted for bounded synthetic/zero interventions. All replacement forwards debit read count. Generation forbids captures/patches.','phase2':'Use fixed64 diagnostic queries, their fixed nearestk1 source, layer sweep and top2 layer head sweep from frozen phase plan. Retained source reads are charged separately.'},
 'reference.attribute':{'records':'{query_id,reference_id,baseline_request,reference_request}; both complete base reads must match canonical class, zero references versus exactly this eligible reference. Values are read from trusted receipts, never accepted from caller.'},
 'python':{'capabilities':['load_tensor(name,candidate=base)','load_capture(request_id,record_id,key)','register_edit_family({...})','select_edit_configuration({family,configuration,name})','save_edit(name,tensor)','commit_candidate(id,hypothesis,method)','reset_candidate(parent=base)'],'persistent':'CPU numeric variables persist between calls. No imports/files/network/GPU/autograd/optimizers. print previews only; full artifacts remain in trusted storage.'},
 'edit.family':{'configurations':'At most5 tuples within frozen operation policy; family includes operation, mechanism, tensor names and read/capture lineage. Renaming cannot reset tuples. Commit consumes active configuration.'},
 'eval.dev':{'candidate':'R3 validated patch ID with profileP00 only; R4 candidate must equal one frozen P00..P11 profile and cannot alter weights. ids must cover exact ordered512 explore membership. Single attempt per profile/candidate; no caller reference override.'},
 'edit.memit':{'keys':'in_features×rank source keys','delta_values':'out_features×rank desired value change, not absolute targets','covariance':'development preservation covariance matrix or diagonal; ridge1e-4×mean diagonal. This operation requires algebraic/mixed method tagging.'}}


def manifest():
    schemas={name:{'type':'object','required':spec['required'],'properties':{key:{} for key in spec['required']+spec['optional']},'additionalProperties':False} for name,spec in OPERATIONS.items()}
    result={'schema':'wsp03-capabilities/1','authorization':'Scientific actions require an external matching preflight review receipt; static context is not authorization','operations':OPERATIONS,'parameter_schemas':schemas,
        'configured_by_recipe':['model_adapter_and_inventory','dataset_membership_and_identity_maps','classes','canonical_label_prompt','generation_prompt','budgets','retrieval_strategies','context_pack_hash'],
        'implemented_and_cpu_tested':['persistent_numeric_process','exact_tensor_replay','regularized_closed_form_edit','checkpoint_export','head_slice_diagnostic','same_pass_attention_reconstruction','durable_soft_stop','grouped_split','child_path_denial'],
        'protocol':{'transport':'one tool-free codex exec response per round; fenced blocks dispatched by trusted supervisor','blocks':['python','read','forward','save_edit','tool','handoff','freeze'],'freeze':'terminal editing stop; trusted finalist runner follows','handoff_every_rounds':40,'usage_source':'supervisor-owned native exec JSON stream only','output_reservation':128000,'input_bound':'full current prompt UTF-8 bytes plus verified native envelope; repacked-history fallback','isolated_tool_uid':'dedicated actual host UID, distinct from supervisor'},
        'limits':{'read_records':32,'read_input_tokens':8192,'forced_target_tokens':64,'retained_read_bytes':268435456,'configs_per_family':5},
        'documentation':'Numeric operations are generic; dataset/model/prompt policies belong to a frozen recipe. No standalone reusable skill is published by this engineering gate.'}
    result['contract_sha256']=digest(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();result=manifest();json_write(Path(a.output),result);print(json.dumps({'output':a.output,'contract_sha256':result['contract_sha256']}))
