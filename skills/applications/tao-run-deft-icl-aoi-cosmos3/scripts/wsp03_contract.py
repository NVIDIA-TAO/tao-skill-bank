# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit honest generic operation capabilities for an experiment context pack."""
import argparse
import json
from pathlib import Path
from wsp03_io import json_write
from wsp03_worker import digest

OPERATIONS={
 'tensor.read':{'executor':'isolated_cpu_kernel','required':['name'],'optional':['candidate','rows','columns']},
 'activation.read':{'executor':'dev_teacher_forced_worker','required':['records'],'optional':['candidate','split']},
 'activation.patch':{'executor':'dev_teacher_forced_worker','required':['records'],'optional':['candidate','split']},
 'reference.attribute':{'executor':'isolated_cpu_kernel','required':['records'],'optional':[]},
 'edit.family':{'executor':'isolated_cpu_kernel','required':['id','mechanism','tensors','lineage','configurations'],'optional':[]},
 'edit.direction':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','direction'],'optional':[]},
 'edit.rank1':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','left','right'],'optional':[]},
 'edit.memit':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','keys','delta_values','covariance','covariance_source'],'optional':[]},
 'edit.head_scale':{'executor':'isolated_cpu_kernel','required':['name','family','configuration','axis','indices'],'optional':[]},
 'edit.commit':{'executor':'isolated_cpu_kernel','required':['id','hypothesis','method'],'optional':[]},
 'edit.replay':{'executor':'isolated_cpu_kernel','required':['candidate'],'optional':[]},
 'edit.revert':{'executor':'isolated_cpu_kernel','required':[],'optional':['parent']},
 'eval.dev':{'executor':'trusted_generation_and_grader','required':['candidate','ids'],'optional':['references','split']},
 'export.checkpoint':{'executor':'isolated_cpu_kernel','required':['candidate'],'optional':[]}}


def manifest():
    schemas={name:{'type':'object','required':spec['required'],'properties':{key:{} for key in spec['required']+spec['optional']},'additionalProperties':False} for name,spec in OPERATIONS.items()}
    result={'schema':'wsp03-capabilities/1','scientific_launch_enabled':False,'operations':OPERATIONS,'parameter_schemas':schemas,
        'configured_by_recipe':['model_adapter_and_inventory','dataset_membership_and_identity_maps','classes','canonical_label_prompt','generation_prompt','budgets','retrieval_strategies','context_pack_hash'],
        'implemented_and_cpu_tested':['persistent_numeric_process','exact_tensor_replay','regularized_closed_form_edit','checkpoint_export','head_slice_diagnostic','same_pass_attention_reconstruction','durable_soft_stop','grouped_split','child_path_denial'],
        'remaining_science_gates':['bind trusted request renderer and frozen retrieval strategies','bind scorer with no editor-visible withheld outputs','bind native editor usage event adapter and provider relay','validate real-model activation shape/span/mask diagnostics','bind numerical-family declarations to scientific hypotheses in review','validate saved-generation-prefix diagnostics before exposing that optional mode'],
        'limits':{'read_records':32,'read_input_tokens':8192,'forced_target_tokens':64,'retained_read_bytes':268435456,'configs_per_family':5},
        'documentation':'Numeric operations are generic; dataset/model/prompt policies belong to a frozen recipe. No standalone reusable skill is published by this engineering gate.'}
    result['contract_sha256']=digest(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();result=manifest();json_write(Path(a.output),result);print(json.dumps({'output':a.output,'contract_sha256':result['contract_sha256']}))
