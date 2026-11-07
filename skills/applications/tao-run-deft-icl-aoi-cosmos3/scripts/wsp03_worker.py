# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated batch-one generation or bounded dev teacher-forced read worker.

The trusted host dispatcher supplies one allowlisted request view. No gold,
provider credentials, evaluator, or editor process is mounted in this worker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import re
import time
from wsp03_io import json_write
from wsp03_resources import append_event, cpu_memory


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_requests(requests, mode, split):
    if mode not in ('generation','read') or not requests or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('nonempty unique requests and known mode required')
    if mode == 'read' and (split not in ('development','explore','diagnostic') or len(requests)>32):
        raise ValueError('reads are dev-only, at most32 records per exposure')
    allowed={'id','messages','references','policy_sha256'}
    if mode=='read': allowed |= {'forced_text','capture','query_id'}
    for r in requests:
        if set(r)-allowed or not isinstance(r['id'],str) or not r['id']:
            raise ValueError('unexpected worker fields; generation forbids hooks and targets')
        if mode=='read' and (not isinstance(r.get('forced_text'),str)):
            raise ValueError('canonical forced label target required')


def recovery(output, requests):
    done={}; byid={r['id']:r for r in requests}
    answers=output/'raw_answers.jsonl'
    if answers.exists():
        for line in answers.read_text().splitlines():
            row=json.loads(line); qid=row['id']
            if qid in done or qid not in byid or row['request_sha256'] != digest(byid[qid]):
                raise ValueError('duplicate, unrelated or changed recovery input')
            done[qid]=row
    attempts=output/'attempts.jsonl'
    if attempts.exists():
        attempted={json.loads(line)['id'] for line in attempts.read_text().splitlines()}
        if attempted-set(done):
            raise RuntimeError('unresolved or failed first attempt; explicit trusted reconciliation required, no automatic replay')
    return done


def encode(processor, request):
    from PIL import Image
    messages=[]; images=[]; image_paths=[]
    try:
        for msg in request['messages']:
            if msg['role'] not in ('user','assistant'): raise ValueError('unsupported role')
            parts=[]
            for part in msg['content']:
                if part['type']=='image':
                    path=Path(part['image']).resolve(strict=True)
                    if not path.is_relative_to('/input'): raise ValueError('image outside private request view')
                    with Image.open(path) as source: image=source.convert('RGB')
                    if image.size!=(256,256): raise ValueError('expected preprocessed256PNG')
                    images.append(image); image_paths.append(str(path));parts.append({'type':'image','image':str(path)})
                elif part['type']=='text':parts.append({'type':'text','text':part['text']})
                else:raise ValueError('unsupported content')
            messages.append({'role':msg['role'],'content':parts})
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=images,return_tensors='pt')
        return inputs,text,image_paths
    finally:
        for image in images:image.close()


def canonical_target(tokenizer, text):
    if text=='':return [],[],None
    parsed=json.loads(text)
    if set(parsed)!= {'labels'} or len(parsed['labels'])!=1 or not isinstance(parsed['labels'][0],str):raise ValueError('canonical single-label continuation required')
    match=re.search(r'"labels"\s*:\s*\[\s*"((?:[^"\\]|\\.)*)"\s*\]',text)
    if match is None:raise ValueError('canonical class span unavailable')
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping'];left,right=match.span(1)
    positions=[i for i,(a,b) in enumerate(offsets) if b>left and a<right]
    if not positions:raise ValueError('class token span empty')
    span={'character_span':[left,right],'label':parsed['labels'][0],'token_indices':positions,
        'offsets':[list(offsets[i]) for i in positions], 'mapping':'tokens overlapping class characters in actual full JSON tokenization; boundary-sharing tokens retained'}
    return ids,positions,span


def apply_candidate(model, directory):
    """Exact replacement replay before CUDA, with immutable-base and edited hashes."""
    from safetensors import safe_open
    from wsp03_kernel import digest_tensor, protected_name, whole_tensor_scaling
    import torch
    directory=Path(directory).resolve(strict=True)
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest.get('valid') or manifest.get('changed_elements',0)<=0:raise ValueError('invalid candidate')
    names=set(); changed=0
    with torch.inference_mode():
        for item in manifest['tensors']:
            name=item['name']
            if name in names or protected_name(name):raise ValueError('duplicate/protected tensor')
            names.add(name)
            parameter=model.get_parameter(name)
            if digest_tensor(parameter)!=item['before_sha256']:raise ValueError('candidate base hash mismatch')
            path=(directory/item['file']).resolve(strict=True)
            if path.parent!=directory:raise ValueError('candidate path escape')
            with safe_open(path,framework='pt',device='cpu') as file:edited=file.get_tensor('value')
            if parameter.shape!=edited.shape or parameter.dtype!=edited.dtype or not bool(torch.isfinite(edited).all()):raise ValueError('candidate shape/dtype/finiteness')
            if digest_tensor(edited)!=item['after_sha256'] or whole_tensor_scaling(parameter,edited):raise ValueError('candidate hash/scaling failure')
            n=int((parameter!=edited).sum());changed+=n
            if n!=item['changed_elements']:raise ValueError('candidate count mismatch')
            parameter.copy_(edited)
    if changed!=manifest['changed_elements'] or len(names)>32 or changed>2000000000:raise ValueError('candidate limits/counts')
    return digest(manifest)


def run(model_path, requests_path, output_dir, *, mode='generation', split='probe', candidate=None, query_seconds=180):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    requests=json.loads(Path(requests_path).read_text());validate_requests(requests,mode,split)
    done=recovery(output,requests)
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from safetensors.torch import save_file
    from wsp03_capture import ActivationCapture,target_logprobs
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    start=time.monotonic();processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    candidate_hash=apply_candidate(model,candidate) if candidate else None
    model.to('cuda');torch.cuda.synchronize()
    profile={'schema':'wsp03-worker/1','mode':mode,'split':split,'torch':torch.__version__,'transformers':transformers.__version__,
        'candidate_manifest_sha256':candidate_hash,'dtype':'bfloat16','attention':'sdpa','batch_size':1,'seed':42,
        'max_new_tokens':4096,'do_sample':False,'query_seconds':query_seconds,'model_load_seconds':time.monotonic()-start,
        'chat_template_sha256':hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
        'requests_sha256':digest(requests),'generation_hooks':False,'reasoning':'prompted_off','native_thinking_disabled':False}
    old_profile=output/'runtime_profile.json'
    if old_profile.exists():
        old=json.loads(old_profile.read_text())
        for key in ('candidate_manifest_sha256','requests_sha256','mode','split','chat_template_sha256'):
            if old[key]!=profile[key]:raise ValueError('resume runtime changed')
    else:json_write(old_profile,profile)
    retained_bytes=0
    for request in requests:
        if request['id'] in done:continue
        inputs,text,paths=encode(processor,request);prompt_length=inputs['input_ids'].shape[1]
        if mode=='read' and prompt_length>8192:raise ValueError('read prompt exceeds8192 tokens')
        inputs=inputs.to('cuda');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before_allocated=torch.cuda.memory_allocated();before_reserved=torch.cuda.memory_reserved()
        started=time.time();tick=time.monotonic();base={'id':request['id'],'request_sha256':digest(request),'started_at_unix':started}
        append_event(output/'attempts.jsonl',dict(base,status='running'))
        old_alarm=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('query deadline')))
        signal.alarm(query_seconds)
        try:
            with torch.inference_mode():
                if mode=='generation':
                    tokens=model.generate(**inputs,do_sample=False,max_new_tokens=4096)
                    completion=tokens[0,prompt_length:];raw=processor.tokenizer.decode(completion,skip_special_tokens=True)
                    result={'raw_answer':raw,'completion_token_ids':completion.cpu().tolist(),
                        'finish_reason':'length' if len(completion)>=4096 else 'stop','inference_output_tokens':len(completion)}
                else:
                    target,class_positions,class_span=canonical_target(processor.tokenizer,request['forced_text'])
                    if len(target)>64:raise ValueError('bounded canonical target required')
                    target_ids=torch.tensor([target],dtype=inputs['input_ids'].dtype,device='cuda')
                    inputs['input_ids']=torch.cat([inputs['input_ids'],target_ids],dim=1)
                    inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(target_ids)],dim=1)
                    # Resolve semantic positions only after the actual processor image expansion.
                    spec=json.loads(json.dumps(request.get('capture',{})))
                    for item in spec.get('points',[])+spec.get('patches',[]):
                        item['positions']=[prompt_length-1 if p=='answer_boundary' else p for p in item['positions']]
                    for item in spec.get('attention',[]):
                        item['rows']=[prompt_length-1 if p=='answer_boundary' else p for p in item['rows']]
                    with ActivationCapture(model,spec,max_bytes=256*1024**2-retained_bytes,allow_patches=True) as capture:
                        logits=model(**inputs,use_cache=False,return_dict=True).logits
                    retained_bytes+=capture.bytes
                    result={'teacher_forced':target_logprobs(logits,inputs['input_ids'],list(range(prompt_length,prompt_length+len(target)))) if target else None,
                        'correct_class':target_logprobs(logits,inputs['input_ids'],[prompt_length+i for i in class_positions]) if target else None,
                        'class_span':class_span,'forced_text':request['forced_text'],'inference_output_tokens':0,'forced_target_tokens':len(target),
                        'diagnostic_only':capture.diagnostic_only,'capture':capture.metadata}
                    if capture.values:
                        filename='capture-'+hashlib.sha256(request['id'].encode()).hexdigest()+'.safetensors'
                        save_file(capture.values,output/filename);result['capture_file']=filename
                    del logits
            torch.cuda.synchronize()
            result.update(base,completed=True,mode=mode,rendered_prompt=text,rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                prompt_token_ids=inputs['input_ids'][0,:prompt_length].cpu().tolist(),image_order=paths,references=request.get('references',[]),
                inference_input_tokens=prompt_length,wall_seconds=time.monotonic()-tick,ended_at_unix=time.time(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                before_allocated_bytes=before_allocated,before_reserved_bytes=before_reserved,after_allocated_bytes=torch.cuda.memory_allocated(),after_reserved_bytes=torch.cuda.memory_reserved(),cpu_memory=cpu_memory(),
                kernel_active_seconds=None,gpu_allocation_accounting='trusted host lease receipt includes load, idle and forward time')
            append_event(output/'raw_answers.jsonl',result)
            append_event(output/'attempts.jsonl',dict(base,status='complete',ended_at_unix=time.time()))
        except BaseException as error:
            append_event(output/'attempts.jsonl',dict(base,status='error',error_type=type(error).__name__,message=str(error),ended_at_unix=time.time()))
            raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
    json_write(output/'completion.json',{'status':'complete','n':len(requests),'mode':mode,'split':split,'requests_sha256':digest(requests),'ended_at_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','requests','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['generation','read'],default='generation');p.add_argument('--split',required=True)
    p.add_argument('--candidate');p.add_argument('--query-seconds',type=int,default=180)
    a=p.parse_args();run(a.model,a.requests,a.output,mode=a.mode,split=a.split,candidate=a.candidate,query_seconds=a.query_seconds)

if __name__=='__main__':main()
