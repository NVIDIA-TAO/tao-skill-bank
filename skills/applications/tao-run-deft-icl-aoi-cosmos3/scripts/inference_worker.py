#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated image inference worker. Receives no gold or evaluator dependency."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())

def run(model_path, requests_path, output_dir, seed=42):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    requests=json.loads(Path(requests_path).read_text())
    if len({r['id'] for r in requests})!=len(requests):
        raise ValueError('duplicate request identities')
    done={}
    answers=output/'raw_answers.jsonl'
    previous=os.environ.get('TAO_RESUME_ANSWERS')
    if previous and not answers.exists():
        # Completed responses, including INVALID, are copied before loading a
        # model. Request hashes below prevent changed-input recovery.
        import shutil
        shutil.copyfile(previous, answers)
    if answers.exists():
        for line in answers.read_text().splitlines():
            row=json.loads(line)
            if row['id'] in done: raise ValueError('duplicate completed response')
            done[row['id']]=row
    if set(done)-{r['id'] for r in requests}:
        raise ValueError('recovery contains unrelated query identities')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start=time.time()
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    profile={'transformers':transformers.__version__,'torch':torch.__version__,
             'processor':processor.to_dict(),'chat_template':processor.chat_template,
             'chat_template_sha256':hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
             'model_path':model_path,'attention':'sdpa','dtype':'bfloat16','seed':seed,
             'max_new_tokens':4096,'do_sample':False,'reasoning':'prompted_off',
             'native_thinking_disabled':False,'model_load_seconds':time.time()-start}
    (output/'runtime_profile.json').write_text(json.dumps(profile,default=str,sort_keys=True,indent=2)+'\n')
    for request in requests:
        request_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if request['id'] in done:
            if done[request['id']]['request_sha256']!=request_hash: raise ValueError('request changed on resume')
            continue
        if set(request)-{'id','messages','references','policy_sha256'}:
            raise ValueError('unexpected classifier input fields')
        messages=[]
        images=[]
        image_paths=[]
        for msg in request['messages']:
            if msg['role'] not in ('user','assistant'):
                raise ValueError('unsupported message role')
            parts=[]
            for part in msg['content']:
                if part['type']=='image':
                    path=Path(part['image']).resolve(strict=True)
                    if not path.is_relative_to('/input'):
                        raise ValueError('image outside isolated input view')
                    image=Image.open(path).convert('RGB')
                    if image.size!=(256,256): raise ValueError('expected preprocessed 256x256 image')
                    images.append(image)
                    image_paths.append(str(path))
                    parts.append({'type':'image','image':str(path)})
                elif part['type']=='text':
                    parts.append({'type':'text','text':part['text']})
                else: raise ValueError('unknown content type')
            messages.append({'role':msg['role'],'content':parts})
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=images,return_tensors='pt').to('cuda')
        prompt_tokens=int(inputs['input_ids'].shape[1])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started=time.time()
        invocation=f"{request['id']}:{started:.6f}"
        append(output/'attempts.jsonl',{'invocation_id':invocation,'id':request['id'],'started_at_unix':started,'status':'running','request_sha256':request_hash})
        try:
            with torch.inference_mode():
                tokens=model.generate(**inputs,do_sample=False,max_new_tokens=4096)
            torch.cuda.synchronize()
            completion=tokens[0,prompt_tokens:]
            raw=processor.tokenizer.decode(completion,skip_special_tokens=True)
            ended=time.time()
            row={'id':request['id'],'completed':True,'raw_answer':raw,
                 'finish_reason':'length' if len(completion)>=4096 else 'stop',
                 'request_sha256':request_hash,'rendered_prompt':text,
                 'rendered_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                 'image_order':image_paths,'references':request.get('references',[]),
                 'policy_sha256':request.get('policy_sha256'),
                 'invocation_id':invocation,'inference_input_tokens':prompt_tokens,
                 'inference_output_tokens':len(completion),'started_at_unix':started,'ended_at_unix':ended,
                 'wall_seconds':ended-started,'gpu_seconds':ended-started,
                 'gpu_time_method':'one allocated GPU times generation wall time',
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            # Persist the first raw final response before parsing/grading anywhere.
            append(answers,row)
            append(output/'attempts.jsonl',{'invocation_id':invocation,'id':request['id'],'status':'complete','ended_at_unix':ended})
        except BaseException as error:
            append(output/'attempts.jsonl',{'invocation_id':invocation,'id':request['id'],'status':'error','started_at_unix':started,'ended_at_unix':time.time(),'error_type':type(error).__name__,'message':str(error),'inference_input_tokens':prompt_tokens,'inference_output_tokens':None})
            raise
        finally:
            for image in images: image.close()
    (output/'completion.json').write_text(json.dumps({'status':'complete','n':len(requests),'ended_at_unix':time.time()})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--requests',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    run(a.model,a.requests,a.output,a.seed)
