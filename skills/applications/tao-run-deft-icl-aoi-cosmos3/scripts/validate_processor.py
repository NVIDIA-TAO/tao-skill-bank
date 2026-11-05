#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real native-processor label-mask and image-cardinality smoke validation."""
import argparse
import json
import os
from pathlib import Path

def check(annotation,media,model,mode):
    os.environ['TAO_CR3_ANNOTATION_MODE']=mode
    from cfw_cr3_aoi_adapter import CR3TwoImageShareGPTDataset,CR3TwoImageVLMProcessor
    from cosmos_framework.data.generator.processors import build_processor
    from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX
    dataset=CR3TwoImageShareGPTDataset(annotation,media)
    native=build_processor(tokenizer_type=model,config_variant='hf')
    processor=CR3TwoImageVLMProcessor(native,ignore_index=IGNORE_INDEX)
    evidence=[]
    for row in dataset:
        value=processor.process(row)
        supervised=value['labels']!=IGNORE_INDEX
        if not supervised.any() or supervised.all(): raise ValueError('loss mask must separate input from assistant target')
        if not (supervised==value['token_mask']).all(): raise ValueError('label/mask disagreement')
        tokens=value['input_ids'][supervised]
        target=native.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        if target!=row['conversations'][-1]['value']: raise ValueError(f'supervised text differs from exact target: {target!r}')
        if int(value['image_grid_thw'].shape[0])!=len(row['images']): raise ValueError('native image cardinality mismatch')
        if not value['attention_mask'].all(): raise ValueError('unexpected pre-collation padding')
        evidence.append({'input_tokens':len(value['input_ids']),'valid_label_count':int(supervised.sum()),
                         'image_count':len(row['images']),'target_exact':True,'prompt_masked':True})
    return {'status':'passed','mode':mode,'records':evidence}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--annotation',required=True);p.add_argument('--media',required=True)
    p.add_argument('--model',required=True);p.add_argument('--mode',choices=['tht_single','bare_okng'],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=check(a.annotation,a.media,a.model,a.mode)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
