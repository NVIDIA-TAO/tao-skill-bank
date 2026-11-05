# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic child-process/event capture inside a fresh isolated Codex container.

Only token_count events leave the private session stream. Credentials are never
read or copied. Native rollout enforcement owns the per-response token gate.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

def capture_usage():
    rows=[]
    for p in Path('/codex/sessions').rglob('*.jsonl'):
        for ordinal,line in enumerate(p.read_text().splitlines()):
            try: event=json.loads(line)
            except ValueError: continue
            if event.get('payload',{}).get('type')=='token_count':
                event['_source_file']=str(p);event['_source_line']=ordinal
                rows.append(event)
    target=Path('/results/codex_usage_events.jsonl')
    if rows:
        temporary=target.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(e)+'\n' for e in rows))
        temporary.replace(target)

if __name__=='__main__':
    process=subprocess.Popen(sys.argv[1:])
    while process.poll() is None:
        capture_usage();time.sleep(.1)
    capture_usage()
    sys.exit(process.returncode)
