#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSP child capture: raw token events plus final source/exit coverage.

Runs inside the isolated container. Never reads auth or exports session text.
Native Codex JSON stdout/stderr is inherited and captured by the trusted host.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def capture_usage(session_root=Path('/codex/sessions'), output=Path('/results'), *, child_exit=None):
    rows, sources = [], []
    for path in sorted(session_root.rglob('*.jsonl')):
        if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
            raise ValueError('invalid or oversized private session source')
        data = path.read_bytes()
        malformed, identities = [], []
        for ordinal, line in enumerate(data.splitlines()):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                malformed.append(ordinal)
                continue
            if event.get('type') == 'session_meta':
                identities.append(event.get('payload', {}).get('id'))
            if event.get('payload', {}).get('type') == 'token_count':
                event['_source_file'] = str(path)
                event['_source_line'] = ordinal
                rows.append(event)
        sources.append({'file': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'session_ids': identities, 'malformed_lines': malformed,
                        'newline_terminated': data.endswith(b'\n')})
    coverage = {'schema': 'wsp-final-capture/1', 'captured_at_unix': time.time(), 'child_exit': child_exit,
                'sources': sources, 'token_events': len(rows), 'events_sha256': digest(rows)}
    for name, data in (('codex_usage_events.jsonl', ''.join(json.dumps(e) + '\n' for e in rows)),
                       ('codex_usage_capture.json', json.dumps(coverage) + '\n')):
        temporary = output / (name + '.tmp')
        temporary.write_text(data)
        temporary.replace(output / name)
    return coverage


if __name__ == '__main__':
    process = subprocess.Popen(sys.argv[1:])
    while process.poll() is None:
        capture_usage()
        time.sleep(.1)
    capture_usage(child_exit=process.returncode)
    sys.exit(process.returncode)
