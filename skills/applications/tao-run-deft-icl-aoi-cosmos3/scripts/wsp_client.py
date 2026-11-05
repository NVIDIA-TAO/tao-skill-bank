#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only editor-facing WSP client: JSON over a single mounted Unix socket."""
import json
import socket
import sys


def call(request, path='/channel/wsp.sock', max_bytes=4 * 1024**2):
    data = json.dumps(request, allow_nan=False).encode() + b'\n'
    if len(data) > max_bytes:
        raise ValueError('request byte cap exceeded')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1200)
        connection.connect(path)
        connection.sendall(data)
        with connection.makefile('rb') as stream:
            response = stream.readline(16 * 1024**2 + 1)
    if not response.endswith(b'\n') or len(response) > 16 * 1024**2:
        raise ValueError('invalid bounded response')
    return json.loads(response)


if __name__ == '__main__':
    raw = sys.stdin.buffer.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise SystemExit('request byte cap exceeded')
    print(json.dumps(call(json.loads(raw)), allow_nan=False))
