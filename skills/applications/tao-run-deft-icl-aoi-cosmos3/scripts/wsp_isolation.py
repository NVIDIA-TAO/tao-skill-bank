#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic editor sandbox entry and deny-by-default provider CONNECT relay.

The editor has no network interface beyond loopback. Its optional API traffic
uses one Unix socket; the host relay accepts only frozen provider hosts on443.
The WSP capability socket is separate and never exposes a holdout operation.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import select
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

DENIED = ['/home/ubuntu/workspace', '/workspace', '/source', '/private', '/holdout',
          '/benchmark', '/proxy', '/mining', '/var/run/docker.sock', '/run/docker.sock',
          '/home/ubuntu/.codex', '/root/.codex', '/model', '/base',
          '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/AGENTS.md', '/task/AGENTS.md']


def probe():
    checks = {path: not (os.path.exists(path) or os.access(path, os.R_OK)) for path in DENIED}
    checks['no_gpu_environment'] = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'void') == 'void'
    checks['no_torch'] = __import__('importlib.util', fromlist=['find_spec']).find_spec('torch') is None
    checks['no_user_config'] = not Path('/codex/config.toml').exists()
    checks['root_readonly'] = False
    try:
        Path('/must-not-write').write_text('x')
    except OSError:
        checks['root_readonly'] = True
    checks['no_direct_network'] = False
    with socket.socket() as connection:
        connection.settimeout(1)
        try:
            connection.connect(('1.1.1.1', 443))
        except OSError:
            checks['no_direct_network'] = True
    checks['no_host_processes'] = not Path('/proc/1/root/home/ubuntu/workspace').exists()
    print(json.dumps({'schema': 'wsp-access-probe/1', 'checks': checks, 'passed': all(checks.values())}), flush=True)
    if not all(checks.values()):
        raise RuntimeError('editor isolation probe failed')
    return checks


def relay(a, b, idle_timeout=60):
    while True:
        ready, _, _ = select.select([a, b], [], [], idle_timeout)
        if not ready:
            return
        for source in ready:
            data = source.recv(65536)
            if not data:
                return
            (b if source is a else a).sendall(data)


def connect_target(request, allowed):
    """Exact authority parsing: no arbitrary HTTP, DNS names, ports or IPs."""
    try:
        text = request.decode('ascii')
        first = text.split('\r\n')[0].split(' ')
        if len(first) != 3 or first[0] != 'CONNECT' or first[2] not in ('HTTP/1.0', 'HTTP/1.1'):
            raise ValueError
        host, port = first[1].split(':')
        if host not in allowed or port != '443':
            raise ValueError
        return host, 443
    except (ValueError, UnicodeError):
        raise ValueError('provider egress denied') from None


class ProviderHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(15)
        data = b''
        while not data.endswith(b'\r\n\r\n') and len(data) <= 8192:
            block = self.request.recv(1)
            if not block:
                return
            data += block
        try:
            if len(data) > 8192:
                raise ValueError('request too long')
            host, port = connect_target(data, self.server.allowed)
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
                raise ValueError('provider resolved to nonpublic address')
            family, kind, protocol, _, address = addresses[0]
            with socket.socket(family, kind, protocol) as upstream:
                upstream.settimeout(15)
                upstream.connect(address)
                self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                relay(self.request, upstream)
        except (OSError, ValueError):
            try:
                self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
            except OSError:
                pass


class UnixProvider(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    def __init__(self, path, allowed):
        if not allowed or set(allowed) - {'chatgpt.com', 'api.openai.com'}:
            raise ValueError('unreviewed provider host')
        self.allowed = frozenset(allowed)
        super().__init__(str(path), ProviderHandler)
        os.chmod(path, 0o600)


class ClosingProvider(UnixProvider):
    """Track and close only this invocation's incoming provider connections."""
    def __init__(self, *args):
        self.connections, self.connection_lock = set(), threading.Lock()
        self.workers = set()
        super().__init__(*args)

    def get_request(self):
        request, address = super().get_request()
        with self.connection_lock: self.connections.add(request)
        return request, address

    def close_request(self, request):
        with self.connection_lock: self.connections.discard(request)
        super().close_request(request)

    def process_request_thread(self, request, client_address):
        worker = threading.current_thread()
        with self.connection_lock: self.workers.add(worker)
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.connection_lock: self.workers.discard(worker)

    def wait_closed(self, timeout=2):
        until = time.monotonic() + timeout
        with self.connection_lock: workers = list(self.workers)
        for worker in workers: worker.join(max(0, until - time.monotonic()))
        with self.connection_lock:
            if self.workers: raise ValueError('owned provider worker did not exit')

    def disconnect(self):
        with self.connection_lock:
            for request in list(self.connections):
                try: request.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                request.close()



class LocalProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect('/channel/provider.sock')
            relay(self.request, upstream)


def editor_entry(command):
    probe()
    proxy = socketserver.ThreadingTCPServer(('127.0.0.1', 3128), LocalProxyHandler)
    proxy.daemon_threads = True
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    env = dict(os.environ, HTTPS_PROXY='http://127.0.0.1:3128', HTTP_PROXY='http://127.0.0.1:3128',
               https_proxy='http://127.0.0.1:3128', http_proxy='http://127.0.0.1:3128', NO_PROXY='')
    handlers = {sig: signal.signal(sig, lambda number, frame: Path('/tmp/editor-stop').touch())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        Path('/results/editor_child.json').write_text(json.dumps({'pid': process.pid}))
        # The capture supervisor owns termination/reap/final capture. Never kill
        # its group: it deliberately launches Codex in a different child group.
        process.wait()
        supervision = json.loads(Path('/results/editor_supervision.json').read_text())
        wrapper_exit = 0 if process.returncode == 0 else 70
        exit_file = Path('/results/editor_exit.json.tmp')
        exit_file.write_text(json.dumps({
            'schema': 'wsp-editor-exit/2', 'child_returncode': supervision['child_returncode'],
            'supervisor_returncode': process.returncode, 'container_returncode': wrapper_exit}))
        exit_file.replace('/results/editor_exit.json')
        while not Path('/tmp/harvested').exists():
            time.sleep(.1)
        return wrapper_exit
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        proxy.shutdown()
        proxy.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'editor-entry'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'probe':
        probe()
    else:
        sys.exit(editor_entry(args.command))
