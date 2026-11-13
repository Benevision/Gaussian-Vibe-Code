"""Smoke test for the worker on Windows (and any OS): start ``splatworker serve`` the way the desktop app does,
check ``/v1/info``, find the worker over mDNS (``_splatworker._tcp``, as a Mac would), then stop it by closing its
stdin and check that it shut down cleanly.

    python windows/scripts/worker_smoke.py [--port 7878]

Needs the worker's dependencies (fastapi, uvicorn, zeroconf); no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7878)
    ap.add_argument("--name", default="SplatWork Windows smoke test")
    ap.add_argument("--no-mdns-check", action="store_true")
    args = ap.parse_args()

    data = Path(tempfile.mkdtemp(prefix="splatwork-smoke-"))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "linux") + os.pathsep + str(ROOT / "tools"),
               SPLATWORKER_STOP_ON_STDIN_EOF="1", SPLATWORKER_TOKEN="smoke-token")
    argv = [sys.executable, "-m", "splatworker", "serve", "--port", str(args.port), "--data-dir", str(data),
            "--name", args.name]
    print("$", " ".join(argv), flush=True)
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, text=True, errors="replace", **kw)
    lines = []

    def pump():
        for line in proc.stdout:
            lines.append(line.rstrip())
            print("  worker |", line.rstrip(), flush=True)

    threading.Thread(target=pump, daemon=True).start()
    try:
        info = None
        deadline = time.time() + 60
        while time.time() < deadline and proc.poll() is None:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/info",
                                             headers={"Authorization": "Bearer smoke-token"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    info = json.load(r)
                break
            except OSError:
                time.sleep(0.5)
        if info is None:
            print("FAIL: /v1/info did not answer", flush=True)
            return 1
        print("info:", json.dumps({k: info.get(k) for k in ("name", "version", "platform", "gpus", "capabilities")
                                   if k in info}), flush=True)
        if info.get("name") != args.name:
            print(f"FAIL: name {info.get('name')!r}", flush=True)
            return 1

        if not args.no_mdns_check:
            found = browse(args.name, timeout=20)
            if not found:
                print("FAIL: _splatworker._tcp not found over mDNS", flush=True)
                return 1
            print("mDNS:", found, flush=True)

        t0 = time.time()
        proc.stdin.close()
        try:
            rc = proc.wait(40)
        except subprocess.TimeoutExpired:
            print("FAIL: worker did not stop within 40 s of closing stdin", flush=True)
            return 1
        print(f"stopped with code {rc} after {time.time() - t0:.1f} s", flush=True)
        if rc not in (0, None) and not any("Shutting down" in l or "Finished server process" in l for l in lines):
            print("FAIL: unclean exit", flush=True)
            return 1
        print("OK", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


def browse(name: str, timeout: float) -> dict:
    from zeroconf import ServiceBrowser, Zeroconf

    found: dict = {}
    zc = Zeroconf()

    class Listener:
        def add_service(self, z, type_, svc):
            info = z.get_service_info(type_, svc, timeout=3000)
            if info is not None and svc.startswith(name):
                found.update(service=svc, port=info.port, addresses=info.parsed_addresses(),
                             properties={k.decode(): (v or b"").decode() for k, v in info.properties.items()})

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    try:
        ServiceBrowser(zc, "_splatworker._tcp.local.", Listener())
        deadline = time.time() + timeout
        while not found and time.time() < deadline:
            time.sleep(0.3)
    finally:
        zc.close()
    return found


if __name__ == "__main__":
    sys.exit(main())
