"""iPhone Inbox: receives ``.splatcapture`` bundles from Splat Capture (PROTOCOL.md §5), like the Mac.

HTTP on port 7879 (next free port if taken), Bonjour ``_splatcapture._tcp`` with TXT ``name`` / ``version=1``.

* ``GET /v1/hello`` → ``{"name", "app": "Splat Studio", "version": 1}``
* ``POST /v1/captures`` (zip; ``X-Capture-Name``, ``X-Pairing-Code``) → ``{"ok": true, "savedAs": "<name>.splatcapture"}``
  403 when pairing is required and the code is wrong.

Captures are unpacked with the worker's safe extractor into ``<inbox>/<name>.splatcapture``.
"""
from __future__ import annotations

import json
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

from ..safezip import UnsafeArchive, safe_extract

MAX_UPLOAD = 64 * 1024 ** 3
SERVICE = "_splatcapture._tcp.local."


def new_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def format_code(code: str) -> str:
    return f"{code[:3]} {code[3:]}"


@dataclass
class Capture:
    path: Path
    name: str
    received: float
    size: int
    sender: str = ""

    @property
    def thumbnail(self) -> Optional[Path]:
        t = self.path / "thumbnail.jpg"
        if t.is_file():
            return t
        imgs = self.path / "images"
        if imgs.is_dir():
            for p in sorted(imgs.iterdir()):
                return p
        return None

    @property
    def frames(self) -> int:
        try:
            return len(json.loads((self.path / "manifest.json").read_text()).get("frames", []))
        except (OSError, ValueError):
            return 0


def list_captures(inbox: Path) -> List[Capture]:
    out = []
    if Path(inbox).is_dir():
        for d in Path(inbox).iterdir():
            if d.suffix == ".splatcapture" and (d / "manifest.json").is_file():
                meta = {}
                try:
                    meta = json.loads((d / ".received.json").read_text())
                except (OSError, ValueError):
                    pass
                size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                out.append(Capture(d, d.stem, meta.get("time", d.stat().st_mtime), size, meta.get("sender", "")))
    return sorted(out, key=lambda c: -c.received)


def _unique(inbox: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in " ._-()" else "_" for ch in name).strip(" .") or "Capture"
    p, k = inbox / f"{safe}.splatcapture", 2
    while p.exists():
        p, k = inbox / f"{safe} {k}.splatcapture", k + 1
    return p


class CaptureReceiver:
    def __init__(self, inbox: Path, name: str, port: int = 7879, require_code: bool = True,
                 on_received: Optional[Callable[[Capture], None]] = None,
                 on_progress: Optional[Callable[[str, int, int], None]] = None, mdns: bool = True):
        self.inbox = Path(inbox)
        self.name = name
        self.port = port
        self.require_code = require_code
        self.code = new_code()
        self.on_received = on_received
        self.on_progress = on_progress
        self.mdns = mdns
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._zc = None
        self._info = None
        self.error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def start(self) -> bool:
        self.inbox.mkdir(parents=True, exist_ok=True)
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def _json(self, status: int, obj) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.split("?")[0] == "/v1/hello":
                    return self._json(200, {"name": receiver.name, "app": "Splat Studio", "version": 1})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                if self.path.split("?")[0] != "/v1/captures":
                    return self._json(404, {"error": "not found"})
                if receiver.require_code and self.headers.get("X-Pairing-Code", "").replace(" ", "") != receiver.code:
                    return self._json(403, {"error": "wrong or missing pairing code"})
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._json(400, {"error": "bad Content-Length"})
                if length <= 0 or length > MAX_UPLOAD:
                    return self._json(413 if length > MAX_UPLOAD else 400, {"error": "bad size"})
                name = urllib.parse.unquote(self.headers.get("X-Capture-Name") or "Capture")[:120]
                try:
                    saved = receiver._receive(self.rfile, length, name, self.client_address[0])
                except UnsafeArchive as e:
                    return self._json(400, {"error": f"rejected archive: {e}"})
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": str(e)})
                self._json(200, {"ok": True, "savedAs": saved.path.name})

        for port in [self.port] + list(range(self.port + 1, self.port + 20)):
            try:
                self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
                self.port = port
                break
            except OSError as e:
                self.error = str(e)
        if self._httpd is None:
            return False
        self.error = None
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="capture-receiver").start()
        if self.mdns:
            self._advertise()
        return True

    def _receive(self, rfile, length: int, name: str, sender: str) -> Capture:
        tmpdir = Path(tempfile.mkdtemp(prefix=".incoming-", dir=self.inbox))
        try:
            zpath = tmpdir / "upload.zip"
            done = 0
            with open(zpath, "wb") as fh:
                while done < length:
                    chunk = rfile.read(min(1 << 20, length - done))
                    if not chunk:
                        raise IOError("connection closed during upload")
                    fh.write(chunk)
                    done += len(chunk)
                    if self.on_progress:
                        self.on_progress(name, done, length)
            out = tmpdir / "x"
            safe_extract(zpath, out, max_bytes=MAX_UPLOAD * 2, max_files=500_000, max_ratio=1000.0)
            roots = [p for p in out.iterdir()]
            root = roots[0] if len(roots) == 1 and roots[0].is_dir() and not (out / "manifest.json").exists() else out
            if not (root / "manifest.json").is_file():
                raise ValueError("not a capture bundle (no manifest.json)")
            dest = _unique(self.inbox, name)
            shutil.move(str(root), dest)
            (dest / ".received.json").write_text(json.dumps({"time": time.time(), "sender": sender}))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        cap = Capture(dest, dest.stem, time.time(), length, sender)
        if self.on_received:
            self.on_received(cap)
        return cap

    def _advertise(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf

            from ..discovery import local_ipv4_addresses

            addrs = [socket.inet_aton(a) for a in local_ipv4_addresses()] or [socket.inet_aton("127.0.0.1")]
            host = socket.gethostname().split(".")[0]
            self._info = ServiceInfo(SERVICE, f"{self.name}.{SERVICE}", addresses=addrs, port=self.port,
                                     properties={"name": self.name, "version": "1"}, server=f"{host}.local.")
            self._zc = Zeroconf()
            self._zc.register_service(self._info, allow_name_change=True)
        except Exception as e:  # noqa: BLE001 — receiving still works with a typed address
            self.error = f"Bonjour advertisement failed: {e}"
            self._zc = None

    def stop(self) -> None:
        if self._zc is not None:
            try:
                self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                pass
            self._zc = None
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
