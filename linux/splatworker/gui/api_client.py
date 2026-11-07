"""Tiny stdlib HTTP client for the worker API (PROTOCOL.md §4), used by the GUI.

The GUI talks to its own child server over 127.0.0.1 with the same bearer token as
the Mac, so everything it shows comes through the public, unchanged API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class WorkerApiError(Exception):
    def __init__(self, status: Optional[int], message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class WorkerClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 5.0,
                 opener: Optional[Callable[..., Any]] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    def _request(self, method: str, path: str, timeout: Optional[float] = None):
        req = urllib.request.Request(self.base_url + path, method=method)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            return self._open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8", "replace")).get("error") or e.reason
            except Exception:
                msg = str(e.reason)
            raise WorkerApiError(e.code, msg)
        except (urllib.error.URLError, OSError) as e:
            raise WorkerApiError(None, f"cannot reach the worker: {getattr(e, 'reason', e)}")

    def _json(self, method: str, path: str) -> Any:
        with self._request(method, path) as resp:
            body = resp.read()
        return json.loads(body) if body else None

    def info(self) -> Dict[str, Any]:
        return self._json("GET", "/v1/info")

    def jobs(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/v1/jobs")

    def cancel(self, job_id: str) -> Dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{job_id}/cancel")

    def delete(self, job_id: str) -> None:
        with self._request("DELETE", f"/v1/jobs/{job_id}"):
            pass

    def log(self, job_id: str) -> str:
        with self._request("GET", f"/v1/jobs/{job_id}/log") as resp:
            return resp.read().decode("utf-8", "replace")

    def download(self, job_id: str, artifact: str, dest: Path,
                 progress: Optional[Callable[[int, int], None]] = None) -> Path:
        """``artifact`` is ``result.ply``, ``preview.ply`` or ``output.zip``."""
        if artifact not in ("result.ply", "preview.ply", "output.zip"):
            raise ValueError(artifact)
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        done = 0
        with self._request("GET", f"/v1/jobs/{job_id}/{artifact}", timeout=60) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        os.replace(part, dest)
        return dest
