"""HTTP client for the worker API (stdlib only), used by the Studio for its own local server and for other
workers. Same requests as the Mac's ``WorkerClient`` (PROTOCOL.md §4, §6)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ApiError(Exception):
    def __init__(self, status: Optional[int], message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Client:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0, opener=None,
                 local_secret: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.local_secret = local_secret
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    def _req(self, method: str, path: str, body: Optional[bytes] = None, ctype: Optional[str] = None,
             timeout: Optional[float] = None):
        req = urllib.request.Request(self.base_url + path, data=body, method=method)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if self.local_secret:
            req.add_header("X-SplatWork-Local", self.local_secret)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            return self._open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8", "replace")).get("error") or e.reason
            except Exception:
                msg = str(e.reason)
            raise ApiError(e.code, msg)
        except (urllib.error.URLError, OSError) as e:
            raise ApiError(None, f"cannot reach the worker: {getattr(e, 'reason', e)}")

    def get(self, path: str) -> Any:
        with self._req("GET", path) as r:
            data = r.read()
        return json.loads(data) if data else None

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        with self._req("POST", path, json.dumps(body or {}).encode(), "application/json") as r:
            data = r.read()
        return json.loads(data) if data else None

    def delete(self, path: str) -> None:
        with self._req("DELETE", path):
            pass

    def download(self, path: str, dest: Path, progress: Optional[Callable[[int, int], None]] = None,
                 missing_ok: bool = False) -> Optional[Path]:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        try:
            resp = self._req("GET", path, timeout=120)
        except ApiError as e:
            if missing_ok and e.status == 404:
                return None
            raise
        with resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
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

    # ------------------------------------------------------------- API
    def info(self) -> Dict[str, Any]:
        return self.get("/v1/info")

    def jobs(self) -> List[Dict[str, Any]]:
        return self.get("/v1/jobs")

    def job(self, job_id: str) -> Dict[str, Any]:
        return self.get(f"/v1/jobs/{job_id}")

    def local_dataset(self, path: Path) -> Dict[str, Any]:
        return self.post("/v1/datasets/local", {"path": str(Path(path).resolve())})

    def export_dataset(self, dataset_id: str, dest: Path) -> Dict[str, Any]:
        return self.post(f"/v1/datasets/{dataset_id}/export-local", {"path": str(Path(dest).resolve())})

    def delete_dataset(self, dataset_id: str) -> None:
        try:
            self.delete(f"/v1/datasets/{dataset_id}")
        except ApiError:
            pass

    def process(self, dataset_id: str, request: Dict[str, Any], name: str, priority: int = 1) -> str:
        return self.post("/v1/process", {"datasetId": dataset_id, "request": request, "name": name,
                                         "priority": priority})["jobId"]

    def train(self, dataset_id: str, config: Dict[str, Any], name: str, priority: int = 1) -> str:
        return self.post("/v1/jobs", {"datasetId": dataset_id, "config": config, "name": name,
                                      "priority": priority})["jobId"]

    def census(self, dataset_id: str, ply_path: Path, name: str, priority: int = 1) -> str:
        return self.post("/v1/census", {"datasetId": dataset_id, "plyPath": str(ply_path), "name": name,
                                        "priority": priority})["jobId"]

    def cancel(self, job_id: str) -> None:
        self.post(f"/v1/jobs/{job_id}/cancel")

    def prioritize(self, job_id: str) -> None:
        self.post(f"/v1/jobs/{job_id}/prioritize")

    # pause / resume / checkpoint (worker capability "pause")
    def pause(self, job_id: str) -> Dict[str, Any]:
        return self.post(f"/v1/jobs/{job_id}/pause")

    def resume(self, job_id: str) -> Dict[str, Any]:
        return self.post(f"/v1/jobs/{job_id}/resume")

    def checkpoint(self, job_id: str, dest_for_step: Callable[[int], Path], wait: float = 120.0) -> tuple:
        """Have the job write a checkpoint of its current splats and download it to ``dest_for_step(step)``.
        Returns ``(path, step)``."""
        with self._req("POST", f"/v1/jobs/{job_id}/checkpoint?wait={wait:g}", b"{}", "application/json",
                       timeout=wait + 30) as r:
            s = json.loads(r.read() or b"{}")
        step = s.get("checkpointStep")
        if step is None:
            raise ApiError(None, "the worker didn't write a checkpoint in time")
        return self.download(f"/v1/jobs/{job_id}/checkpoint.ply", dest_for_step(int(step))), int(step)

    def log(self, job_id: str) -> str:
        with self._req("GET", f"/v1/jobs/{job_id}/log") as r:
            return r.read().decode("utf-8", "replace")
