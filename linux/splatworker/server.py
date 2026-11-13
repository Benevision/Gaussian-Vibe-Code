"""HTTP API (PROTOCOL.md §4 and §6), job queue and persistence.

Layout under ``data_dir``::

    datasets/<datasetId>/meta.json        upload metadata (root = dataset root, relative)
    datasets/<datasetId>/files/...        extracted upload
    jobs/<jobId>/job.json                 spec + JobStatus (persisted)
    jobs/<jobId>/train.log                log (served by /log)
    jobs/<jobId>/preview.splat            latest compact snapshot (train; .splat, ≤400 k splats)
    jobs/<jobId>/preview.ply              full snapshot (only written by test/fake backends)
    jobs/<jobId>/result.ply               final splat (train)
    jobs/<jobId>/checkpoint.ply           on-demand / pause-time snapshot of the splats (train, capability "pause")
    jobs/<jobId>/resume.pt                full trainer state of a paused train job (removed when the job ends)
    jobs/<jobId>/output.zip               output of a processing job
    tmp/                                  in-flight uploads
"""

from __future__ import annotations

import collections
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import PRODUCT_NAME, PROTOCOL_VERSION, __version__
from .backends import Backend, JobCancelled, JobContext, JobPaused, SubprocessBackend
from .config import ConfigError, ServerSettings, TrainingConfig
from .dataset import DatasetError, find_upload_root, parse_dataset
from .processing import ProcessingError, homevideo_available, validate_request
from .safezip import UnsafeArchive, safe_extract

log = logging.getLogger("splatworker")

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ACTIVE_STATES = ("loading", "training", "running")
FINAL_STATES = ("completed", "failed", "cancelled")
# "paused" (worker ≥ 1.4.0, capability "pause") is neither active nor final: the job holds no GPU and waits for
# POST /v1/jobs/{id}/resume. Old clients see an unknown, non-terminal state and simply keep polling.
STATUS_KEYS = ("jobId", "kind", "name", "state", "step", "totalSteps", "loss", "psnr", "evalPsnr",
               "numGaussians", "elapsedSeconds", "etaSeconds", "previewStep", "message",
               "datasetId", "createdAt", "gpu", "outputDatasetId", "priority", "origin", "checkpointStep",
               "pausedAt", "stage", "stageProgress", "stages", "trainJobId", "parentJobId")
REMOTE_PAUSE_LIMIT = 3600.0         # another computer's paused job resumes by itself after this (s)
LOCAL_PAUSE_LIMIT = 14 * 86400.0    # a paused job of this PC's app is cancelled (resume.pt freed) after this
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
LOCAL_HEADER = "X-SplatWork-Local"   # per-launch secret of the desktop app that started this server


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


_IF_CACHE: dict = {"at": 0.0, "value": []}


def _lan_interfaces() -> list:
    """(address, kind) pairs from discovery.local_ipv4_interfaces, cached for a minute (polled by /v1/info)."""
    if time.monotonic() - _IF_CACHE["at"] > 60:
        try:
            from .discovery import local_ipv4_interfaces

            _IF_CACHE["value"] = local_ipv4_interfaces()
        except Exception:  # noqa: BLE001
            _IF_CACHE["value"] = []
        _IF_CACHE["at"] = time.monotonic()
    return _IF_CACHE["value"]


def _host_memory() -> dict:
    """{"totalMB", "availableMB"} (/proc/meminfo on Linux, GlobalMemoryStatusEx on Windows; empty if unknown)."""
    from .platform_compat import memory

    m = memory()
    return {("totalMB" if k == "total" else "availableMB"): v // (1024 * 1024) for k, v in m.items()}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise ApiError(404, f"{what} not found")
    return value


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_dir: Path, spec: Dict[str, Any], status: Dict[str, Any]):
        self.dir = job_dir
        self.spec = spec          # {"kind","datasetId","name","config"|"request"}
        self.status = status
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.checkpoint_event = threading.Event()
        self.checkpoints = 0            # checkpoints written since the server started (POST …/checkpoint?wait=)
        self.deleted = False
        self.started_monotonic: Optional[float] = None
        self.lock = threading.Lock()
        self._last_persist = 0.0

    @property
    def id(self) -> str:
        return self.status["jobId"]

    @property
    def state(self) -> str:
        return self.status["state"]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            s = {k: self.status.get(k) for k in STATUS_KEYS}
            if self.started_monotonic is not None and s["state"] in ACTIVE_STATES:
                s["elapsedSeconds"] = round(time.monotonic() - self.started_monotonic, 1)
            return s

    def update(self, persist: bool = False, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                if k in ("state", "step", "totalSteps", "loss", "psnr", "evalPsnr", "numGaussians",
                         "elapsedSeconds", "etaSeconds", "previewStep", "message", "gpu", "priority",
                         "checkpointStep", "pausedAt", "stage", "stageProgress", "stages", "trainJobId"):
                    if isinstance(v, float) and v != v:  # NaN -> null
                        v = None
                    self.status[k] = v
            now = time.monotonic()
            if persist or now - self._last_persist > 5.0:
                self._last_persist = now
                self._persist_locked()

    def persist(self) -> None:
        with self.lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        if self.deleted or not self.dir.exists():
            return
        try:
            _write_json_atomic(self.dir / "job.json", {"spec": self.spec, "status": self.status})
        except OSError as e:  # pragma: no cover
            log.warning("failed to persist job %s: %s", self.id, e)

    def log(self, msg: str) -> None:
        if self.deleted:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            with open(self.dir / "train.log", "a", encoding="utf-8") as fh:
                for line in str(msg).rstrip("\n").splitlines() or [""]:
                    fh.write(f"[{stamp}] {line}\n")
        except OSError:
            pass


class JobManager:
    def __init__(self, settings: ServerSettings, backend: Backend, devices: List[Optional[int]]):
        self.settings = settings
        self.backend = backend
        self.devices = devices or [None]
        self.jobs: Dict[str, Job] = {}
        self.queue: Deque[str] = collections.deque()
        self.cond = threading.Condition()
        self.running: Dict[Optional[int], str] = {}
        self.stopping = False
        self.threads: List[threading.Thread] = []

    # ---------------------------------------------------------------- setup
    def load(self) -> None:
        jobs_dir = self.settings.jobs_dir
        loaded = []
        for d in sorted(jobs_dir.iterdir()) if jobs_dir.exists() else []:
            f = d / "job.json"
            if not f.is_file():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                job = Job(d, data["spec"], data["status"])
            except Exception as e:
                log.warning("skipping unreadable job %s: %s", d.name, e)
                continue
            if job.spec.get("kind") == "pipeline":
                pass       # a whole-pipeline job continues after the restart (start() restarts its runner)
            elif job.state in ACTIVE_STATES and (d / "resume.pt").is_file():
                # a resumed job that was running when the worker stopped: its last pause is still on disk
                job.status["state"] = "paused"
                job.status["message"] = "interrupted by a worker restart; resume continues from the last pause"
                job.status["etaSeconds"] = None
                job.status["pausedAt"] = time.time()     # so the paused-job clean-up still applies
                job.persist()
                job.log("job was interrupted by a worker restart; it can be resumed from its last pause")
            elif job.state in ACTIVE_STATES:
                job.status["state"] = "failed"
                job.status["message"] = "interrupted by worker restart"
                job.status["etaSeconds"] = None
                job.persist()
                job.log("job was interrupted by a worker restart")
            loaded.append(job)
        loaded.sort(key=lambda j: j.status.get("createdAt") or "")
        for job in loaded:
            self.jobs[job.id] = job
            if job.spec.get("kind") == "pipeline":
                continue
            if job.state == "paused" and job.status.get("origin") != "local":
                # another computer's paused job: its client may never resume it, so a restart does
                job.update(persist=True, state="queued", pausedAt=None, message="queued (resumed after a restart)")
                job.log("resumed automatically after a worker restart")
            if job.state == "queued":
                self.queue.append(job.id)

    def start(self) -> None:
        for dev in self.devices:
            t = threading.Thread(target=self._worker_loop, args=(dev,), daemon=True,
                                 name=f"splatworker-gpu{dev}")
            t.start()
            self.threads.append(t)
        from .pipeline import start_runner

        for job in list(self.jobs.values()):
            if job.spec.get("kind") == "pipeline" and job.state not in FINAL_STATES:
                try:       # elapsed time counts from when the job was created (worker downtime included)
                    created = datetime.fromisoformat(str(job.status.get("createdAt")).replace("Z", "+00:00"))
                    job.started_monotonic = time.monotonic() - max(0.0, time.time() - created.timestamp())
                except ValueError:
                    job.started_monotonic = time.monotonic()
                job.log("worker started: continuing the pipeline")
                start_runner(self, job)

    def shutdown(self, timeout: float = 20.0) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
            running = [self.jobs[j] for j in self.running.values() if j in self.jobs]
        for job in running:
            # A training job that can pause saves its state and resumes after the restart; others are cancelled.
            if self.pausable(job) and not job.cancel_event.is_set():
                job.pause_event.set()
            else:
                job.cancel_event.set()
        deadline = time.time() + timeout
        for t in self.threads:
            t.join(max(0.1, deadline - time.time()))

    # ---------------------------------------------------------------- queue
    def submit(self, kind: str, dataset_id: str, name: str, payload_key: str,
               payload: Dict[str, Any], total_steps: int, priority: int = 0, origin: Optional[str] = None,
               parent: Optional[str] = None) -> Job:
        prefix = {"train": "j", "census": "c"}.get(kind, "p")
        job_id = prefix + uuid.uuid4().hex[:11]
        job_dir = self.settings.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        spec = {"kind": kind, "datasetId": dataset_id, "name": name, payload_key: payload}
        if parent:
            spec["parentJobId"] = parent
        status = {
            "jobId": job_id, "kind": kind, "name": name, "state": "queued",
            "step": 0, "totalSteps": total_steps, "loss": None, "psnr": None,
            "numGaussians": 0 if kind == "train" else None,
            "elapsedSeconds": 0.0, "etaSeconds": None, "previewStep": None,
            "message": "queued", "datasetId": dataset_id, "createdAt": _now_iso(), "gpu": None,
            "priority": int(priority), "origin": origin,
        }
        if parent:
            status["parentJobId"] = parent
        job = Job(job_dir, spec, status)
        job.persist()
        job.log(f"{kind} job created for dataset {dataset_id}: {json.dumps(payload)}")
        with self.cond:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.cond.notify()
        return job

    def get(self, job_id: str) -> Job:
        _check_id(job_id, "job")
        job = self.jobs.get(job_id)
        if job is None:
            raise ApiError(404, "job not found")
        return job

    # ------------------------------------------------------------ pipelines
    def submit_pipeline(self, dataset_id: str, name: str, spec: Dict[str, Any], priority: int = 0,
                        origin: Optional[str] = None) -> Job:
        """A whole-pipeline job (capability "pipeline"): coordinated by its own thread, never in the GPU queue —
        its steps are child jobs that are. Its working dataset holds the project as it is prepared."""
        from .pipeline import initial_stages, start_runner
        from .processing import link_tree

        job_id = "w" + uuid.uuid4().hex[:11]
        job_dir = self.settings.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        work_id = _new_dataset_id()
        wd = self.settings.datasets_dir / work_id
        (wd / "files").mkdir(parents=True)
        meta: Dict[str, Any] = {"datasetId": work_id, "createdAt": _now_iso(), "root": "files", "fromJob": job_id,
                                "pipelineWork": True, "hasDatasetJson": False, "bytes": 0}
        if origin == "local":
            meta["localOnly"] = True
        if not spec.get("frames"):                # frames the client ingested: they are the starting point
            src = dataset_root(self.settings, dataset_id)
            link_tree(src, wd / "files", [src])
        _write_json_atomic(wd / "meta.json", meta)
        stages = initial_stages(spec)
        total = 1000
        status = {
            "jobId": job_id, "kind": "pipeline", "name": name, "state": "running",
            "step": 0, "totalSteps": total, "loss": None, "psnr": None, "numGaussians": None,
            "elapsedSeconds": 0.0, "etaSeconds": None, "previewStep": None, "message": "starting",
            "datasetId": dataset_id, "createdAt": _now_iso(), "gpu": None, "priority": int(priority),
            "origin": origin, "stage": stages[0]["id"], "stages": stages, "trainJobId": None,
        }
        job = Job(job_dir, {"kind": "pipeline", "datasetId": dataset_id, "name": name, "pipeline": spec,
                            "workDatasetId": work_id}, status)
        job.started_monotonic = time.monotonic()
        job.persist()
        job.log(f"pipeline job created for dataset {dataset_id}: {json.dumps(spec)}")
        with self.cond:
            self.jobs[job_id] = job
        start_runner(self, job)
        return job

    def children(self, job: Job) -> List[Job]:
        return [j for j in self.jobs.values() if j.spec.get("parentJobId") == job.id]

    def _current_child(self, job: Job) -> Optional[Job]:
        for st in job.status.get("stages") or []:
            if st.get("state") == "running" and st.get("jobId"):
                return self.jobs.get(st["jobId"])
        return None

    def publish_work_dataset(self, job: Job) -> None:
        """The prepared dataset is ready: describe it (so it can be trained) and make it the job's output."""
        work_id = job.spec["workDatasetId"]
        wd = self.settings.datasets_dir / work_id
        meta = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
        meta["bytes"] = _dir_bytes(wd / "files")
        _describe_dataset(wd / "files", meta)
        _write_json_atomic(wd / "meta.json", meta)
        with job.lock:
            job.status["outputDatasetId"] = work_id
        job.persist()

    def drop_dataset(self, dataset_id: str) -> None:
        if ID_RE.match(dataset_id or ""):
            shutil.rmtree(self.settings.datasets_dir / dataset_id, ignore_errors=True)

    def finish_pipeline(self, job: Job, state: str, message: str) -> None:
        if state != "completed" and not job.status.get("outputDatasetId"):
            self.drop_dataset(job.spec.get("workDatasetId", ""))   # nothing usable was prepared
        self._finish(job, state, message)
        if job.deleted:
            shutil.rmtree(job.dir, ignore_errors=True)
            self.drop_dataset(job.spec.get("workDatasetId", ""))

    def list(self) -> List[Dict[str, Any]]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.status.get("createdAt") or "")
        return [j.snapshot() for j in jobs]

    def busy(self) -> bool:
        return bool(self.running)

    def queue_length(self) -> int:
        with self.cond:
            return sum(1 for j in self.queue if j in self.jobs and self.jobs[j].state == "queued")

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.spec.get("kind") == "pipeline":
            if job.state not in FINAL_STATES:
                job.cancel_event.set()          # the runner cancels its current step and ends the job
                job.update(message="cancelling")
            return job
        with self.cond:
            if job.state == "queued":
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
                job.update(persist=True, state="cancelled", message="cancelled before start",
                           etaSeconds=None)
                job.log("cancelled while queued")
                return job
            if job.state == "paused":
                job.update(persist=True, state="cancelled", message="cancelled while paused", etaSeconds=None)
                (job.dir / "resume.pt").unlink(missing_ok=True)
                job.log("cancelled while paused")
                return job
        if job.state in ACTIVE_STATES:
            job.cancel_event.set()
            job.update(message="cancelling")
        return job

    # --------------------------------------------------------- pause / resume
    def pausable(self, job: Job) -> bool:
        return job.spec.get("kind") == "train" and bool(getattr(self.backend, "pausable", False))

    def pause(self, job_id: str) -> Job:
        """Queued job (any kind): held back. Running train job: the trainer saves its state and stops, freeing the
        GPU for the next queued job (the state becomes "paused" once it has). Paused: nothing to do.
        A pipeline pauses its current step (its training, or a step still waiting in the queue)."""
        job = self.get(job_id)
        if job.spec.get("kind") == "pipeline":
            if job.state == "paused":
                return job
            child = self._current_child(job)
            if job.state in FINAL_STATES or child is None:
                raise ApiError(409, "a pipeline can be paused while its training runs or a step is queued")
            self.pause(child.id)
            return job
        with self.cond:
            if job.state == "paused":
                return job
            if job.state == "queued":
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
                job.update(persist=True, state="paused", message="paused before it started", etaSeconds=None,
                           pausedAt=time.time())
                job.log("paused while queued")
                return job
        if job.state in ACTIVE_STATES:
            if not self.pausable(job):
                raise ApiError(409, "only training jobs can be paused while they run")
            if job.cancel_event.is_set():
                raise ApiError(409, "the job is being cancelled")
            job.pause_event.set()
            job.update(message="pausing")
            job.log("pause requested")
            return job
        raise ApiError(409, f"a {job.state} job can't be paused")

    def resume(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.spec.get("kind") == "pipeline":
            child = self._current_child(job)
            if job.state != "paused" or child is None:
                if job.state in ("running",) and child is not None and child.state != "paused":
                    return job
                raise ApiError(409, f"a {job.state} pipeline can't be resumed")
            self.resume(child.id)
            return job
        with self.cond:
            if job.state in ("queued",) + ACTIVE_STATES:
                if job.pause_event.is_set():
                    raise ApiError(409, "the job is still pausing; resume it once it has paused")
                return job
            if job.state != "paused":
                raise ApiError(409, f"a {job.state} job can't be resumed")
            self._requeue_paused_locked(job, "resumed: back in the queue")
        return job

    def request_checkpoint(self, job_id: str) -> Job:
        """Ask a running train job to write checkpoint.ply now. A paused job already has one (written at pause)."""
        job = self.get(job_id)
        if job.spec.get("kind") == "pipeline":
            child = self._current_child(job)
            if child is None or child.spec.get("kind") != "train":
                raise ApiError(409, "the pipeline isn't training yet")
            self.request_checkpoint(child.id)
            return child
        if job.state == "paused":
            if not (job.dir / "checkpoint.ply").is_file():
                raise ApiError(409, "the job was paused before training started; there is nothing to save yet")
            return job
        if job.state not in ACTIVE_STATES or not self.pausable(job):
            raise ApiError(409, "checkpoints can only be taken of a running or paused training job")
        job.checkpoint_event.set()
        job.log("checkpoint requested")
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.spec.get("kind") == "pipeline":
            for child in self.children(job):
                try:
                    self.delete(child.id)
                except ApiError:
                    pass
            with self.cond:
                self.jobs.pop(job.id, None)
                job.deleted = True
            if job.state in FINAL_STATES:
                shutil.rmtree(job.dir, ignore_errors=True)
                self.drop_dataset(job.spec.get("workDatasetId", ""))
            else:
                job.cancel_event.set()        # the runner stops; finish_pipeline removes its folder and dataset
            return
        with self.cond:
            if job.id in self.queue:
                self.queue.remove(job.id)
            self.jobs.pop(job.id, None)
            active = job.state in ACTIVE_STATES or job.id in self.running.values()
            job.deleted = True
        if active:
            job.cancel_event.set()  # the worker thread removes the directory when it stops
        else:
            shutil.rmtree(job.dir, ignore_errors=True)

    def prioritize(self, job_id: str) -> Job:
        """Move a queued job to the front: priority above every other queued job."""
        job = self.get(job_id)
        with self.cond:
            if job.state != "queued":
                raise ApiError(409, "only queued jobs can be moved")
            top = max([int(self.jobs[j].status.get("priority") or 0) for j in self.queue if j in self.jobs] + [0])
            job.update(persist=True, priority=top + 1)
            job.log(f"moved to the front of the queue (priority {top + 1})")
            self.cond.notify()
        return job

    def dataset_in_use(self, dataset_id: str) -> bool:
        return any((j.spec.get("datasetId") == dataset_id or j.spec.get("workDatasetId") == dataset_id)
                   and j.state not in FINAL_STATES for j in self.jobs.values())

    # --------------------------------------------------------------- worker
    def _next_job(self, device: Optional[int]) -> Optional[Job]:
        with self.cond:
            while True:
                if self.stopping:
                    return None
                self._housekeep_locked()
                # drop stale entries, then take the highest priority (ties: oldest first). Nothing running is
                # ever pre-empted; priority only decides which queued job a free GPU takes next.
                live = [j for j in self.queue if j in self.jobs and self.jobs[j].state == "queued"]
                self.queue = collections.deque(live)
                if live:
                    best = max(live, key=lambda j: (int(self.jobs[j].status.get("priority") or 0), -live.index(j)))
                    self.queue.remove(best)
                    self.running[device] = best
                    job = self.jobs[best]
                    # taken: no longer "queued" from here on, so a pause / resume / cancel arriving before the job
                    # really starts goes through the running path (events the job checks) instead of re-queueing it
                    job.update(persist=True, state="loading", message="starting")
                    return job
                self.cond.wait(timeout=1.0)

    def _housekeep_locked(self) -> None:
        """(Holding ``cond``, at most every 30 s.) Paused jobs mustn't wait forever: another computer's (its app
        may have no Resume button and would poll for ever) resume after REMOTE_PAUSE_LIMIT; this PC's are
        cancelled after LOCAL_PAUSE_LIMIT so their multi-GB resume.pt doesn't stay behind."""
        now = time.time()
        if now - getattr(self, "_housekept", 0.0) < 30.0:
            return
        self._housekept = now
        for job in list(self.jobs.values()):
            if job.state != "paused" or job.spec.get("kind") == "pipeline":
                continue
            age = now - float(job.status.get("pausedAt") or now)
            if job.status.get("origin") != "local" and age >= REMOTE_PAUSE_LIMIT:
                self._requeue_paused_locked(job, "resumed automatically (paused by another computer for too long)")
            elif job.status.get("origin") == "local" and age >= LOCAL_PAUSE_LIMIT:
                job.update(persist=True, state="cancelled", message="cancelled: paused for more than 14 days",
                           etaSeconds=None)
                (job.dir / "resume.pt").unlink(missing_ok=True)
                job.log("cancelled after 14 days paused; its saved state was removed")

    def _requeue_paused_locked(self, job: Job, why: str) -> None:
        job.pause_event.clear()
        job.checkpoint_event.clear()
        step = job.status.get("step") or 0
        resumable = (job.dir / "resume.pt").is_file()
        job.update(persist=True, state="queued", pausedAt=None,
                   message=f"queued (continues from step {step:,})" if resumable else "queued")
        job.log(why + (f", continues from step {step}" if resumable else ""))
        self.queue.append(job.id)
        self.cond.notify()

    def _worker_loop(self, device: Optional[int]) -> None:
        while True:
            job = self._next_job(device)
            if job is None:
                return
            try:
                self._run_job(job, device)
            except Exception:  # pragma: no cover - defensive
                log.exception("unexpected error running job %s", job.id)
            finally:
                with self.cond:
                    self.running.pop(device, None)
                if job.deleted:
                    shutil.rmtree(job.dir, ignore_errors=True)

    def _run_job(self, job: Job, device: Optional[int]) -> None:
        from .gpu import list_gpus

        gpu_name = None
        if device is not None:
            gpus = list_gpus([device])
            gpu_name = gpus[0]["name"] if gpus else f"GPU {device}"
        resuming = job.spec.get("kind") == "train" and (job.dir / "resume.pt").is_file()
        prior = float(job.status.get("elapsedSeconds") or 0.0) if resuming else 0.0
        job.started_monotonic = time.monotonic() - prior     # elapsed time carries on across a pause
        job.update(persist=True, state="loading", message="resuming" if resuming else "starting", gpu=gpu_name)
        job.log(f"starting on {'GPU ' + str(device) if device is not None else 'default device'}"
                f"{' (' + gpu_name + ')' if gpu_name else ''}")

        dataset_dir = dataset_root(self.settings, job.spec["datasetId"])
        if dataset_dir is None:
            self._finish(job, "failed", "dataset no longer exists")
            return
        if job.cancel_event.is_set():           # cancelled between being taken and starting
            self._finish(job, "cancelled", "cancelled before start")
            return
        if job.pause_event.is_set():            # paused between being taken and starting: nothing ran yet
            job.pause_event.clear()
            job.update(persist=True, state="paused", etaSeconds=None, pausedAt=time.time(),
                       message="paused before it started" if not resuming else "paused again before it resumed")
            job.log("paused before it started")
            return

        def status(**fields):
            fields.pop("elapsedSeconds", None)  # server measures wall time itself
            if fields.get("state") not in (None, "loading", "training", "running"):
                fields.pop("state")
            if job.cancel_event.is_set() or job.pause_event.is_set():
                fields.pop("message", None)
            job.update(**fields)

        def preview(step: int):
            job.update(persist=True, previewStep=step)

        def checkpoint(step: int):
            job.checkpoints += 1
            job.update(persist=True, checkpointStep=step)

        kind = job.spec["kind"]
        ctx = JobContext(
            job_id=job.id, kind=kind, job_dir=job.dir, dataset_dir=dataset_dir,
            spec=job.spec.get({"train": "config", "census": "census"}.get(kind, "request")) or {},
            device_index=device, status=status, log=job.log, preview=preview,
            cancel_event=job.cancel_event, pause_event=job.pause_event, checkpoint_event=job.checkpoint_event,
            checkpoint=checkpoint,
        )
        try:
            stats = self.backend.run(ctx) or {}
        except JobPaused as e:
            job.pause_event.clear()
            if job.cancel_event.is_set():        # cancelled while pausing: cancel wins
                self._finish(job, "cancelled", "cancelled")
                return
            elapsed = round(time.monotonic() - job.started_monotonic, 1)
            job.update(persist=True, state="paused", step=e.step, message=f"paused at step {e.step:,}",
                       etaSeconds=None, elapsedSeconds=elapsed, pausedAt=time.time())
            job.log(f"paused at step {e.step}; the GPU is free (resume continues from here)")
            return
        except JobCancelled:
            if self.stopping and not job.deleted and (job.dir / "resume.pt").is_file():
                # stopped before it could save again: its last pause is still on disk, keep it resumable
                job.update(persist=True, state="paused", pausedAt=time.time(), etaSeconds=None,
                           message="interrupted by a worker stop; resume continues from the last pause")
                job.log("worker stopped; the job can be resumed from its last pause")
                return
            if self.stopping and not job.deleted:
                self._finish(job, "failed", "worker shut down while the job was running")
            else:
                self._finish(job, "cancelled", "cancelled")
            return
        except Exception as e:
            msg = str(e) or e.__class__.__name__
            job.log(f"FAILED: {msg}")
            self._finish(job, "failed", msg)
            return

        if kind == "train":
            if not (job.dir / "result.ply").is_file():
                self._finish(job, "failed", "trainer finished without writing result.ply")
                return
            extra = {k: stats[k] for k in ("numGaussians", "psnr", "loss") if stats.get(k) is not None}
            job.update(step=job.status.get("totalSteps"), **extra)
        elif kind == "census":
            if not (job.dir / "census.npz").is_file():
                self._finish(job, "failed", "census finished without census.npz")
                return
            job.update(step=job.status.get("totalSteps"))
        else:
            out = job.dir / "output"
            if not out.is_dir():
                self._finish(job, "failed", "processing finished without output")
                return
            try:
                out_id = _adopt_output_as_dataset(self.settings, out, job)
            except Exception as e:  # pragma: no cover - disk trouble
                self._finish(job, "failed", f"couldn't keep the output: {e}")
                return
            job.status["outputDatasetId"] = out_id
            job.log(f"output kept as dataset {out_id}")
            job.update(step=job.status.get("totalSteps"))
        self._finish(job, "completed", "completed")

    def _finish(self, job: Job, state: str, message: str) -> None:
        elapsed = None
        if job.started_monotonic is not None:
            elapsed = round(time.monotonic() - job.started_monotonic, 1)
        job.update(persist=True, state=state, message=message, etaSeconds=0.0 if state == "completed" else None,
                   elapsedSeconds=elapsed)
        job.pause_event.clear()
        (job.dir / "resume.pt").unlink(missing_ok=True)      # only a paused job needs its saved state
        job.log(f"job {state}: {message}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_RAW_NAME = re.compile(r"[A-Za-z0-9._ -]{1,200}")


def dataset_root(settings: ServerSettings, dataset_id: str, local: bool = True) -> Optional[Path]:
    """The dataset's folder. ``local=False`` (a request from another computer) hides datasets that are folders
    of the desktop app on this PC (``POST /v1/datasets/local``): remote clients can neither see nor use them."""
    if not ID_RE.match(dataset_id or ""):
        return None
    d = settings.datasets_dir / dataset_id
    meta = d / "meta.json"
    if not meta.is_file():
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        rel = m.get("root", "files")
    except Exception:
        return None
    if m.get("localOnly") and not local:
        return None                       # the desktop app's processing output (see _adopt_output_as_dataset)
    if m.get("external"):
        # a folder of the desktop app on this PC (POST /v1/datasets/local), used in place
        if not settings.local_enabled or not local:
            return None
        root = Path(m["external"]).resolve()
        return root if root.is_dir() else None
    root = (d / rel).resolve()
    if d.resolve() not in root.parents:
        return None
    return root


def _new_dataset_id() -> str:
    return "d" + uuid.uuid4().hex[:11]


def _describe_dataset(root: Path, info: Dict[str, Any]) -> None:
    info["hasDatasetJson"] = (root / "dataset.json").is_file()
    if info["hasDatasetJson"]:
        ds = parse_dataset(root, check_files=True)
        info["cameras"] = len(ds.cameras)
        info["hasDepth"] = any(c.depth for c in ds.cameras)
        info["name"] = ds.raw.get("name")


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _adopt_output_as_dataset(settings: ServerSettings, out_dir: Path, job: "Job") -> str:
    """A processing job's output (already hard links under the job dir) becomes a dataset: moved, not copied,
    so the next step (masks → poses → training) can use it without a round trip through the client."""
    dataset_id = _new_dataset_id()
    partial = settings.datasets_dir / f".{dataset_id}.partial"
    partial.mkdir(parents=True)
    try:
        os.replace(out_dir, partial / "files")
        info: Dict[str, Any] = {"datasetId": dataset_id, "createdAt": _now_iso(),
                                "bytes": _dir_bytes(partial / "files"), "fromJob": job.id, "root": "files"}
        if job.status.get("origin") == "local":
            info["localOnly"] = True      # the desktop app's processing output: private to this PC, like its folders
        try:
            _describe_dataset(partial / "files", info)
        except DatasetError:
            info["hasDatasetJson"] = False
        _write_json_atomic(partial / "meta.json", info)
        os.replace(partial, settings.datasets_dir / dataset_id)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return dataset_id


def _compose_dataset(settings: ServerSettings, layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """New dataset = union of existing datasets' files (hard links; later layers win; optional ``prefix``
    puts a layer under a sub-folder)."""
    from .processing import link_tree

    dataset_id = _new_dataset_id()
    partial = settings.datasets_dir / f".{dataset_id}.partial"
    files = partial / "files"
    files.mkdir(parents=True)
    try:
        for layer in layers:
            root = dataset_root(settings, layer["datasetId"])
            dst = files / layer["prefix"] if layer["prefix"] else files
            link_tree(root, dst, [root])
        info: Dict[str, Any] = {"datasetId": dataset_id, "createdAt": _now_iso(), "bytes": _dir_bytes(files),
                                "composedOf": [l["datasetId"] for l in layers], "root": "files"}
        _describe_dataset(files, info)
        _write_json_atomic(partial / "meta.json", info)
        os.replace(partial, settings.datasets_dir / dataset_id)
        return info
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _ingest_dataset(settings: ServerSettings, zip_path: Path, dataset_id: str) -> Dict[str, Any]:
    final_dir = settings.datasets_dir / dataset_id
    partial = settings.datasets_dir / f".{dataset_id}.partial"
    files = partial / "files"
    try:
        size = safe_extract(zip_path, files, max_bytes=settings.max_extracted_bytes,
                            max_files=settings.max_files, max_ratio=settings.max_compression_ratio)
        root = find_upload_root(files)
        info: Dict[str, Any] = {"datasetId": dataset_id, "createdAt": _now_iso(),
                                "bytes": size, "hasDatasetJson": (root / "dataset.json").is_file()}
        if info["hasDatasetJson"]:
            ds = parse_dataset(root, check_files=False)        # malformed → rejected
            missing = [c.image for c in ds.cameras if not ds.resolve(c.image).is_file()]
            if missing:
                # images the worker already has elsewhere (a layer for /v1/datasets/compose): can't train on its own
                info["hasDatasetJson"] = False
                info["partial"] = True
            else:
                info["cameras"] = len(ds.cameras)
                info["hasDepth"] = any(c.depth for c in ds.cameras)
                info["name"] = ds.raw.get("name")
        info["root"] = str(root.relative_to(partial))
        _write_json_atomic(partial / "meta.json", info)
        os.replace(partial, final_dir)
        return info
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: ServerSettings, backend: Optional[Backend] = None,
               devices: Optional[List[Optional[int]]] = None) -> FastAPI:
    from .gpu import adapters as gpu_adapters
    from .gpu import backend_description, list_gpus

    backend = backend or SubprocessBackend()
    for d in (settings.datasets_dir, settings.jobs_dir, settings.tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(settings.data_dir, 0o700)
    except OSError:
        pass
    for leftover in settings.tmp_dir.iterdir():
        shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
    for leftover in settings.datasets_dir.glob(".*.partial"):
        shutil.rmtree(leftover, ignore_errors=True)

    if devices is None:
        if settings.gpus is not None:
            devices = list(settings.gpus)
        else:
            devices = [g["index"] for g in list_gpus()] or [None]
    manager = JobManager(settings, backend, devices)
    manager.load()

    advertiser = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal advertiser
        manager.start()
        if settings.mdns:
            from .discovery import Advertiser

            gpus = list_gpus(settings.gpus)
            gpu_name = gpus[0]["name"] if gpus else "none"
            if len(gpus) > 1:
                gpu_name = f"{len(gpus)}x {gpu_name}"
            try:
                advertiser = Advertiser(settings.name, settings.port, gpu_name,
                                        settings.token is not None, settings.advertise_ips)
                await run_in_threadpool(advertiser.start)
            except Exception as e:  # pragma: no cover
                log.warning("mDNS advertisement failed: %s", e)
                advertiser = None
        try:
            yield
        finally:
            if advertiser is not None:
                await run_in_threadpool(advertiser.stop)
            await run_in_threadpool(manager.shutdown)

    async def require_auth(request: Request) -> None:
        if settings.token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
                value.strip().encode(), settings.token.encode()):
            raise ApiError(401, "missing or invalid bearer token")

    app = FastAPI(title=PRODUCT_NAME, version=__version__, lifespan=lifespan,
                  dependencies=[Depends(require_auth)], docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.manager = manager
    app.state.settings = settings

    # ------------------------------------------------------------- errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return JSONResponse({"error": exc.message}, status_code=exc.status, headers=headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        errs = exc.errors()
        msg = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg')}" for e in errs)
        return JSONResponse({"error": msg or "invalid request"}, status_code=400)

    async def json_body(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise ApiError(400, "body must be valid JSON")
        if not isinstance(body, dict):
            raise ApiError(400, "body must be a JSON object")
        return body

    # --------------------------------------------------------------- info
    @app.get("/v1/info")
    def info():
        gpus = [{"name": g["name"], "memoryMB": g["memoryMB"], "freeMemoryMB": g["freeMemoryMB"],
                 "vendor": g.get("vendor", "nvidia")}
                for g in list_gpus(settings.gpus)]
        try:
            caps = backend.capabilities()
        except Exception:
            caps = []
        if "process" in caps or "frames" in caps:
            caps = caps + ["chain"]       # outputDatasetId, /v1/datasets/compose, /v1/datasets/{id}/archive.zip
            caps = caps + ["stream"]      # ≥ 1.3.13: training images streamed from disk when RAM is short
        if "process" in caps:
            caps = caps + ["maskpreview"]  # ≥ 1.4.0: processing task "maskpreview" (SAM 2 clicks, mask editor)
        if "frames" in caps:
            caps = caps + ["stereo"]      # ≥ 1.4.0: frames request "stereo" (two eyes → images/*_L, *_R)
            if homevideo_available():
                caps = caps + ["homevideo"]   # ≥ 1.6.0: frames request "vhs" (old tape restoration, clips), sfm "cameraGroups"
        caps = caps + ["priority"]        # ≥ 1.4.0: POST /v1/jobs/{id}/prioritize, JobStatus.priority / origin
        if "train" in caps:
            caps = caps + ["census"]      # ≥ 1.4.0: POST /v1/census (Auto clean)
        if "train" in caps and getattr(backend, "pausable", False):
            caps = caps + ["pause"]       # ≥ 1.4.0: POST /v1/jobs/{id}/pause|resume|checkpoint, checkpoint.ply
        if "train" in caps and "process" in caps:
            caps = caps + ["pipeline"]    # ≥ 1.6.0: POST /v1/pipeline (whole pipeline, client may go offline)
        if settings.local_enabled:
            caps = caps + ["local"]       # ≥ 1.4.0: POST /v1/datasets/local (the desktop app on this PC only)
        try:
            models = backend.models() if "process" in caps else []
        except Exception:
            models = []
        return {
            "name": settings.name,
            "version": PROTOCOL_VERSION,
            "workerVersion": __version__,
            "gpus": gpus,
            # ≥ 1.6.0: every graphics card of any vendor (sysfs), also ones jobs can't use (AMD / Intel without a
            # GPU runtime, integrated GPUs): {vendor, name, pci, driver, integrated, memoryMB?, freeMemoryMB?}
            "adapters": gpu_adapters(),
            "backend": backend_description(),
            "busy": manager.busy(),
            "queueLength": manager.queue_length(),
            "capabilities": caps,
            "models": models,
            "auth": settings.token is not None,
            # LAN addresses, preferred first (ethernet, then Wi-Fi) — the Mac switches to the first that answers
            "addresses": [{"address": a, "kind": k} for a, k in _lan_interfaces() if k != "virtual"],
            # host RAM (worker ≥ 1.3.14): the Mac checks a job fits before sending it
            "memory": _host_memory(),
        }

    # ------------------------------------------------------------ runtime
    from .runtime_install import RuntimeInstaller
    installer = RuntimeInstaller()

    def _origin(request: Request) -> str:
        """"local" = the desktop app that started this server: a loopback connection that also carries the
        per-launch secret only that process knows. Loopback alone proves nothing (SSH tunnels, ``tailscale serve``
        and reverse proxies all arrive from 127.0.0.1), so without the secret a request counts as remote."""
        host = request.client.host if request.client else ""
        if host in LOOPBACK and settings.local_enabled:
            given = request.headers.get(LOCAL_HEADER, "")
            if given and hmac.compare_digest(given.encode(), settings.local_secret.encode()):
                return "local"
        return host or "unknown"

    def _is_local(request: Request) -> bool:
        return _origin(request) == "local"

    def _root(request: Request, dataset_id: str) -> Optional[Path]:
        return dataset_root(settings, dataset_id, local=_is_local(request))

    def _priority(request: Request, body: Dict[str, Any]) -> int:
        """Only clients on this PC (the Studio) may ask to go first; remote jobs always queue at 0."""
        if _origin(request) != "local":
            return 0
        p = body.get("priority", 0)
        if not isinstance(p, int) or isinstance(p, bool):
            raise ApiError(400, "priority must be an integer")
        return max(0, min(p, 1000))

    @app.get("/v1/runtime")
    def runtime_status():
        """Optional components and the progress of a remote install."""
        return installer.status()

    @app.post("/v1/runtime/install")
    async def runtime_install(request: Request):
        body = await json_body(request)
        comp = (body or {}).get("component", "processing") if isinstance(body, dict) else "processing"
        try:
            return installer.start(str(comp))
        except ValueError as e:
            raise ApiError(400, str(e))

    # ----------------------------------------------------------- datasets
    @app.post("/v1/datasets")
    async def upload_dataset(request: Request):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > settings.max_upload_bytes:
                    raise ApiError(413, f"upload exceeds {settings.max_upload_bytes} bytes")
            except ValueError:
                raise ApiError(400, "invalid Content-Length")
        ctype = request.headers.get("content-type", "application/zip").split(";")[0].strip().lower()
        if ctype not in ("application/zip", "application/x-zip-compressed", "application/octet-stream",
                         "multipart/x-zip", ""):
            raise ApiError(415, "body must be a zip archive (Content-Type: application/zip)")

        dataset_id = "d" + uuid.uuid4().hex[:11]
        tmp = settings.tmp_dir / f"{dataset_id}.zip"
        received = 0
        try:
            with open(tmp, "wb") as fh:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > settings.max_upload_bytes:
                        raise ApiError(413, f"upload exceeds {settings.max_upload_bytes} bytes")
                    fh.write(chunk)
            if received == 0:
                raise ApiError(400, "empty body")
            try:
                info = await run_in_threadpool(_ingest_dataset, settings, tmp, dataset_id)
            except UnsafeArchive as e:
                raise ApiError(400, f"rejected archive: {e}")
            except DatasetError as e:
                raise ApiError(400, f"invalid dataset: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        log.info("dataset %s uploaded (%d bytes zip, %s)", dataset_id, received, info)
        return {"datasetId": dataset_id}

    @app.post("/v1/datasets/file")
    async def upload_file(request: Request, name: str):
        """One raw file (e.g. a multi-GB video) as a dataset, streamed to disk: no zip, no client-side copy.
        The file lands at the dataset root as ``name`` (a plain file name)."""
        if not _RAW_NAME.fullmatch(name or "") or name.startswith("."):
            raise ApiError(400, "name must be a plain file name")
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > settings.max_upload_bytes:
                    raise ApiError(413, f"upload exceeds {settings.max_upload_bytes} bytes")
            except ValueError:
                raise ApiError(400, "invalid Content-Length")
        dataset_id = "d" + uuid.uuid4().hex[:11]
        partial = settings.datasets_dir / f".{dataset_id}.partial"
        files = partial / "files"
        files.mkdir(parents=True)
        received = 0
        try:
            with open(files / name, "wb") as fh:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > settings.max_upload_bytes:
                        raise ApiError(413, f"upload exceeds {settings.max_upload_bytes} bytes")
                    fh.write(chunk)
            if received == 0:
                raise ApiError(400, "empty body")
            info = {"datasetId": dataset_id, "createdAt": _now_iso(), "bytes": received,
                    "hasDatasetJson": False, "file": name, "root": "files"}
            _write_json_atomic(partial / "meta.json", info)
            os.replace(partial, settings.datasets_dir / dataset_id)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        log.info("file dataset %s uploaded (%s, %d bytes)", dataset_id, name, received)
        return {"datasetId": dataset_id}

    @app.post("/v1/datasets/compose")
    async def compose_dataset(request: Request):
        """``{"layers": [{"datasetId": A}, {"datasetId": B, "prefix": "sub"}]}`` → a new dataset holding the union
        of the layers' files (hard links, later layers win). Lets a client send only what changed (dataset.json,
        masks) and reuse the images already here."""
        body = await json_body(request)
        raw = body.get("layers")
        if not isinstance(raw, list) or not raw or len(raw) > 16:
            raise ApiError(400, "layers must be a list of 1–16 {datasetId, prefix?}")
        layers = []
        for l in raw:
            if not isinstance(l, dict):
                raise ApiError(400, "each layer must be an object")
            did = l.get("datasetId")
            if not isinstance(did, str) or _root(request, did) is None:
                raise ApiError(404, f"dataset {did} not found")
            prefix = l.get("prefix") or ""
            if prefix:
                from .processing import ProcessingError, _safe_rel
                try:
                    prefix = _safe_rel(prefix, "prefix")
                except ProcessingError as e:
                    raise ApiError(400, str(e))
            layers.append({"datasetId": did, "prefix": prefix})
        try:
            info = await run_in_threadpool(_compose_dataset, settings, layers)
        except DatasetError as e:
            raise ApiError(400, f"invalid dataset: {e}")
        log.info("dataset %s composed from %s", info["datasetId"], [l["datasetId"] for l in layers])
        return {"datasetId": info["datasetId"], "hasDatasetJson": info["hasDatasetJson"], "bytes": info["bytes"]}

    @app.post("/v1/datasets/local")
    async def register_local_dataset(request: Request):
        """``{"path": "/abs/folder"}`` → a dataset that IS that folder (nothing copied). Only the desktop app that
        started this server (``--allow-local-datasets`` + the per-launch secret, see ``_origin``)."""
        if not settings.local_enabled:
            raise ApiError(403, "local datasets are disabled on this worker")
        if not _is_local(request):
            raise ApiError(403, "local datasets can only be registered by the desktop app on this computer")
        body = await json_body(request)
        path = body.get("path")
        if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
            raise ApiError(400, "path must be an absolute folder path")
        root = Path(path).resolve()
        if not root.is_dir():
            raise ApiError(404, "folder not found")
        if not _registrable(root):
            raise ApiError(403, "that folder is too broad to use as a dataset")
        dataset_id = _new_dataset_id()
        d = settings.datasets_dir / dataset_id
        d.mkdir(parents=True)
        info: Dict[str, Any] = {"datasetId": dataset_id, "createdAt": _now_iso(), "external": str(root),
                                "local": True}
        try:
            _describe_dataset(root, info)
        except DatasetError as e:
            info["hasDatasetJson"] = False
            info["datasetError"] = str(e)
        _write_json_atomic(d / "meta.json", info)
        return {k: v for k, v in info.items() if k != "external"} | {"path": str(root)}

    @app.post("/v1/datasets/{dataset_id}/export-local")
    async def export_local_dataset(dataset_id: str, request: Request):
        """``{"path": "/abs/folder"}``: recreate the dataset's files there with hard links (copies across file
        systems), merging into what is already there. Loopback clients with ``--allow-local-datasets`` only."""
        from .processing import link_tree

        if not settings.local_enabled or not _is_local(request):
            raise ApiError(403, "only the desktop app on this computer can export datasets to a folder")
        _check_id(dataset_id, "dataset")
        root = _root(request, dataset_id)
        if root is None:
            raise ApiError(404, "dataset not found")
        body = await json_body(request)
        dest = body.get("path")
        if not isinstance(dest, str) or not os.path.isabs(dest):
            raise ApiError(400, "path must be an absolute folder path")
        dest_p = Path(dest).resolve()
        if not any(dest_p == r or r in dest_p.parents for r in _local_roots()):
            # only into folders the user chose in the app (registered with POST /v1/datasets/local)
            raise ApiError(403, "export-local can only write into a folder registered with /v1/datasets/local")
        dest_p.mkdir(parents=True, exist_ok=True)
        n = await run_in_threadpool(link_tree, root, dest_p, [root])
        return {"files": n, "path": str(dest_p)}

    def _is_external(d: Path) -> bool:
        try:
            return bool(json.loads((d / "meta.json").read_text(encoding="utf-8")).get("external"))
        except Exception:
            return False

    def _local_roots() -> List[Path]:
        out = []
        for meta in settings.datasets_dir.glob("*/meta.json"):
            try:
                ext = json.loads(meta.read_text(encoding="utf-8")).get("external")
            except Exception:
                continue
            if ext:
                out.append(Path(ext).resolve())
        return out

    def _registrable(root: Path) -> bool:
        """Not ``/``, the home folder, or a folder holding this worker's own data."""
        home = Path.home().resolve()
        data = settings.data_dir.resolve()
        return len(root.parts) > 2 and root != home and root not in home.parents \
            and root != data and root not in data.parents and data not in root.parents

    @app.get("/v1/datasets")
    def list_datasets(request: Request):
        local = _is_local(request)
        out = []
        for d in sorted(settings.datasets_dir.iterdir()):
            meta = d / "meta.json"
            if d.name.startswith(".") or not meta.is_file():
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            m.pop("root", None)
            if m.get("localOnly") and not local:
                continue               # the desktop app's processing outputs: not shown to other computers
            if m.get("external"):
                if not local:
                    continue           # the desktop app's own folders: not shown to other computers
                m.pop("external", None)
            out.append(m)
        return out

    @app.delete("/v1/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str, request: Request):
        _check_id(dataset_id, "dataset")
        d = settings.datasets_dir / dataset_id
        if not (d / "meta.json").is_file() or (_is_external(d) and not _is_local(request)):
            raise ApiError(404, "dataset not found")
        if manager.dataset_in_use(dataset_id):
            raise ApiError(409, "dataset is used by a queued or running job")
        shutil.rmtree(d, ignore_errors=True)
        return Response(status_code=204)

    # --------------------------------------------------------------- jobs
    def _require_dataset(body: Dict[str, Any], request: Request) -> tuple:
        dataset_id = body.get("datasetId")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ApiError(400, "datasetId is required")
        root = _root(request, dataset_id)
        if root is None:
            raise ApiError(404, f"dataset {dataset_id} not found")
        return dataset_id, root

    def _name(body: Dict[str, Any], default: str) -> str:
        name = body.get("name")
        if name is None:
            return default
        if not isinstance(name, str):
            raise ApiError(400, "name must be a string")
        return name[:200]

    @app.post("/v1/jobs")
    async def create_job(request: Request):
        body = await json_body(request)
        dataset_id, root = _require_dataset(body, request)
        if not (root / "dataset.json").is_file():
            raise ApiError(400, "dataset has no dataset.json; it can only be used for /v1/process")
        try:
            parse_dataset(root, check_files=True)
        except DatasetError as e:
            raise ApiError(400, f"invalid dataset: {e}")
        try:
            cfg = TrainingConfig.from_dict(body.get("config"))
        except ConfigError as e:
            raise ApiError(400, f"invalid config: {e}")
        # Keep only keys the client sent, so context-dependent defaults resolve in the trainer.
        cfg_dict = {k: v for k, v in cfg.to_dict().items() if k in cfg.explicit}
        job = manager.submit("train", dataset_id, _name(body, dataset_id), "config", cfg_dict,
                             total_steps=cfg.iterations, priority=_priority(request, body),
                             origin=_origin(request))
        return {"jobId": job.id}

    @app.post("/v1/process")
    async def create_process_job(request: Request):
        body = await json_body(request)
        dataset_id, _root = _require_dataset(body, request)
        try:
            req = validate_request(body.get("request"))
        except ProcessingError as e:
            raise ApiError(400, f"invalid request: {e}")
        from .processing import PROGRESS_SCALE

        job = manager.submit("process", dataset_id, _name(body, f"{req['task']} {dataset_id}"),
                             "request", req, total_steps=PROGRESS_SCALE, priority=_priority(request, body),
                             origin=_origin(request))
        return {"jobId": job.id}

    @app.post("/v1/pipeline")
    async def create_pipeline_job(request: Request):
        """Whole pipeline on this worker (capability "pipeline"): frames → masks → sky → camera solve → dataset →
        training, each step a child job, so the client may go offline until it is done (PROTOCOL.md §4)."""
        from .pipeline import PipelineError, validate_spec

        body = await json_body(request)
        dataset_id, root = _require_dataset(body, request)
        if "process" not in backend.capabilities():
            raise ApiError(400, "this worker can't process (install the processing models)")
        try:
            spec = validate_spec(body, root)
        except PipelineError as e:
            raise ApiError(400, f"invalid pipeline: {e}")
        if spec.get("frames") and "frames" not in backend.capabilities():
            raise ApiError(400, "this worker can't split videos (install the processing models)")
        job = manager.submit_pipeline(dataset_id, _name(body, dataset_id), spec, priority=_priority(request, body),
                                      origin=_origin(request))
        return {"jobId": job.id}

    @app.post("/v1/census")
    async def create_census_job(request: Request):
        """Auto clean census: ``{"datasetId", "jobId": <completed train job>}`` or, from this PC only,
        ``{"datasetId", "plyPath": "/abs/scene.ply"}``. Result: ``GET /v1/jobs/{id}/census.npz``."""
        body = await json_body(request)
        dataset_id, root = _require_dataset(body, request)
        if not (root / "dataset.json").is_file():
            raise ApiError(400, "the census needs a dataset with dataset.json (the photos' cameras)")
        if body.get("jobId"):
            src = _job(request, str(body["jobId"]))     # the desktop app's jobs don't exist for other computers
            ply = src.dir / "result.ply"
            if not ply.is_file():
                raise ApiError(409, "that job has no result.ply")
        elif body.get("plyPath"):
            if not _is_local(request):
                raise ApiError(403, "plyPath can only be used by the desktop app on this computer")
            ply = Path(str(body["plyPath"]))
            if not ply.is_absolute() or not ply.is_file():
                raise ApiError(404, "plyPath not found")
        else:
            raise ApiError(400, "jobId or plyPath is required")
        try:
            n_cams = len(parse_dataset(root, check_files=False).cameras)
        except DatasetError as e:
            raise ApiError(400, f"invalid dataset: {e}")
        spec = {"plyPath": str(ply.resolve())}
        if isinstance(body.get("maxImageSize"), int):
            spec["maxImageSize"] = body["maxImageSize"]
        job = manager.submit("census", dataset_id, _name(body, f"census {dataset_id}"), "census", spec,
                             total_steps=max(n_cams, 1), priority=_priority(request, body), origin=_origin(request))
        return {"jobId": job.id}

    @app.post("/v1/jobs/{job_id}/prioritize")
    def prioritize_job(job_id: str, request: Request):
        """Moving a job ahead of everybody else's is for the person at this PC (the desktop app) only."""
        if not _is_local(request):
            raise ApiError(403, "only the desktop app on this computer can reorder the queue")
        return manager.prioritize(job_id).snapshot()

    def _may_control(request: Request, job: Job, resume: bool = False) -> None:
        """Pause / checkpoint: only whoever submitted the job (same origin; "local" = the desktop app here).
        The app doesn't park other computers' jobs: their apps may have no Resume button and would wait for
        ever (they resume by themselves after REMOTE_PAUSE_LIMIT anyway). Resume: also the desktop app, for any
        paused job (so the person at this PC can always get a parked job going again)."""
        origin = _origin(request)
        if origin not in ("", "unknown") and origin == job.status.get("origin"):
            return
        if resume and origin == "local":
            return
        raise ApiError(403, "only the client that submitted the job can pause or checkpoint it")

    def _job(request: Request, job_id: str) -> Job:
        """A job for this request: the desktop app's own jobs don't exist for other computers' file requests."""
        job = manager.get(job_id)
        if job.status.get("origin") == "local" and not _is_local(request):
            raise ApiError(404, f"job {job_id} not found")
        return job

    def _public(request: Request, snap: Dict[str, Any]) -> Dict[str, Any]:
        """JobStatus as another computer may see it: the desktop app's jobs keep what a queue estimate needs
        (state, progress, priority, GPU) but not their name, dataset or log text; other clients' addresses
        become "remote" (a client still sees its own)."""
        me = _origin(request)
        if me == "local":
            return snap
        s = dict(snap)
        if s.get("origin") == "local":
            s.update(name="Job on this PC", datasetId="", outputDatasetId=None, message=None)
        elif s.get("origin") not in (None, me):
            s["origin"] = "remote"
        return s

    @app.post("/v1/jobs/{job_id}/pause")
    def pause_job(job_id: str, request: Request):
        job = manager.get(job_id)
        _may_control(request, job)
        return manager.pause(job_id).snapshot()

    @app.post("/v1/jobs/{job_id}/resume")
    def resume_job(job_id: str, request: Request):
        job = manager.get(job_id)
        _may_control(request, job, resume=True)
        return manager.resume(job_id).snapshot()

    @app.post("/v1/jobs/{job_id}/checkpoint")
    async def checkpoint_job(job_id: str, request: Request, wait: float = 0.0):
        """Ask for checkpoint.ply of the current splats. ``?wait=<s>`` (≤ 300) returns once it is written (or the
        job stops running, or the time is up); the JobStatus' ``checkpointStep`` says which step it holds."""
        import asyncio

        job = manager.get(job_id)
        _may_control(request, job)
        before = job.checkpoints
        manager.request_checkpoint(job_id)
        deadline = time.monotonic() + max(0.0, min(float(wait), 300.0))
        while job.state in ACTIVE_STATES and job.checkpoints == before and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return job.snapshot()

    @app.get("/v1/jobs/{job_id}/checkpoint.ply")
    def get_checkpoint(job_id: str, request: Request):
        job = _job(request, job_id)
        path = job.dir / "checkpoint.ply"
        if not path.is_file():
            raise ApiError(404, "no checkpoint yet")
        step = job.status.get("checkpointStep")
        headers = {"X-Checkpoint-Step": str(step)} if step is not None else None
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{job_id}-step{step if step is not None else 0}.ply", headers=headers)

    @app.get("/v1/jobs")
    def list_jobs(request: Request):
        return [_public(request, s) for s in manager.list()]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        return _public(request, manager.get(job_id).snapshot())

    def _may_end(request: Request, job: Job) -> None:
        """Cancel / delete: anybody may stop other computers' jobs (as before 1.4.0), but only the desktop app
        may stop its own."""
        if job.status.get("origin") == "local" and not _is_local(request):
            raise ApiError(403, "that job belongs to the desktop app on this computer")

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request):
        _may_end(request, manager.get(job_id))
        return _public(request, manager.cancel(job_id).snapshot())

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request):
        _may_end(request, manager.get(job_id))
        manager.delete(job_id)
        return Response(status_code=204)

    def _job_file(request: Request, job_id: str, name: str) -> Path:
        job = _job(request, job_id)
        path = job.dir / name
        if not path.exists() and job.spec.get("kind") == "pipeline" and job.status.get("trainJobId") \
                and name != "train.log":
            child = manager.jobs.get(job.status["trainJobId"])      # previews / checkpoint of its training
            if child is not None:
                return child.dir / name
        return path

    @app.get("/v1/jobs/{job_id}/preview.ply")
    def get_preview(job_id: str, request: Request):
        path = _job_file(request, job_id, "preview.ply")
        if not path.is_file():
            raise ApiError(404, "no preview yet")
        step = _job(request, job_id).status.get("previewStep")
        headers = {"X-Preview-Step": str(step)} if step is not None else None
        return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}-preview.ply",
                            headers=headers)

    @app.get("/v1/jobs/{job_id}/preview.splat")
    def get_preview_splat(job_id: str, request: Request):
        """Compact live preview (antimatter15 .splat, the most visible ≤400 k splats)."""
        path = _job_file(request, job_id, "preview.splat")
        if not path.is_file():
            raise ApiError(404, "no preview yet")
        step = _job(request, job_id).status.get("previewStep")
        headers = {"X-Preview-Step": str(step)} if step is not None else None
        return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}-preview.splat",
                            headers=headers)

    @app.get("/v1/jobs/{job_id}/cameras")
    def get_job_cameras(job_id: str, request: Request):
        """The training job's cameras (same coordinates as its splats) — lets a viewer that watches the job
        draw the right camera frustums instead of whatever project it has open."""
        job = _job(request, job_id)
        root = _root(request, job.spec.get("datasetId", "")) if job.spec.get("datasetId") else None
        dj = root / "dataset.json" if root is not None else None
        if dj is None or not dj.is_file():
            raise ApiError(404, "job has no dataset cameras")
        try:
            raw = json.loads(dj.read_text(encoding="utf-8"))
        except Exception:
            raise ApiError(500, "unreadable dataset.json")
        keys = ("image", "width", "height", "fx", "fy", "cx", "cy", "cameraToWorld", "holdout")
        cams = [{"id": i, **{k: c.get(k) for k in keys if k in c}} for i, c in enumerate(raw.get("cameras") or [])]
        return {"cameras": cams, "sceneExtent": raw.get("sceneExtent"), "upVector": raw.get("upVector")}

    @app.get("/v1/jobs/{job_id}/result.ply")
    def get_result(job_id: str, request: Request):
        job = _job(request, job_id)
        path = job.dir / "result.ply"
        if job.state != "completed" or not path.is_file():
            raise ApiError(404, f"result not available (job is {job.state})")
        return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}.ply")

    @app.get("/v1/jobs/{job_id}/census.npz")
    def get_census(job_id: str, request: Request):
        path = _job_file(request, job_id, "census.npz")
        if not path.is_file():
            raise ApiError(404, "census not available yet")
        return FileResponse(path, media_type="application/octet-stream", filename="census.npz")

    @app.get("/v1/jobs/{job_id}/sky.png")
    def get_sky(job_id: str, request: Request):
        """The learned skybox (worker ≥ 1.3.15): equirectangular panorama; orientation in sky.json."""
        job = _job(request, job_id)
        path = job.dir / "result.sky.png"
        if job.state != "completed" or not path.is_file():
            raise ApiError(404, "no skybox for this job")
        return FileResponse(path, media_type="image/png", filename=f"{job_id}.sky.png")

    @app.get("/v1/jobs/{job_id}/sky.json")
    def get_sky_meta(job_id: str, request: Request):
        job = _job(request, job_id)
        path = job.dir / "result.sky.json"
        if job.state != "completed" or not path.is_file():
            raise ApiError(404, "no skybox for this job")
        return json.loads(path.read_text())

    def _zip_filter(only: Optional[str], skip: Optional[str]):
        """``only=images,masks`` / ``skip=images,sfm/images``: folders (or files) to include / leave out — a
        top-level name or a nested path (worker ≥ 1.3.12)."""
        def names(v):
            return {x.strip().strip("/") for x in (v or "").split(",") if x.strip().strip("/")}
        o, k = names(only), names(skip)
        if not o and not k:
            return None

        def under(rel, prefixes):
            return any(rel == p or rel.startswith(p + "/") for p in prefixes)
        return lambda rel: (not o or under(rel, o)) and not under(rel, k)

    async def _serve_zip(root: Path, name: str, keep, cache: Optional[Path] = None):
        from .processing import zip_output

        if cache is not None and keep is None and cache.is_file():
            return FileResponse(cache, media_type="application/zip", filename=name)
        target = cache if (cache is not None and keep is None) else settings.tmp_dir / f"{uuid.uuid4().hex}.zip"
        await run_in_threadpool(zip_output, root, target, [root], keep)
        cleanup = None if target == cache else BackgroundTask(target.unlink, missing_ok=True)
        return FileResponse(target, media_type="application/zip", filename=name, background=cleanup)

    @app.get("/v1/jobs/{job_id}/output.zip")
    async def get_output(job_id: str, request: Request, only: Optional[str] = None, skip: Optional[str] = None):
        job = _job(request, job_id)
        legacy = job.dir / "output.zip"           # jobs from workers before 1.3.8
        out_id = job.status.get("outputDatasetId")
        root = dataset_root(settings, out_id) if out_id else None
        if job.state != "completed" or (root is None and not legacy.is_file()):
            raise ApiError(404, f"output not available (job is {job.state})")
        if root is None:
            return FileResponse(legacy, media_type="application/zip", filename=f"{job_id}-output.zip")
        return await _serve_zip(root, f"{job_id}-output.zip", _zip_filter(only, skip), cache=legacy)

    @app.get("/v1/datasets/{dataset_id}/archive.zip")
    async def get_dataset_archive(dataset_id: str, request: Request, only: Optional[str] = None,
                                  skip: Optional[str] = None):
        """A dataset's files as a zip (``only`` / ``skip`` top-level folders) — e.g. a video's extracted frames
        fetched in the background while the next steps already run here."""
        _check_id(dataset_id, "dataset")
        root = _root(request, dataset_id)
        if root is None:
            raise ApiError(404, "dataset not found")
        return await _serve_zip(root, f"{dataset_id}.zip", _zip_filter(only, skip))

    @app.get("/v1/jobs/{job_id}/log")
    def get_log(job_id: str, request: Request):
        path = _job_file(request, job_id, "train.log")
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        return PlainTextResponse(text)

    return app
