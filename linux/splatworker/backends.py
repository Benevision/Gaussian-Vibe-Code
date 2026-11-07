"""Job execution backends.

The server hands a :class:`JobContext` to a backend's ``run``. The production
:class:`SubprocessBackend` runs each job in a fresh ``spawn``-ed Python
process (one per GPU slot), which keeps the API process free of CUDA state,
returns all GPU memory to the driver after each job, and isolates crashes.
Tests substitute a fake backend.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class JobCancelled(Exception):
    pass


class JobFailed(RuntimeError):
    pass


@dataclass
class JobContext:
    job_id: str
    kind: str                      # "train" | "process"
    job_dir: Path
    dataset_dir: Path              # root of the uploaded dataset
    spec: Dict[str, Any]           # TrainingConfig dict (train) or ProcessingRequest (process)
    device_index: Optional[int]    # GPU index, None if no GPU detected
    status: Callable[..., None]    # update JobStatus fields
    log: Callable[[str], None]
    preview: Callable[[int], None]  # preview.ply for this step has been written
    cancel_event: threading.Event = field(default_factory=threading.Event)


class Backend:
    def capabilities(self) -> List[str]:
        raise NotImplementedError

    def models(self) -> List[str]:
        return []

    def run(self, ctx: JobContext) -> Dict[str, Any]:
        """Run the job to completion; raise JobCancelled or an exception on failure."""
        raise NotImplementedError


class SubprocessBackend(Backend):
    def __init__(self, cancel_grace_seconds: float = 30.0):
        self.cancel_grace = cancel_grace_seconds
        self._mp = mp.get_context("spawn")

    def capabilities(self) -> List[str]:
        from .gpu import training_available
        from .processing import splatprep_available

        caps = []
        if training_available():
            caps.append("train")
        if splatprep_available():
            caps.append("process")
        return caps

    def models(self) -> List[str]:
        from .processing import available_models

        return available_models()

    def run(self, ctx: JobContext) -> Dict[str, Any]:
        q = self._mp.Queue()
        cancel = self._mp.Event()
        if ctx.kind == "train":
            from .trainer import subprocess_entry

            device = f"cuda:{ctx.device_index}" if ctx.device_index is not None else "cuda:0"
            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), device, q, cancel)
        elif ctx.kind == "process":
            from .processing import subprocess_entry

            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), ctx.device_index, q, cancel)
        else:
            raise JobFailed(f"unknown job kind {ctx.kind}")

        proc = self._mp.Process(target=subprocess_entry, args=args, daemon=True,
                                name=f"splatworker-{ctx.job_id}")
        proc.start()
        ctx.log(f"started worker process pid {proc.pid}")
        cancel_sent_at: Optional[float] = None
        outcome: Optional[tuple] = None
        try:
            while True:
                if ctx.cancel_event.is_set() and cancel_sent_at is None:
                    cancel.set()
                    cancel_sent_at = time.time()
                    ctx.log("cancellation requested")
                if cancel_sent_at is not None and time.time() - cancel_sent_at > self.cancel_grace:
                    ctx.log("worker did not stop in time; terminating")
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                    raise JobCancelled()
                try:
                    kind, payload = q.get(timeout=0.5)
                except queue_mod.Empty:
                    if not proc.is_alive():
                        # drain anything left, then decide
                        try:
                            kind, payload = q.get(timeout=1.0)
                        except queue_mod.Empty:
                            if cancel_sent_at is not None:
                                raise JobCancelled()
                            raise JobFailed(f"worker process exited unexpectedly (code {proc.exitcode})")
                    else:
                        continue
                if kind == "status":
                    ctx.status(**payload)
                elif kind == "log":
                    ctx.log(payload)
                elif kind == "preview":
                    ctx.preview(int(payload))
                elif kind in ("done", "cancelled", "error"):
                    outcome = (kind, payload)
                    break
        finally:
            proc.join(15)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
        kind, payload = outcome
        if kind == "cancelled":
            raise JobCancelled()
        if kind == "error":
            raise JobFailed(payload)
        return payload or {}
