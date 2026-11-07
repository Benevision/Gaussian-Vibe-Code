"""Remote install of optional processing components (``POST /v1/runtime/install``).

Lets Splat Studio on the Mac add the camera-solving / masking models to this worker's GPU runtime
without anyone touching the PC: the same pip steps as the setup assistant's "Processing models" option,
run with the server's own interpreter (the runtime venv), with live progress for the Mac to show.
Only a fixed component list is installable — the request cannot name arbitrary packages.
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from .gui.pip_progress import PipProgressParser
from .gui.runtime_plan import GPU_PROFILES, LIGHTGLUE_DEPS, PROCESSING_DEPS, SAM2_REQ

COMPONENTS = ("processing",)


def _has(m: str) -> bool:
    try:
        return importlib.util.find_spec(m) is not None
    except (ImportError, ValueError):
        return False


def installed_components() -> Dict[str, bool]:
    importlib.invalidate_caches()
    return {"lightglue": _has("lightglue"), "pycolmap": _has("pycolmap"), "sam2": _has("sam2"),
            "transformers": _has("transformers"), "torchvision": _has("torchvision"),
            "processing": all(_has(m) for m in ("lightglue", "pycolmap", "sam2", "transformers", "torchvision"))}


def _profile():
    """The GPU profile matching the installed torch build (None if torch is missing or unknown)."""
    try:
        import torch  # noqa: F401

        base = torch.__version__.split("+")[0]
        cuda = (torch.version.cuda or "").replace(".", "")
    except Exception:
        return None
    for p in GPU_PROFILES:
        if p.torch == base and p.id == f"cu{cuda}":
            return p
    return None


class RuntimeInstaller:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict = {"state": "idle", "component": None, "step": None, "fraction": 0.0,
                             "message": "", "log": [], "startedAt": None, "finishedAt": None}
        self._thread: Optional[threading.Thread] = None

    def status(self) -> Dict:
        with self._lock:
            s = dict(self._state)
            s["log"] = list(self._state["log"][-40:])
        s["installed"] = installed_components()
        return s

    def start(self, component: str) -> Dict:
        if component not in COMPONENTS:
            raise ValueError(f"unknown component {component!r} (installable: {', '.join(COMPONENTS)})")
        with self._lock:
            if self._state["state"] == "running":
                return dict(self._state)
            self._state.update(state="running", component=component, step=None, fraction=0.0, message="starting",
                               log=[], startedAt=time.time(), finishedAt=None)
        self._thread = threading.Thread(target=self._run, args=(component,), daemon=True)
        self._thread.start()
        return self.status()

    def _steps(self) -> List[tuple]:
        py = sys.executable
        base = [py, "-m", "pip", "install", "--progress-bar", "raw", "--prefer-binary"]
        p = _profile()
        # Every step pins torch / torchvision to the runtime's build (gsplat's CUDA kernels are built for it).
        pins = ([f"torch=={p.torch}", f"torchvision=={p.torchvision}", "--extra-index-url", p.torch_index]
                if p is not None else [])
        tv = ([f"torchvision=={p.torchvision}", f"torch=={p.torch}", "--index-url", p.torch_index]
              if p is not None else ["torchvision"])
        return [("torchvision", "Installing torchvision", base + tv, 0.3),
                ("lightglue", "Installing LightGlue", base + list(LIGHTGLUE_DEPS) + pins, 0.15),
                ("processing", "Installing COLMAP mapper and OWLv2", base + list(PROCESSING_DEPS) + pins, 0.45),
                ("sam2", "Installing SAM 2", base + ["--no-deps", "--no-build-isolation", SAM2_REQ], 0.1)]

    def _log(self, line: str) -> None:
        with self._lock:
            self._state["log"].append(line.rstrip()[:300])
            if len(self._state["log"]) > 400:
                del self._state["log"][:-400]

    def _run(self, component: str) -> None:
        done = 0.0
        try:
            for sid, title, argv, weight in self._steps():
                with self._lock:
                    self._state.update(step=sid, message=title)
                parser = PipProgressParser()
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._log(line)
                    try:
                        p = parser.feed(line)
                        f = getattr(p, "fraction", None) if p is not None else None
                    except Exception:
                        f = None
                    if f is not None:
                        with self._lock:
                            self._state["fraction"] = min(0.99, done + weight * max(0.0, min(1.0, f)))
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"{title} failed (pip exit {rc})")
                done += weight
                with self._lock:
                    self._state["fraction"] = min(0.99, done)
            ok = installed_components().get(component, False)
            with self._lock:
                self._state.update(state="done" if ok else "failed", fraction=1.0 if ok else self._state["fraction"],
                                   message="installed" if ok else "installed, but some modules still can't be imported",
                                   finishedAt=time.time())
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._state.update(state="failed", message=str(e), finishedAt=time.time())
