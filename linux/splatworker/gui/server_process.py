"""Lifecycle of the worker server (``python -m splatworker serve``) as a child process.

The GUI does not re-implement any server logic: it starts the unchanged CLI with
arguments built from :class:`~splatworker.gui.settings.AppSettings`, reads its output
(worker log + uvicorn access log) and talks to it over the normal HTTP API on
127.0.0.1. A child process (rather than a thread) is required because the GPU runtime
uses a different Python (3.10, see :mod:`runtime_plan`) than the GTK app, and it keeps
CUDA and crashes away from the UI.

GTK-free; the GUI polls :meth:`ServerProcess.snapshot` from a GLib timeout.
"""

from __future__ import annotations

import collections
import errno
import os
import re
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence

from ..platform_compat import IS_WINDOWS, kill_tree, new_group_kwargs

STOPPED, STARTING, RUNNING, STOPPING, FAILED = "stopped", "starting", "running", "stopping", "failed"

_ACCESS_RE = re.compile(
    r'(?:^|\s)(?P<client>\[?[0-9A-Fa-f:.]+\]?):(?P<port>\d+) - "(?P<method>[A-Z]+) (?P<path>\S+) '
    r'HTTP/[\d.]+" (?P<status>\d{3})')
_RUNNING_RE = re.compile(r"Uvicorn running on (\S+)")
_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


@dataclass
class AccessRecord:
    time: float
    client: str
    method: str
    path: str
    status: int


def parse_access_line(line: str, now: Optional[float] = None) -> Optional[AccessRecord]:
    m = _ACCESS_RE.search(line)
    if not m:
        return None
    return AccessRecord(time=time.time() if now is None else now, client=m.group("client").strip("[]"),
                        method=m.group("method"), path=m.group("path"), status=int(m.group("status")))


@dataclass
class ClientInfo:
    address: str
    first_seen: float
    last_seen: float
    requests: int = 0
    last_path: str = ""
    unauthorized: int = 0


class RequestLog:
    """Recent requests and distinct clients, fed from uvicorn's access log."""

    def __init__(self, maxlen: int = 200, ignore_local_polls: bool = True):
        self.recent: Deque[AccessRecord] = collections.deque(maxlen=maxlen)
        self.clients: Dict[str, ClientInfo] = {}
        self.ignore_local_polls = ignore_local_polls
        self._lock = threading.Lock()

    def add(self, rec: AccessRecord) -> None:
        # The GUI itself polls /v1/jobs and /v1/info on 127.0.0.1; hide that noise.
        if self.ignore_local_polls and rec.client in ("127.0.0.1", "::1") and rec.method == "GET":
            return
        with self._lock:
            self.recent.append(rec)
            c = self.clients.get(rec.client)
            if c is None:
                c = self.clients[rec.client] = ClientInfo(rec.client, rec.time, rec.time)
            c.last_seen = rec.time
            c.requests += 1
            c.last_path = f"{rec.method} {rec.path}"
            if rec.status == 401:
                c.unauthorized += 1

    def active_clients(self, within_seconds: float = 300.0, now: Optional[float] = None) -> List[ClientInfo]:
        now = time.time() if now is None else now
        with self._lock:
            return sorted((c for c in self.clients.values() if now - c.last_seen <= within_seconds),
                          key=lambda c: -c.last_seen)

    def recent_requests(self, n: int = 20) -> List[AccessRecord]:
        with self._lock:
            return list(self.recent)[-n:][::-1]

    def clear(self) -> None:
        with self._lock:
            self.recent.clear()
            self.clients.clear()


class LogBuffer:
    def __init__(self, maxlen: int = 5000):
        self._lines: Deque[str] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.total = 0   # monotonically increasing line counter, for incremental UI updates

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            self.total += 1

    def tail(self, n: Optional[int] = None) -> List[str]:
        with self._lock:
            lines = list(self._lines)
        return lines if n is None else lines[-n:]

    def since(self, total_seen: int) -> List[str]:
        """Lines appended after the caller had seen ``total_seen`` lines."""
        with self._lock:
            new = self.total - total_seen
            if new <= 0:
                return []
            lines = list(self._lines)
            return lines[-min(new, len(lines)):]


def port_problem(port: int, host: str = "0.0.0.0") -> Optional[str]:
    """None if ``port`` can be bound, else a friendly explanation."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if IS_WINDOWS:   # SO_REUSEADDR on Windows would let us bind a port another program is using
            s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", -5), 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
    except OSError as e:
        winerror = getattr(e, "winerror", None)
        if IS_WINDOWS and winerror == 10013:   # WSAEACCES: held exclusively by another program, or reserved
            return (f"Port {port} is already in use or reserved by Windows (Hyper-V / WSL reserve port ranges). "
                    "Stop the other program or choose a different port in Settings.")
        if e.errno == errno.EADDRINUSE or winerror == 10048:
            return (f"Port {port} is already in use. Another SplatWork may be running (for example "
                    "the systemd service or Docker container), or another program uses this port. Stop "
                    "it or choose a different port in Settings.")
        if e.errno == errno.EACCES:
            return f"Not allowed to use port {port}. Choose a port above 1024 in Settings."
        return f"Cannot listen on port {port}: {e.strerror or e}"
    finally:
        s.close()
    return None


def explain_exit(lines: Sequence[str], returncode: Optional[int], port: int) -> str:
    text = "\n".join(lines[-80:])
    low = text.lower()
    if ("address already in use" in low or "errno 98" in low or "errno 10048" in low
            or "only one usage of each socket address" in low):
        return port_problem(port) or f"Port {port} is already in use."
    if "no module named" in low:
        m = re.search(r"No module named '([^']+)'", text)
        mod = m.group(1) if m else "a module"
        return (f"The worker could not start because {mod} is missing. Open Settings → GPU runtime and "
                "press Repair.")
    if "permission denied" in low:
        return "The worker could not access its data folder (permission denied). Choose another folder in Settings."
    tail = next((ln for ln in reversed(lines) if ln.strip()), "")
    return f"The worker stopped unexpectedly (exit code {returncode}). {tail}".strip()


@dataclass
class ServerSnapshot:
    state: str
    message: str = ""
    pid: Optional[int] = None
    started_at: Optional[float] = None
    url: Optional[str] = None
    exit_code: Optional[int] = None


class ServerProcess:
    """Start/stop ``<python> -m splatworker serve <args>`` and collect its output."""

    def __init__(self, on_change: Optional[Callable[[ServerSnapshot], None]] = None,
                 log_lines: int = 5000):
        self.on_change = on_change
        self.log = LogBuffer(log_lines)
        self.requests = RequestLog()
        self._proc: Optional[subprocess.Popen] = None
        self._state = STOPPED
        self._message = ""
        self._started_at: Optional[float] = None
        self._url: Optional[str] = None
        self._port = 0
        self._exit_code: Optional[int] = None
        self._lock = threading.RLock()
        self._stop_requested = False
        self._reader: Optional[threading.Thread] = None
        self._recent: Deque[str] = collections.deque(maxlen=200)

    # ------------------------------------------------------------------ api
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in (STARTING, RUNNING, STOPPING)

    def snapshot(self) -> ServerSnapshot:
        with self._lock:
            return ServerSnapshot(self._state, self._message, self._proc.pid if self._proc else None,
                                  self._started_at, self._url, self._exit_code)

    def start(self, python: str, serve_args: Sequence[str], env: Mapping[str, str], port: int,
              check_port: bool = True, cwd: Optional[str] = None) -> bool:
        """Start the server. Returns False (state FAILED + message) if it cannot."""
        with self._lock:
            if self.is_active:
                return True
            self._port = port
            self._exit_code = None
            self._url = None
            self._recent.clear()
            self._stop_requested = False
            problem = port_problem(port) if check_port else None
            if problem:
                self._set(FAILED, problem)
                return False
            argv = [python, "-m", "splatworker", "serve", *serve_args]
            self.log.append(f"$ {' '.join(argv)}")
            env = dict(env)
            if IS_WINDOWS:  # stop = close stdin (no SIGTERM for a child without a console)
                env["SPLATWORKER_STOP_ON_STDIN_EOF"] = "1"
            try:
                self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              stdin=subprocess.PIPE if IS_WINDOWS else subprocess.DEVNULL,
                                              env=env, text=True, bufsize=1, errors="replace",
                                              cwd=cwd, **new_group_kwargs())
            except OSError as e:
                self._proc = None
                self._set(FAILED, f"Could not start the worker: {e}")
                return False
            self._started_at = time.time()
            self.requests.clear()
            self._set(STARTING, "Starting…")
            self._reader = threading.Thread(target=self._read_loop, args=(self._proc,), daemon=True,
                                            name="splatworker-server-reader")
            self._reader.start()
            return True

    def stop(self, timeout: float = 40.0, wait: bool = True) -> None:
        """Graceful stop (SIGTERM → uvicorn shutdown → running jobs are cancelled)."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            self._stop_requested = True
            self._set(STOPPING, "Stopping…")
            # SIGTERM only the server: it cancels its trainer processes itself (they are
            # in the same process group, which is SIGKILLed only if shutdown hangs).
            try:
                if IS_WINDOWS and proc.stdin is not None:
                    proc.stdin.close()
                else:
                    proc.terminate()
            except OSError:
                pass

        def killer():
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.log.append("worker did not stop in time; killing it")
                try:
                    kill_tree(proc, force=True)
                except OSError:
                    pass

        if wait:
            killer()
            if self._reader is not None:
                self._reader.join(5)
        else:
            threading.Thread(target=killer, daemon=True).start()

    def wait_until(self, states: Sequence[str], timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._state in states:
                return True
            time.sleep(0.05)
        return self._state in states

    # ------------------------------------------------------------- internal
    def _set(self, state: str, message: str = "") -> None:
        with self._lock:
            self._state = state
            self._message = message
            snap = self.snapshot()
        if self.on_change is not None:
            try:
                self.on_change(snap)
            except Exception:
                pass

    def _read_loop(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            self.log.append(line)
            self._recent.append(line)
            rec = parse_access_line(line)
            if rec is not None:
                self.requests.add(rec)
                continue
            m = _RUNNING_RE.search(line)
            if m and self._state == STARTING:
                self._url = m.group(1)
                self._set(RUNNING, "Running")
        rc = proc.wait()
        with self._lock:
            self._exit_code = rc
            if self._proc is proc:
                self._proc = None
            if self._stop_requested:
                self._set(STOPPED, "Stopped")
            else:
                self._set(FAILED, explain_exit(list(self._recent), rc, self._port))
        self.log.append(f"worker exited with code {rc}")


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LaunchSpec:
    python: str
    env: Dict[str, str] = field(default_factory=dict)
    runtime_ready: bool = False
    warning: Optional[str] = None


def launch_spec(paths, environ: Optional[Mapping[str, str]] = None, fallback_python: Optional[str] = None,
                lib_dir: Optional[Path] = None) -> LaunchSpec:
    """Which interpreter runs the worker, with which environment.

    The GPU runtime venv if it is installed; otherwise the app's own Python, which has the
    lightweight server dependencies but no torch/gsplat (the worker then starts but
    advertises no ``train`` capability).
    """
    import sys

    from .installer import RuntimeState, base_env
    from .paths import splatprep_dir, worker_lib_dir
    from .runtime_plan import child_pythonpath

    state = RuntimeState.load(paths)
    env = base_env(paths, environ)
    lib = lib_dir or worker_lib_dir()
    sp: Optional[Path] = None
    if state.with_splatprep:
        sp = Path(state.splatprep_dir) if state.splatprep_dir else splatprep_dir(environ)
        if sp is not None and not (sp / "splatprep").is_dir():
            sp = splatprep_dir(environ)
    env["PYTHONPATH"] = child_pythonpath(lib, sp)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.pop("TMPDIR", None)
    if state.is_installed(paths):
        return LaunchSpec(str(paths.venv_python), env, True)
    return LaunchSpec(fallback_python or sys.executable, env, False,
                      "The GPU runtime is not installed, so this worker cannot train yet. "
                      "Run the setup assistant to install it.")
