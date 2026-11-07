"""Application state shared by all windows: settings, token, runtime, server, GPU stats.

GTK-free so it can be tested; the GTK layer calls :meth:`AppState.poll` from a worker
thread every couple of seconds and renders the returned :class:`Poll`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .api_client import WorkerApiError, WorkerClient
from .installer import RuntimeState
from .nvidia import GpuMonitor, GpuStats
from .paths import AppPaths
from .server_process import RUNNING, ServerProcess, ServerSnapshot, launch_spec
from .settings import AppSettings, TokenStore


@dataclass
class Poll:
    server: ServerSnapshot
    info: Optional[Dict[str, Any]] = None
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    gpus: List[GpuStats] = field(default_factory=list)
    gpu_error: Optional[str] = None
    api_error: Optional[str] = None
    time: float = field(default_factory=time.time)


class AppState:
    def __init__(self, paths: Optional[AppPaths] = None, environ: Optional[Mapping[str, str]] = None,
                 monitor: Optional[GpuMonitor] = None, fallback_python: Optional[str] = None):
        self.paths = paths or AppPaths.from_env(environ)
        self.paths.ensure()
        self.environ = environ
        self.fallback_python = fallback_python
        self.settings = AppSettings.load(self.paths.settings_file)
        self.tokens = TokenStore(self.paths.token_file)
        self.server = ServerProcess()
        self.monitor = monitor if monitor is not None else GpuMonitor()
        self.runtime = RuntimeState.load(self.paths)
        self._lock = threading.Lock()
        self._running_port: Optional[int] = None
        self._running_token: Optional[str] = None
        self._running_settings: Optional[Dict[str, Any]] = None
        self.last_info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------- settings
    def save_settings(self) -> None:
        self.settings.save(self.paths.settings_file)

    def token(self) -> Optional[str]:
        return self.tokens.get_or_create() if self.settings.require_token else None

    def regenerate_token(self) -> str:
        return self.tokens.regenerate()

    def needs_restart(self) -> bool:
        """True when the running server was started with different settings or token."""
        if not self.server.is_active or self._running_settings is None:
            return False
        return self._running_settings != self.settings.to_dict() or self._running_token != self.token()

    def reload_runtime(self) -> RuntimeState:
        self.runtime = RuntimeState.load(self.paths)
        return self.runtime

    @property
    def runtime_installed(self) -> bool:
        return self.runtime.is_installed(self.paths)

    # --------------------------------------------------------------- server
    def start_server(self) -> bool:
        with self._lock:
            token = self.token()
            spec = launch_spec(self.paths, self.environ, fallback_python=self.fallback_python)
            ok = self.server.start(spec.python, self.settings.serve_args(self.paths), spec.env,
                                   self.settings.port)
            if ok:
                self._running_port = self.settings.port
                self._running_token = token
                self._running_settings = self.settings.to_dict()
                self.last_info = None
            return ok

    def stop_server(self, wait: bool = True) -> None:
        self.server.stop(wait=wait)

    def restart_server(self) -> bool:
        self.stop_server(wait=True)
        return self.start_server()

    def client(self) -> Optional[WorkerClient]:
        if self._running_port is None:
            return None
        return WorkerClient(f"http://127.0.0.1:{self._running_port}", self._running_token, timeout=4)

    @property
    def running_port(self) -> Optional[int]:
        return self._running_port

    # ----------------------------------------------------------------- poll
    def poll(self, with_gpus: bool = True) -> Poll:
        snap = self.server.snapshot()
        p = Poll(server=snap)
        if with_gpus:
            p.gpus = self.monitor.sample()
            p.gpu_error = self.monitor.error
        if snap.state == RUNNING:
            client = self.client()
            try:
                if self.last_info is None or int(p.time) % 10 == 0:
                    self.last_info = client.info()
                p.info = self.last_info
                p.jobs = client.jobs()
            except WorkerApiError as e:
                p.api_error = e.message
        return p

    def shutdown(self) -> None:
        self.server.stop(wait=True)
        self.monitor.close()

    def lan_addresses(self) -> List[str]:
        if self.settings.advertise_ip:
            return [self.settings.advertise_ip]
        from splatworker.discovery import local_ipv4_addresses

        return local_ipv4_addresses()
