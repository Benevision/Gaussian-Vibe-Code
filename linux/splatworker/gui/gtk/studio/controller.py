"""Glue between the Studio pages and the app: the local worker server (started on demand), the Studio
client, running pipelines, recent files and the iPhone receiver. Callbacks into GTK go through GLib.idle_add."""
from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from gi.repository import GLib

from ....studio.client import ApiError, Client
from ....studio.inbox import CaptureReceiver
from ....studio.pipeline import LocalPipeline
from ....studio.prefs import StudioPrefs
from ....studio.presets import ProjectOptions
from ....studio.projects import Project, Recents, default_root
from ...server_process import RUNNING


class Run:
    """A pipeline running in a thread, with the events seen so far (pages attach / detach as they like)."""

    def __init__(self, pipeline: LocalPipeline):
        self.pipeline = pipeline
        self.project = pipeline.project
        self.events: List[tuple] = []
        self.listeners: List[Callable[[str, object], None]] = []
        self.state = "running"          # running | done | failed | cancelled
        self.output: Optional[Path] = None
        self.message = ""
        self.stage = ""
        self.fraction = 0.0
        self.last_job: Optional[Dict] = None
        self.history: List[tuple] = []   # (step, psnr, evalPsnr, splats) for the charts

    def handle(self, kind: str, data) -> None:          # main thread
        self.events.append((kind, data))
        if kind == "stage":
            self.stage = data["title"]
            self.fraction = 0.0
        elif kind == "progress":
            self.fraction = data["fraction"]
            self.message = data["message"]
        elif kind == "job":
            self.last_job = data
            if data.get("kind") == "train" and data.get("step"):
                if not self.history or self.history[-1][0] != data["step"]:
                    self.history.append((data["step"], data.get("psnr"), data.get("evalPsnr"),
                                         data.get("numGaussians")))
        elif kind == "done":
            self.state = "done"
            self.output = data.get("output")
        elif kind == "failed":
            self.state = "failed"
            self.message = str(data)
        elif kind == "cancelled":
            self.state = "cancelled"
        for cb in list(self.listeners):
            cb(kind, data)

    @property
    def active(self) -> bool:
        return self.state == "running"


class StudioController:
    def __init__(self, app_state, toast: Callable[[str], None]):
        self.state = app_state
        self.toast = toast
        cfg = app_state.paths.config_dir
        self.prefs_path = cfg / "studio.json"
        self.prefs = StudioPrefs.load(self.prefs_path)
        self.recents = Recents(cfg / "recent.json")
        self.runs: Dict[str, Run] = {}
        self.listeners: List[Callable[[], None]] = []      # "something changed" (runs, inbox)
        self.receiver: Optional[CaptureReceiver] = None
        self.receiver_listeners: List[Callable[[str, object], None]] = []

    # ------------------------------------------------------------- prefs / paths
    def save_prefs(self) -> None:
        self.prefs.save(self.prefs_path)

    @property
    def projects_root(self) -> Path:
        return Path(self.prefs.projects_dir).expanduser() if self.prefs.projects_dir.strip() else default_root()

    @property
    def inbox_dir(self) -> Path:
        return self.state.paths.data_root / "inbox"

    def changed(self) -> None:
        for cb in list(self.listeners):
            cb()

    # ------------------------------------------------------------- server
    def client(self) -> Optional[Client]:
        port = self.state.running_port
        if port is None or self.state.server.snapshot().state != RUNNING:
            return None
        return Client(f"http://127.0.0.1:{port}", self.state.token(), timeout=30,
                      local_secret=getattr(self.state, "local_secret", None))

    def with_server(self, then: Callable[[Client], None], fail: Callable[[str], None]) -> None:
        """Run ``then(client)`` on a worker thread once the local server answers (starting it if needed)."""
        if not self.state.runtime_installed:
            GLib.idle_add(fail, "The GPU runtime is not installed yet: open the menu → Set Up GPU Runtime…")
            return
        if not self.state.server.is_active:
            self.toast("Starting the worker server…")
            if not self.state.start_server():
                GLib.idle_add(fail, self.state.server.snapshot().message or "the worker server did not start")
                return

        def work():
            import time

            deadline = time.time() + 90
            while time.time() < deadline:
                c = self.client()
                if c is not None:
                    try:
                        caps = c.info().get("capabilities", [])
                        if "local" not in caps:
                            GLib.idle_add(fail, "The running server is too old or was started without "
                                                "--allow-local-datasets: restart it.")
                            return
                        then(c)
                        return
                    except ApiError:
                        pass
                if not self.state.server.is_active:
                    GLib.idle_add(fail, self.state.server.snapshot().message or "the worker server stopped")
                    return
                time.sleep(0.5)
            GLib.idle_add(fail, "the worker server did not answer within 90 s")
        threading.Thread(target=work, daemon=True).start()

    @property
    def priority(self) -> int:
        return 1 if self.prefs.local_jobs_first else 0

    # ------------------------------------------------------------- pipelines
    def start_pipeline(self, project: Project, options: ProjectOptions, train: Optional[bool] = None,
                       train_only: bool = False, config: Optional[Dict] = None,
                       on_error: Optional[Callable[[str], None]] = None) -> None:
        key = str(project.root)
        if key in self.runs and self.runs[key].active:
            self.toast(f"{project.name} is already running")
            return

        def go(client: Client) -> None:
            def emit(kind, data):
                GLib.idle_add(self._emit, key, kind, data)
            p = LocalPipeline(client, project, options, emit, train=train, priority=self.priority)
            run = Run(p)
            GLib.idle_add(self._register_run, key, run)
            if train_only:
                p._titles = ["Training"]

                def target():
                    try:
                        out = p.train(config)
                        emit("done", {"output": out, "dataset": project.dataset_dir})
                    except Exception as e:  # noqa: BLE001
                        from ....studio.pipeline import Cancelled

                        if isinstance(e, Cancelled):
                            emit("cancelled", None)
                        else:
                            emit("failed", getattr(e, "message", None) or str(e))
                    finally:
                        for d in p._registered:
                            client.delete_dataset(d)
                threading.Thread(target=target, daemon=True).start()
            else:
                threading.Thread(target=p.run, daemon=True).start()

        def fail(msg: str) -> bool:
            self.toast(msg)
            if on_error:
                on_error(msg)
            return False
        self.with_server(go, fail)

    def _register_run(self, key: str, run: Run) -> bool:
        self.runs[key] = run
        self.changed()
        return False

    def _emit(self, key: str, kind: str, data) -> bool:
        run = self.runs.get(key)
        if run is not None:
            run.handle(kind, data)
            if kind in ("done", "failed", "cancelled", "stage"):
                self.changed()
        return False

    def run_for(self, project_root: Path) -> Optional[Run]:
        return self.runs.get(str(project_root))

    # ------------------------------------------------------------- iPhone inbox
    def start_receiver(self) -> bool:
        if self.receiver and self.receiver.running:
            return True
        self.receiver = CaptureReceiver(
            self.inbox_dir, self.state.settings.effective_name() or socket.gethostname(),
            port=self.prefs.capture_port, require_code=self.prefs.require_pairing,
            on_received=lambda cap: GLib.idle_add(self._receiver_event, "received", cap),
            on_progress=lambda n, d, t: GLib.idle_add(self._receiver_event, "progress", (n, d, t)),
            on_code_changed=lambda code: GLib.idle_add(self._receiver_event, "code", code))
        ok = self.receiver.start()
        self._receiver_event("state", ok)
        return ok

    def stop_receiver(self) -> None:
        if self.receiver:
            self.receiver.stop()
        self._receiver_event("state", False)

    def _receiver_event(self, kind: str, data) -> bool:
        if kind == "received":
            self.toast(f"Received “{data.name}” from the iPhone")
        for cb in list(self.receiver_listeners):
            cb(kind, data)
        if kind == "received":
            self.changed()
        return False

    @property
    def busy(self) -> bool:
        """A pipeline is running (between the server's jobs too: importing photos, stacking masks, …)."""
        return any(r.active for r in self.runs.values())

    def busy_names(self) -> List[str]:
        return [r.project.name for r in self.runs.values() if r.active]

    def shutdown(self) -> None:
        """On quit: cancel running pipelines (their server jobs too), stop the iPhone receiver + its Bonjour."""
        for run in list(self.runs.values()):
            if run.active:
                try:
                    run.pipeline.cancel()
                except Exception:  # noqa: BLE001
                    pass
        if self.receiver:
            try:
                self.receiver.stop()
            except Exception:  # noqa: BLE001
                pass
            self.receiver = None
