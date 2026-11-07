"""Runs a :class:`~splatworker.gui.runtime_plan.Plan`: download, venv, pip, verify.

GTK-free. The GUI runs :meth:`Installer.run` in a thread and marshals the events to
the main loop; ``flatpak run com.example.SplatWorker --install-runtime`` runs it in a
terminal. Steps are idempotent, so *Retry* simply runs the plan again from the step
that failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .paths import AppPaths, worker_lib_dir
from .pip_progress import PipProgressParser, human_bytes
from .runtime_plan import PYTHON_RUNTIME, Plan, Step, child_pythonpath, verify_problems

RUNTIME_STATE_VERSION = 1
VERIFY_PREFIX = "SPLATWORKER-VERIFY "

EventCallback = Callable[[str, Dict[str, Any]], None]


class InstallCancelled(Exception):
    pass


@dataclass
class InstallError(Exception):
    title: str
    detail: str = ""
    hint: str = ""
    step_index: int = -1

    def __str__(self) -> str:
        return f"{self.title}: {self.detail}" if self.detail else self.title


@dataclass
class InstallResult:
    ok: bool
    cancelled: bool = False
    error: Optional[InstallError] = None
    failed_step: Optional[int] = None
    verify: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# error classification
# ---------------------------------------------------------------------------

def classify_failure(lines: List[str], returncode: Optional[int], step: Step) -> InstallError:
    text = "\n".join(lines[-200:])
    low = text.lower()
    errors = [ln.split("ERROR:", 1)[1].strip() for ln in lines if "ERROR:" in ln]
    detail = errors[-1] if errors else (lines[-1].strip() if lines else f"exit code {returncode}")
    if "no space left on device" in low or "errno 28" in low:
        return InstallError("The disk is full", detail,
                            "Free some disk space (the GPU runtime needs about 6 GB) and press Retry.")
    if any(k in low for k in ("newconnectionerror", "connection broken", "temporary failure in name resolution",
                              "name or service not known", "network is unreachable", "read timed out",
                              "connecttimeout", "connection reset", "max retries exceeded",
                              "failed to establish a new connection", "urlopen error")):
        return InstallError("Network problem while downloading", detail,
                            "Check the internet connection (and any proxy), then press Retry. Downloads "
                            "already finished are cached and will not be repeated.")
    if "certificate verify failed" in low or "sslerror" in low or "ssl: " in low:
        return InstallError("Secure connection failed", detail,
                            "The download server's certificate was rejected. Check the system clock and "
                            "any TLS-intercepting proxy, then press Retry.")
    if "hashes are required" in low or "do not match the hashes" in low or "sha256 mismatch" in low:
        return InstallError("A download was corrupted", detail,
                            "Press Retry to download it again.")
    if "no matching distribution" in low or "could not find a version that satisfies" in low:
        return InstallError(f"{step.title}: the package is not available", detail,
                            "The package server does not offer this build (yet or any more). Try a different "
                            "runtime in the assistant's advanced options, or update Splat Worker.")
    if "permission denied" in low or "errno 13" in low:
        return InstallError("Permission denied", detail,
                            "The app could not write to its data folder. Remove the runtime in Settings and "
                            "try again.")
    if "killed" in low or returncode in (-9, 137):
        return InstallError("The installer was stopped by the system", detail,
                            "This usually means the computer ran out of memory. Close other programs and "
                            "press Retry.")
    return InstallError(f"{step.title} failed", detail,
                        "Press Retry. If it fails again, use Show log and include it in a bug report.")


# ---------------------------------------------------------------------------
# runtime state (runtime.json)
# ---------------------------------------------------------------------------

@dataclass
class RuntimeState:
    profile_id: Optional[str] = None
    can_train: bool = False
    with_splatprep: bool = False
    with_lightglue: bool = False
    splatprep_dir: Optional[str] = None
    verify: Dict[str, Any] = field(default_factory=dict)
    installed_at: Optional[str] = None
    python_version: Optional[str] = None

    @classmethod
    def load(cls, paths: AppPaths) -> "RuntimeState":
        try:
            d = json.loads(paths.runtime_state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(d, dict) or d.get("version") != RUNTIME_STATE_VERSION:
            return cls()
        return cls(profile_id=d.get("profile"), can_train=bool(d.get("canTrain")),
                   with_splatprep=bool(d.get("withSplatprep")), with_lightglue=bool(d.get("withLightglue")),
                   splatprep_dir=d.get("splatprepDir"), verify=d.get("verify") or {},
                   installed_at=d.get("installedAt"), python_version=d.get("pythonVersion"))

    def save(self, paths: AppPaths) -> None:
        paths.data_root.mkdir(parents=True, exist_ok=True)
        data = {"version": RUNTIME_STATE_VERSION, "profile": self.profile_id, "canTrain": self.can_train,
                "withSplatprep": self.with_splatprep, "withLightglue": self.with_lightglue,
                "splatprepDir": self.splatprep_dir, "verify": self.verify,
                "installedAt": self.installed_at, "pythonVersion": self.python_version}
        tmp = paths.runtime_state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, paths.runtime_state_file)

    def is_installed(self, paths: AppPaths) -> bool:
        return self.profile_id is not None and paths.venv_python.exists()

    def describe(self) -> str:
        if self.profile_id is None:
            return "Not installed"
        v = self.verify
        if self.can_train:
            parts = [f"PyTorch {v.get('torch') or '?'}", f"gsplat {v.get('gsplat') or '?'}"]
        else:
            parts = ["processing only (no training)"]
        if self.with_splatprep:
            parts.append("splatprep" + (" + LightGlue" if self.with_lightglue else ""))
        return ", ".join(parts)


def remove_runtime(paths: AppPaths) -> None:
    """Delete the downloaded Python, the venv and runtime.json (keeps jobs and settings)."""
    for d in (paths.venv_dir, paths.python_dir):
        shutil.rmtree(d, ignore_errors=True)
    paths.runtime_state_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# installer
# ---------------------------------------------------------------------------

def base_env(paths: AppPaths, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "PIP_USER", "PIP_TARGET",
              "PIP_PREFIX"):
        env.pop(k, None)
    env.update({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1", "TMPDIR": str(paths.tmp_dir)})
    return env


class Installer:
    def __init__(self, plan: Plan, paths: AppPaths, on_event: Optional[EventCallback] = None,
                 environ: Optional[Mapping[str, str]] = None, lib_dir: Optional[Path] = None,
                 opener: Callable[..., Any] = urllib.request.urlopen):
        self.plan = plan
        self.paths = paths
        self.on_event = on_event or (lambda kind, data: None)
        self.environ = environ
        self.lib_dir = lib_dir or worker_lib_dir()
        self.opener = opener
        self._cancel = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log_fh = None
        self.lines: List[str] = []
        self.overall = 0.0

    # ------------------------------------------------------------------ api
    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            _terminate(proc)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self, start_at: int = 0) -> InstallResult:
        self._cancel.clear()
        for d in (self.paths.data_root, self.paths.pip_cache, self.paths.tmp_dir, self.paths.downloads_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.paths.install_log, "a" if start_at else "w", encoding="utf-8")
        total_w = sum(s.weight for s in self.plan.steps) or 1.0
        done_w = sum(s.weight for s in self.plan.steps[:start_at])
        verify: Dict[str, Any] = {}
        try:
            self._log(f"=== Splat Worker runtime setup ({self.plan.profile_id}) "
                      f"{datetime.now().isoformat(timespec='seconds')}")
            if start_at == 0:
                self._preflight()
            for i in range(start_at, len(self.plan.steps)):
                step = self.plan.steps[i]
                if self._cancel.is_set():
                    raise InstallCancelled()
                self._emit("step", index=i, step=step, count=len(self.plan.steps))
                self._log(f"--- [{i + 1}/{len(self.plan.steps)}] {step.title}")

                def progress(frac: float, message: str, _w=step.weight, _base=done_w, _i=i) -> None:
                    frac = min(1.0, max(0.0, frac))
                    self.overall = (_base + _w * frac) / total_w
                    self._emit("progress", index=_i, step_fraction=frac, overall=self.overall,
                               message=message)

                try:
                    if step.kind == "download":
                        self._run_download(step, progress)
                    elif step.kind == "command":
                        self._run_command(step, progress)
                    elif step.kind == "verify":
                        verify = self._run_verify(step, progress)
                    else:  # pragma: no cover
                        raise InstallError(f"unknown step kind {step.kind}")
                except InstallError as e:
                    e.step_index = i
                    raise
                done_w += step.weight
                progress(1.0, "Done")
            self._swap_in()
            state = RuntimeState(profile_id=self.plan.profile_id, can_train=self.plan.can_train,
                                 with_splatprep=self.plan.with_splatprep,
                                 with_lightglue=self.plan.with_lightglue,
                                 splatprep_dir=str(self.plan.splatprep_dir) if self.plan.splatprep_dir else None,
                                 verify=verify, python_version=verify.get("python") or PYTHON_RUNTIME.version,
                                 installed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            state.save(self.paths)
            self._log("=== setup finished")
            self._emit("done", verify=verify)
            return InstallResult(ok=True, verify=verify)
        except InstallCancelled:
            self._log("=== cancelled")
            self._emit("cancelled")
            return InstallResult(ok=False, cancelled=True)
        except InstallError as e:
            if self.paths.venv_python.exists() and "has not been changed" not in (e.hint or ""):
                e.hint = ((e.hint or "") + " Your current runtime has not been changed and still works.").strip()
            self._log(f"=== FAILED: {e}\n{e.hint}")
            self._emit("failed", error=e)
            return InstallResult(ok=False, error=e, failed_step=e.step_index)
        except Exception as e:  # pragma: no cover - defensive
            err = InstallError("Unexpected error", str(e) or e.__class__.__name__,
                               "Press Retry. If it fails again, include the log in a bug report.")
            self._log(f"=== FAILED: {err}")
            self._emit("failed", error=err)
            return InstallResult(ok=False, error=err, failed_step=None)
        finally:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None

    # ------------------------------------------------------ preflight / swap
    def _preflight(self) -> None:
        """Fail before downloading gigabytes when the chosen runtime's gsplat build isn't published."""
        from .runtime_plan import gsplat_wheel_published, profile_by_id

        prof = profile_by_id(self.plan.profile_id)
        if prof is not None and gsplat_wheel_published(prof, self.opener, local_dir=self.paths.local_wheels_dir) is False:
            raise InstallError(f"gsplat {prof.gsplat} for {prof.label} is not published yet",
                               f"{prof.gsplat_find_links} has no gsplat-{prof.gsplat}+{prof.gsplat_index} wheel.",
                               "Choose \"PyTorch 2.4 + CUDA 12.4\" in the advanced options, or wait for the next "
                               "Splat Worker update. Your current runtime has not been changed.")

    def _swap_in(self) -> None:
        """Replace the live runtime with the verified staging one (old one removed only after the swap)."""
        new = self.plan.venv_dir
        live = self.paths.venv_dir
        if new is None or Path(new) == Path(live):
            return
        new, live = Path(new), Path(live)
        old = live.with_name(live.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if live.exists():
            os.replace(live, old)
        try:
            os.replace(new, live)
        except OSError:
            if old.exists() and not live.exists():
                os.replace(old, live)   # put the working runtime back
            raise
        shutil.rmtree(old, ignore_errors=True)
        self._log(f"new runtime moved into place: {live}")

    # -------------------------------------------------------------- helpers
    def _emit(self, kind: str, **data: Any) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # never let the UI break the installer
            pass

    def _log(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > 5000:
            del self.lines[:1000]
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()
        self._emit("log", line=line)

    def _env(self) -> Dict[str, str]:
        return base_env(self.paths, self.environ)

    # ------------------------------------------------------------- download
    def _run_download(self, step: Step, progress: Callable[[float, str], None]) -> None:
        target_dir = self.paths.python_dir
        marker = target_dir / ".splatworker-sha256"
        if marker.is_file() and marker.read_text().strip() == step.sha256 and \
                (target_dir / "bin" / "python3").exists():
            self._log("Python already downloaded")
            progress(1.0, "Already downloaded")
            return
        name = step.url.rsplit("/", 1)[-1].replace("%2B", "+")
        archive = self.paths.downloads_dir / name
        if not (archive.is_file() and step.sha256 and _sha256(archive) == step.sha256):
            self._download(step.url, archive, step.sha256, step.size_bytes, progress)
        progress(0.95, "Unpacking Python…")
        if self._cancel.is_set():
            raise InstallCancelled()
        staging = target_dir.with_name(target_dir.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            _safe_untar(archive, staging)
        except (tarfile.TarError, OSError, ValueError) as e:
            shutil.rmtree(staging, ignore_errors=True)
            archive.unlink(missing_ok=True)
            raise InstallError("The Python download could not be unpacked", str(e), "Press Retry.")
        root = staging / "python" if (staging / "python").is_dir() else staging
        shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(root, target_dir)
        shutil.rmtree(staging, ignore_errors=True)
        if not (target_dir / "bin" / "python3").exists():
            raise InstallError("The Python download is incomplete", "bin/python3 missing", "Press Retry.")
        marker.write_text(step.sha256 or "")
        archive.unlink(missing_ok=True)
        self._log(f"Python unpacked to {target_dir}")

    def _download(self, url: str, dest: Path, sha256: Optional[str], size: Optional[int],
                  progress: Callable[[float, str], None]) -> None:
        part = dest.with_name(dest.name + ".part")
        h = hashlib.sha256()
        done = 0
        self._log(f"downloading {url}")
        try:
            with self.opener(url, timeout=60) as resp, open(part, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0) if getattr(resp, "headers", None) else 0
                total = total or (size or 0)
                last = 0.0
                while True:
                    if self._cancel.is_set():
                        raise InstallCancelled()
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last > 0.2:
                        last = now
                        frac = (done / total) if total else 0.0
                        progress(0.9 * frac, f"Downloading Python ({human_bytes(done)}"
                                             f"{' of ' + human_bytes(total) if total else ''})")
        except InstallCancelled:
            part.unlink(missing_ok=True)
            raise
        except OSError as e:
            part.unlink(missing_ok=True)
            err = classify_failure([f"urlopen error {e}"], None, Step("python", "Downloading Python", "", "download"))
            if "no space" in str(e).lower() or getattr(e, "errno", None) == 28:
                err = classify_failure(["No space left on device"], None, Step("python", "", "", "download"))
            err.detail = str(e)
            raise err
        if sha256 and h.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise InstallError("A download was corrupted",
                               f"sha256 mismatch for {dest.name}: got {h.hexdigest()}",
                               "Press Retry to download it again.")
        os.replace(part, dest)

    # -------------------------------------------------------------- command
    def _run_command(self, step: Step, progress: Callable[[float, str], None],
                     env: Optional[Dict[str, str]] = None) -> List[str]:
        parser = PipProgressParser(expected_bytes=step.download_mb * 1_000_000) if step.parser == "pip" else None
        lines: List[str] = []
        self._log("$ " + " ".join(step.argv))
        progress(0.0, step.title + "…")
        try:
            proc = subprocess.Popen(step.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env or self._env(), text=True,
                                    bufsize=1, errors="replace", start_new_session=True)
        except OSError as e:
            raise InstallError(f"{step.title} could not start", str(e),
                               "The runtime looks damaged. Remove it in Settings and run the setup again.")
        with self._lock:
            self._proc = proc
        last_emit = 0.0
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                for line in raw.replace("\r", "\n").splitlines():
                    if not line.strip():
                        continue
                    is_progress = line.startswith("Progress ")
                    if not is_progress:
                        lines.append(line)
                        self._log(line)
                    if parser is not None:
                        st = parser.feed(line)
                        now = time.monotonic()
                        if not is_progress or now - last_emit > 0.2:
                            last_emit = now
                            progress(st.fraction, st.message)
            rc = proc.wait()
        finally:
            with self._lock:
                self._proc = None
        if self._cancel.is_set():
            raise InstallCancelled()
        if rc != 0:
            raise classify_failure(lines, rc, step)
        if parser is not None:
            parser.finish(True)
        return lines

    # --------------------------------------------------------------- verify
    def _run_verify(self, step: Step, progress: Callable[[float, str], None]) -> Dict[str, Any]:
        env = self._env()
        env["PYTHONPATH"] = child_pythonpath(self.lib_dir, self.plan.splatprep_dir)
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        lines = self._run_command(step, progress, env=env)
        result: Dict[str, Any] = {}
        for line in lines:
            if line.startswith(VERIFY_PREFIX):
                try:
                    result = json.loads(line[len(VERIFY_PREFIX):])
                except ValueError:
                    pass
        if not result:
            raise InstallError("The installation check did not run", lines[-1] if lines else "",
                               "Press Retry.")
        problems = verify_problems(result, self.plan)
        if problems:
            raise InstallError("The installation check found a problem", " ".join(problems),
                               "Press Retry, or choose a different runtime in the advanced options.")
        return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_untar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        dest_resolved = dest.resolve()
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise ValueError(f"unsafe path in archive: {m.name}")
            if m.isdev():
                raise ValueError(f"device file in archive: {m.name}")
        if hasattr(tarfile, "data_filter"):
            # 'tar' keeps the symlinks python-build-standalone uses (bin/python3 -> python3.10)
            tf.extractall(dest, filter="tar")
        else:  # pragma: no cover - Python < 3.10.12
            tf.extractall(dest)


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass

    def reaper():
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()

    threading.Thread(target=reaper, daemon=True).start()
