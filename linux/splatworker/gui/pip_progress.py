"""Turn ``pip install --progress-bar raw`` output into a progress fraction and a message.

pip >= 24.1 prints ``Progress <done> of <total>`` lines (bytes of the current download)
with ``--progress-bar raw``. Older pips print nothing per chunk; the parser then only
advances per file, which still gives a usable (if jumpy) bar. Recognised lines::

    Collecting torch==2.4.1
      Downloading torch-2.4.1+cu124-cp310-cp310-linux_x86_64.whl (797.1 MB)
      Downloading https://host/path/torch-2.4.1%2Bcu124-...whl (797.1 MB)   (older pip)
    Progress 104857600 of 797100000
      Using cached torch-2.4.1+cu124-...whl (797.1 MB)
    Requirement already satisfied: numpy in ./venv/lib/python3.10/site-packages
    Building wheel for lightglue (pyproject.toml) ... done
    Installing collected packages: nvidia-cublas-cu12, ..., torch
    Successfully installed nvidia-cublas-cu12-12.4.2.65 ... torch-2.4.1+cu124
    ERROR: No matching distribution found for gsplat==1.5.3+pt24cu124
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import unquote

_SIZE_UNITS = {"bytes": 1, "b": 1, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3,
               "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}

_PROGRESS_RE = re.compile(r"^\s*Progress (\d+) of (\d+)\s*$")
_DOWNLOAD_RE = re.compile(r"^\s*(Downloading|Using cached)\s+(\S+)(?:\s+\(([\d.]+)\s*([kKMG]i?B|bytes)\))?")
_COLLECT_RE = re.compile(r"^\s*Collecting\s+(\S+)")
_SATISFIED_RE = re.compile(r"^\s*Requirement already satisfied:\s+(\S+)")
_BUILDING_RE = re.compile(r"^\s*(?:Building wheel for|Preparing metadata|Getting requirements to build|"
                          r"Installing build dependencies)\s*(?:for\s+)?(\S+)?")
_INSTALLING_RE = re.compile(r"^\s*Installing collected packages:\s*(.*)$")
_SUCCESS_RE = re.compile(r"^\s*Successfully installed\s+(.*)$")
_ERROR_RE = re.compile(r"^\s*ERROR:\s*(.*)$")


def parse_size(number: str, unit: str) -> int:
    return int(float(number) * _SIZE_UNITS[unit.lower()])


def human_bytes(n: float) -> str:
    n = float(n)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if unit != "kB" else f"{n / scale:.0f} {unit}"
    return f"{int(n)} B"


def package_from_file(path: str) -> str:
    """``https://h/torch-2.4.1%2Bcu124-cp310-...whl`` -> ``torch``."""
    name = unquote(path.rsplit("/", 1)[-1])
    for suffix in (".whl.metadata", ".metadata", ".whl", ".tar.gz", ".zip", ".tar.bz2"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    m = re.match(r"^([A-Za-z0-9_.]+?)-\d", name)
    return (m.group(1) if m else name).replace("_", "-")


@dataclass
class PipProgress:
    phase: str = "starting"          # starting | resolving | downloading | building | installing | done | error
    message: str = "Starting…"
    fraction: float = 0.0            # 0..1 for this pip invocation
    bytes_done: int = 0              # all completed + current downloads
    current_package: Optional[str] = None
    current_done: int = 0
    current_total: int = 0
    errors: List[str] = field(default_factory=list)


class PipProgressParser:
    """Feed lines with :meth:`feed`; read :attr:`state`.

    ``expected_bytes`` is the planner's estimate of how much this invocation downloads;
    it is only used to scale the bar (the parser grows it when pip announces more).
    Downloads fill 0-90 % of the bar, "Installing collected packages" jumps to 90 %,
    "Successfully installed" to 100 %.
    """

    DOWNLOAD_SHARE = 0.9

    def __init__(self, expected_bytes: int = 0):
        self.expected = max(0, int(expected_bytes))
        self.state = PipProgress()
        self._completed_bytes = 0     # finished downloads
        self._announced_total = 0     # sum of announced sizes (including current)
        self._current_announced = 0
        self._current_counted = False

    # ------------------------------------------------------------------ api
    def feed(self, line: str) -> PipProgress:
        line = line.rstrip("\r\n")
        s = self.state
        m = _PROGRESS_RE.match(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total and total != s.current_total:
                # raw progress gives the exact size; prefer it over the rounded announcement
                self._announced_total += total - (self._current_announced or 0)
                self._current_announced = total
            s.current_done, s.current_total = done, total or s.current_total
            s.phase = "downloading"
            if total and done >= total:
                self._finish_current()
            self._update_bytes()
            self._set_download_message()
            return s
        m = _DOWNLOAD_RE.match(line)
        if m:
            self._finish_current()
            verb, path = m.group(1), m.group(2)
            size = parse_size(m.group(3), m.group(4)) if m.group(3) else 0
            pkg = package_from_file(path)
            is_metadata = path.endswith(".metadata")
            s.current_package = pkg
            s.current_done = 0
            s.current_total = size
            self._current_announced = size
            self._current_counted = False
            self._announced_total += size
            if verb == "Using cached" or is_metadata:
                # instant: count it as done right away
                s.current_done = size
                self._finish_current()
                if not is_metadata:
                    s.phase = "downloading"
                    s.message = f"Using cached {pkg} ({human_bytes(size)})" if size else f"Using cached {pkg}"
            else:
                s.phase = "downloading"
                self._set_download_message()
            self._update_bytes()
            return s
        m = _COLLECT_RE.match(line)
        if m:
            self._finish_current()
            if s.phase in ("starting", "resolving"):
                s.phase = "resolving"
            s.message = f"Looking up {m.group(1)}"
            return s
        m = _SATISFIED_RE.match(line)
        if m:
            if s.phase == "starting":
                s.phase = "resolving"
            s.message = f"{m.group(1)} is already installed"
            return s
        m = _BUILDING_RE.match(line)
        if m and line.strip().startswith(("Building wheel for", "Preparing metadata",
                                          "Getting requirements", "Installing build dependencies")):
            self._finish_current()
            s.phase = "building"
            s.message = line.strip().rstrip(".").replace(" ... ", " … ")
            return s
        m = _INSTALLING_RE.match(line)
        if m:
            self._finish_current()
            pkgs = [p.strip() for p in m.group(1).split(",") if p.strip()]
            s.phase = "installing"
            s.message = (f"Installing {len(pkgs)} package{'s' if len(pkgs) != 1 else ''} "
                         "(this can take a few minutes)…")
            s.fraction = max(s.fraction, self.DOWNLOAD_SHARE)
            return s
        m = _SUCCESS_RE.match(line)
        if m:
            s.phase = "done"
            s.message = "Installed " + ", ".join(m.group(1).split()[:4]) + (
                "…" if len(m.group(1).split()) > 4 else "")
            s.fraction = 1.0
            return s
        m = _ERROR_RE.match(line)
        if m:
            s.errors.append(m.group(1).strip())
            s.phase = "error"
            s.message = m.group(1).strip()
            return s
        return s

    def finish(self, ok: bool) -> PipProgress:
        if ok:
            self.state.fraction = 1.0
            if self.state.phase != "done":
                self.state.phase = "done"
                self.state.message = "Done"
        return self.state

    # -------------------------------------------------------------- internal
    def _finish_current(self) -> None:
        s = self.state
        if s.current_package is not None and not self._current_counted:
            self._completed_bytes += max(s.current_total, s.current_done)
            self._current_counted = True
        s.current_done = s.current_total
        self._update_bytes()

    def _update_bytes(self) -> None:
        s = self.state
        in_flight = 0 if self._current_counted else s.current_done
        s.bytes_done = self._completed_bytes + in_flight
        denom = max(self.expected, self._announced_total, 1)
        if s.phase in ("installing", "done"):
            return
        s.fraction = max(s.fraction, min(self.DOWNLOAD_SHARE, self.DOWNLOAD_SHARE * s.bytes_done / denom))

    def _set_download_message(self) -> None:
        s = self.state
        if s.current_package is None:
            return
        if s.current_total:
            s.message = (f"Downloading {s.current_package} "
                         f"({human_bytes(s.current_done)} of {human_bytes(s.current_total)})")
        else:
            s.message = f"Downloading {s.current_package}"
