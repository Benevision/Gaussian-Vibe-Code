"""SfM mappers: GLOMAP CLI, COLMAP CLI (``mapper`` / ``global_mapper``) and pycolmap.

Executables are looked up on PATH (plus Homebrew prefixes) and can be forced
with ``SPLATPREP_COLMAP`` / ``SPLATPREP_GLOMAP``.
"""

from __future__ import annotations

import sys
import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .colmap_model import Reconstruction, read_largest_model
from .device import has_module
from .errors import MapperUnavailable, SplatPrepError

_EXTRA_PATHS = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]

#: With ``mapper: auto``, sets larger than this use global mapping (GLOMAP / colmap global_mapper);
#: smaller ones use COLMAP's incremental mapper, which is more robust and still fast at that size.
AUTO_GLOBAL_MIN_IMAGES = 300


def _prior_focal_fraction(database) -> float:
    """Fraction of cameras in a COLMAP database flagged with a prior focal length (1.0 if unreadable)."""
    import sqlite3

    try:
        with sqlite3.connect(str(database)) as c:
            rows = c.execute("SELECT prior_focal_length FROM cameras").fetchall()
        return sum(1 for (p,) in rows if p) / len(rows) if rows else 1.0
    except Exception:
        return 1.0


def find_executable(name: str) -> Optional[str]:
    env = os.environ.get(f"SPLATPREP_{name.upper()}")
    if env and Path(env).exists():
        return env
    p = shutil.which(name)
    if p:
        return p
    for d in _EXTRA_PATHS:
        c = Path(d) / name
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


@functools.lru_cache(maxsize=None)
def colmap_commands(colmap: str) -> frozenset:
    try:
        out = subprocess.run([colmap, "help"], capture_output=True, text=True, timeout=30)
        text = out.stdout + out.stderr
    except Exception:
        return frozenset()
    return frozenset(re.findall(r"^\s+([a-z_]+)\s*$", text, re.M)) | frozenset(re.findall(r"\b([a-z]+_[a-z_]+)\b", text))


@functools.lru_cache(maxsize=None)
def colmap_options(colmap: str, command: str) -> frozenset:
    """Option names accepted by ``colmap <command>`` (parsed from -h)."""
    try:
        out = subprocess.run([colmap, command, "-h"], capture_output=True, text=True, timeout=30)
        text = out.stdout + out.stderr
    except Exception:
        return frozenset()
    return frozenset(re.findall(r"--([A-Za-z_][A-Za-z0-9_.]*)", text))


def pick_option(colmap: str, command: str, *candidates: str) -> Optional[str]:
    """First candidate option that this COLMAP version accepts (handles 3.9–3.13 renames)."""
    opts = colmap_options(colmap, command)
    for c in candidates:
        if c in opts:
            return c
    return candidates[0] if not opts else None


def available_mappers() -> Dict[str, bool]:
    colmap = find_executable("colmap")
    return {
        "glomap": bool(find_executable("glomap")) or bool(colmap and "global_mapper" in colmap_commands(colmap)),
        "colmap": bool(colmap),
        "pycolmap": has_module("pycolmap"),
    }


def any_mapper_available() -> bool:
    return any(available_mappers().values())


def run_logged(cmd: Sequence[str], log_path: Optional[Path], on_line: Optional[Callable[[str], None]] = None, cwd=None) -> None:
    """Run a subprocess, appending stdout/stderr to log_path; never writes to our stdout."""
    log = open(log_path, "a") if log_path else None
    try:
        if log:
            log.write("$ " + " ".join(map(str, cmd)) + "\n")
            log.flush()
        proc = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd, bufsize=1)
        tail: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            if log:
                log.write(line)
            tail.append(line)
            tail = tail[-40:]
            if on_line:
                on_line(line)
        rc = proc.wait()
        if rc != 0:
            raise SplatPrepError(f"{Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''} failed (exit {rc}): " + "".join(tail[-8:]).strip())
    finally:
        if log:
            log.close()


def run_mapper(
    database: Path,
    image_dir: Path,
    out_dir: Path,
    preference: str = "auto",
    log_path: Optional[Path] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    num_images: int = 0,
) -> tuple[Reconstruction, str]:
    """Run the mapper; returns (largest reconstruction, mapper name used)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    avail = available_mappers()
    preference = (preference or "auto").lower()
    order = {
        "glomap": ["glomap", "colmap", "pycolmap"],
        "colmap": ["colmap", "pycolmap", "glomap"],
        "pycolmap": ["pycolmap", "colmap", "glomap"],
        # auto: incremental up to AUTO_GLOBAL_MIN_IMAGES — it is only seconds-to-minutes there and far
        # more robust (benchmark: synthetic 360° orbit without EXIF focal, AUC@5° 0.98 incremental vs
        # 0.38 global even after view-graph calibration). Global mapping pays off on large sets.
        "auto": ["glomap", "colmap", "pycolmap"] if num_images > AUTO_GLOBAL_MIN_IMAGES else ["colmap", "pycolmap", "glomap"],
    }.get(preference, ["glomap", "colmap", "pycolmap"])
    chosen = next((m for m in order if avail.get(m)), None)
    if chosen is None:
        raise MapperUnavailable("no SfM mapper available: install pycolmap (pip install pycolmap) or COLMAP/GLOMAP (brew install colmap glomap)")
    progress = progress or (lambda f, m="": None)
    registered = [0]

    def on_line(line: str):
        m = re.search(r"Registering image #(\d+)", line) or re.search(r"Registered (\d+) images", line)
        if m:
            registered[0] += 1
            if num_images:
                progress(min(0.95, registered[0] / num_images), f"registered {registered[0]}/{num_images}")

    if chosen == "glomap":
        glomap = find_executable("glomap")
        if glomap:
            cmd = [glomap, "mapper", "--database_path", database, "--image_path", image_dir, "--output_path", out_dir]
        else:
            colmap = find_executable("colmap")
            cmd = [colmap, "global_mapper", "--database_path", database, "--image_path", image_dir, "--output_path", out_dir]
            if "view_graph_calibrator" in colmap_commands(colmap) and _prior_focal_fraction(database) < 0.5:
                # COLMAP's global mapper needs focal priors; estimate them from the view graph first
                # (video frames and EXIF-less images have none).
                progress(0.03, "calibrating focal length from the view graph")
                try:
                    run_logged([colmap, "view_graph_calibrator", "--database_path", database], log_path, None)
                except SplatPrepError:
                    pass  # fall through: global mapper still runs with the initial guess
        progress(0.05, "global mapping (GLOMAP)")
        run_logged(cmd, log_path, on_line)
    elif chosen == "colmap":
        progress(0.05, "incremental mapping (COLMAP)")
        run_logged(
            [find_executable("colmap"), "mapper", "--database_path", database, "--image_path", image_dir, "--output_path", out_dir],
            log_path,
            on_line,
        )
    else:
        progress(0.05, "incremental mapping (pycolmap)")
        run_pycolmap_subprocess(database, image_dir, out_dir, log_path)
    rec = read_largest_model(out_dir)
    if rec is None or not rec.images:
        raise SplatPrepError("mapper produced no reconstruction (too few matches / images could not be registered)")
    progress(1.0, f"registered {len(rec.images)} images")
    return rec, chosen


_PYCOLMAP_SCRIPT = r"""
import sys, pathlib, pycolmap
db, img, out = sys.argv[1:4]
maps = pycolmap.incremental_mapping(database_path=db, image_path=img, output_path=out)
for idx, rec in maps.items():
    d = pathlib.Path(out) / str(idx)
    d.mkdir(parents=True, exist_ok=True)
    rec.write(str(d))
print(f"Registered {max((len(r.images) for r in maps.values()), default=0)} images")
"""


def run_pycolmap_subprocess(database, image_dir, out_dir, log_path) -> None:
    """pycolmap mapping in a child interpreter, so its libomp never meets torch's (see isolation.py)."""
    import subprocess
    cmd = [sys.executable, "-c", _PYCOLMAP_SCRIPT, str(database), str(image_dir), str(out_dir)]
    run_logged(cmd, log_path, lambda line: None)
