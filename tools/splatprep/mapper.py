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

from .colmap_model import Reconstruction, find_models, read_model, read_largest_model
from .device import has_module
from .errors import MapperUnavailable, SplatPrepError

_EXTRA_PATHS = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]

#: With ``mapper: auto``, sets larger than this use global mapping (GLOMAP / colmap global_mapper);
#: smaller ones use COLMAP's incremental mapper, which is more robust and still fast at that size.
AUTO_GLOBAL_MIN_IMAGES = 300
#: With ``mapper: auto`` and at most this many images, both mappers run and the better result is kept.
AUTO_BOTH_MAX = 1000
#: Cameras flagged with a prior focal length (EXIF / ARKit) keep it fixed while mapping when at least this
#: fraction of them has one. Free focal lengths collapse on flat scenes — a nadir drone survey shrank a
#: 2795 px EXIF focal to 573 px and registered 33/120 photos — because depth and focal trade off there.
FIX_FOCAL_MIN_PRIOR = 0.9
#: After mapping with fixed focal lengths, one bundle adjustment frees them; it is kept only if no focal
#: moves by more than this fraction (EXIF values are usually within a few percent).
FOCAL_REFINE_MAX_DRIFT = 0.08


def _largest_model_dir(sparse_root: Path) -> Optional[Path]:
    best, n = None, -1
    for p in find_models(sparse_root):
        k = len(read_model(p).images)
        if k > n:
            best, n = p, k
    return best


def _refine_focal(colmap: str, out_dir: Path, log_path, progress) -> Optional[str]:
    """Free focal lengths in a final bundle adjustment; keep it only if they stay near the priors."""
    src = _largest_model_dir(out_dir)
    if src is None:
        return None
    before = read_model(src)
    dst = out_dir.parent / (out_dir.name + "_focal_refined")
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    progress(0.96, "refining focal length")
    try:
        run_logged([colmap, "bundle_adjuster", "--input_path", src, "--output_path", dst,
                    "--BundleAdjustment.refine_focal_length", "1", "--BundleAdjustment.refine_extra_params", "1"],
                   log_path, None)
        after = read_model(dst)
    except Exception as e:  # noqa: BLE001 — keep the fixed-focal model
        return f"focal refinement failed ({e}); kept EXIF focal"
    drift = max((abs(after.cameras[c].params[0] / before.cameras[c].params[0] - 1)
                 for c in before.cameras if c in after.cameras), default=1.0)
    if drift > FOCAL_REFINE_MAX_DRIFT:
        shutil.rmtree(dst, ignore_errors=True)
        return f"focal refinement moved focal {drift:.0%}; kept EXIF focal"
    for f in src.iterdir():
        f.unlink()
    for f in dst.iterdir():
        shutil.move(str(f), src / f.name)
    dst.rmdir()
    return f"focal refined ({drift:.1%} from EXIF)"


def _prior_focal_fraction(database, default: float = 1.0) -> float:
    """Fraction of cameras in a COLMAP database flagged with a prior focal length (``default`` if unreadable)."""
    import sqlite3

    try:
        with sqlite3.connect(str(database)) as c:
            rows = c.execute("SELECT prior_focal_length FROM cameras").fetchall()
        return sum(1 for (p,) in rows if p) / len(rows) if rows else default
    except Exception:
        return default


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
    """Run the mapper; returns (largest reconstruction, mapper name used).

    ``auto`` with up to AUTO_BOTH_MAX images runs the incremental *and* the global mapper on the same matches
    and keeps the better reconstruction: more registered images first, then the longer mean track length,
    then the lower reprojection error. On the synthetic benchmark neither mapper won everywhere (incremental:
    orbit AUC@5° 0.52–0.96 vs 0.12–0.29; global: walk 0.90–0.96 vs 0.14–0.50 for learned matchers), and the
    longer track length picked the right one in every pair. Mapping is cheap next to matching.
    """
    progress = progress or (lambda f, m="": None)
    pref = (preference or "auto").lower()
    avail = available_mappers()
    inc = next((m for m in ("colmap", "pycolmap") if avail.get(m)), None)
    if pref != "auto" or num_images > AUTO_BOTH_MAX or not (inc and avail.get("glomap")):
        return _run_single(database, image_dir, out_dir, preference, log_path, progress, num_images)
    out_dir = Path(out_dir)
    results = []
    # incremental first: the global path may calibrate focal lengths in the shared database
    for name, sub, lo, hi in ((inc, "incremental", 0.0, 0.55), ("glomap", "global", 0.55, 1.0)):
        sub_p = (lambda f, m="", lo=lo, hi=hi, name=name: progress(lo + (hi - lo) * f, f"{name}: {m}"))
        try:
            rec, used = _run_single(database, image_dir, out_dir / sub, name, log_path, sub_p, num_images)
            st = rec.stats()
            results.append((rec, used, st))
            progress(hi, f"{used}: {st.get('numRegistered')} images, track length {st.get('meanTrackLength') or 0:.2f}")
        except Exception as e:  # noqa: BLE001 — one mapper failing is fine if the other works
            progress(hi, f"{name} failed: {e}")
    if not results:
        raise SplatPrepError("both mappers failed to produce a reconstruction")

    def key(r):
        st = r[2]
        reg = int(st.get("numRegistered") or 0)
        return (round(reg / max(num_images or reg or 1, 1) * 50),   # ~2 % registration buckets
                float(st.get("meanTrackLength") or 0), -float(st.get("meanReprojError") or 1e9))
    best = max(results, key=key)
    if log_path:
        with open(log_path, "a") as fh:
            for r in results:
                fh.write(f"auto mapper candidate {r[1]}: {r[2]}\n")
            fh.write(f"auto mapper chose {best[1]}\n")
    progress(1.0, f"chose {best[1]} (registered {best[2].get('numRegistered')}, track length {best[2].get('meanTrackLength') or 0:.2f})")
    return best[0], f"auto:{best[1]}"


def _run_single(
    database: Path,
    image_dir: Path,
    out_dir: Path,
    preference: str = "auto",
    log_path: Optional[Path] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    num_images: int = 0,
) -> tuple[Reconstruction, str]:
    """Run one mapper (by preference order); returns (largest reconstruction, mapper name used)."""
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
    fix_focal = _prior_focal_fraction(database, default=0.0) >= FIX_FOCAL_MIN_PRIOR
    note = None

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
        if fix_focal and not glomap:
            cmd += ["--GlobalMapper.ba_refine_focal_length", "0"]
        progress(0.05, "global mapping (GLOMAP)")
        run_logged(cmd, log_path, on_line)
        if fix_focal and not glomap:
            note = _refine_focal(cmd[0], out_dir, log_path, progress)
    elif chosen == "colmap":
        progress(0.05, "incremental mapping (COLMAP)")
        colmap = find_executable("colmap")
        cmd = [colmap, "mapper", "--database_path", database, "--image_path", image_dir, "--output_path", out_dir]
        if fix_focal:
            cmd += ["--Mapper.ba_refine_focal_length", "0"]
        run_logged(cmd, log_path, on_line)
        if fix_focal:
            note = _refine_focal(colmap, out_dir, log_path, progress)
    else:
        progress(0.05, "incremental mapping (pycolmap)")
        run_pycolmap_subprocess(database, image_dir, out_dir, log_path, fix_focal)
    rec = read_largest_model(out_dir)
    if rec is None or not rec.images:
        raise SplatPrepError("mapper produced no reconstruction (too few matches / images could not be registered)")
    if note:
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(note + "\n")
    progress(1.0, f"registered {len(rec.images)} images" + (f"; {note}" if note else ""))
    return rec, chosen


_PYCOLMAP_SCRIPT = r"""
import sys, pathlib, pycolmap
db, img, out, fix = sys.argv[1:5]
opts = pycolmap.IncrementalPipelineOptions()
if fix == "1":
    opts.ba_refine_focal_length = False
maps = pycolmap.incremental_mapping(database_path=db, image_path=img, output_path=out, options=opts)
for idx, rec in maps.items():
    d = pathlib.Path(out) / str(idx)
    d.mkdir(parents=True, exist_ok=True)
    rec.write(str(d))
print(f"Registered {max((len(r.images) for r in maps.values()), default=0)} images")
"""


def run_pycolmap_subprocess(database, image_dir, out_dir, log_path, fix_focal: bool = False) -> None:
    """pycolmap mapping in a child interpreter, so its libomp never meets torch's (see isolation.py)."""
    import subprocess
    cmd = [sys.executable, "-c", _PYCOLMAP_SCRIPT, str(database), str(image_dir), str(out_dir), "1" if fix_focal else "0"]
    run_logged(cmd, log_path, lambda line: None)
