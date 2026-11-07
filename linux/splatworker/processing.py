"""Processing jobs (PROTOCOL.md §6): run the shared ``splatprep`` toolkit.

``splatprep`` is a separate package (``tools/splatprep`` in the repo). It is
imported lazily; when it is missing the job fails with a clear message and
``/v1/info`` simply does not advertise the ``process`` capability.
"""

from __future__ import annotations

import importlib.util
import os
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

PROGRESS_SCALE = 1000  # step/totalSteps = progress x 1000


class ProcessingError(RuntimeError):
    pass


def splatprep_available() -> bool:
    try:
        return importlib.util.find_spec("splatprep") is not None
    except (ImportError, ValueError):
        return False


def frames_available() -> bool:
    """Video → frames (splatprep ``task: frames``) needs PyAV, installed with the processing models."""
    try:
        return splatprep_available() and importlib.util.find_spec("av") is not None
    except (ImportError, ValueError):
        return False


def available_models() -> List[str]:
    """``splatprep.models.available_models()`` if importable, else []."""
    if not splatprep_available():
        return []
    try:
        from splatprep.models import available_models as _am  # type: ignore

        models = _am()
        return [str(m) for m in models]
    except Exception:
        return []


def _safe_rel(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ProcessingError(f"request.{name} must be a relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ProcessingError(f"request.{name} must not escape the working directory")
    return str(p)


def validate_request(request: Any) -> Dict[str, Any]:
    """Light validation; splatprep owns the full schema. Unknown keys pass through."""
    if not isinstance(request, dict):
        raise ProcessingError("request must be a JSON object")
    task = request.get("task")
    if task not in ("sfm", "masks", "benchmark", "frames"):
        raise ProcessingError("request.task must be one of 'sfm', 'masks', 'benchmark', 'frames'")
    req = dict(request)
    req["imagesDir"] = _safe_rel(req.get("imagesDir", "images"), "imagesDir")
    req["outputDir"] = _safe_rel(req.get("outputDir", "out"), "outputDir")
    if req.get("priorPoses") is not None:
        req["priorPoses"] = _safe_rel(req["priorPoses"], "priorPoses")
    if task == "frames":
        req["video"] = _safe_rel(req.get("video"), "video")
    return req


def prepare_workdir(upload_root: Path, workdir: Path) -> None:
    """Populate ``workdir`` with symlinks to the (read-only) uploaded dataset.

    Outputs are written into ``workdir`` so the uploaded dataset is never
    modified and can be reused by other jobs.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for entry in Path(upload_root).iterdir():
        link = workdir / entry.name
        if not link.exists() and not link.is_symlink():
            os.symlink(entry.resolve(), link, target_is_directory=entry.is_dir())


def _inside(path: Path, roots: List[Path]) -> bool:
    return any(path == r or r in path.parents for r in roots)


def zip_output(out_dir: Path, zip_path: Path, allowed_roots: List[Path]) -> int:
    """Zip ``out_dir`` with its contents at the archive root.

    Symlinks (e.g. images linked back to the upload) are followed only when
    they resolve inside ``allowed_roots``; directory cycles are skipped.
    Returns the number of files written.
    """
    out_dir = Path(out_dir)
    roots = [Path(r).resolve() for r in allowed_roots]
    count = 0
    seen_dirs = set()
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for dirpath, dirnames, filenames in os.walk(out_dir, followlinks=True):
            real = Path(dirpath).resolve()
            if real in seen_dirs or not _inside(real, roots):
                dirnames[:] = []
                continue
            seen_dirs.add(real)
            rel_dir = Path(dirpath).relative_to(out_dir)
            for fn in sorted(filenames):
                target = (Path(dirpath) / fn).resolve()
                if not _inside(target, roots) or not target.is_file():
                    continue
                arc = str(PurePosixPath(*(rel_dir / fn).parts))
                stored = fn.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".zip"))
                zf.write(target, arcname=arc,
                         compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
                count += 1
    os.replace(tmp, zip_path)
    return count


def run_processing(upload_root: Path, request: Dict[str, Any], job_dir: Path,
                   progress_cb, log, should_stop) -> Dict[str, Any]:
    """Run ``splatprep.runner.run_request`` and zip its output directory."""
    req = validate_request(request)
    if not splatprep_available():
        raise ProcessingError("splatprep is not installed on this worker "
                              "(pip install -e ../tools/splatprep, see README)")
    try:
        from splatprep.runner import run_request  # type: ignore
    except Exception as e:
        raise ProcessingError(f"failed to import splatprep.runner: {e}") from e

    workdir = job_dir / "work"
    prepare_workdir(upload_root, workdir)
    reported_error: List[str] = []

    def progress(evt: Dict[str, Any]) -> None:
        if should_stop():
            raise KeyboardInterrupt("cancelled")
        if not isinstance(evt, dict):
            return
        if evt.get("error"):
            reported_error.append(str(evt["error"]))
            log(f"error: {evt['error']}")
            return
        if evt.get("done"):
            return
        stage = str(evt.get("stage", "processing"))
        frac = evt.get("progress")
        detail = str(evt.get("message") or "").strip()
        # "stage: detail" (e.g. "masks: SAM 2 forward: frame 120/649") so the Mac can show what is
        # actually happening; state "running" once splatprep reports (not "loading" for the whole job).
        fields: Dict[str, Any] = {"message": f"{stage}: {detail}" if detail else stage,
                                  "totalSteps": PROGRESS_SCALE, "state": "running"}
        if isinstance(frac, (int, float)) and not isinstance(frac, bool):
            fields["step"] = int(round(max(0.0, min(1.0, float(frac))) * PROGRESS_SCALE))
        progress_cb(**fields)
        if evt.get("message"):
            log(f"[{stage}] {evt['message']}")

    log(f"splatprep request: {req}")
    result = run_request(req, workdir, progress)
    if isinstance(result, dict) and result.get("error"):
        raise ProcessingError(str(result["error"]))
    if reported_error:
        raise ProcessingError(reported_error[-1])

    out_dir = workdir / req["outputDir"]
    if not out_dir.is_dir():
        raise ProcessingError(f"splatprep produced no output directory '{req['outputDir']}'")
    n = zip_output(out_dir, job_dir / "output.zip", [workdir, Path(upload_root)])
    log(f"output.zip written ({n} files)")
    stats = result.get("stats") if isinstance(result, dict) else None
    return {"files": n, "stats": stats}


def subprocess_entry(upload_root: str, request: dict, job_dir: str, device_index,
                     queue, cancel_event) -> None:
    if device_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    try:
        stats = run_processing(
            Path(upload_root), request, Path(job_dir),
            progress_cb=lambda **f: queue.put(("status", f)),
            log=lambda m: queue.put(("log", m)),
            should_stop=cancel_event.is_set,
        )
        queue.put(("done", stats))
    except KeyboardInterrupt:
        queue.put(("cancelled", None))
    except BaseException as e:  # noqa: BLE001
        queue.put(("log", traceback.format_exc()))
        queue.put(("error", str(e) or e.__class__.__name__))
