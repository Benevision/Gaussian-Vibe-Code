"""ProcessingRequest execution (PROTOCOL §6).

``run_request(request, workdir, progress) -> stats`` is the single entry point
used by the CLI and by the Linux worker.  Relative paths in the request are
resolved against ``workdir``.  Errors are raised (``ModelUnavailable``,
``SplatPrepError`` …); the caller turns them into ``{"error": ...}``.

Extra (optional) request keys understood here, beyond §6:

* ``masksDir``        — keep-masks used to drop keypoints (sfm).
* ``exportImages``    — default true; false skips copying/undistorting images.
* ``allowMatchingOnly`` — sfm without a mapper returns matching stats + ``matches.npz``
  instead of failing (used by the benchmark).
* ``device``          — "auto" | "cuda" | "mps" | "cpu".
* ``datasets`` / ``models`` — task = benchmark (see benchmark.py).
* mask keys (task = masks / maskpreview) — ``maskPrompts``, ``maskClasses``, ``maskThreshold``,
  ``dilatePixels``, ``replaceMasks``, ... (see masks/prompts.py and README).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .backends.base import SFM_STAGES, SfMContext
from .device import peak_gpu_mb, peak_rss_mb, select_device
from .errors import RequestError, SplatPrepError
from .export import export_dataset
from .geometry import robust_umeyama
from .images import list_images
from .models import get_backend
from .priors import load_priors, match_priors
from .progress import StagedProgress, null_progress

ProgressFn = Callable[[dict], None]

DEFAULTS = {
    "imagesDir": "images",
    "outputDir": "out",
    "features": "superpoint_lightglue",
    "matching": "auto",
    "retrievalTopK": 30,
    "sequentialOverlap": 10,
    "mapper": "auto",
    "cameraModel": "auto",
    "singleCamera": True,
    "maxKeypoints": 4096,
    "resizeMax": 1600,
    "invertMask": False,
}


def _resolve(workdir: Path, p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    q = Path(p).expanduser()
    return q if q.is_absolute() else (workdir / q)


def normalize_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise RequestError("request must be a JSON object")
    req = dict(DEFAULTS)
    req.update({k: v for k, v in request.items() if v is not None})
    task = req.get("task")
    if task not in ("sfm", "masks", "benchmark", "maskpreview", "frames"):
        raise RequestError(f"unknown task: {task!r} (expected sfm | masks | benchmark | maskpreview | frames)")
    if task == "frames" and not req.get("video"):
        raise RequestError("task frames needs 'video'")
    if task == "sfm" and req["matching"] not in ("sequential", "exhaustive", "retrieval", "auto"):
        raise RequestError(f"unknown matching: {req['matching']!r}")
    if task == "sfm" and req["mapper"] not in ("glomap", "colmap", "pycolmap", "auto"):
        raise RequestError(f"unknown mapper: {req['mapper']!r}")
    return req


def run_request(request: dict, workdir: Path, progress: Optional[ProgressFn] = None) -> dict:
    progress = progress or null_progress
    workdir = Path(workdir).resolve()
    req = normalize_request(request)
    t0 = time.perf_counter()
    if req["task"] == "sfm":
        stats = run_sfm(req, workdir, progress)
    elif req["task"] == "masks":
        stats = run_mask_task(req, workdir, progress)
    elif req["task"] == "maskpreview":
        stats = run_maskpreview_task(req, workdir, progress)
    elif req["task"] == "frames":
        stats = run_frames_task(req, workdir, progress)
    else:
        from .benchmark import run_benchmark_request

        stats = run_benchmark_request(req, workdir, progress)
    stats["wallTimeSec"] = round(time.perf_counter() - t0, 3)
    stats["peakMemoryMB"] = peak_rss_mb()
    g = peak_gpu_mb()
    if g is not None:
        stats["peakGpuMemoryMB"] = g
    return stats


# ---------------------------------------------------------------------------


def align_to_priors(rec, priors: dict) -> Optional[dict]:
    names = [im.name for im in rec.images.values() if im.name in priors]
    if len(names) < 3:
        return None
    by_name = rec.image_by_name()
    src = np.stack([by_name[n].center() for n in names])
    dst = np.stack([priors[n].c2w[:3, 3] for n in names])
    if np.linalg.matrix_rank(src - src.mean(0), tol=1e-6) < 1:
        return None
    T, inl = robust_umeyama(src, dst)
    rec.transform(T)
    res = np.linalg.norm(np.stack([by_name[n].center() for n in names]) - dst, axis=1)
    return {**T.to_json(), "numCorrespondences": len(names), "inliers": int(inl.sum()), "rmse": float(np.sqrt(np.mean(res[inl] ** 2)))}


def run_sfm(req: dict, workdir: Path, progress: ProgressFn) -> dict:
    model_id = req["features"]
    backend = get_backend(model_id)
    image_dir = _resolve(workdir, req["imagesDir"])
    out_dir = _resolve(workdir, req["outputDir"])
    assert image_dir is not None and out_dir is not None
    names = list_images(image_dir)
    if len(names) < 2:
        raise SplatPrepError(f"need at least 2 images in {image_dir}, found {len(names)}")
    allow_mo = bool(req.get("allowMatchingOnly", False))
    backend.require(None if not allow_mo else _AllowMO)  # fail fast with "model unavailable: ..."

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir.parent / f".splatprep_work_{out_dir.name}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    priors = {}
    if req.get("priorPoses"):
        ppath = _resolve(workdir, req["priorPoses"])
        if ppath is None or not ppath.exists():
            raise RequestError(f"priorPoses not found: {req['priorPoses']}")
        priors = match_priors(load_priors(ppath), names)
    sp = StagedProgress(progress, SFM_STAGES)
    device = select_device(req.get("device"))
    sp.message("pairs", f"{model_id}: {len(names)} images, device {device}")
    ctx = SfMContext(
        image_dir=image_dir,
        names=names,
        work_dir=work,
        request=req,
        progress=sp,
        device=device,
        priors=priors,
        masks_dir=_resolve(workdir, req.get("masksDir")),
        sky_masks_dir=_resolve(workdir, req.get("skyMasksDir")),
        allow_matching_only=allow_mo,
        log_path=out_dir / "splatprep.log",
    )
    result = backend.run(ctx)
    stats: dict = {"task": "sfm", "model": model_id, "device": device, "numImages": len(names), **result.info}
    if result.reconstruction is None:
        if result.matching is not None:
            result.matching.save(out_dir / "matches.npz")
        stats["status"] = "matching-only"
        stats["numRegistered"] = 0
        return stats

    rec = result.reconstruction
    stats.update(rec.stats())
    stats["registeredPct"] = round(100.0 * len(rec.images) / len(names), 2)
    if result.matching is not None and req.get("saveMatches"):
        result.matching.save(out_dir / "matches.npz")
    if priors:
        sp.update("alignment", 0.1, "aligning to prior poses (Sim3)")
        stats["priorAlignment"] = align_to_priors(rec, priors)
        stats["numPriors"] = len(priors)
    known_up = None
    if not priors and req.get("gpsCheck", True):
        from .gps import check_against_gps

        sp.update("alignment", 0.1, "checking camera positions against GPS")
        chk = check_against_gps(rec, image_dir, names)
        if chk is not None:
            T = chk.pop("_T")
            stats["gpsCheck"] = chk
            if chk["ok"]:
                rec.transform(T)  # metres, +Z up (east / north / up)
                known_up = np.array([0.0, 0.0, 1.0])
                # The few cameras far from their GPS position are mis-registered (e.g. a take-off shot of the
                # horizon placed 290 m away); one such camera alone can wreck a training run's framing.
                drop = set(chk["outliers"])
                if drop:
                    for iid in [i for i, im in rec.images.items() if im.name in drop]:
                        del rec.images[iid]
                    stats["numRegistered"] = len(rec.images)
                    stats["registeredPct"] = round(100.0 * len(rec.images) / len(names), 2)
                sp.message("alignment", f"camera positions match GPS (median {chk['medianM']} m over {chk['extentM']} m) — scene in metres, Z up"
                                        + (f"; dropped {len(drop)} camera(s) far from their GPS position" if drop else ""))
            else:
                sp.message("alignment", f"WARNING: {chk['badFraction']:.0%} of cameras are more than {chk['limitM']} m from their GPS position "
                                        f"(flight {chk['extentM']} m) — the solve is wrong in places; scene left in SfM units")
    sp.update("alignment", 1.0, "alignment done")
    exp = export_dataset(
        rec,
        image_dir,
        out_dir,
        work,
        sp.sub("export"),
        export_images=bool(req.get("exportImages", True)),
        log_path=out_dir / "splatprep.log",
        extra={"model": model_id, "mapper": stats.get("mapper"), "registeredPct": stats["registeredPct"]},
        up=known_up,
    )
    stats.update({"export": exp, "status": "ok"})
    if not req.get("keepWork", False):
        shutil.rmtree(work, ignore_errors=True)
    return stats


class _AllowMO:  # minimal stand-in so Backend.require() accepts matching-only
    allow_matching_only = True


def run_mask_task(req: dict, workdir: Path, progress: ProgressFn) -> dict:
    from .masks import run_masks

    model = req.get("maskModel") or ("sam2_track" if req.get("maskPrompts") else "sam2_auto")
    image_dir = _resolve(workdir, req["imagesDir"])
    out_dir = _resolve(workdir, req["outputDir"])
    names = list_images(image_dir)
    sp = StagedProgress(progress, [("masks", 1.0)])
    device = select_device(req.get("device"))
    sp.message("masks", f"{model}: {len(names)} images, device {device}")
    info = run_masks(model, image_dir, names, out_dir, device, sp.sub("masks"), req.get("maskPrompt"), bool(req.get("invertMask")), request=req)
    return {"task": "masks", "device": device, "status": "ok", **info}


def run_frames_task(req: dict, workdir: Path, progress: ProgressFn) -> dict:
    """Video → ``outputDir/images`` (+ ``masks``) and ``frames.json`` — the Mac app's frame extraction."""
    from .frames import extract_frames

    video = _resolve(workdir, req["video"])
    if not video.is_file():
        raise RequestError(f"video not found: {req['video']}")
    out_dir = _resolve(workdir, req["outputDir"])
    stereo = req.get("stereo")
    right = None
    if stereo is not None:
        if not isinstance(stereo, dict):
            raise RequestError("stereo must be an object")
        if stereo.get("rightVideo"):
            right = _resolve(workdir, stereo["rightVideo"])
            if not right.is_file():
                raise RequestError(f"right-eye video not found: {stereo['rightVideo']}")
    stage = ("splitting 360° video" if req.get("equirect") else
             "extracting stereo frames" if stereo is not None else "extracting frames")
    sp = StagedProgress(progress, [(stage, 1.0)])
    device = select_device(req.get("device"))
    info = extract_frames(video, out_dir, req.get("videoOptions") or {}, req.get("equirect"), device,
                          sp.sub(stage), lambda m: sp.message(stage, m), stereo=stereo, right_video=right)
    return {"task": "frames", "device": device, "status": "ok", **info}


def run_maskpreview_task(req: dict, workdir: Path, progress: ProgressFn) -> dict:
    """Single-frame SAM 2 image prediction → ``outputDir/preview.png`` (click-to-mask UI)."""
    from .masks.server import run_preview_request

    image_dir = _resolve(workdir, req["imagesDir"])
    out_dir = _resolve(workdir, req["outputDir"])
    device = select_device(req.get("device"))
    progress({"stage": "masks", "progress": 0.0, "message": f"maskpreview on {device}"})
    stats = run_preview_request(req, image_dir, out_dir, device)
    progress({"stage": "masks", "progress": 1.0, "message": f"preview written: {stats['preview']}"})
    return {"device": device, **stats}
