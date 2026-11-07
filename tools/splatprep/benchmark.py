"""Benchmark harness: every available feature backend x one or more datasets.

Datasets (auto-detected per path):

* a prepared/synthetic dataset directory containing ``dataset.json`` (+ ``images/``) — GT poses
  (``splatprep synth`` output is exactly this),
* a ``.splatcapture`` directory or ``.zip`` (PROTOCOL §1) — ARKit poses as reference,
* a plain image directory (or directory with ``images/``) — no GT; only SfM metrics.

Each (dataset, model) run is a separate ``python -m splatprep run`` subprocess by
default (isolation + true per-run peak RSS); ``in_process=True`` runs in this
interpreter (used by tests, peak memory is then the process high-water mark).

Writes ``benchmark.json`` (schema in README.md) and ``benchmark.md``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import __version__
from .backends.base import MatchingData
from .device import select_device
from .errors import SplatPrepError
from .geometry import (
    angle_between_deg,
    evaluate_poses,
    fundamental_from_poses,
    intrinsics_matrix,
    invert_pose,
    pose_auc,
    rotation_angle_deg,
    sampson_distance,
)
from .images import list_images
from .models import model_status
from .priors import _parse_matrix
from .progress import null_progress

ProgressFn = Callable[[dict], None]


@dataclass
class BenchDataset:
    name: str
    path: Path
    image_dir: Path
    names: List[str]
    gt_source: Optional[str] = None  # "synthetic" | "dataset" | "arkit" | None
    gt_c2w: Dict[str, np.ndarray] = field(default_factory=dict)
    gt_K: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "numImages": len(self.names),
            "groundTruth": self.gt_source,
            "numGroundTruthPoses": len(self.gt_c2w),
        }


def _unique_name(base: str, taken: set) -> str:
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    taken.add(name)
    return name


def load_dataset(path: Path, scratch: Path) -> BenchDataset:
    path = Path(path).expanduser().resolve()
    if path.suffix == ".zip":
        dest = scratch / path.stem
        if not dest.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        inner = [p for p in dest.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        path = inner[0] if inner and not (dest / "manifest.json").exists() else dest
    name = path.name.replace(".splatcapture", "")
    ds_json = path / "dataset.json"
    manifest = path / "manifest.json"
    if ds_json.exists():
        data = json.loads(ds_json.read_text())
        image_dir = path / "images"
        gt, Ks = {}, {}
        for cam in data.get("cameras", []):
            rel = cam["image"][len("images/") :] if cam["image"].startswith("images/") else cam["image"]
            gt[rel] = _parse_matrix(cam["cameraToWorld"], False)
            Ks[rel] = intrinsics_matrix(cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        src = "synthetic" if (path / "synth.json").exists() else "dataset"
        names = [n for n in list_images(image_dir) if n in gt] or list_images(image_dir)
        return BenchDataset(name, path, image_dir, names, src, gt, Ks)
    if manifest.exists():
        data = json.loads(manifest.read_text())
        from .geometry import arkit_to_opencv_c2w

        image_dir = path / "images"
        gt, Ks = {}, {}
        for fr in data.get("frames", []):
            if fr.get("trackingState", "normal") != "normal":
                continue
            rel = fr["image"][len("images/") :] if fr["image"].startswith("images/") else fr["image"]
            gt[rel] = arkit_to_opencv_c2w(_parse_matrix(fr["cameraToWorld"], True))
            if fr.get("intrinsics"):
                fx, fy, cx, cy = fr["intrinsics"]
                Ks[rel] = intrinsics_matrix(fx, fy, cx, cy)
        return BenchDataset(name, path, image_dir, list_images(image_dir), "arkit", gt, Ks)
    image_dir = path / "images" if (path / "images").is_dir() else path
    return BenchDataset(name, path, image_dir, list_images(image_dir))


# ---------------------------------------------------------------------------
# metrics


def matching_metrics(md: MatchingData, ds: BenchDataset, thresholds=(5.0, 10.0, 20.0), max_axis_angle_deg: float = 60.0) -> dict:
    """Correctness of matches under the GT epipolar geometry + relative pose from matches.

    Only pairs whose GT optical axes differ by <= ``max_axis_angle_deg`` are scored (pairs looking in
    very different directions usually share no content; exhaustive matching would otherwise be
    dominated by impossible pairs).
    """
    import cv2

    precisions_raw, precisions_ver, n_raw, n_ver, n_correct, pose_errs = [], [], [], [], [], []
    n_skipped = 0
    for (i, j), m in md.matches.items():
        a, b = md.names[i], md.names[j]
        if a not in ds.gt_c2w or b not in ds.gt_c2w:
            continue
        K0 = ds.gt_K.get(a)
        K1 = ds.gt_K.get(b)
        if K0 is None or K1 is None:
            continue
        w0, w1 = invert_pose(ds.gt_c2w[a]), invert_pose(ds.gt_c2w[b])
        # skip pure-rotation pairs (epipolar geometry undefined)
        base = np.linalg.norm(ds.gt_c2w[a][:3, 3] - ds.gt_c2w[b][:3, 3])
        if base < 1e-6:
            continue
        if angle_between_deg(ds.gt_c2w[a][:3, 2], ds.gt_c2w[b][:3, 2]) > max_axis_angle_deg:
            n_skipped += 1
            continue
        F = fundamental_from_poses(K0, K1, w0, w1)
        W, H = md.sizes[i]
        thr = 2.0 * max(1.0, max(W, H) / 1000.0)
        inl = md.inliers.get((i, j), np.zeros(len(m), bool))
        n_raw.append(len(m))
        n_ver.append(int(inl.sum()))
        if len(m) == 0:
            pose_errs.append(np.inf)
            continue
        x0 = md.keypoints[i][m[:, 0], :2]
        x1 = md.keypoints[j][m[:, 1], :2]
        e = sampson_distance(F, x0, x1)
        ok = e < thr
        precisions_raw.append(float(ok.mean()))
        if inl.any():
            precisions_ver.append(float(ok[inl].mean()))
        n_correct.append(int(ok.sum()))
        # relative pose from the (verified, else raw) matches with GT intrinsics
        sel = inl if inl.sum() >= 8 else np.ones(len(m), bool)
        if sel.sum() < 8:
            pose_errs.append(np.inf)
            continue
        p0 = cv2.undistortPoints(x0[sel].reshape(-1, 1, 2).astype(np.float64), K0, None).reshape(-1, 2)
        p1 = cv2.undistortPoints(x1[sel].reshape(-1, 1, 2).astype(np.float64), K1, None).reshape(-1, 2)
        try:
            E, emask = cv2.findEssentialMat(p0, p1, np.eye(3), method=cv2.RANSAC, prob=0.9999, threshold=thr / ((K0[0, 0] + K1[0, 0]) / 2))
            if E is None or E.shape != (3, 3):
                raise cv2.error
            _, R, t, _ = cv2.recoverPose(E, p0, p1, np.eye(3), mask=emask)
        except cv2.error:
            pose_errs.append(np.inf)
            continue
        T = w1 @ invert_pose(w0)
        r_err = rotation_angle_deg(T[:3, :3].T @ R)
        t_err = angle_between_deg(t.ravel(), T[:3, 3])
        pose_errs.append(max(r_err, t_err))
    out = {
        "numPairsEvaluated": len(n_raw),
        "numPairsSkippedNoOverlap": n_skipped,
        "maxAxisAngleDeg": max_axis_angle_deg,
        "pooledInlierRatio": float(np.sum(n_correct) / max(1, np.sum([n for n in n_raw]))) if n_raw else 0.0,
        "meanMatches": float(np.mean(n_raw)) if n_raw else 0.0,
        "meanVerifiedMatches": float(np.mean(n_ver)) if n_ver else 0.0,
        "meanCorrectMatches": float(np.mean(n_correct)) if n_correct else 0.0,
        "meanInlierRatio": float(np.mean(precisions_raw)) if precisions_raw else 0.0,
        "meanVerifiedPrecision": float(np.mean(precisions_ver)) if precisions_ver else 0.0,
        "meanKeypoints": float(np.mean([len(k) for k in md.keypoints])) if md.keypoints else 0.0,
    }
    for t, a in zip(thresholds, pose_auc(pose_errs, thresholds)):
        out[f"pairPoseAucAt{int(t)}"] = round(a, 4)
    return out


def _est_poses_from_dataset_json(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    out = {}
    for cam in data["cameras"]:
        rel = cam["image"][len("images/") :] if cam["image"].startswith("images/") else cam["image"]
        out[rel] = np.asarray(cam["cameraToWorld"], float)
    return out


# ---------------------------------------------------------------------------
# runs


def _child_env() -> dict:
    env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_one(
    ds: BenchDataset,
    model: str,
    run_dir: Path,
    options: dict,
    in_process: bool,
    forward: Callable[[dict], None],
    timeout: Optional[float] = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    request = {
        "task": "sfm",
        "imagesDir": str(ds.image_dir),
        "outputDir": str(out_dir),
        "features": model,
        "matching": options.get("matching", "auto"),
        "mapper": options.get("mapper", "auto"),
        "maxKeypoints": options.get("maxKeypoints", 4096),
        "resizeMax": options.get("resizeMax", 1600),
        "retrievalTopK": options.get("retrievalTopK", 30),
        "sequentialOverlap": options.get("sequentialOverlap", 10),
        "cameraModel": options.get("cameraModel", "auto"),
        "singleCamera": True,
        "exportImages": False,
        "allowMatchingOnly": True,
        "saveMatches": True,
    }
    (run_dir / "request.json").write_text(json.dumps(request, indent=2))
    t0 = time.perf_counter()
    stats: Optional[dict] = None
    error: Optional[str] = None
    if in_process:
        from .runner import run_request

        try:
            stats = run_request(request, run_dir, forward)
        except Exception as e:
            error = str(e)
    else:
        cmd = [sys.executable, "-m", "splatprep", "run", "--request", str(run_dir / "request.json"), "--progress-jsonl", "-", "--workdir", str(run_dir)]
        with open(run_dir / "stderr.log", "w") as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, env=_child_env(), cwd=str(run_dir))
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("done"):
                        stats = msg.get("stats", {})
                    elif "error" in msg:
                        error = msg["error"]
                    else:
                        forward(msg)
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                error = f"timeout after {timeout}s"
        if stats is None and error is None:
            error = f"process exited with {proc.returncode} (see {run_dir / 'stderr.log'})"
    wall = time.perf_counter() - t0
    res: dict = {"model": model, "dataset": ds.name}
    if error is not None:
        res.update({"status": "unavailable" if error.startswith("model unavailable") else "failed", "error": error, "metrics": {"wallTimeSec": round(wall, 3)}})
        return res
    assert stats is not None
    status = stats.get("status", "ok")
    m: dict = {
        "numImages": len(ds.names),
        "numRegistered": int(stats.get("numRegistered", 0)),
        "registeredPct": float(stats.get("registeredPct", 0.0)),
        "meanReprojError": stats.get("meanReprojError"),
        "meanTrackLength": stats.get("meanTrackLength"),
        "numPoints3D": int(stats.get("numPoints3D", 0)),
        "wallTimeSec": float(stats.get("wallTimeSec", wall)),
        "peakMemoryMB": stats.get("peakMemoryMB"),
        "peakGpuMemoryMB": stats.get("peakGpuMemoryMB"),
        "device": stats.get("device"),
        "mapper": stats.get("mapper"),
        "numPairs": stats.get("numPairs") or (stats.get("pairs") or {}).get("numPairs"),
    }
    if status == "ok" and ds.gt_c2w and (out_dir / "dataset.json").exists():
        est = _est_poses_from_dataset_json(out_dir / "dataset.json")
        gt = {n: ds.gt_c2w[n] for n in ds.names if n in ds.gt_c2w}
        m["pose"] = evaluate_poses(est, gt)
    if (out_dir / "matches.npz").exists() and ds.gt_c2w and ds.gt_K:
        try:
            m["matching"] = matching_metrics(MatchingData.load(out_dir / "matches.npz"), ds)
        except Exception as e:  # pragma: no cover
            m["matching"] = {"error": str(e)}
    res.update({"status": status, "metrics": m})
    return res


def _summary(results: List[dict]) -> Dict[str, dict]:
    by_model: Dict[str, List[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    out = {}
    for model, rs in by_model.items():
        ok = [r for r in rs if r["status"] in ("ok", "matching-only")]

        def mean(path):
            vals = []
            for r in ok:
                v = r["metrics"]
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                    vals.append(float(v))
            return round(float(np.mean(vals)), 4) if vals else None

        out[model] = {
            "status": "ok" if any(r["status"] == "ok" for r in rs) else (rs[0]["status"] if rs else "unknown"),
            "numDatasets": len(rs),
            "numSucceeded": len(ok),
            "meanRegisteredPct": mean(["registeredPct"]),
            "meanReprojError": mean(["meanReprojError"]),
            "meanTrackLength": mean(["meanTrackLength"]),
            "meanNumPoints3D": mean(["numPoints3D"]),
            "meanWallTimeSec": mean(["wallTimeSec"]),
            "maxPeakMemoryMB": max([r["metrics"].get("peakMemoryMB") or 0 for r in ok], default=None),
            "meanAucAt5": mean(["pose", "aucAt5"]),
            "meanAucAt10": mean(["pose", "aucAt10"]),
            "meanAteRmseRel": mean(["pose", "ateRmseRel"]),
            "meanInlierRatio": mean(["matching", "meanInlierRatio"]),
            "meanPairPoseAucAt10": mean(["matching", "pairPoseAucAt10"]),
        }
    return out


def _fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown_report(report: dict) -> str:
    lines = [f"# splatprep benchmark — {report['createdAt']}", ""]
    h = report["host"]
    lines.append(f"Host: {h['platform']} {h['machine']}, device `{h['device']}`, Python {h['python']}, splatprep {report['splatprepVersion']}")
    lines.append("")
    for ds in report["datasets"]:
        lines.append(f"## {ds['name']} ({ds['numImages']} images, ground truth: {ds['groundTruth'] or 'none'})")
        lines.append("")
        lines.append("| model | status | reg % | reproj px | track len | #3D pts | AUC@5° | AUC@10° | ATE (rel) | match inlier ratio | pair-pose AUC@10° | time s | peak MB |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in report["results"]:
            if r["dataset"] != ds["name"]:
                continue
            m = r.get("metrics", {})
            p = m.get("pose") or {}
            mm = m.get("matching") or {}
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    r["model"],
                    r["status"] + (f" ({r['error'][:60]})" if r.get("error") else ""),
                    _fmt(m.get("registeredPct"), 1),
                    _fmt(m.get("meanReprojError")),
                    _fmt(m.get("meanTrackLength")),
                    _fmt(m.get("numPoints3D")),
                    _fmt(p.get("aucAt5"), 3),
                    _fmt(p.get("aucAt10"), 3),
                    _fmt(p.get("ateRmseRel"), 4),
                    _fmt(mm.get("meanInlierRatio"), 3),
                    _fmt(mm.get("pairPoseAucAt10"), 3),
                    _fmt(m.get("wallTimeSec"), 1),
                    _fmt(m.get("peakMemoryMB"), 0),
                )
            )
        lines.append("")
    lines.append("## Summary (mean over datasets)")
    lines.append("")
    lines.append("| model | status | reg % | AUC@5° | AUC@10° | inlier ratio | time s | peak MB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, s in report["summary"].items():
        lines.append(
            f"| {model} | {s['status']} | {_fmt(s['meanRegisteredPct'], 1)} | {_fmt(s['meanAucAt5'], 3)} | {_fmt(s['meanAucAt10'], 3)} | "
            f"{_fmt(s['meanInlierRatio'], 3)} | {_fmt(s['meanWallTimeSec'], 1)} | {_fmt(s['maxPeakMemoryMB'], 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    datasets: List[Path],
    out_dir: Path,
    models: Optional[List[str]] = None,
    options: Optional[dict] = None,
    in_process: bool = False,
    progress: Optional[ProgressFn] = None,
    timeout: Optional[float] = None,
) -> dict:
    progress = progress or null_progress
    options = options or {}
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_datasets"
    status = model_status()
    feature_ids = [m for m, s in status.items() if s["kind"] != "mask"]
    if not models or models == ["all"]:
        models = [m for m in feature_ids if m != "mock"]
    taken: set = set()
    dsets = []
    for p in datasets:
        d = load_dataset(Path(p), scratch)
        d.name = _unique_name(d.name, taken)
        dsets.append(d)
    if not dsets:
        raise SplatPrepError("benchmark: no datasets given")
    results: List[dict] = []
    total = len(dsets) * len(models)
    k = 0
    for ds in dsets:
        for model in models:
            st = status.get(model)
            base = k / total
            if st is None or not st["available"]:
                results.append(
                    {"model": model, "dataset": ds.name, "status": "unavailable", "error": (st or {}).get("reason", "unknown model"), "metrics": {}}
                )
                k += 1
                continue
            progress({"stage": "benchmark", "progress": round(base, 4), "message": f"{ds.name}: {model}"})

            def forward(msg, base=base):
                p = float(msg.get("progress", 0.0))
                progress({"stage": "benchmark", "progress": round(base + p / total, 4), "message": f"{ds.name}/{model}: {msg.get('message', '')}"})

            r = run_one(ds, model, out_dir / "runs" / ds.name / model, options, in_process, forward, timeout)
            results.append(r)
            k += 1
    report = {
        "format": "splatbenchmark",
        "version": 1,
        "splatprepVersion": __version__,
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "device": select_device(),
            "cpuCount": os.cpu_count(),
        },
        "options": options,
        "inProcess": in_process,
        "models": {m: status.get(m, {"available": False, "sfm": False, "reason": "unknown"}) for m in models},
        "datasets": [d.to_json() for d in dsets],
        "results": results,
        "summary": _summary(results),
    }
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=_json_default))
    (out_dir / "benchmark.md").write_text(markdown_report(report))
    progress({"stage": "benchmark", "progress": 1.0, "message": f"wrote {out_dir / 'benchmark.json'}"})
    return report


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_benchmark_request(req: dict, workdir: Path, progress: ProgressFn) -> dict:
    """task = "benchmark": ``datasets`` (list of paths, default: the working dir) and ``models``."""
    datasets = req.get("datasets")
    if not datasets:
        datasets = [str(workdir)] if ((workdir / "dataset.json").exists() or (workdir / "manifest.json").exists()) else [req["imagesDir"]]
    paths = [Path(d) if Path(d).is_absolute() else workdir / d for d in datasets]
    out = Path(req["outputDir"])
    out = out if out.is_absolute() else workdir / out
    models = req.get("models")
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    opts = {k: req[k] for k in ("matching", "mapper", "maxKeypoints", "resizeMax", "retrievalTopK", "sequentialOverlap", "cameraModel") if k in req}
    rep = run_benchmark(paths, out, models, opts, bool(req.get("inProcess", False)), progress)
    return {"task": "benchmark", "status": "ok", "benchmark": str(out / "benchmark.json"), "summary": rep["summary"]}
