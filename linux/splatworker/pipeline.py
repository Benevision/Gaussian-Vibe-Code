"""Whole-pipeline jobs (PROTOCOL.md §4, "Whole pipeline jobs", capability ``pipeline``).

A client that processes *and* trains a project on this worker sends the source (a video, or frames it ingested
itself) and every setting once; this module then runs each stage here on its own — frames → fixed parts of the
frame → masks → sky → camera solve → dataset assembly + levelling → training — so the client can go offline for
the whole job and only download the result (and the prepared dataset) when it comes back.

Each GPU step is an ordinary child job of the server's queue (``process`` / ``train``, ``parentJobId`` = the
pipeline), so it shares the per-GPU queue, priorities and pause / resume with every other job. The CPU steps between
them follow SplatWork's ``DatasetBuilder`` exactly (mask combining, sky layer, ``PoseSanity``, 360° rig completion,
``PoseAssembly``, levelling — ``Core/Orientation.swift``), so a project prepared here matches one prepared on the Mac.

State lives in the job's ``stages`` list (persisted in ``job.json``) and in the pipeline's working dataset
(``spec.workDatasetId``): after a worker restart the job picks up at the first unfinished stage, re-attaching to a
child job that is still queued / paused or running it again when the restart interrupted it.

The working dataset becomes the job's ``outputDatasetId`` as soon as it is ready to train, with
``.pipeline/`` snapshots of each stage (the frame list after the frames and after the masks, the solved dataset
before levelling) so the client can record the same per-stage checkpoints a local run leaves.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("splatworker")

PROGRESS_SCALE = 1000
SNAPSHOTS = ".pipeline"
STAGE_TITLES = {
    "frames": "Extracting frames",
    "static": "Finding fixed parts of the frame",
    "masks": "Masks",
    "sky": "Finding the sky",
    "sfm": "Camera solving",
    "level": "Finalising dataset",
    "train": "Training",
}
# share of the whole job (training's share is what is left), for the overall progress bar
WEIGHTS = {"frames": 0.12, "static": 0.02, "masks": 0.08, "sky": 0.04, "sfm": 0.30, "level": 0.01, "train": 0.43}
# what the Mac's project page calls each stage's checkpoint (Pipeline/PrepCheckpoints.swift PrepStage)
PREP_STAGE = {"frames": "frames", "static": "masks", "masks": "masks", "sky": "masks", "sfm": "poses",
              "level": "done", "train": None}
MAX_ATTEMPTS = 3          # a child job interrupted by a worker restart is run again (at most this often)


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(Exception):
    pass


class PipelineStopped(Exception):
    """The worker is shutting down: leave the job as it is, it continues after the restart."""


# ---------------------------------------------------------------------------
# Validation (POST /v1/pipeline)
# ---------------------------------------------------------------------------

def validate_spec(body: Dict[str, Any], upload_root: Path) -> Dict[str, Any]:
    """The ``pipeline`` object of the request, checked and normalised. Raises PipelineError (→ 400)."""
    from .config import ConfigError, TrainingConfig
    from .processing import ProcessingError, validate_request

    p = body.get("pipeline")
    if not isinstance(p, dict):
        raise PipelineError("pipeline must be an object")
    out: Dict[str, Any] = {}

    def req(section: str, task: str) -> Optional[Dict[str, Any]]:
        s = p.get(section)
        if s is None:
            return None
        if not isinstance(s, dict) or not isinstance(s.get("request"), dict):
            raise PipelineError(f"pipeline.{section} must be an object with a request")
        r = dict(s["request"], task=task)
        try:
            r = validate_request(r)
        except ProcessingError as e:
            raise PipelineError(f"pipeline.{section}.request: {e}")
        return dict(s, request=r)

    out["frames"] = req("frames", "frames")
    if out["frames"] is None:
        fs = upload_root / "frames.json"
        if not fs.is_file():
            raise PipelineError("without a frames stage the upload must hold images/ and frames.json")
        try:
            frames = json.loads(fs.read_text(encoding="utf-8"))
        except ValueError as e:
            raise PipelineError(f"frames.json: {e}")
        if not isinstance(frames, dict) or not isinstance(frames.get("frames"), list) or not frames["frames"]:
            raise PipelineError("frames.json has no frames")
        for f in frames["frames"]:
            if not isinstance(f, dict) or not isinstance(f.get("image"), str):
                raise PipelineError("frames.json: every frame needs an image")
            for k in ("image", "mask", "sky", "depth"):
                if f.get(k) is not None:
                    _safe_rel(f[k])
    out["staticRegions"] = bool(p.get("staticRegions", False))
    out["masks"] = req("masks", "masks")
    if out["masks"] is not None:
        out["masks"]["dilatePixels"] = max(0, min(64, int(out["masks"].get("dilatePixels", 4) or 0)))
    out["sky"] = req("sky", "masks")
    sfm = req("sfm", "sfm")
    if sfm is None:
        raise PipelineError("pipeline.sfm is required")
    sfm["sequential"] = bool(sfm.get("sequential", False))
    sfm["enforceRig"] = bool(sfm.get("enforceRig", True))
    out["sfm"] = sfm
    hi = p.get("holdoutInterval")
    out["holdoutInterval"] = int(hi) if isinstance(hi, (int, float)) and not isinstance(hi, bool) and hi > 1 else None
    out["level"] = bool(p.get("level", True))
    train = p.get("train")
    if train is not None:
        if not isinstance(train, dict):
            raise PipelineError("pipeline.train must be an object")
        try:
            cfg = TrainingConfig.from_dict(train.get("config"))
        except ConfigError as e:
            raise PipelineError(f"pipeline.train.config: {e}")
        out["train"] = {"config": {k: v for k, v in cfg.to_dict().items() if k in cfg.explicit},
                        "iterations": cfg.iterations}
    else:
        out["train"] = None
    cps = p.get("checkpoints")
    out["checkpoints"] = {str(k): str(v) for k, v in cps.items()} if isinstance(cps, dict) else {}
    return out


def _safe_rel(value: str) -> str:
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(part == ".." for part in p.parts) or "\\" in value:
        raise PipelineError(f"path escapes the dataset: {value}")
    return value


def initial_stages(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = []
    if spec.get("frames"):
        ids.append("frames")
    ids.append("static")          # also runs for circular fisheye footage, found once the frames are there
    if spec.get("masks"):
        ids.append("masks")
    if spec.get("sky"):
        ids.append("sky")
    ids += ["sfm", "level"]
    if spec.get("train"):
        ids.append("train")
    return [{"id": i, "title": STAGE_TITLES[i], "state": "pending", "jobId": None, "message": None,
             "prepStage": PREP_STAGE[i], "attempts": 0} for i in ids]


# ---------------------------------------------------------------------------
# Small helpers (images, matrices, PLY)
# ---------------------------------------------------------------------------

def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _stem(rel: str) -> str:
    return PurePosixPath(rel).stem


def _gray(path: Path, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        g = im.convert("L")
        if g.size != size:
            g = g.resize(size, Image.BILINEAR)
        return np.asarray(g, dtype=np.uint8).copy()


def _write_gray(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    Image.fromarray(arr.astype(np.uint8), "L").save(tmp)
    tmp.replace(path)


def erode(keep: np.ndarray, radius: int) -> np.ndarray:
    """Min filter over a (2r+1)² square, edges extended (vImageMin_Planar8 on the Mac)."""
    if radius <= 0:
        return keep
    from PIL import Image, ImageFilter

    return np.asarray(Image.fromarray(keep, "L").filter(ImageFilter.MinFilter(2 * radius + 1)), dtype=np.uint8)


def circular_fisheye(paths: List[Path]) -> bool:
    """FrameSet.looksCircularFisheye: most sampled frames have ≥ 2 uniformly black corners and a bright centre."""
    from PIL import Image

    if not paths:
        return False
    picks = paths[::max(1, len(paths) // 5)][:5]
    votes = seen = 0
    for p in picks:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((64, 64))
                a = np.asarray(im, dtype=np.int32)
        except Exception:  # noqa: BLE001
            continue
        seen += 1
        h, w = a.shape[:2]
        if w < 16 or h < 16 or abs(w / h - 1) >= 0.35:
            continue
        v = a.max(axis=2)
        c, r = max(2, w // 10), max(2, h // 10)
        corners = [v[0:r, 0:c], v[0:r, w - c:w], v[h - r:h, 0:c], v[h - r:h, w - c:w]]
        black = sum(1 for q in corners if q.mean() < 10 and q.max() < 40)
        if black >= 2 and v[h // 3:2 * h // 3, w // 3:2 * w // 3].mean() > 30:
            votes += 1
    return seen > 0 and votes * 2 > seen


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(x, y, z, w) of a rotation matrix (columns normalised first, as `orthonormalized` does)."""
    m = m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-9)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orientation.rotation(from:to:): the shortest rotation taking unit vector a onto b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 0.99999:
        return np.eye(3)
    if d < -0.99999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-3:
            axis = np.cross(a, [0, 0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)       # half turn about axis
    v = np.cross(a, b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + d))


def estimate_up(c2ws: List[np.ndarray]) -> Optional[np.ndarray]:
    """Orientation.estimateUp (OpenCV cameras): see Core/Orientation.swift for the reasoning."""
    if not c2ws:
        return None
    avg_up = np.zeros(3)
    avg_fwd = np.zeros(3)
    S = np.zeros((3, 3))
    for m in c2ws:
        f = m[:3, 2] / max(np.linalg.norm(m[:3, 2]), 1e-12)
        avg_up -= m[:3, 1]
        avg_fwd += f
        S += np.outer(f, f)
    n = float(len(c2ws))

    def unit(v):
        return v / np.linalg.norm(v)

    if np.linalg.norm(avg_up) / n > 0.7:
        return unit(avg_up)
    vals, vecs = np.linalg.eigh(S / n)            # ascending
    if vals[1] > 0.15 and vals[0] < 0.6 * vals[1]:
        u = vecs[:, 0].copy()
        vote = 0.0
        for m in c2ws:
            cam_up, cam_left = -m[:3, 1], -m[:3, 0]
            a, b = float(np.dot(cam_up, u)), float(np.dot(cam_left, u))
            d = a if abs(a) >= abs(b) else b
            if abs(d) > 0.6:
                vote += d
        if vote == 0:
            vote = float(np.dot(avg_up, u))
        if vote < 0:
            u = -u
        return unit(u)
    if np.linalg.norm(avg_up) / n > 0.3:
        return unit(avg_up)
    if np.linalg.norm(avg_fwd) / n > 0.5:
        return -unit(avg_fwd)
    return unit(avg_up) if np.linalg.norm(avg_up) > 1e-6 else None


def read_points(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    from .ply import read_ply_vertices

    v = read_ply_vertices(path)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(k in v for k in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def write_points(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray]) -> None:
    """Binary little-endian x y z (float) + red green blue (uchar), as PLYCodec.writePointCloud."""
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 128, np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec = np.zeros(n, dtype=dt)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header.encode())
        fh.write(rec.tobytes())
    tmp.replace(path)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Stage logic on a working folder (pure functions: tested directly)
# ---------------------------------------------------------------------------

def load_frames(work: Path) -> Dict[str, Any]:
    fs = json.loads((work / "frames.json").read_text(encoding="utf-8"))
    fs["root"] = "."
    fs.setdefault("rejected", [])
    fs.setdefault("isMultiViewRig", False)
    for f in fs["frames"]:
        f.setdefault("sharpness", 0.0)
        f["width"], f["height"] = int(f["width"]), int(f["height"])
    return fs


def save_frames(work: Path, fs: Dict[str, Any]) -> None:
    _write_json(work / "frames.json", fs)


def combine_model_masks(work: Path, fs: Dict[str, Any], layer: Path, dilate: int, model: str) -> Dict[str, int]:
    """FrameMasker.applySplatPrep after the model ran: binarise, refuse when most masks keep nothing, leave
    frames whose mask is empty unmasked, erode the keep region, AND with the frame's mask, write masks/<stem>.png."""
    def path(f):
        return layer / (_stem(f["image"]) + ".png")

    empty, seen = set(), 0
    for i, f in enumerate(fs["frames"]):
        m = path(f)
        if not m.is_file():
            continue
        keep = _gray(m, (f["width"], f["height"]))
        seen += 1
        if (keep >= 128).sum() < 0.01 * max(keep.size, 1):
            empty.add(i)
    if seen and len(empty) > 0.5 * seen:
        hint = ("“Isolate one object” keeps only a single central object — for rooms and scenes choose “Remove "
                "moving things” or no masks." if model == "sam2_auto" else "Try a different mask mode or no masks.")
        raise PipelineError(f"Masking with {model} kept almost nothing in {len(empty)} of {seen} frames, so it "
                            f"was not applied. {hint}")
    applied = 0
    for i, f in enumerate(fs["frames"]):
        m = path(f)
        if i in empty or not m.is_file():
            continue
        keep = np.where(_gray(m, (f["width"], f["height"])) >= 128, 255, 0).astype(np.uint8)
        keep = erode(keep, dilate)
        f["mask"] = _and_into_frame_mask(work, f, keep)
        applied += 1
    return {"masked": applied, "empty": len(empty), "seen": seen}


def _and_into_frame_mask(work: Path, f: Dict[str, Any], keep: np.ndarray) -> str:
    """FrameMasker.combineAndWrite."""
    old = f.get("mask")
    if old and (work / old).is_file():
        keep = np.minimum(keep, _gray(work / old, (f["width"], f["height"])))
    rel = f"masks/{_stem(f['image'])}.png"
    _write_gray(keep, work / rel)
    return rel


def apply_static_layer(work: Path, fs: Dict[str, Any], layer: Path) -> int:
    """The fixed-parts layer (255 = keep) ANDed into each frame's mask."""
    n = 0
    for f in fs["frames"]:
        m = layer / (_stem(f["image"]) + ".png")
        if not m.is_file():
            continue
        keep = _gray(m, (f["width"], f["height"]))
        if keep.min() >= 128:
            continue
        f["mask"] = _and_into_frame_mask(work, f, np.where(keep >= 128, 255, 0).astype(np.uint8))
        n += 1
    return n


def record_sky(work: Path, fs: Dict[str, Any]) -> int:
    """FrameMasker.applySky after the model ran: frames with > 0.2 % sky (on a 64 px wide copy) get ``sky``."""
    d = work / "splatprep-sky" / "masks"
    n = 0
    for f in fs["frames"]:
        stem = _stem(f["image"])
        p = d / f"{stem}.png"
        if not p.is_file():
            continue
        w = 64
        h = max(8, 64 * f["height"] // max(f["width"], 1))
        px = _gray(p, (w, h))
        if (px >= 128).sum() > 0.002 * px.size:
            f["sky"] = f"splatprep-sky/masks/{stem}.png"
            n += 1
    return n


def sfm_request(base: Dict[str, Any], fs: Dict[str, Any], fisheye: bool) -> Tuple[Dict[str, Any], bool]:
    """SplatPrepPoseSolver.request(for:) + the circular-fisheye camera model. Returns (request, preferOutput)."""
    r = dict(base)
    r.update(task="sfm", imagesDir="images", outputDir="sfm")
    frames = fs["frames"]
    multiview = bool(fs.get("isMultiViewRig"))
    if fisheye and not multiview and r.get("cameraModel") in ("pinhole", "auto", None):
        r["cameraModel"] = "fisheye"
    r["singleCamera"] = len({(f["width"], f["height"]) for f in frames}) == 1 and fs.get("rig") is None
    if any(f.get("mask") for f in frames):
        r["masksDir"] = "masks"
    if any(f.get("sky") for f in frames):
        r["skyMasksDir"] = "splatprep-sky/masks"
    if multiview:
        groups: Dict[int, Dict[str, Any]] = {}
        for f in frames:
            if f.get("group") is not None and f.get("rotationToRig") is not None:
                groups.setdefault(int(f["group"]), {})[PurePosixPath(f["image"]).name] = f["rotationToRig"]
        hint: Dict[str, Any] = {"kind": "multiview", "groups": [groups[k] for k in sorted(groups)]}
        f0 = frames[0]
        fp = f0.get("focalPx")
        if fp and all(f.get("focalPx") == fp and f["width"] == f0["width"] and f["height"] == f0["height"]
                      for f in frames):
            hint["intrinsics"] = [fp, fp, f0.get("cx") or f0["width"] / 2, f0.get("cy") or f0["height"] / 2]
        r["rig"] = hint
    return r, r.get("cameraModel") != "pinhole"


def read_solution(work: Path, fs: Dict[str, Any], prefix: str = "sfm",
                  prefer_output: bool = False) -> Tuple[Dict[str, Dict[str, Any]], Optional[Path]]:
    """SplatPrepPoseSolver.readSolution: {frame image: pose} and the solve's point cloud."""
    sfm = work / prefix
    raw = json.loads((sfm / "dataset.json").read_text(encoding="utf-8"))
    undistorted = bool((raw.get("splatprep") or {}).get("undistorted", False))
    by_base: Dict[str, Dict[str, Any]] = {}
    for f in fs["frames"]:
        name = PurePosixPath(f["image"]).name
        by_base[name] = f
        by_base[PurePosixPath(name).stem] = f
    poses: Dict[str, Dict[str, Any]] = {}
    for c in raw.get("cameras", []):
        base = PurePosixPath(c["image"]).name
        f = by_base.get(base) or by_base.get(PurePosixPath(base).stem)
        if f is None:
            continue
        p: Dict[str, Any] = {"c2w": np.asarray(c["cameraToWorld"], float).reshape(4, 4),
                             "fx": c.get("fx"), "fy": c.get("fy"), "cx": c.get("cx"), "cy": c.get("cy")}
        exported = (sfm / c["image"]).is_file()
        if exported and (prefer_output or c["width"] != f["width"] or c["height"] != f["height"]):
            p.update(image=f"{prefix}/{c['image']}", width=int(c["width"]), height=int(c["height"]))
            if undistorted:
                p["undistorted"] = True
                p["mask"] = f"{prefix}/{c['mask']}" if c.get("mask") else None
        poses[f["image"]] = p
    pts = raw.get("points")
    points = sfm / pts if pts and (sfm / pts).is_file() else None
    return poses, points


def pose_outliers(positions: List[np.ndarray], ordered: bool) -> set:
    """PoseSanity.outliers."""
    n = len(positions)
    if n < 8:
        return set()
    P = np.asarray(positions, float)

    def median(v):
        return float(np.sort(np.asarray(v))[len(v) // 2])

    centre = np.array([median(P[:, 0]), median(P[:, 1]), median(P[:, 2])])
    dist = np.linalg.norm(P - centre, axis=1)
    typical = max(median(dist), 1e-9)
    drop = {i for i in range(n) if dist[i] > 20 * typical}
    if ordered:
        steps = np.linalg.norm(P[1:] - P[:-1], axis=1)
        step = max(median(steps), 1e-9)
        for i in range(1, n - 1):
            if steps[i - 1] > 10 * step and steps[i] > 10 * step and np.linalg.norm(P[i + 1] - P[i - 1]) < 4 * step:
                drop.add(i)
    return drop


def pose_sanity(poses: Dict[str, Dict[str, Any]], fs: Dict[str, Any], ordered: bool) -> int:
    """PoseSanity.apply: one track per eye / view, in the frames' order."""
    tracks: Dict[str, List[str]] = {}
    for f in fs["frames"]:
        if f["image"] in poses:
            tracks.setdefault(f.get("eye") or "", []).append(f["image"])
    removed = 0
    for names in tracks.values():
        for i in pose_outliers([poses[n]["c2w"][:3, 3] for n in names], ordered):
            poses.pop(names[i], None)
            removed += 1
    return removed


def enforce_multiview_rig(poses: Dict[str, Dict[str, Any]], fs: Dict[str, Any], min_registered: int = 2) -> int:
    """RigConstraints.enforceMultiViewRig: each 360° frame's rig pose from its registered views (mean rotation,
    mean centre), then every view of the frame re-derived from it — fills views the solve missed."""
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for f in fs["frames"]:
        if f.get("group") is not None and f.get("rotationToRig") is not None:
            groups.setdefault(int(f["group"]), []).append(f)
    added = 0
    for views in groups.values():
        qs, centres = [], []
        for v in views:
            p = poses.get(v["image"])
            if p is None:
                continue
            Rc = np.asarray(v["rotationToRig"], float)
            qs.append(_quat_from_matrix(p["c2w"][:3, :3] @ Rc.T))
            centres.append(p["c2w"][:3, 3])
        if len(qs) < min_registered:
            continue
        acc = np.zeros(4)
        for q in qs:
            acc += -q if np.dot(q, qs[0]) < 0 else q
        R_rig = _matrix_from_quat(acc / np.linalg.norm(acc))
        c = np.mean(centres, axis=0)
        for v in views:
            Rc = np.asarray(v["rotationToRig"], float)
            p = poses.get(v["image"])
            if p is None:
                p = {"fx": None, "fy": None, "cx": None, "cy": None}
                poses[v["image"]] = p
                added += 1
            m = np.eye(4)
            m[:3, :3] = R_rig @ Rc
            m[:3, 3] = c
            p["c2w"] = m
    return added


def recompute_bounds(doc: Dict[str, Any]) -> None:
    """PreparedDataset.recomputeSceneBounds."""
    cams = doc["cameras"]
    if not cams:
        return
    c2ws = [np.asarray(c["cameraToWorld"], float) for c in cams]
    centres = np.array([m[:3, 3] for m in c2ws])
    mean = centres.mean(axis=0)
    r = float(np.max(np.linalg.norm(centres - mean, axis=1)))
    doc["sceneCenter"] = [float(x) for x in mean]
    doc["sceneExtent"] = max(r * 1.1, 1e-3)
    g = doc.get("gravityUp")
    if isinstance(g, list) and len(g) == 3:
        doc["upVector"] = g
    else:
        u = estimate_up(c2ws)
        if u is not None:
            doc["upVector"] = [float(x) for x in u]


def assemble(work: Path, fs: Dict[str, Any], poses: Dict[str, Dict[str, Any]], points: Optional[Path],
             holdout: Optional[int]) -> Dict[str, Any]:
    """PoseAssembly.makeDataset + DatasetAssembly.finalize → ``dataset.json`` (+ ``points3d.ply``)."""
    cams = []
    for i, f in enumerate(fs["frames"]):
        p = poses.get(f["image"])
        if p is None:
            continue
        w, h = p.get("width") or f["width"], p.get("height") or f["height"]
        fx = p.get("fx") or f.get("focalPx") or float(max(w, h))
        c: Dict[str, Any] = {"id": i, "image": p.get("image") or f["image"], "width": int(w), "height": int(h),
                             "fx": float(fx), "fy": float(p.get("fy") or p.get("fx") or fx),
                             "cx": float(p.get("cx") or f.get("cx") or w / 2),
                             "cy": float(p.get("cy") or f.get("cy") or h / 2),
                             "cameraToWorld": [[float(x) for x in row] for row in np.asarray(p["c2w"], float)]}
        if p.get("undistorted"):
            if p.get("mask"):
                c["mask"] = p["mask"]
        elif p.get("image") is None or (p.get("width") == f["width"] and p.get("height") == f["height"]):
            for k in ("mask", "sky", "depth", "depthWidth", "depthHeight"):
                if f.get(k) is not None:
                    c[k] = f[k]
        cams.append(c)
    if not cams:
        raise PipelineError("no camera could be registered")
    cams.sort(key=lambda c: _natural(c["image"]))
    for i, c in enumerate(cams):
        c["id"] = i
        if holdout and holdout > 1 and i % holdout == 0:
            c["holdout"] = True
    doc: Dict[str, Any] = {"format": "splatdataset", "version": 1, "convention": "opencv",
                           "sceneCenter": [0.0, 0.0, 0.0], "sceneExtent": 1.0, "upVector": [0.0, -1.0, 0.0],
                           "cameras": cams}
    if points is not None:
        xyz, rgb = read_points(points)
        if len(xyz):
            write_points(work / "points3d.ply", xyz, rgb)
            doc["points"] = "points3d.ply"
    recompute_bounds(doc)
    _write_json(work / "dataset.json", doc)
    return doc


def level(work: Path) -> Optional[List[float]]:
    """PreparedDataset.normalizeOrientation: estimated up → −Y, scene centre → origin. Returns the up used."""
    doc = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
    g = doc.get("gravityUp")
    c2ws = [np.asarray(c["cameraToWorld"], float) for c in doc["cameras"]]
    up = np.asarray(g, float) / np.linalg.norm(g) if isinstance(g, list) and len(g) == 3 else estimate_up(c2ws)
    if up is None:
        return None
    R = rotation_between(up, np.array([0.0, -1.0, 0.0]))
    centre = np.asarray(doc.get("sceneCenter") or [0, 0, 0], float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ centre
    for c, m in zip(doc["cameras"], c2ws):
        c["cameraToWorld"] = [[float(x) for x in row] for row in M @ m]
    pts = doc.get("points")
    if pts and (work / pts).is_file():
        xyz, rgb = read_points(work / pts)
        write_points(work / pts, ((xyz.astype(np.float64) - centre) @ R.T).astype(np.float32), rgb)
    recompute_bounds(doc)
    doc["upVector"] = [0.0, -1.0, 0.0]
    if doc.get("gravityUp") is not None:
        doc["gravityUp"] = [0.0, -1.0, 0.0]
    _write_json(work / "dataset.json", doc)
    return [float(x) for x in up]


def snapshot_frames(work: Path, fs: Dict[str, Any], name: str) -> None:
    """``.pipeline/<name>.frames.json`` and copies of the mask / sky files it names (``.pipeline/<name>.files/``):
    later stages write combined masks under the same names."""
    snap = work / SNAPSHOTS
    files = snap / f"{name}.files"
    shutil.rmtree(files, ignore_errors=True)
    for f in fs["frames"]:
        for k in ("mask", "sky"):
            rel = f.get(k)
            if rel and (work / rel).is_file():
                dst = files / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / rel, dst)
    _write_json(snap / f"{name}.frames.json", fs)


# ---------------------------------------------------------------------------
# The runner (one thread per active pipeline job)
# ---------------------------------------------------------------------------

class PipelineRunner:
    poll_seconds = 0.25

    def __init__(self, manager, job):
        self.m = manager
        self.job = job
        self.spec = job.spec["pipeline"]
        self.settings = manager.settings

    # ------------------------------------------------------------- helpers
    @property
    def stages(self) -> List[Dict[str, Any]]:
        return self.job.status["stages"]

    def _work_id(self) -> str:
        return self.job.spec["workDatasetId"]

    def _work(self) -> Path:
        from .server import dataset_root

        root = dataset_root(self.settings, self._work_id())
        if root is None:
            raise PipelineError("the pipeline's working dataset is gone")
        return root

    def _log(self, msg: str) -> None:
        self.job.log(msg)

    def _check(self) -> None:
        if self.job.cancel_event.is_set() or self.job.deleted:
            raise PipelineCancelled()
        if self.m.stopping:
            raise PipelineStopped()

    def _overall(self, stage_id: str, fraction: float) -> int:
        ids = [s["id"] for s in self.stages]
        total = sum(WEIGHTS[i] for i in ids) or 1.0
        before = 0.0
        for s in self.stages:
            if s["id"] == stage_id:
                break
            before += WEIGHTS[s["id"]]
        return int(round((before + WEIGHTS[stage_id] * max(0.0, min(1.0, fraction))) / total * PROGRESS_SCALE))

    def _set_stage(self, st: Dict[str, Any], persist: bool = True, **fields) -> None:
        with self.job.lock:
            st.update(fields)
        self.job.update(persist=persist, stages=self.stages)

    def _progress(self, st: Dict[str, Any], fraction: float, message: str, **extra) -> None:
        if self.job.cancel_event.is_set():
            return
        self.job.update(stage=st["id"], step=self._overall(st["id"], fraction), totalSteps=PROGRESS_SCALE,
                        message=f"{st['title']}: {message}" if message else st["title"], **extra)

    # ------------------------------------------------------------- children
    def _child(self, st: Dict[str, Any], kind: str, dataset_id: str, payload_key: str, payload: Dict[str, Any],
               total: int) -> Dict[str, Any]:
        """Run (or re-attach to) the stage's child job; returns its final status snapshot."""
        if self.m.stopping:
            raise PipelineStopped()
        child_id = st.get("jobId")
        child = self.m.jobs.get(child_id) if child_id else None
        if child is not None and child.state == "completed" and kind == "process":
            from .server import dataset_root

            out = child.status.get("outputDatasetId")
            if not out or dataset_root(self.settings, out) is None:
                child = None          # its output was already taken (a restart came before the stage ended)
        if child is not None and child.state in ("failed", "cancelled"):
            interrupted = (child.status.get("message") or "").startswith(("interrupted by worker restart",
                                                                         "worker shut down"))
            if not interrupted or self.job.cancel_event.is_set():
                if child.state == "cancelled" or self.job.cancel_event.is_set():
                    raise PipelineCancelled()
                raise PipelineError(child.status.get("message") or f"{st['title']} failed")
            child = None              # a restart interrupted it: run that step again
        if child is None:
            if st.get("attempts", 0) >= MAX_ATTEMPTS:
                raise PipelineError(f"{st['title']} was interrupted {MAX_ATTEMPTS} times")
            name = f"{self.job.status.get('name') or 'pipeline'}: {st['title']}"
            child = self.m.submit(kind, dataset_id, name, payload_key, payload, total_steps=total,
                                  priority=int(self.job.status.get("priority") or 0),
                                  origin=self.job.status.get("origin"), parent=self.job.id)
            self._set_stage(st, jobId=child.id, attempts=st.get("attempts", 0) + 1)
            if kind == "train":
                self.job.update(persist=True, trainJobId=child.id)
            self._log(f"{st['title']}: job {child.id}")
        was_paused = False
        while True:
            if self.job.cancel_event.is_set() or self.job.deleted:
                self.m.cancel(child.id)
                raise PipelineCancelled()
            if self.m.stopping:
                raise PipelineStopped()
            s = child.snapshot()
            state = s["state"]
            total_steps = s.get("totalSteps") or 0
            frac = (s.get("step") or 0) / total_steps if total_steps else 0.0
            extra = {}
            if kind == "train":
                extra = {k: s.get(k) for k in ("loss", "psnr", "evalPsnr", "numGaussians", "previewStep",
                                               "checkpointStep")}
                if state == "training" and s.get("etaSeconds") is not None:
                    extra["etaSeconds"] = s["etaSeconds"]
            msg = s.get("message") or state
            if kind == "train" and state == "training":
                msg = f"step {s.get('step') or 0:,} of {total_steps:,}"
            if state == "paused":
                if not was_paused:
                    was_paused = True
                    self.job.update(persist=True, state="paused", pausedAt=s.get("pausedAt") or time.time(),
                                    etaSeconds=None, message=f"{st['title']}: paused")
            else:
                if was_paused:
                    was_paused = False
                    self.job.update(persist=True, state="running", pausedAt=None)
                self._progress(st, frac, "waiting for a free GPU" if state == "queued" else msg,
                               gpu=s.get("gpu"), **extra)
            if state in ("completed", "failed", "cancelled"):
                if state == "completed":
                    return s
                return self._child(st, kind, dataset_id, payload_key, payload, total)   # decides retry / error
            time.sleep(self.poll_seconds)

    def _take_output(self, status: Dict[str, Any], into: Path) -> int:
        """Merge a child's output dataset into ``into`` (hard links), then drop that dataset."""
        from .processing import link_tree
        from .server import dataset_root

        out_id = status.get("outputDatasetId")
        root = dataset_root(self.settings, out_id) if out_id else None
        if root is None:
            raise PipelineError("the step finished without output")
        n = link_tree(root, into, [root])
        self.m.drop_dataset(out_id)
        return n

    # ------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self.job.update(persist=True, state="running", message="starting")
            for st in self.stages:
                if st["state"] in ("done", "skipped"):
                    continue
                self._check()
                if st["state"] != "running":
                    self._set_stage(st, state="running", startedAt=time.time())
                self._progress(st, 0.0, "")
                self._log(f"== {st['title']}")
                result = getattr(self, f"_stage_{st['id']}")(st)
                skipped = isinstance(result, tuple) and result[0] == "skipped"
                summary = result[1] if isinstance(result, tuple) else result
                self._set_stage(st, state="skipped" if skipped else "done", summary=summary, finishedAt=time.time())
                self._write_manifest()
            self.m.finish_pipeline(self.job, "completed", "completed")
        except PipelineStopped:
            self._log("worker stopping: the pipeline continues after the restart")
        except PipelineCancelled:
            self._fail_running("cancelled")
            self.m.finish_pipeline(self.job, "cancelled", "cancelled")
        except Exception as e:  # noqa: BLE001 — every failure ends the job with its message
            msg = str(e) or e.__class__.__name__
            if not isinstance(e, PipelineError):
                log.exception("pipeline %s failed", self.job.id)
            self._log(f"FAILED: {msg}")
            self._fail_running("failed", msg)
            self.m.finish_pipeline(self.job, "failed", msg)

    def _fail_running(self, state: str, msg: Optional[str] = None) -> None:
        for st in self.stages:
            if st["state"] == "running":
                self._set_stage(st, state=state, message=msg)

    def _write_manifest(self) -> None:
        try:
            work = self._work()
        except PipelineError:
            return
        _write_json(work / SNAPSHOTS / "pipeline.json", {
            "format": "splatpipeline", "version": 1, "jobId": self.job.id,
            "stages": [{k: s.get(k) for k in ("id", "title", "state", "summary", "message", "prepStage",
                                                "startedAt", "finishedAt")} for s in self.stages],
            "checkpoints": self.spec.get("checkpoints") or {},
        })

    # ------------------------------------------------------------- stages
    def _stage_frames(self, st):
        req = dict(self.spec["frames"]["request"], outputDir="_frames")
        s = self._child(st, "process", self.job.spec["datasetId"], "request", req, PROGRESS_SCALE)
        work = self._work()
        for sub in ("images", "masks"):
            shutil.rmtree(work / sub, ignore_errors=True)
        self._take_output(s, work)
        if not (work / "frames.json").is_file():
            raise PipelineError("the frames step returned no frames.json")
        fs = load_frames(work)
        if not fs["frames"]:
            raise PipelineError("no frames extracted from the video")
        save_frames(work, fs)
        snapshot_frames(work, fs, "frames")
        return {"frames": len(fs["frames"]), "rejected": len(fs.get("rejected") or [])}

    def _fisheye(self, work: Path, fs: Dict[str, Any]) -> bool:
        st = self.job.status.setdefault("facts", {})
        if "fisheye" not in st:
            st["fisheye"] = (not fs.get("isMultiViewRig")
                             and circular_fisheye([work / f["image"] for f in fs["frames"]]))
            self.job.persist()
        return bool(st["fisheye"])

    def _stage_static(self, st):
        work = self._work()
        fs = load_frames(work)
        if not any(s["id"] == "frames" for s in self.stages) and not (work / SNAPSHOTS / "frames.frames.json").is_file():
            snapshot_frames(work, fs, "frames")         # frames ingested by the client
        fisheye = self._fisheye(work, fs)
        if not (self.spec.get("staticRegions") or fisheye):
            return ("skipped", {"reason": "off"})
        from .studio import static_regions

        if fisheye:
            self._log("circular fisheye footage: masking the fixed lens border")
        reps = static_regions.apply(work, progress=lambda f, m="": (self._check(), self._progress(st, f, m)))
        for r in reps:
            if r.get("staticFraction", 0) > 0:
                self._log(f"{r['camera']}: {r['staticFraction'] * 100:.1f} % of the frame "
                          + ("masked" if r["applied"] else "looks fixed — not masked (the camera barely moves)"))
        n = apply_static_layer(work, fs, work / static_regions.LAYER)
        save_frames(work, fs)
        return {"masked": n, "fisheye": fisheye}

    def _stage_masks(self, st):
        conf = self.spec["masks"]
        req = dict(conf["request"], imagesDir="images", outputDir="splatprep-masks")
        s = self._child(st, "process", self._work_id(), "request", req, PROGRESS_SCALE)
        work = self._work()
        layer = work / "splatprep-masks"
        shutil.rmtree(layer, ignore_errors=True)
        self._take_output(s, layer)
        fs = load_frames(work)
        res = combine_model_masks(work, fs, layer / "masks", conf.get("dilatePixels", 4),
                                  str(conf["request"].get("maskModel") or ""))
        if res["empty"]:
            self._log(f"{res['empty']} frames had empty masks and were left unmasked")
        save_frames(work, fs)
        return res

    def _stage_sky(self, st):
        req = dict(self.spec["sky"]["request"], imagesDir="images", outputDir="splatprep-sky")
        try:
            s = self._child(st, "process", self._work_id(), "request", req, PROGRESS_SCALE)
            work = self._work()
            shutil.rmtree(work / "splatprep-sky", ignore_errors=True)
            self._take_output(s, work / "splatprep-sky")
        except (PipelineCancelled, PipelineStopped):
            raise
        except Exception as e:  # noqa: BLE001 — DatasetBuilder.skyOrSkip: no sky model is not a failure
            self._log(f"sky skipped: the sky model couldn't run ({e})")
            self._set_stage(st, message=f"skipped: the sky model couldn't run ({e})")
            return ("skipped", {"reason": str(e)})
        work = self._work()
        fs = load_frames(work)
        n = record_sky(work, fs)
        save_frames(work, fs)
        self._log(f"sky in {n} of {len(fs['frames'])} photos")
        return {"withSky": n, "frames": len(fs["frames"])}

    def _stage_sfm(self, st):
        work = self._work()
        fs = load_frames(work)
        snapshot_frames(work, fs, "masks")       # the frames and masks as the camera solve gets them
        conf = self.spec["sfm"]
        req, prefer = sfm_request(conf["request"], fs, self._fisheye(work, fs))
        if req.get("cameraModel") == "fisheye" and conf["request"].get("cameraModel") != "fisheye":
            self._log("fisheye footage: solving with a fisheye camera model and undistorting")
        s = self._child(st, "process", self._work_id(), "request", req, PROGRESS_SCALE)
        shutil.rmtree(work / "sfm", ignore_errors=True)
        self._take_output(s, work / "sfm")
        poses, points = read_solution(work, fs, "sfm", prefer)
        multiview = bool(fs.get("isMultiViewRig"))
        dropped = pose_sanity(poses, fs, ordered=conf.get("sequential", False) and not multiview)
        if dropped:
            self._log(f"left out {dropped} camera(s) placed far off the camera path (likely mis-registered)")
        n = len(fs["frames"])
        registered = len(poses)
        if n >= 10 and registered < 0.3 * n:
            masked = self.spec.get("masks") is not None
            raise PipelineError(f"Camera solving placed only {registered} of {n} photos. "
                                + ("Masks may be hiding most of each photo — try without masks. " if masked else "")
                                + "Try another camera solver, or capture with more overlap between photos.")
        if conf.get("enforceRig", True) and multiview:
            added = enforce_multiview_rig(poses, fs)
            if added:
                self._log(f"360° rig: {added} views the solve missed placed from their frame's other views")
        doc = assemble(work, fs, poses, points, self.spec.get("holdoutInterval"))
        snap = work / SNAPSHOTS
        snap.mkdir(exist_ok=True)
        shutil.copyfile(work / "dataset.json", snap / "poses.dataset.json")
        if doc.get("points"):
            shutil.copyfile(work / doc["points"], snap / "poses.points.ply")
        self._log(f"{len(doc['cameras'])} of {n} frames placed")
        return {"cameras": len(doc["cameras"]), "frames": n, "dropped": dropped}

    def _stage_level(self, st):
        work = self._work()
        if self.spec.get("level", True):
            up = level(work)
            if up is not None:
                self._log("Levelled scene: up = −Y, centred at origin")
        doc = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
        self.m.publish_work_dataset(self.job)        # outputDatasetId: the client may fetch it from now on
        return {"cameras": len(doc["cameras"]), "levelled": bool(self.spec.get("level", True))}

    def _stage_train(self, st):
        conf = self.spec["train"]
        s = self._child(st, "train", self._work_id(), "config", conf["config"], int(conf.get("iterations") or 0))
        child = self.m.jobs.get(s["jobId"])
        if child is None or not (child.dir / "result.ply").is_file():
            raise PipelineError("training finished without a result")
        for name in ("result.ply", "sky.png", "sky.json"):
            src = child.dir / name
            if src.is_file():
                dst = self.job.dir / name
                dst.unlink(missing_ok=True)
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copyfile(src, dst)
        self.job.update(persist=True, numGaussians=s.get("numGaussians"), psnr=s.get("psnr"), loss=s.get("loss"),
                        evalPsnr=s.get("evalPsnr"))
        return {"numGaussians": s.get("numGaussians"), "psnr": s.get("psnr")}


def start_runner(manager, job) -> threading.Thread:
    t = threading.Thread(target=PipelineRunner(manager, job).run, daemon=True, name=f"pipeline-{job.id}")
    t.start()
    return t
