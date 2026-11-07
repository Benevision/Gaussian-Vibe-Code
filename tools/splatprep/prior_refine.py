"""Refine known camera poses (iPhone / ARKit captures) instead of solving them from scratch.

ARKit poses are good but drift slowly in heading (≈ 1° over a room scan, measured on a 649-frame bedroom
capture against full SfM). Full SfM fixes that (+2.5 dB held-out PSNR) but re-solves what ARKit already
knows. This module keeps the ARKit poses and intrinsics as the starting point:

1. pairs: frames that look at the same place (look-at points close, view axes < 60° apart) plus a short
   sequential window — the revisits are what pin down drift;
2. ``point_triangulator`` with the prior poses (no mapper, nothing to register or fail);
3. ``bundle_adjuster`` refining poses only (intrinsics stay ARKit's per-frame values), then triangulate and
   adjust once more with the corrected poses;
4. sanity: if the corrections look implausible, keep the prior poses.

The caller Sim3-aligns the result back to the priors (runner.align_to_priors), so scale and gravity are
the capture's.
"""

from __future__ import annotations

import math
import shutil
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .colmap_model import Reconstruction, find_models, read_model
from .errors import SplatPrepError
from .mapper import run_logged

#: Use refinement when at least this share of images has a prior pose with intrinsics.
MIN_PRIOR_COVERAGE = 0.9
#: Corrections larger than this (median) mean the priors and the images disagree badly: keep the priors.
MAX_MEDIAN_ROT_DEG = 3.0
MAX_MEDIAN_POS_FRACTION = 0.05   # of the camera extent

Pair = Tuple[int, int]


def lookat_pairs(c2ws: Dict[int, np.ndarray], k: int = 10, window: int = 4, max_angle_deg: float = 60.0,
                 depth: Optional[float] = None) -> List[Pair]:
    """Frames whose look-at points (a typical scene depth ahead) are closest, plus ``window`` sequential
    neighbours. Two views of the same wall from different spots — revisits — get paired, which is what
    corrects slow heading drift; plain centre-distance pairs mostly link consecutive frames."""
    ids = sorted(c2ws)
    if len(ids) < 2:
        return []
    C = np.stack([c2ws[i][:3, 3] for i in ids])
    Z = np.stack([c2ws[i][:3, 2] for i in ids])
    if depth is None:
        spread = np.linalg.norm(C - C.mean(0), axis=1)
        depth = max(0.5, float(np.median(spread)) * 1.2)
    T = C + Z * depth
    D = np.linalg.norm(T[:, None] - T[None], axis=2)
    D[(Z @ Z.T) < math.cos(math.radians(max_angle_deg))] = np.inf
    np.fill_diagonal(D, np.inf)
    out = set()
    kk = min(k, len(ids) - 1)
    for a in range(len(ids)):
        for b in np.argsort(D[a])[:kk]:
            if np.isfinite(D[a, b]):
                i, j = ids[a], ids[int(b)]
                out.add((min(i, j), max(i, j)))
        for d in range(1, window + 1):
            if a + d < len(ids):
                out.add((ids[a], ids[a + d]))
    return sorted(out)


def set_prior_cameras(db: Path, names: Sequence[str], priors: dict) -> int:
    """Give every image with prior intrinsics its own PINHOLE camera (ARKit focal varies per frame with
    focus). Returns how many images got one."""
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        rows = cur.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        cams = {cid: (w, h) for cid, w, h in cur.execute("SELECT camera_id, width, height FROM cameras")}
        nxt = (cur.execute("SELECT MAX(camera_id) FROM cameras").fetchone()[0] or 0) + 1
        tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rigs = {"rigs", "frames", "frame_data"} <= tables   # COLMAP 4: every frame belongs to a rig of cameras
        done = 0
        for iid, name, cid in rows:
            p = priors.get(name)
            if p is None or p.intrinsics is None:
                continue
            w, h = cams[cid]
            fx, fy, cx, cy = [float(v) for v in p.intrinsics]
            params = np.array([fx, fy, cx, cy], np.float64).tobytes()
            cur.execute("INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
                        "VALUES (?, ?, ?, ?, ?, 1)", (nxt, 1, w, h, params))   # 1 = PINHOLE
            cur.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", (nxt, iid))
            if rigs:
                # its own single-camera rig, and the image's frame moved onto it (a frame whose rig doesn't
                # contain its camera aborts COLMAP: "existing_frame.RigId() == frame.RigId()")
                cur.execute("INSERT INTO rigs (ref_sensor_id, ref_sensor_type) VALUES (?, 0)", (nxt,))
                rig = cur.lastrowid
                row = cur.execute("SELECT frame_id FROM frame_data WHERE data_id = ? AND sensor_type = 0", (iid,)).fetchone()
                if row:
                    cur.execute("UPDATE frames SET rig_id = ? WHERE frame_id = ?", (rig, row[0]))
                    cur.execute("UPDATE frame_data SET sensor_id = ? WHERE data_id = ? AND sensor_type = 0", (nxt, iid))
            nxt += 1
            done += 1
        if rigs:   # drop rigs no frame uses any more (the original shared camera's)
            cur.execute("DELETE FROM rigs WHERE rig_id NOT IN (SELECT DISTINCT rig_id FROM frames)")
            cur.execute("DELETE FROM rig_sensors WHERE rig_id NOT IN (SELECT rig_id FROM rigs)")
        con.commit()
        return done
    finally:
        con.close()


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def write_prior_model(db: Path, priors: dict, out: Path) -> int:
    """COLMAP text model (cameras, images with prior poses, no points) using the database's ids."""
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    try:
        cams = con.execute("SELECT camera_id, model, width, height, params FROM cameras").fetchall()
        imgs = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    finally:
        con.close()
    names = {1: "PINHOLE", 0: "SIMPLE_PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
    used = set()
    n = 0
    with open(out / "images.txt", "w") as fi:
        for iid, name, cid in imgs:
            p = priors.get(name)
            if p is None:
                continue
            R = p.c2w[:3, :3].T
            t = -R @ p.c2w[:3, 3]
            q = _rotmat_to_qvec(R)
            fi.write(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cid} {name}\n\n")
            used.add(cid)
            n += 1
    with open(out / "cameras.txt", "w") as fc:
        for cid, model, w, h, params in cams:
            if cid in used:
                pr = np.frombuffer(params, np.float64)
                fc.write(f"{cid} {names.get(model, 'PINHOLE')} {w} {h} " + " ".join(f"{v:.10g}" for v in pr) + "\n")
    (out / "points3D.txt").write_text("")
    return n


def _largest(root: Path) -> Path:
    ms = find_models(root)
    if not ms:
        raise SplatPrepError(f"COLMAP wrote no model in {root}")
    return max(ms, key=lambda p: len(read_model(p).images))


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


_PYCOLMAP_REFINE = r"""
import sys, pathlib, pycolmap
db, img, src, tri, ba, rnd = sys.argv[1:7]
rec = pycolmap.Reconstruction(src)
rec = pycolmap.triangulate_points(rec, db, img, tri, clear_points=True, refine_intrinsics=False)
o = pycolmap.BundleAdjustmentOptions()
o.refine_focal_length = False
o.refine_principal_point = False
o.refine_extra_params = False
pycolmap.bundle_adjustment(rec, o)
pathlib.Path(ba).mkdir(parents=True, exist_ok=True)
rec.write(ba)
print(f"round {rnd}: {rec.num_reg_images()} images, {rec.num_points3D()} points")
"""


def _round_pycolmap(db: Path, image_dir: Path, src: Path, tri: Path, ba: Path, rnd: int, log_path) -> Path:
    """One triangulate + pose-BA round with pycolmap in a child interpreter (its libomp never meets torch's)."""
    import sys

    run_logged([sys.executable, "-c", _PYCOLMAP_REFINE, str(db), str(image_dir), str(src), str(tri), str(ba), str(rnd)],
               log_path, None)
    return ba


def refine_poses(colmap: Optional[str], db: Path, image_dir: Path, priors: dict, work: Path,
                 log_path: Optional[Path], progress: Callable[[float, str], None]) -> Tuple[Reconstruction, dict]:
    """Triangulate with the prior poses, then bundle-adjust poses (twice). Returns (reconstruction, info).
    ``colmap``: the COLMAP executable, or None to use pycolmap (Linux worker: no COLMAP CLI)."""
    fixed_intr = ["--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_principal_point", "0",
                  "--BundleAdjustment.refine_extra_params", "0"]
    tri_intr = ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0"]
    prior_dir = work / "prior_model"
    shutil.rmtree(prior_dir, ignore_errors=True)
    n = write_prior_model(db, priors, prior_dir)
    src = prior_dir
    for rnd in (1, 2):
        tri = work / f"tri{rnd}"
        ba = work / f"ba{rnd}"
        for d in (tri, ba):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        progress(0.1 + 0.4 * (rnd - 1), f"triangulating with {'prior' if rnd == 1 else 'refined'} poses")
        if colmap is None:   # triangulate_points also writes the triangulation (before BA) into ``tri``
            _round_pycolmap(db, image_dir, src, tri, ba, rnd, log_path)
            src = _largest(ba)
            continue
        cmd = [colmap, "point_triangulator", "--database_path", db, "--image_path", image_dir,
               "--input_path", src, "--output_path", tri] + tri_intr
        if rnd > 1:
            cmd += ["--clear_points", "1"]
        run_logged(cmd, log_path, None)
        progress(0.3 + 0.4 * (rnd - 1), "adjusting camera poses")
        run_logged([colmap, "bundle_adjuster", "--input_path", _largest(tri), "--output_path", ba] + fixed_intr,
                   log_path, None)
        src = _largest(ba)
    rec = read_model(src)
    first = read_model(_largest(work / "tri1"))

    # How far did the poses move? (after a Sim3 back onto the priors, as the runner will do)
    from .geometry import robust_umeyama

    by = rec.image_by_name()
    common = [nm for nm in by if nm in priors]
    src_c = np.stack([by[nm].center() for nm in common])
    dst_c = np.stack([priors[nm].c2w[:3, 3] for nm in common])
    T, _ = robust_umeyama(src_c, dst_c)
    rot = np.array([_rot_deg(T.R @ by[nm].c2w()[:3, :3], priors[nm].c2w[:3, :3]) for nm in common])
    pos = np.linalg.norm(T.apply_points(src_c) - dst_c, axis=1)
    extent = float(np.linalg.norm(dst_c - dst_c.mean(0), axis=1).max()) or 1.0
    st_before, st_after = first.stats(), rec.stats()
    info = {"mode": "prior-refine", "numPriorImages": n, "numRegistered": len(rec.images),
            "medianRotationChangeDeg": round(float(np.median(rot)), 3),
            "p90RotationChangeDeg": round(float(np.percentile(rot, 90)), 3),
            "medianPositionChangeM": round(float(np.median(pos)), 4),
            "reprojErrorBefore": st_before.get("meanReprojError"), "reprojErrorAfter": st_after.get("meanReprojError")}
    ok = (np.median(rot) <= MAX_MEDIAN_ROT_DEG and np.median(pos) <= MAX_MEDIAN_POS_FRACTION * extent
          and (st_after.get("meanReprojError") or 99) <= (st_before.get("meanReprojError") or 99) + 1e-6)
    info["accepted"] = bool(ok)
    if not ok:
        info["note"] = "refinement looked implausible — kept the prior poses"
        rec = first   # triangulated with the prior poses, which stay as they were
    progress(1.0, (f"refined {len(rec.images)} poses: median {info['medianRotationChangeDeg']}°, "
                   f"{info['medianPositionChangeM'] * 100:.1f} cm; reprojection "
                   f"{(st_before.get('meanReprojError') or 0):.2f} → {(st_after.get('meanReprojError') or 0):.2f} px")
             if ok else info["note"])
    return rec, info


def use_prior_refine(names: Sequence[str], priors: dict, request: dict) -> bool:
    mode = (request.get("priorMode") or "auto").lower()
    if mode == "solve" or not priors:
        return False
    have = sum(1 for n in names if n in priors and priors[n].intrinsics is not None)
    return mode == "refine" or have >= MIN_PRIOR_COVERAGE * len(names)
