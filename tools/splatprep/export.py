"""COLMAP reconstruction -> prepared training dataset (PROTOCOL §2).

* Cameras with distortion are undistorted: ``colmap image_undistorter`` when the
  binary exists, else OpenCV (``cv2.undistort`` / ``cv2.fisheye``) with a
  border-free new pinhole matrix; pinhole cameras are copied verbatim.
* ``dataset.json``: OpenCV convention, ROW-major ``cameraToWorld``.
* ``points3d.ply``: binary little endian xyz float + rgb uchar.
* ``sparse/0``: the (undistorted, aligned) COLMAP model, binary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .colmap_model import Camera, Reconstruction, read_largest_model, write_binary_model
from .mapper import find_executable, run_logged
from .ply import write_points_ply

ProgressCb = Callable[[float, str], None]


def scene_metrics(c2ws: np.ndarray, points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """(sceneCenter, sceneExtent, upVector) from camera poses.

    * centre: mean camera centre (the same as 3DGS' ``getNerfppNorm``),
    * extent: 1.1 x max distance of a camera from the centre (min 1e-3), ignoring stray cameras,
    * up: minus the mean camera +Y axis (OpenCV +Y is down in the image).
    """
    C = c2ws[:, :3, 3]
    center = C.mean(0)
    if len(C):
        # Ignore stray cameras (> 3x the median distance): one mis-registered camera 290 m out tripled a
        # drone scene's extent, and trainers scale position learning rates / densification by it.
        dist = np.linalg.norm(C - C.mean(0), axis=1)
        core = dist <= 3 * max(float(np.median(dist)), 1e-9)
        center = C[core].mean(0) if core.any() else C.mean(0)
        dist = np.linalg.norm(C[core] - center, axis=1) if core.any() else dist
        extent = float(max(1.1 * np.max(dist), 1e-3))
    else:
        extent = 1.0
    up = -c2ws[:, :3, 1].mean(0)
    nrm = np.linalg.norm(up)
    up = up / nrm if nrm > 1e-6 else np.array([0.0, -1.0, 0.0])
    return center, extent, up


def _undistort_cv2(rec: Reconstruction, image_dir: Path, out_images: Path, progress: Optional[ProgressCb]) -> Reconstruction:
    import cv2

    new_cams: Dict[int, Camera] = {}
    maps: Dict[int, tuple] = {}
    for cid, cam in rec.cameras.items():
        fx, fy, cx, cy = cam.pinhole()
        # COLMAP pixel centres are at +0.5; OpenCV's at integers.
        K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]])
        d = cam.distortion()
        size = (cam.width, cam.height)
        if cam.model == "OPENCV_FISHEYE":
            D = np.asarray(d[:4], float).reshape(4, 1)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, size, cv2.CV_32FC1)
        elif cam.model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"):
            if cam.model == "SIMPLE_RADIAL":
                D = np.array([d[0], 0, 0, 0], float)
            elif cam.model == "RADIAL":
                D = np.array([d[0], d[1], 0, 0], float)
            elif cam.model == "OPENCV":
                D = np.array(d[:4], float)
            else:  # FULL_OPENCV: k1 k2 p1 p2 k3 k4 k5 k6
                D = np.array(d[:8], float)
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0.0, size)
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, size, cv2.CV_32FC1)
        else:
            raise RuntimeError(f"cannot undistort camera model {cam.model} without COLMAP (brew install colmap)")
        maps[cid] = (m1, m2)
        new_cams[cid] = Camera(cid, "PINHOLE", cam.width, cam.height, np.array([newK[0, 0], newK[1, 1], newK[0, 2] + 0.5, newK[1, 2] + 0.5]))
    ims = sorted(rec.images.values(), key=lambda i: i.id)
    for k, im in enumerate(ims):
        src = image_dir / im.name
        dst = out_images / im.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {src}")
        m1, m2 = maps[im.camera_id]
        und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if dst.suffix.lower() in (".heic", ".heif"):
            dst = dst.with_suffix(".jpg")
            im.name = str(Path(im.name).with_suffix(".jpg").as_posix())
        cv2.imwrite(str(dst), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if progress:
            progress((k + 1) / len(ims), f"undistorted {k + 1}/{len(ims)}")
    # Points2D are kept in distorted coordinates (only used for stats)
    return Reconstruction(new_cams, rec.images, rec.points3D)


def _undistort_colmap(colmap: str, sparse_dir: Path, image_dir: Path, work: Path, log_path: Optional[Path]) -> Tuple[Reconstruction, Path]:
    out = work / "undistorted"
    if out.exists():
        shutil.rmtree(out)
    run_logged(
        [colmap, "image_undistorter", "--image_path", image_dir, "--input_path", sparse_dir, "--output_path", out, "--output_type", "COLMAP"],
        log_path,
    )
    rec = read_largest_model(out / "sparse")
    if rec is None:
        raise RuntimeError("image_undistorter produced no model")
    return rec, out / "images"


def export_dataset(
    rec: Reconstruction,
    image_dir: Path,
    out_dir: Path,
    work_dir: Path,
    progress: Optional[ProgressCb] = None,
    export_images: bool = True,
    log_path: Optional[Path] = None,
    undistort: str = "auto",
    extra: Optional[dict] = None,
    up: Optional[np.ndarray] = None,
) -> dict:
    """Write dataset.json + images/ + points3d.ply + sparse/0 into ``out_dir``.

    ``up``: known world up (GPS-aligned scenes: +Z). Otherwise it is guessed from the cameras, which fails
    for straight-down (nadir) photos whose image-down axes all lie in the ground plane.
    """
    image_dir, out_dir, work_dir = Path(image_dir), Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    progress = progress or (lambda f, m="": None)
    needs_undistort = any(c.has_distortion() for c in rec.cameras.values())
    method = "copy"
    src_images = image_dir
    if needs_undistort and export_images:
        colmap = find_executable("colmap") if undistort in ("auto", "colmap") else None
        if colmap:
            tmp_sparse = work_dir / "sparse_for_undistort"
            if tmp_sparse.exists():
                shutil.rmtree(tmp_sparse)
            write_binary_model(rec, tmp_sparse)
            rec, src_images = _undistort_colmap(colmap, tmp_sparse, image_dir, work_dir, log_path)
            method = "colmap"
        else:
            progress(0.0, "undistorting with OpenCV")
            rec = _undistort_cv2(rec, image_dir, out_images, lambda f, m: progress(0.8 * f, m))
            src_images = None
            method = "opencv"

    ims = sorted(rec.images.values(), key=lambda i: i.name)
    if export_images and src_images is not None:
        for k, im in enumerate(ims):
            dst = out_images / im.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_images / im.name, dst)
            if k % 10 == 0:
                progress(0.8 * (k + 1) / len(ims), f"copied {k + 1}/{len(ims)} images")

    cams = []
    c2ws = []
    for idx, im in enumerate(ims):
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.pinhole()
        c2w = im.c2w()
        c2ws.append(c2w)
        cams.append(
            {
                "id": idx,
                "image": f"images/{im.name}",
                "width": int(cam.width),
                "height": int(cam.height),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "cameraToWorld": [[float(v) for v in row] for row in c2w],
                "holdout": False,
            }
        )
    c2ws_a = np.stack(c2ws) if c2ws else np.zeros((0, 4, 4))
    xyz, rgb = rec.points_xyz_rgb()
    known_up = up
    center, extent, up = scene_metrics(c2ws_a, xyz) if len(c2ws_a) else (np.zeros(3), 1.0, np.array([0, -1.0, 0]))
    if known_up is not None:
        up = np.asarray(known_up, float) / np.linalg.norm(known_up)
    ds = {
        "format": "splatdataset",
        "version": 1,
        "convention": "opencv",
        "sceneCenter": [float(v) for v in center],
        "sceneExtent": float(extent),
        "upVector": [float(v) for v in up],
        "cameras": cams,
    }
    if len(xyz):
        write_points_ply(out_dir / "points3d.ply", xyz, rgb)
        ds["points"] = "points3d.ply"
    if extra:
        ds["splatprep"] = extra
    (out_dir / "dataset.json").write_text(json.dumps(ds, indent=2))
    sparse = out_dir / "sparse" / "0"
    if sparse.exists():
        shutil.rmtree(sparse)
    write_binary_model(rec, sparse)
    progress(1.0, f"exported {len(cams)} cameras, {len(xyz)} points")
    return {"numCameras": len(cams), "numPoints3D": int(len(xyz)), "undistortion": method, "sceneExtent": float(extent)}
