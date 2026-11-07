"""Helpers shared by the feed-forward / dense backends (MASt3R, VGGT)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..colmap_model import Camera, Image, Point3D, Reconstruction


def add_repo_paths(env_var: str, subdirs: Sequence[str] = ("",)) -> None:
    """Prepend a cloned repo (and sub-folders such as ``dust3r``) to sys.path.

    Uses ``$<env_var>`` if set, else ``<venv>/src/<repo>`` (where the Models pack installer clones it,
    e.g. ``SPLATPREP_MAST3R_PATH`` → ``src/mast3r``), so the app and CLI work without extra setup.
    """
    root = os.environ.get(env_var)
    if not root:
        repo = env_var.removeprefix("SPLATPREP_").removesuffix("_PATH").lower()
        guess = Path(sys.prefix) / "src" / repo
        if not guess.is_dir():
            return
        root = str(guess)
    for sub in subdirs:
        p = str(Path(root) / sub) if sub else str(Path(root))
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)


def build_reconstruction(
    names: List[str],
    sizes: List[tuple],
    c2ws: np.ndarray,
    intrinsics: np.ndarray,  # S x 4 (fx, fy, cx, cy) in ORIGINAL image pixels (COLMAP convention)
    xyz: np.ndarray,
    rgb: np.ndarray,
    shared_camera: bool = False,
) -> Reconstruction:
    rec = Reconstruction()
    for i, name in enumerate(names):
        w, h = sizes[i]
        if shared_camera and i > 0 and (w, h) == sizes[0]:
            cid = 1
        else:
            cid = i + 1
            rec.cameras[cid] = Camera(cid, "PINHOLE", int(w), int(h), np.asarray(intrinsics[i], float))
        im = Image(i + 1, np.array([1.0, 0, 0, 0]), np.zeros(3), cid, name)
        im.set_c2w(c2ws[i])
        rec.images[i + 1] = im
    for k in range(len(xyz)):
        rec.points3D[k + 1] = Point3D(k + 1, np.asarray(xyz[k], float), np.asarray(rgb[k], np.uint8))
    return rec


def subsample_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int, seed: int = 0):
    if len(xyz) <= max_points:
        return xyz, rgb
    idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
    return xyz[idx], rgb[idx]


def map_intrinsics_to_original(K_proc: np.ndarray, scale_x: float, scale_y: float, off_x: float, off_y: float) -> np.ndarray:
    """Processed-image pinhole (fx, fy, cx, cy; integer-centre pixels) -> original-image COLMAP pixels.

    The processed image is ``resize(original, scale)`` followed by a crop starting at ``(off_x, off_y)``.
    """
    fx, fy, cx, cy = [float(v) for v in K_proc]
    return np.array([fx / scale_x, fy / scale_y, (cx + off_x + 0.5) / scale_x, (cy + off_y + 0.5) / scale_y])
