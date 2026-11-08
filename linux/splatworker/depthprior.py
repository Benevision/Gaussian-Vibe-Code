"""Monocular depth priors for photos without sensor depth (no LiDAR / stereo).

Depth Anything V2 (Small, Apache-2.0, through ``transformers``) predicts relative inverse depth per photo. That has
no scale, so each prediction is fitted to the scene: the dataset's own 3D points (camera solve / SfM) are projected
into the photo and ``1/z ≈ a·pred + b`` is solved robustly (iteratively reweighted least squares, so points hidden
behind a surface don't drag it). The result is a metric-ish depth map at about LiDAR resolution (256 px on the long
side), used like LiDAR depth for the depth-normal loss (normals don't depend on the fit's scale) and, with a small
weight, the depth loss.

Results are cached next to the dataset (``depth-prior/<camera id>.npy`` + ``meta.json``), so later jobs and the
other blocks of a block-parallel run reuse them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CACHE_VERSION = 1
LONG_SIDE = 256


def available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def fit_inverse_depth(pred: np.ndarray, z: np.ndarray, iters: int = 8) -> Optional[Tuple[float, float, float]]:
    """Robust fit of 1/z ≈ a·pred + b over sample pairs. Returns (a, b, median relative depth error) or None."""
    ok = np.isfinite(pred) & np.isfinite(z) & (z > 0)
    pred, z = pred[ok].astype(np.float64), z[ok].astype(np.float64)
    if pred.size < 20:
        return None
    y = 1.0 / z
    w = np.ones_like(y)
    a = b = 0.0
    for _ in range(iters):
        A = np.stack([pred * w, w], 1)
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(a * pred + b - y)
        s = 1.4826 * np.median(r) + 1e-12
        w = 1.0 / np.maximum(1.0, r / (2.0 * s))     # Huber-style: full weight within 2σ, then 1/r
    if a <= 0:
        return None
    inv = a * pred + b
    good = inv > 1e-9
    if good.sum() < 20:
        return None
    rel = np.abs(1.0 / inv[good] - z[good]) / z[good]
    return a, b, float(np.median(rel))


def _project(points: np.ndarray, w2c: np.ndarray, K: np.ndarray, width: int, height: int):
    Xc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    front = z > 1e-6
    u = K[0, 0] * Xc[:, 0] / np.where(front, z, 1) + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / np.where(front, z, 1) + K[1, 2]
    inside = front & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    return u[inside], v[inside], z[inside]


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = img.shape
    x0 = np.clip(np.floor(u).astype(np.int64), 0, W - 2); y0 = np.clip(np.floor(v).astype(np.int64), 0, H - 2)
    fx = np.clip(u - x0, 0, 1); fy = np.clip(v - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


class DepthPriors:
    """Per-camera prior depth maps (float32, ~256 px long side; 0 = no data)."""

    def __init__(self, cache_dir: Path, log: Callable[[str], None] = print):
        self.cache_dir = Path(cache_dir)
        self.log = log
        self.maps: Dict[int, np.ndarray] = {}

    def _meta_ok(self) -> bool:
        try:
            m = json.loads((self.cache_dir / "meta.json").read_text())
            return m.get("model") == MODEL_ID and m.get("version") == CACHE_VERSION
        except (OSError, ValueError):
            return False

    def compute(self, ds, cams: List, points: Optional[np.ndarray], device: str = "cuda",
                should_stop: Callable[[], bool] = lambda: False,
                status: Callable[[str], None] = lambda m: None) -> int:
        """Fills ``self.maps`` for ``cams`` (cached ones are read, the rest predicted). Returns how many have a map."""
        from PIL import Image

        cached = self._meta_ok()
        todo = []
        for c in cams:
            f = self.cache_dir / f"{c.id}.npy"
            if cached and f.is_file():
                try:
                    self.maps[c.id] = np.load(f)
                    continue
                except (OSError, ValueError):
                    pass
            todo.append(c)
        if todo:
            if points is None or len(points) < 100:
                self.log("depth priors: the dataset has too few 3D points to scale a depth model's output; skipped")
                return len(self.maps)
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            proc = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "meta.json").write_text(json.dumps({"model": MODEL_ID, "version": CACHE_VERSION}))
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) > 200_000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 200_000, replace=False)]
            fitted, errs = 0, []
            for i, c in enumerate(todo):
                if should_stop():
                    break
                if i % 10 == 0:
                    status(f"depth priors {i + 1}/{len(todo)} (Depth Anything V2)")
                try:
                    img = Image.open(ds.resolve(c.image)).convert("RGB")
                except OSError:
                    continue
                W, H = img.size
                s = LONG_SIDE / max(W, H)
                ow, oh = max(8, round(W * s)), max(8, round(H * s))
                with torch.no_grad():
                    inp = proc(images=img, return_tensors="pt").to(device)
                    out = model(**inp).predicted_depth            # relative inverse depth (bigger = nearer)
                    pred = torch.nn.functional.interpolate(out[:, None], size=(oh, ow), mode="bilinear",
                                                           align_corners=False)[0, 0].float().cpu().numpy()
                K = np.array([[c.fx * ow / W, 0, c.cx * ow / W], [0, c.fy * oh / H, c.cy * oh / H], [0, 0, 1]])
                w2c = np.linalg.inv(c.camera_to_world)
                u, v, z = _project(pts, w2c, K, ow, oh)
                fit = fit_inverse_depth(_sample(pred, u, v), z) if len(z) >= 20 else None
                if fit is None or fit[2] > 0.25:
                    continue
                a, b, err = fit
                inv = a * pred + b
                depth = np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), 0.0).astype(np.float32)
                # beyond 50× the median fitted depth is sky / noise, not a surface
                depth[depth > 50 * np.median(z)] = 0
                self.maps[c.id] = depth
                np.save(self.cache_dir / f"{c.id}.npy", depth)
                fitted += 1
                errs.append(err)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if errs:
                self.log(f"depth priors: {fitted} of {len(todo)} photos fitted to the scene "
                         f"(median depth error vs its 3D points {100 * float(np.median(errs)):.0f} %)")
        return len(self.maps)
