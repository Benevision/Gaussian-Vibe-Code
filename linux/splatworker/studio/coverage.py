"""Coverage display mode: "how well did the capture see this?" for every splat (Mac: ``Render/Coverage.swift``).

For each splat, every dataset camera is checked: is the splat in front of it and inside its image, and (when the
camera has a LiDAR depth map) not hidden behind a nearer surface? That gives the number of photos that saw it and
the best detail any of them had (``fx / depth``, pixels per scene unit). The score combines both, relative to this
capture: seen by only one or two photos, or from 4× further away than typical, is poor. So it works for a bedroom
in metres and an SfM scene in arbitrary units alike.

CPU numpy, chunked (``chunk`` splats × one camera at a time), so peak memory stays at a few hundred MB for any
scene size; run it on a background thread. Pure functions, no GTK.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

# Same constants as the Mac.
POOR_VIEWS = 2.0          # views at or below this score 0 …
GOOD_VIEWS = 8.0          # … at or above this score 1
POOR_DETAIL_RATIO = 0.25  # detail ¼ of the median (seen from 4× further) scores 0, median or better 1
POOR_SCORE = 0.34         # below this a seen splat counts as "poorly seen" in the summary

# Ramp of the Coverage display mode (red → yellow → green), shared with the shader and the inspector legend.
RAMP = ((0.86, 0.16, 0.12), (0.96, 0.78, 0.12), (0.20, 0.72, 0.32))


class CoverageCancelled(Exception):
    pass


@dataclass
class CoverageCamera:
    w2c: np.ndarray                    # (3, 4) world → OpenCV camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth: Optional[np.ndarray] = None   # (dh, dw) float16 Z depth, metres (scene units), or None


@dataclass
class CoverageResult:
    score: np.ndarray          # float32 per splat, 0 = poor … 1 = good
    views: np.ndarray          # uint16 per splat: photos that saw it (0 = none)
    poor_fraction: float       # share of the scene (size × opacity) seen, but poorly
    unseen_fraction: float     # share no photo saw at all (usually floaters beyond the walls)
    median_views: int
    used_depth: bool

    def summary(self) -> str:
        return (f"{self.poor_fraction * 100:.0f} % of the scene was seen only poorly (red/orange areas the photos did "
                f"see). {self.unseen_fraction * 100:.0f} % was seen by no photo at all — usually floaters beyond the "
                f"walls, safe to delete in Edit. A typical part was seen by {self.median_views} photos"
                + ("." if self.used_depth else " (no depth maps, so walls don't block the view in this estimate)."))


def load_cameras(dataset_dir: Path, use_depth: bool = True) -> List[CoverageCamera]:
    """Cameras of a prepared dataset (PROTOCOL.md §2: row-major OpenCV ``cameraToWorld``), with their depth maps
    (``depth`` + ``depthWidth``/``depthHeight``: float16 little-endian Z depth) when present and readable."""
    dataset_dir = Path(dataset_dir)
    doc = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    out: List[CoverageCamera] = []
    for c in doc.get("cameras", []):
        try:
            c2w = np.asarray(c["cameraToWorld"], np.float64).reshape(4, 4)
            w2c = np.linalg.inv(c2w)[:3]
            cam = CoverageCamera(w2c.astype(np.float64), float(c["fx"]), float(c["fy"]), float(c["cx"]),
                                 float(c["cy"]), int(c["width"]), int(c["height"]))
        except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
            continue
        dw, dh, rel = c.get("depthWidth"), c.get("depthHeight"), c.get("depth")
        if use_depth and rel and dw and dh:
            try:
                raw = (dataset_dir / rel).read_bytes()
                if len(raw) >= int(dw) * int(dh) * 2:
                    cam.depth = np.frombuffer(raw, "<f2", count=int(dw) * int(dh)).reshape(int(dh), int(dw))
            except (OSError, ValueError):
                cam.depth = None
        out.append(cam)
    return out


def raw_counts(positions: np.ndarray, cameras: List[CoverageCamera], chunk: int = 200_000,
               progress: Optional[Callable[[float], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None):
    """(views float32, best detail float32) per splat — the Mac's ``coverage_compute`` kernel, in chunks."""
    pos = np.asarray(positions, np.float32)
    n = len(pos)
    cnt = np.zeros(n, np.float32)
    best = np.zeros(n, np.float32)
    if n == 0 or not cameras:
        return cnt, best
    total = (n + chunk - 1) // chunk
    for k, s in enumerate(range(0, n, chunk)):
        if cancelled is not None and cancelled():
            raise CoverageCancelled()
        p = pos[s:s + chunk]
        c_cnt = cnt[s:s + chunk]
        c_best = best[s:s + chunk]
        for cam in cameras:
            R = cam.w2c[:, :3].astype(np.float32)
            t = cam.w2c[:, 3].astype(np.float32)
            z = p @ R[2] + t[2]
            front = z >= 1e-3
            if not front.any():
                continue
            zi = np.where(front, z, 1.0)
            u = cam.fx * (p @ R[0] + t[0]) / zi + cam.cx
            v = cam.fy * (p @ R[1] + t[1]) / zi + cam.cy
            ok = front & (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
            if cam.depth is not None and ok.any():
                dh, dw = cam.depth.shape
                idx = np.nonzero(ok)[0]
                du = np.minimum((u[idx] * (dw / cam.width)).astype(np.int64), dw - 1)
                dv = np.minimum((v[idx] * (dh / cam.height)).astype(np.int64), dh - 1)
                d = cam.depth[dv, du].astype(np.float32)
                hidden = (d > 0) & (z[idx] > d * 1.1 + 0.05)      # behind the surface this camera saw
                ok[idx[hidden]] = False
            c_cnt += ok
            np.maximum(c_best, np.where(ok, cam.fx / zi, 0.0), out=c_best)
        if progress is not None:
            progress((k + 1) / total)
    return cnt, best


def scores(cnt: np.ndarray, best: np.ndarray, opacities: np.ndarray, sample: int = 200_000):
    """Score per splat (0–1) and the median detail and views of reasonably opaque splats (a strided sample)."""
    n = len(cnt)
    stride = max(1, n // sample)
    sel = np.arange(0, n, stride)
    sel = sel[opacities[sel] > 0.1]
    views_s = cnt[sel]
    seen_best = best[sel][views_s > 0]
    median_best = float(np.sort(seen_best)[len(seen_best) // 2]) if len(seen_best) else 1.0
    median_views = int(np.sort(views_s)[len(views_s) // 2]) if len(views_s) else 0
    v = np.clip((cnt - POOR_VIEWS) / (GOOD_VIEWS - POOR_VIEWS), 0, 1)
    lp = math.log2(POOR_DETAIL_RATIO)
    ratio = np.maximum(best / max(median_best, 1e-12), 1e-6)
    d = np.clip((np.log2(ratio) - lp) / -lp, 0, 1)
    s = np.where(cnt > 0, np.minimum(v, d), 0.0).astype(np.float32)
    return s, median_best, median_views


def compute(cloud, dataset_dir: Path, chunk: int = 200_000, use_depth: bool = True,
            progress: Optional[Callable[[float], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> CoverageResult:
    """Coverage of ``cloud`` (a :class:`~.splats.GaussianCloud`) by the cameras of the dataset in ``dataset_dir``."""
    cams = load_cameras(dataset_dir, use_depth)
    if not cams:
        raise ValueError("the dataset has no cameras")
    if cloud.count == 0:
        raise ValueError("the scene has no splats")
    cnt, best = raw_counts(cloud.positions, cams, chunk, progress, cancelled)
    op = cloud.opacities.astype(np.float32)
    s, _mb, median_views = scores(cnt, best, op)
    # Summarise by visible size, not splat count (poorly seen areas have few, large splats); size, not area,
    # so a few huge floaters don't dominate. sqrt(trace(Σ)) = sqrt(Σ exp(2·log scale)).
    poor = unseen = counted = 0.0
    for a in range(0, cloud.count, chunk):
        o = op[a:a + chunk]
        m = o > 0.1
        w = np.sqrt(np.exp(2.0 * cloud.log_scales[a:a + chunk].astype(np.float64)).sum(1)) * o
        w = np.where(m, w, 0.0)
        c = cnt[a:a + chunk]
        counted += float(w.sum())
        unseen += float(w[c == 0].sum())
        poor += float(w[(c > 0) & (s[a:a + chunk] < POOR_SCORE)].sum())
    return CoverageResult(score=s, views=np.minimum(cnt, 65535).astype(np.uint16),
                          poor_fraction=poor / counted if counted > 0 else 0.0,
                          unseen_fraction=unseen / counted if counted > 0 else 0.0,
                          median_views=median_views, used_depth=any(c.depth is not None for c in cams))


def ramp(score: float):
    """Display colour of a score (the shader's mapping), for legends and tests."""
    c = min(max(float(score), 0.0), 1.0)
    a, b, g = (np.array(x) for x in RAMP)
    return tuple(a + (b - a) * c * 2) if c < 0.5 else tuple(b + (g - b) * (c - 0.5) * 2)
