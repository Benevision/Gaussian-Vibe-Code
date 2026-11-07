"""Procedural synthetic dataset generator with exact ground-truth cameras (numpy + Pillow).

Scene (world: right-handed, +Y up, metres):

* textured ground plane (multi-octave noise),
* back wall with a *repeated* brick pattern (stresses matchers),
* boxes: rich noise, coloured noise, checkerboard (repeated), *low-texture* flat box, striped box,
* a noise-textured sphere,
* smooth gradient sky (texture-less).

Rendering is a vectorised ray caster with Lambert shading, optional supersampling,
and per-frame photometric variation (exposure, white balance, gamma, vignetting, noise).

Trajectories: ``orbit`` (360° ring with height wobble), ``walk`` (forward walk with yaw sway,
video-like frame names), ``sparse`` (15 wide-baseline views on a hemisphere).

Output (a valid prepared dataset, PROTOCOL §2, doubling as ground truth)::

    out/images/*.png     out/dataset.json     out/points3d.ply   out/synth.json   [out/depth/*.bin]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .export import scene_metrics
from .geometry import look_at
from .ply import write_points_ply

# ---------------------------------------------------------------------------
# textures


def _value_noise(rng: np.random.Generator, size: int, octaves: Tuple[int, ...], channels: int = 1) -> np.ndarray:
    from PIL import Image

    acc = np.zeros((size, size, channels), np.float32)
    amp_total = 0.0
    for k, cells in enumerate(octaves):
        amp = 0.6**k
        for c in range(channels):
            g = rng.random((cells, cells)).astype(np.float32)
            up = np.asarray(Image.fromarray(g).resize((size, size), Image.BICUBIC), np.float32)
            acc[..., c] += amp * up
        amp_total += amp
    acc /= amp_total
    acc -= acc.min()
    acc /= acc.max() + 1e-6
    return acc


def make_textures(rng: np.random.Generator, size: int = 512) -> Dict[str, np.ndarray]:
    """All textures are HxWx3 float32 in [0, 1] (linear albedo)."""
    tex: Dict[str, np.ndarray] = {}
    n = _value_noise(rng, size, (8, 32, 128, 256), 1)
    tex["ground"] = np.concatenate([0.35 + 0.35 * n, 0.32 + 0.30 * n, 0.25 + 0.25 * n], 2)
    # bricks (repeated pattern) with tiny noise
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rows = np.floor(yy * 16)
    shift = (rows % 2) * 0.5
    bx = ((xx * 8 + shift) % 1.0)
    by = (yy * 16) % 1.0
    mortar = (bx < 0.06) | (by < 0.12)
    fine = _value_noise(rng, size, (256,), 1)[..., 0]
    brick = np.stack([0.62 + 0.05 * fine, 0.30 + 0.04 * fine, 0.22 + 0.03 * fine], 2)
    brick[mortar] = [0.75, 0.74, 0.70]
    tex["brick"] = brick
    tex["noise"] = _value_noise(rng, size, (16, 64, 256), 1).repeat(3, 2) * 0.9 + 0.05
    tex["colornoise"] = _value_noise(rng, size, (8, 32, 128, 256), 3) * 0.85 + 0.1
    check = ((np.floor(xx * 8) + np.floor(yy * 8)) % 2).astype(np.float32)[..., None]
    tex["checker"] = np.concatenate([0.1 + 0.8 * check, 0.1 + 0.8 * check, 0.15 + 0.7 * check], 2)
    tex["flat"] = np.stack([0.55 + 0.03 * yy, 0.6 + 0.02 * xx, 0.65 + 0.0 * xx], 2)  # low texture
    stripes = (0.5 + 0.5 * np.sin(xx * 2 * np.pi * 12)).astype(np.float32)[..., None]
    sn = _value_noise(rng, size, (64, 256), 1)
    tex["stripes"] = np.concatenate([0.2 + 0.6 * stripes * 0.8 + 0.2 * sn, 0.3 + 0.3 * sn, 0.7 - 0.4 * stripes], 2).clip(0, 1)
    return {k: v.astype(np.float32) for k, v in tex.items()}


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear, wrapping texture lookup; u, v in texture repeats."""
    H, W = tex.shape[:2]
    x = (u % 1.0) * W - 0.5
    y = (v % 1.0) * H - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)
    x0 %= W
    y0 %= H
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    return (tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy


# ---------------------------------------------------------------------------
# scene


@dataclass
class Box:
    center: np.ndarray
    half: np.ndarray
    yaw: float
    texture: str
    tex_scale: float = 1.0

    def R(self) -> np.ndarray:  # local -> world
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


SCENE_BOXES = [
    Box(np.array([0.0, 0.6, 0.0]), np.array([0.6, 0.6, 0.6]), np.radians(20), "noise", 1.2),
    Box(np.array([-1.8, 0.4, 0.8]), np.array([0.4, 0.4, 0.9]), np.radians(-35), "colornoise", 1.8),
    Box(np.array([1.7, 0.9, -0.6]), np.array([0.5, 0.9, 0.5]), np.radians(10), "checker", 1.0),
    Box(np.array([0.9, 0.25, 1.6]), np.array([0.5, 0.25, 0.35]), np.radians(60), "flat", 1.0),
    Box(np.array([-1.0, 1.2, -1.8]), np.array([0.7, 1.2, 0.4]), 0.0, "stripes", 1.0),
]
SPHERE = (np.array([-0.4, 0.45, 1.7]), 0.45, "colornoise")
WALL_Z = -7.0
GROUND_EXTENT = 14.0
LIGHT = np.array([0.4, 0.8, 0.45]) / np.linalg.norm([0.4, 0.8, 0.45])
SCENE_TARGET = np.array([0.0, 0.6, 0.0])


def render(
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    textures: Dict[str, np.ndarray],
    supersample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear RGB float image (HxWx3) and Z-depth (HxW, inf = sky)."""
    ss = max(1, int(supersample))
    W, H = width * ss, height * ss
    fx, fy, cx, cy = K[0, 0] * ss, K[1, 1] * ss, K[0, 2] * ss, K[1, 2] * ss
    u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)  # camera rays, z = 1
    R, o = c2w[:3, :3], c2w[:3, 3]
    d = dc @ R.T  # world directions (not normalised: t is then Z-depth)
    N = d.shape[0]
    t_best = np.full(N, np.inf)
    color = np.zeros((N, 3), np.float32)
    normal = np.zeros((N, 3), np.float32)

    def commit(t, mask, col_fn, nrm):
        upd = mask & (t > 1e-4) & (t < t_best)
        if not upd.any():
            return
        idx = np.nonzero(upd)[0]
        t_best[idx] = t[idx]
        P = o + d[idx] * t[idx, None]
        color[idx] = col_fn(P, idx)
        normal[idx] = nrm(P, idx) if callable(nrm) else nrm

    with np.errstate(divide="ignore", invalid="ignore"):
        # ground y = 0
        t = -o[1] / d[:, 1]
        P = o + d * t[:, None]
        m = (np.abs(P[:, 0]) < GROUND_EXTENT) & (np.abs(P[:, 2]) < GROUND_EXTENT)
        def ground_col(P, i):
            # Two detail layers with incommensurate periods and a 37° twist, so the ground never
            # repeats exactly (a single 1.5 m tile made orbits "fold" onto themselves in SfM —
            # repeated structure belongs to the brick wall / checker box, not the whole floor).
            a = _sample(textures["ground"], P[:, 0] / 1.5, P[:, 2] / 1.5)
            c, s = np.cos(0.65), np.sin(0.65)
            u, v = (c * P[:, 0] - s * P[:, 2]) / 2.37 + 0.31, (s * P[:, 0] + c * P[:, 2]) / 2.37 + 0.17
            b = _sample(textures["colornoise"], u, v).mean(1, keepdims=True) * np.array([0.95, 0.88, 0.72], np.float32)
            return np.clip(0.5 + 1.6 * (0.55 * a + 0.45 * b - 0.5), 0, 1)

        commit(t, m, ground_col, np.array([0, 1, 0], np.float32))
        # back wall z = WALL_Z (faces +z)
        t = (WALL_Z - o[2]) / d[:, 2]
        P = o + d * t[:, None]
        m = (np.abs(P[:, 0]) < 8.0) & (P[:, 1] > 0) & (P[:, 1] < 4.0)
        commit(t, m, lambda P, i: _sample(textures["brick"], P[:, 0] / 2.0, -P[:, 1] / 2.0), np.array([0, 0, 1], np.float32))
        # boxes (slab test in local frame)
        for box in SCENE_BOXES:
            Rb = box.R()
            ol = (o - box.center) @ Rb
            dl = d @ Rb
            t1 = (-box.half - ol) / dl
            t2 = (box.half - ol) / dl
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmax > 0)
            tex = textures[box.texture]

            def col(P, idx, box=box, Rb=Rb, tex=tex):
                L = (P - box.center) @ Rb / box.half
                ax = np.argmax(np.abs(L), axis=1)
                Ls = (P - box.center) @ Rb
                uu = np.where(ax == 0, Ls[:, 2], Ls[:, 0])
                vv = np.where(ax == 1, Ls[:, 2], Ls[:, 1])
                face = ax * 2 + (np.take_along_axis(L, ax[:, None], 1)[:, 0] > 0)
                return _sample(tex, uu / box.tex_scale + 0.37 * face, vv / box.tex_scale + 0.21 * face)

            def nrm(P, idx, box=box, Rb=Rb):
                L = (P - box.center) @ Rb / box.half
                ax = np.argmax(np.abs(L), axis=1)
                n = np.zeros_like(L)
                n[np.arange(len(L)), ax] = np.sign(L[np.arange(len(L)), ax])
                return (n @ Rb.T).astype(np.float32)

            commit(np.where(tmin > 0, tmin, tmax), hit, col, nrm)
        # sphere
        c, r, tname = SPHERE
        oc = o - c
        a = (d * d).sum(1)
        b = 2 * (d @ oc)
        cc = oc @ oc - r * r
        disc = b * b - 4 * a * cc
        hit = disc > 0
        sq = np.sqrt(np.maximum(disc, 0))
        t = (-b - sq) / (2 * a)

        def scol(P, idx):
            n = (P - c) / r
            return _sample(textures[tname], np.arctan2(n[:, 2], n[:, 0]) / (2 * np.pi), np.arccos(np.clip(n[:, 1], -1, 1)) / np.pi)  # one wrap: no repeats

        commit(t, hit, scol, lambda P, idx: ((P - c) / r).astype(np.float32))

    sky = ~np.isfinite(t_best)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    elev = np.clip(dn[:, 1], 0, 1)[:, None].astype(np.float32)
    sky_col = (1 - elev) * np.array([0.80, 0.86, 0.95], np.float32) + elev * np.array([0.35, 0.55, 0.90], np.float32)
    shade = 0.35 + 0.65 * np.clip(normal @ LIGHT.astype(np.float32), 0, 1)
    rgb = np.where(sky[:, None], sky_col, color * shade[:, None])
    rgb = rgb.reshape(H, W, 3)
    depth = t_best.reshape(H, W)
    if ss > 1:
        rgb = rgb.reshape(height, ss, width, ss, 3).mean((1, 3))
        depth = depth.reshape(height, ss, width, ss)[:, ss // 2, :, ss // 2]
    return rgb.astype(np.float32), depth


def photometric(rgb_linear: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Exposure, white balance, gamma, vignetting and sensor noise -> uint8 sRGB-ish."""
    H, W = rgb_linear.shape[:2]
    gain = float(np.exp(rng.normal(0, 0.25 * strength)))
    wb = (1 + rng.normal(0, 0.05 * strength, 3)).astype(np.float32)
    gamma = 2.2 + float(rng.normal(0, 0.1 * strength))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    vig = 1 - 0.25 * strength * r2 / 2
    img = np.clip(rgb_linear * gain * wb * vig[..., None], 0, 1) ** (1 / gamma)
    img = img + rng.normal(0, 1.5 * strength / 255.0, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# trajectories


def trajectory(kind: str, n: Optional[int], rng: np.random.Generator) -> Tuple[List[np.ndarray], List[str]]:
    poses, names = [], []
    if kind == "orbit":
        n = n or 60
        for i in range(n):
            th = 2 * np.pi * i / n
            r = 5.0  # stays inside the back wall (z = WALL_Z) all the way round
            eye = np.array([r * np.sin(th), 2.3 + 0.35 * np.sin(2 * th), r * np.cos(th)])
            target = SCENE_TARGET + rng.normal(0, 0.05, 3)
            poses.append(look_at(eye, target))
            names.append(f"orbit_{i:04d}.png")
    elif kind == "walk":
        n = n or 60
        for i in range(n):
            s = i / max(1, n - 1)
            eye = np.array([0.6 * np.sin(2 * np.pi * s), 1.4 + 0.03 * np.sin(14 * np.pi * s), 6.5 - 3.5 * s])
            yaw = np.radians(10) * np.sin(4 * np.pi * s)
            fwd = np.array([np.sin(yaw), -0.15, -np.cos(yaw)])
            poses.append(look_at(eye, eye + fwd))
            names.append(f"frame_{i:05d}.png")
    elif kind == "sparse":
        n = n or 15
        az = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.15, 0.15, n)
        for i in range(n):
            elev = np.radians(rng.uniform(12, 50))
            r = rng.uniform(3.8, 5.5)
            eye = SCENE_TARGET + r * np.array([np.cos(elev) * np.sin(az[i]), np.sin(elev), np.cos(elev) * np.cos(az[i])])
            eye[1] = max(eye[1], 0.4)
            poses.append(look_at(eye, SCENE_TARGET + rng.normal(0, 0.25, 3)))
            names.append(f"view_{i:02d}.png")
        order = rng.permutation(n)  # sparse sets are unordered
        poses = [poses[k] for k in order]
        names = [f"view_{k:02d}.png" for k in range(n)]
    else:
        raise ValueError(f"unknown trajectory {kind!r} (orbit | walk | sparse)")
    return poses, names


# ---------------------------------------------------------------------------


def generate(
    out_dir: Path,
    trajectory_kind: str = "orbit",
    num_views: Optional[int] = None,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    photometric_strength: float = 0.5,
    supersample: int = 2,
    write_depth: bool = False,
    hfov_deg: float = 60.0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    textures = make_textures(np.random.default_rng(seed + 1000))
    poses, names = trajectory(trajectory_kind, num_views, rng)
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    from PIL import Image

    cams = []
    pts, cols = [], []
    for i, (c2w, name) in enumerate(zip(poses, names)):
        lin, depth = render(c2w, K, width, height, textures, supersample)
        img = photometric(lin, np.random.default_rng(seed * 100003 + i), photometric_strength)
        Image.fromarray(img).save(out_dir / "images" / name, optimize=False)
        cam = {
            "id": i,
            "image": f"images/{name}",
            "width": width,
            "height": height,
            "fx": float(f),
            "fy": float(f),
            "cx": width / 2.0,
            "cy": height / 2.0,
            "cameraToWorld": [[float(v) for v in row] for row in c2w],
            "holdout": False,
        }
        if write_depth:
            (out_dir / "depth").mkdir(exist_ok=True)
            dpath = out_dir / "depth" / (Path(name).stem + ".bin")
            np.where(np.isfinite(depth), depth, 0).astype("<f2").tofile(dpath)
            cam.update({"depth": f"depth/{dpath.name}", "depthWidth": width, "depthHeight": height})
        cams.append(cam)
        # sample GT surface points for points3d.ply
        if i % max(1, len(poses) // 8) == 0:
            sub = np.random.default_rng(seed + i).choice(width * height, min(3000, width * height), replace=False)
            vv, uu = np.divmod(sub, width)
            z = depth.reshape(-1)[sub]
            ok = np.isfinite(z)
            xc = np.stack([(uu + 0.5 - K[0, 2]) / f * z, (vv + 0.5 - K[1, 2]) / f * z, z], 1)[ok]
            pts.append(xc @ c2w[:3, :3].T + c2w[:3, 3])
            cols.append(img.reshape(-1, 3)[sub][ok])
        if progress:
            progress((i + 1) / len(poses), f"rendered {name}")
    c2ws = np.stack(poses)
    center, extent, up = scene_metrics(c2ws)
    xyz = np.concatenate(pts) if pts else np.zeros((0, 3))
    rgb = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    write_points_ply(out_dir / "points3d.ply", xyz, rgb)
    ds = {
        "format": "splatdataset",
        "version": 1,
        "convention": "opencv",
        "sceneCenter": center.tolist(),
        "sceneExtent": extent,
        "upVector": up.tolist(),
        "points": "points3d.ply",
        "cameras": cams,
    }
    (out_dir / "dataset.json").write_text(json.dumps(ds, indent=2))
    meta = {
        "format": "splatsynth",
        "version": 1,
        "trajectory": trajectory_kind,
        "numViews": len(poses),
        "width": width,
        "height": height,
        "seed": seed,
        "photometricStrength": photometric_strength,
        "supersample": supersample,
        "hfovDeg": hfov_deg,
        "worldUp": [0, 1, 0],
        "groundTruth": "dataset.json",
    }
    (out_dir / "synth.json").write_text(json.dumps(meta, indent=2))
    return meta


def dataset_hash(out_dir: Path) -> str:
    """SHA-256 over images + dataset.json (determinism checks)."""
    h = hashlib.sha256()
    out_dir = Path(out_dir)
    for p in sorted((out_dir / "images").iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update((out_dir / "dataset.json").read_bytes())
    return h.hexdigest()
