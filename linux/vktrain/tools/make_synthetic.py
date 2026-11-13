#!/usr/bin/env python3
"""Deterministic synthetic training dataset (docs/PROTOCOL.md §2) for the Vulkan trainer's end-to-end tests.

A ray-cast scene — a round checkered ground plate, three striped and shaded spheres, a smooth backdrop sphere around it all —
seen by cameras on a ring, written as images/*.png, dataset.json and a noisy coloured points3d.ply (as a
camera solve would give). Every ``--holdout-every``-th camera is a holdout; camera 1 has a "moving object"
(a magenta block in its photo only) and a mask that removes it. numpy + Pillow only.

Options for the stage-2b features: --sky (sky at infinity + sky masks), --color-shift (per-photo exposure / tint
on the training photos), --depth (float16 sensor depth maps), --pose-noise SIGMA (perturbed training poses),
--transients N (unmasked moving objects in N training photos).

    python3 make_synthetic.py OUT_DIR [--cameras 16] [--width 128] [--height 96] [--points 3000] [feature options]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

SPHERES = [  # centre, radius, base colour, stripe frequency
    ((0.0, 0.35, 0.0), 0.65, (0.85, 0.25, 0.2), 9.0),
    ((-1.1, 0.6, 0.6), 0.4, (0.2, 0.55, 0.9), 14.0),
    ((1.0, 0.55, -0.5), 0.45, (0.3, 0.8, 0.3), 6.0),
]
GROUND_Y = 1.0          # OpenCV-style world: +y is down, the ground is below the spheres
GROUND_R = 3.0          # a round ground plate; beyond it the backdrop
BACKDROP_R = 7.0
LIGHT = np.array([0.4, -1.0, 0.3]) / np.linalg.norm([0.4, -1.0, 0.3])


def look_at(eye, target):
    """OpenCV camera-to-world (x right, y down, z forward), world up = −y."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    up = np.array([0.0, -1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def shade_sphere(p, k):
    c, rad, base, freq = SPHERES[k]
    n = (p - np.asarray(c)) / rad
    lat = np.arcsin(np.clip(-n[:, 1], -1, 1))
    lon = np.arctan2(n[:, 2], n[:, 0])
    stripe = 0.5 + 0.5 * np.sin(freq * lat + 2.0 * np.sin(3 * lon))
    lam = 0.35 + 0.65 * np.clip(n @ -LIGHT, 0, 1)
    col = np.asarray(base)[None] * (0.55 + 0.45 * stripe[:, None]) * lam[:, None]
    return np.clip(col, 0, 1)


def shade_ground(p):
    chk = (np.floor(p[:, 0] / 0.5) + np.floor(p[:, 2] / 0.5)) % 2
    a, b = np.array([0.88, 0.82, 0.7]), np.array([0.25, 0.3, 0.38])
    col = np.where(chk[:, None] > 0, a[None], b[None])
    # soft shadow-ish darkening under the big sphere
    d = np.hypot(p[:, 0], p[:, 2])
    return np.clip(col * (0.6 + 0.4 * np.clip(d / 1.2, 0, 1))[:, None], 0, 1)


def shade_backdrop(d):
    return np.clip(np.stack([0.55 + 0.25 * d[:, 0], 0.6 - 0.3 * d[:, 1], 0.7 + 0.2 * d[:, 2]], 1), 0, 1)


def shade_sky(d):
    """Sky at infinity (--sky): a colour of the viewing direction only — brighter towards the horizon, with bands."""
    el = np.clip(-d[:, 1], 0, 1)                 # up = −y
    az = np.arctan2(d[:, 2], d[:, 0])
    band = 0.08 * np.sin(4 * az) * (1 - el)
    return np.clip(np.stack([0.55 - 0.35 * el + band, 0.68 - 0.25 * el + band, 0.92 - 0.1 * el - band], 1), 0, 1)


def trace(o, d, sky=False, info=False):
    """Colour of rays o + t d (o [3], d [N, 3] unit). With sky, rays that leave the scene above the horizon see
    shade_sky. info=True also returns the hit distance (inf for the sky) and the sky flags."""
    n = d.shape[0]
    best = np.full(n, np.inf)
    col = shade_backdrop(d)
    is_sky = np.zeros(n, bool)
    if sky:
        is_sky = -d[:, 1] > 0.02
        col[is_sky] = shade_sky(d[is_sky])
    # backdrop sphere (from inside): far root
    b = d @ o
    cc = o @ o - BACKDROP_R ** 2
    t_bd = -b + np.sqrt(np.maximum(b * b - cc, 0))
    best = np.where((t_bd > 0) & ~is_sky, t_bd, best)
    # ground plane
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (GROUND_Y - o[1]) / d[:, 1]
    hit = (d[:, 1] > 1e-6) & (t > 0) & (t < best)
    if hit.any():
        p = o + t[hit, None] * d[hit]
        g = np.hypot(p[:, 0], p[:, 2]) < GROUND_R
        idx = np.nonzero(hit)[0][g]
        col[idx] = shade_ground(p[g])
        best[idx] = t[hit][g]
    for k, (c, rad, _, _) in enumerate(SPHERES):
        oc = o - np.asarray(c)
        b = d @ oc
        cc = oc @ oc - rad * rad
        disc = b * b - cc
        t = -b - np.sqrt(np.maximum(disc, 0))
        hit = (disc > 0) & (t > 0) & (t < best)
        if hit.any():
            p = o + t[hit, None] * d[hit]
            col[hit] = shade_sphere(p, k)
            best[hit] = t[hit]
            is_sky[hit] = False
    if sky:
        is_sky &= ~np.isfinite(best)
    return (col, best, is_sky) if info else col


def render(c2w, fx, fy, cx, cy, w, h, ss=3, sky=False):
    """Supersampled (ss × ss per pixel) render; pixel (i, j) covers [i, i+1) × [j, j+1). Returns the image and the
    fraction of each pixel that sees the sky."""
    acc = np.zeros((h, w, 3))
    sky_frac = np.zeros((h, w))
    for sy in range(ss):
        for sx in range(ss):
            u = np.arange(w) + (sx + 0.5) / ss
            v = np.arange(h) + (sy + 0.5) / ss
            uu, vv = np.meshgrid(u, v)
            dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).reshape(-1, 3)
            dw = dc @ c2w[:3, :3].T
            dw /= np.linalg.norm(dw, axis=1, keepdims=True)
            col, _, s = trace(c2w[:3, 3], dw, sky=sky, info=True)
            acc += col.reshape(h, w, 3)
            sky_frac += s.reshape(h, w)
    return acc / (ss * ss), sky_frac / (ss * ss)


def depth_map(c2w, fx, fy, cx, cy, w, h, sky=False, max_range=6.0):
    """Camera-space z of the first surface at the pixel centres of a w × h sensor (0 = no return: sky or beyond
    max_range, like a LiDAR)."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
    n = np.linalg.norm(dc, axis=1)
    dw = dc @ c2w[:3, :3].T / n[:, None]
    _, t, _ = trace(c2w[:3, 3], dw, sky=sky, info=True)
    z = t / n
    z[~np.isfinite(z) | (z > max_range)] = 0
    return z.reshape(h, w)


def small_rotation(rng, sigma):
    a = rng.normal(scale=sigma, size=3)
    th = np.linalg.norm(a)
    if th < 1e-12:
        return np.eye(3)
    k = a / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def surface_points(rng, n, sky=False):
    pts, cols = [], []
    n_s = int(n * 0.45)
    for k, (c, rad, _, _) in enumerate(SPHERES):
        m = max(1, int(n_s * rad / sum(s[1] for s in SPHERES)))
        v = rng.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        p = np.asarray(c) + rad * v
        p = p[p[:, 1] < GROUND_Y]
        pts.append(p)
        cols.append(shade_sphere(p, k))
    m = int(n * 0.4)
    r = GROUND_R * np.sqrt(rng.random(m))
    a = rng.random(m) * 2 * np.pi
    p = np.stack([r * np.cos(a), np.full(m, GROUND_Y), r * np.sin(a)], 1)
    pts.append(p)
    cols.append(shade_ground(p))
    m = n - sum(len(x) for x in pts)
    v = rng.normal(size=(m, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if sky:   # no backdrop above the horizon: that is sky at infinity (no points)
        v = v[-v[:, 1] <= 0.02]
    pts.append(v * BACKDROP_R * 0.98)
    cols.append(shade_backdrop(v))
    p = np.concatenate(pts)
    c = np.concatenate(cols)
    p = p + rng.normal(scale=0.01, size=p.shape)   # a camera solve's points are not exact
    c = np.clip(c + rng.normal(scale=0.03, size=c.shape), 0, 1)
    return p.astype(np.float32), (c * 255).round().astype(np.uint8)


def write_points(path: Path, p, c):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(p)).encode())
        for (x, y, z), (r, g, b) in zip(p, c):
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--cameras", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # stage 2b features (each off by default; the default dataset is unchanged)
    ap.add_argument("--sky", action="store_true", help="sky at infinity above the horizon, sky masks, cameras aimed higher")
    ap.add_argument("--color-shift", action="store_true", help="a different exposure / tint per training photo")
    ap.add_argument("--depth", action="store_true", help="depth/*.bin sensor depth (float16, half resolution)")
    ap.add_argument("--pose-noise", type=float, default=0.0, help="perturb the training cameras' poses (rad / scene units)")
    ap.add_argument("--transients", type=int, default=0, help="unmasked moving objects in this many training photos")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    frng = np.random.default_rng(a.seed + 1000)   # the features' own numbers: the default dataset stays the same
    w, h = a.width, a.height
    fx = fy = 0.9 * w
    cx, cy = w / 2, h / 2
    cams = []
    train_ids = [i for i in range(a.cameras) if not (a.holdout_every > 0 and i % a.holdout_every == a.holdout_every - 1)]
    transient_ids = set(train_ids[2::max(1, len(train_ids) // max(a.transients, 1))][:a.transients]) if a.transients else set()
    for i in range(a.cameras):
        ang = 2 * math.pi * i / a.cameras
        height = -0.9 - 0.5 * math.sin(3 * ang)   # up and down a little: not all views in one plane
        rad = 3.6 + 0.4 * math.cos(2 * ang)
        eye = np.array([rad * math.cos(ang), height, rad * math.sin(ang)])
        c2w = look_at(eye, (0.0, -0.6 if a.sky else 0.4, 0.0))
        img, sky_frac = render(c2w, fx, fy, cx, cy, w, h, sky=a.sky)
        holdout = a.holdout_every > 0 and i % a.holdout_every == a.holdout_every - 1
        cam = {"id": i, "image": f"images/{i:03d}.png", "width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
               "cameraToWorld": c2w.tolist(), "holdout": holdout}
        if i == 1:   # a "moving object" only this photo has, removed by its mask
            y0, y1, x0, x1 = h // 3, h // 3 + h // 4, w // 5, w // 5 + w // 4
            img[y0:y1, x0:x1] = (1.0, 0.0, 1.0)
            m = np.full((h, w), 255, np.uint8)
            m[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 0
            Image.fromarray(m).save(out / "masks" / f"{i:03d}.png")
            cam["mask"] = f"masks/{i:03d}.png"
        if a.sky:
            (out / "sky").mkdir(exist_ok=True)
            Image.fromarray((sky_frac * 255).round().astype(np.uint8)).save(out / "sky" / f"{i:03d}.png")
            cam["sky"] = f"sky/{i:03d}.png"
        if a.depth:
            (out / "depth").mkdir(exist_ok=True)
            dw, dh = w // 2, h // 2
            z = depth_map(c2w, fx * dw / w, fy * dh / h, cx * dw / w, cy * dh / h, dw, dh, sky=a.sky)
            z.astype("<f2").tofile(out / "depth" / f"{i:03d}.bin")
            cam.update({"depth": f"depth/{i:03d}.bin", "depthWidth": dw, "depthHeight": dh})
        if i in transient_ids:   # a moving object (unmasked): a solid block of a random colour somewhere in the photo
            bw, bh = w // 4, h // 3
            x0, y0 = int(frng.integers(0, w - bw)), int(frng.integers(0, h - bh))
            img[y0:y0 + bh, x0:x0 + bw] = frng.uniform(0, 1, 3)
        if a.color_shift and not holdout:   # exposure / white balance of this photo (holdouts: the scene's own)
            gain = frng.uniform(0.7, 1.3) * frng.uniform(0.9, 1.1, 3)
            img = img * gain + frng.uniform(-0.03, 0.03, 3)
        if a.pose_noise > 0 and not holdout:   # the photo was taken from c2w, but the dataset says this
            noisy = c2w.copy()
            noisy[:3, :3] = small_rotation(frng, a.pose_noise) @ c2w[:3, :3]
            noisy[:3, 3] += frng.normal(scale=a.pose_noise, size=3)
            cam["cameraToWorld"] = noisy.tolist()
        Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(out / cam["image"])
        cams.append(cam)
    p, c = surface_points(rng, a.points, sky=a.sky)
    write_points(out / "points3d.ply", p, c)
    ds = {"format": "splatdataset", "version": 1, "convention": "opencv", "sceneCenter": [0.0, 0.4, 0.0],
          "sceneExtent": 4.0, "upVector": [0.0, -1.0, 0.0], "points": "points3d.ply", "cameras": cams}
    (out / "dataset.json").write_text(json.dumps(ds, indent=1))
    print(f"{out}: {len(cams)} cameras ({sum(c['holdout'] for c in cams)} holdout), {w}x{h}, {len(p)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
