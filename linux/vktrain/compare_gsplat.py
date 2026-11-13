#!/usr/bin/env python3
"""Compare the Vulkan trainer's kernels with gsplat (CUDA) on one random scene.

Renders the same scene through gsplat.rasterization (packed=False, eps2d=0.3, its OpenCV camera convention) and
through ``splatwork-vk-train --dump``, then prints max / mean differences of the image and of the gradients of
L = sum(W * image) (W random) w.r.t. means, log-scales, quaternions, opacity logits and SH coefficients.

Three steps, so gsplat can run inside the Flatpak's GPU runtime while the Vulkan program runs on the host:

    python compare_gsplat.py make-scene --dir D [--n 2000 --width 320 --height 240 --sh-degree 3 --antialiased]
    splatwork-vk-train --dump D/scene.bin D/vk_out.bin           # host (and --atomics cas -> D/vk_out_cas.bin)
    python compare_gsplat.py compare --dir D                      # needs torch + gsplat + CUDA

``all`` does the three in one go when one Python has gsplat and the binary can run:
    python compare_gsplat.py all --dir D --vk-train build/splatwork-vk-train

On the RTX 4080 PC (fish shell), see docs/dev/VULKAN_TRAINER.md for the exact commands.
``compare --no-gsplat`` only checks the Vulkan output against the double-precision CPU reference image that
the dump also contains (used by linux/tests/test_vktrain.py).

Expected differences (by design, not bugs): the Vulkan kernels follow the Mac trainer, which clamps alpha at
0.99 (gsplat 0.999; this script keeps opacities <= 0.9 so neither clamp triggers) and bounds each splat's tiles
by its k-sigma ellipse with k = min(3, sqrt(2 ln(255 o))); pixels near a splat's 3-sigma edge can differ by a
few 1e-3. The EWA Jacobian clamp is symmetric (1.3 tan half-fov) - equal to gsplat's for a centred principal
point, which this script uses.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

SCENE_MAGIC = b"SWVS"
OUT_MAGIC = b"SWVO"


# ------------------------------------------------------------------ scene

def look_at(eye, target=(0.0, 0.0, 0.0), roll=0.0):
    """World-to-camera 4x4 (OpenCV: x right, y down, z forward) looking from eye at target."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    up = np.array([0.0, -1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * r + s * d, -s * r + c * d
    w2c = np.eye(4)
    w2c[0, :3], w2c[1, :3], w2c[2, :3] = x, y, f
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def make_scene(n=2000, width=320, height=240, sh_degree=3, antialiased=False, seed=0):
    rng = np.random.default_rng(seed)
    eye = np.array([0.7, -0.9, -4.0])
    u = rng.random(n)
    means = rng.uniform(-1.5, 1.5, (n, 3))
    back = eye / np.linalg.norm(eye)
    behind = (u >= 0.85) & (u < 0.92)            # behind / next to the camera
    means[behind] = eye + back * rng.uniform(0.05, 1.5, (behind.sum(), 1)) + rng.uniform(-0.4, 0.4, (behind.sum(), 3))
    side = u >= 0.92                              # beside the view: partly / fully off-screen
    a = rng.uniform(0, 2 * np.pi, side.sum())
    m = rng.uniform(2.0, 3.5, side.sum())
    means[side] = np.stack([np.cos(a) * m, np.sin(a) * m, rng.uniform(-0.5, 0.5, side.sum())], 1)
    log_scales = rng.uniform(-4.0, -2.0, (n, 3))
    quats = rng.normal(size=(n, 4)) * rng.uniform(0.5, 2.0, (n, 1))
    op_logits = rng.uniform(-2.5, 2.2, n)       # sigmoid <= 0.9: no alpha clamp on either side
    sh = rng.uniform(-0.3, 0.3, (n, 16, 3))
    sh[:, 0, :] = rng.uniform(-1.2, 1.2, (n, 3))
    fx = fy = 0.95 * width
    cam = dict(w2c=look_at(eye, roll=0.2), fx=fx, fy=fy, cx=width / 2, cy=height / 2, width=width, height=height)
    dl_dimg = rng.normal(size=(3, height, width)) / (width * height)
    return dict(means=means, log_scales=log_scales, quats=quats, op_logits=op_logits, sh=sh, cam=cam,
                sh_degree=sh_degree, antialiased=antialiased, bg=np.array([0.1, 0.2, 0.3]), dilation=0.3,
                dl_dimg=dl_dimg)


def write_scene(path: Path, sc) -> None:
    n = len(sc["means"])
    cam = sc["cam"]
    with open(path, "wb") as f:
        f.write(SCENE_MAGIC)
        f.write(struct.pack("<6I", 1, n, cam["width"], cam["height"], sc["sh_degree"], int(sc["antialiased"])))
        f.write(np.asarray(cam["w2c"], "<f8").reshape(16).tobytes())
        f.write(struct.pack("<4d", cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
        f.write(np.asarray(sc["bg"], "<f4").tobytes())
        f.write(struct.pack("<f", sc["dilation"]))
        for k in ("means", "log_scales", "quats", "op_logits", "sh", "dl_dimg"):
            f.write(np.ascontiguousarray(sc[k], "<f4").tobytes())


def read_scene(path: Path):
    b = path.read_bytes()
    if b[:4] != SCENE_MAGIC:
        raise SystemExit(f"{path}: not a scene file")
    ver, n, w, h, deg, aa = struct.unpack_from("<6I", b, 4)
    o = 28
    w2c = np.frombuffer(b, "<f8", 16, o).reshape(4, 4); o += 128
    fx, fy, cx, cy = struct.unpack_from("<4d", b, o); o += 32
    bg = np.frombuffer(b, "<f4", 3, o).copy(); o += 12
    (dil,) = struct.unpack_from("<f", b, o); o += 4

    def take(count, shape):
        nonlocal o
        a = np.frombuffer(b, "<f4", count, o).reshape(shape).copy()
        o += 4 * count
        return a

    sc = dict(means=take(n * 3, (n, 3)), log_scales=take(n * 3, (n, 3)), quats=take(n * 4, (n, 4)),
              op_logits=take(n, (n,)), sh=take(n * 48, (n, 16, 3)), dl_dimg=take(3 * h * w, (3, h, w)))
    sc.update(cam=dict(w2c=w2c, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h), sh_degree=deg,
              antialiased=bool(aa), bg=bg, dilation=dil)
    return sc


def read_vk(path: Path):
    b = path.read_bytes()
    if b[:4] != OUT_MAGIC:
        raise SystemExit(f"{path}: not a splatwork-vk-train dump")
    ver, n, w, h, m = struct.unpack_from("<5I", b, 4)
    o = 24

    def take(count, shape, dt="<f4"):
        nonlocal o
        a = np.frombuffer(b, dt, count, o).reshape(shape).copy()
        o += 4 * count
        return a

    out = dict(image=take(3 * h * w, (3, h, w)), alpha=take(h * w, (h, w)), image_cpu=take(3 * h * w, (3, h, w)),
               means=take(n * 3, (n, 3)), log_scales=take(n * 3, (n, 3)), quats=take(n * 4, (n, 4)),
               op_logits=take(n, (n,)), sh=take(n * 48, (n, 16, 3)), tiles=take(n, (n,), "<u4"), tile_instances=m)
    return out


# ------------------------------------------------------------------ gsplat

def gsplat_render(sc):
    import torch
    from gsplat import rasterization

    dev = torch.device("cuda")
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    p = {k: t(sc[k]).requires_grad_(True) for k in ("means", "log_scales", "quats", "op_logits", "sh")}
    cam = sc["cam"]
    K = t([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    deg = sc["sh_degree"]
    renders, alphas, meta = rasterization(
        means=p["means"], quats=p["quats"], scales=torch.exp(p["log_scales"]),
        opacities=torch.sigmoid(p["op_logits"]), colors=p["sh"], viewmats=t(cam["w2c"])[None], Ks=K[None],
        width=cam["width"], height=cam["height"], near_plane=0.01, far_plane=1e10, eps2d=sc["dilation"],
        sh_degree=deg, packed=False, backgrounds=t(sc["bg"])[None],
        rasterize_mode="antialiased" if sc["antialiased"] else "classic", render_mode="RGB")
    img = renders[0].permute(2, 0, 1)                      # [3, H, W]
    loss = (img * t(sc["dl_dimg"])).sum()
    loss.backward()
    radii = meta["radii"][0]
    visible = (radii > 0).all(-1) if radii.ndim == 2 else radii > 0
    out = {k: v.grad.detach().cpu().numpy() for k, v in p.items()}
    out.update(image=img.detach().cpu().numpy(), alpha=alphas[0, ..., 0].detach().cpu().numpy(),
               visible=visible.cpu().numpy())
    return out


# ------------------------------------------------------------------ comparison

def stats(a, b):
    """Differences of a vs reference b: max |a-b| / max|b|, mean |a-b| / mean|b|, cosine similarity."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0:
        return dict(max_abs=0.0, max_rel=0.0, mean_rel=0.0, cos=1.0)
    d = np.abs(a - b)
    bm = np.abs(b).max() or 1.0
    bmean = np.abs(b).mean() or 1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(max_abs=float(d.max()), max_rel=float(d.max() / bm), mean_rel=float(d.mean() / bmean),
                cos=float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0)


def print_row(name, s):
    print(f"  {name:<34} max|d| {s['max_abs']:10.3e}   max|d|/max|ref| {s['max_rel']:9.3e}   "
          f"mean|d|/mean|ref| {s['mean_rel']:9.3e}   cos {s['cos']:.6f}")


def compare(d: Path, no_gsplat: bool, tol_cpu: float = 2e-2) -> int:
    # tol_cpu: the reference makes its own culling / alpha-threshold decisions in double precision, so a
    # borderline splat can flip at a pixel (a few 1e-3); a convention error shows up as O(0.1).
    sc = read_scene(d / "scene.bin")
    outs = {name: read_vk(d / f) for name, f in (("vulkan", "vk_out.bin"), ("vulkan-cas", "vk_out_cas.bin"))
            if (d / f).exists()}
    if not outs:
        raise SystemExit(f"no vk_out.bin in {d}: run splatwork-vk-train --dump {d}/scene.bin {d}/vk_out.bin first")
    n = len(sc["means"])
    k = (sc["sh_degree"] + 1) ** 2
    rc = 0
    print(f"scene: {n} splats, {sc['cam']['width']}x{sc['cam']['height']}, SH degree {sc['sh_degree']}, "
          f"{'antialiased' if sc['antialiased'] else 'classic'}")
    for name, vk in outs.items():
        print(f"\n{name}: {int((vk['tiles'] > 0).sum())} splats drawn, {vk['tile_instances']} tile instances")
        s = stats(vk["image"], vk["image_cpu"])
        print_row("image vs CPU double reference", s)
        if not all(np.all(np.isfinite(vk[g])) for g in ("image", "means", "log_scales", "quats", "op_logits", "sh")):
            print("  non-finite gradients!")
            rc = 1
        if s["max_abs"] > tol_cpu:
            print(f"  FAIL: image differs from the CPU reference by more than {tol_cpu}")
            rc = 1
    if no_gsplat:
        return rc
    try:
        g = gsplat_render(sc)
    except ImportError as e:
        print(f"\ngsplat / torch not importable ({e}); run 'compare' with the GPU runtime's Python, or --no-gsplat")
        return 2
    print(f"\ngsplat: {int(g['visible'].sum())} splats visible")
    for name, vk in outs.items():
        both = (vk["tiles"] > 0) & g["visible"]
        print(f"\n{name} vs gsplat ({int(both.sum())} splats drawn by both; "
              f"{int(((vk['tiles'] > 0) != g['visible']).sum())} drawn by only one):")
        print_row("image", stats(vk["image"], g["image"]))
        print_row("alpha (1 - T)", stats(vk["alpha"], g["alpha"]))
        for key, label in (("means", "dL/dmeans"), ("log_scales", "dL/dlog_scales"), ("quats", "dL/dquats"),
                           ("op_logits", "dL/dopacity_logits")):
            print_row(label, stats(vk[key][both], g[key][both]))
        print_row(f"dL/dsh (first {k} coefficients)", stats(vk["sh"][both][:, :k], g["sh"][both][:, :k]))
        print_row("dL/dsh DC only", stats(vk["sh"][both][:, 0], g["sh"][both][:, 0]))
    return rc


def run_vk(vk_train: str, d: Path, atomics: str, out_name: str) -> None:
    cmd = [vk_train, "--dump", str(d / "scene.bin"), str(d / out_name), "--atomics", atomics]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make-scene", "compare", "all"])
    ap.add_argument("--dir", type=Path, required=True, help="work directory (scene.bin, vk_out*.bin)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--antialiased", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vk-train", help="splatwork-vk-train binary (step 'all')")
    ap.add_argument("--no-gsplat", action="store_true", help="compare only against the CPU reference image")
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)
    if a.step in ("make-scene", "all"):
        sc = make_scene(a.n, a.width, a.height, a.sh_degree, a.antialiased, a.seed)
        write_scene(a.dir / "scene.bin", sc)
        print(f"wrote {a.dir / 'scene.bin'}")
        for stale in ("vk_out.bin", "vk_out_cas.bin"):
            (a.dir / stale).unlink(missing_ok=True)
    if a.step == "all":
        if not a.vk_train:
            ap.error("'all' needs --vk-train")
        run_vk(a.vk_train, a.dir, "auto", "vk_out.bin")
        run_vk(a.vk_train, a.dir, "cas", "vk_out_cas.bin")
    if a.step in ("compare", "all"):
        return compare(a.dir, a.no_gsplat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
