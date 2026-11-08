"""gsplat-based CUDA trainer.

torch / gsplat / fused_ssim are imported lazily inside functions so that the
rest of the package (server, dataset parsing, PLY IO) works without them.

gsplat APIs relied on (gsplat >= 1.4, tested against the 1.4/1.5 API surface):

* ``gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks,
  width, height, near_plane, far_plane, sh_degree, packed, absgrad,
  rasterize_mode, render_mode)`` returning ``(renders, alphas, info)``.
* ``gsplat.strategy.DefaultStrategy(prune_opa, grow_grad2d, refine_start_iter,
  refine_stop_iter, reset_every, refine_every, absgrad, verbose)`` with
  ``check_sanity``, ``initialize_state(scene_scale=)``, ``step_pre_backward``,
  ``step_post_backward(..., packed=)``.
* ``gsplat.strategy.MCMCStrategy(cap_max, noise_lr, refine_start_iter,
  refine_stop_iter, refine_every, min_opacity, verbose)`` with
  ``initialize_state()`` and ``step_post_backward(..., lr=)``.
"""

from __future__ import annotations

import math
import random
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import TrainingConfig
from .dataset import DatasetInfo, LoadedView, load_view, parse_dataset, select_cameras
from .ply import read_point_cloud, write_gaussian_ply

SH_C0 = 0.28209479177387814


class TrainingCancelled(Exception):
    pass


class TrainingPaused(Exception):
    """Raised after the full training state was saved to ``resume_path``; ``train`` continues from it next time."""

    def __init__(self, step: int):
        super().__init__(f"paused at step {step}")
        self.step = step


RESUME_VERSION = 1


def _np_state_out(torch):
    """numpy's global RNG state with the key array as a tensor, so ``torch.load(weights_only=True)`` reads it."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached": float(cached)}


def _np_state_in(st) -> tuple:
    return (st["name"], st["keys"].numpy().astype(np.uint32), st["pos"], st["has_gauss"], st["cached"])


def _check_gsplat_kernels(torch, gsplat) -> None:
    """gsplat's prebuilt CUDA kernels only load with the torch build they were compiled for; if something
    replaced torch (e.g. a package that pulled a newer PyTorch), fail with a fix instead of a cryptic
    "'NoneType' object has no attribute 'CameraModelType'" on the first render."""
    import importlib

    ok = False
    for mod in ("gsplat.csrc", "gsplat.cuda._backend"):   # 1.6 / prebuilt wheels ship gsplat.csrc
        try:
            m = importlib.import_module(mod)
            ok = getattr(m, "_C", m) is not None
            if ok:
                break
        except Exception:
            continue
    if not ok:
        raise TrainingError(
            f"gsplat {getattr(gsplat, '__version__', '?')}'s GPU kernels can't load with PyTorch {torch.__version__} — "
            "the GPU runtime's PyTorch was replaced. Open SplatWork Node → menu → Set Up… and run it again to repair "
            "(downloads and settings are kept).")


class TrainingError(RuntimeError):
    pass


class Progress:
    """Callback sink. Subclass or pass callables; all methods are optional."""

    def __init__(self, status: Optional[Callable[..., None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 preview: Optional[Callable[[int], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 should_pause: Optional[Callable[[], bool]] = None,
                 take_checkpoint_request: Optional[Callable[[], bool]] = None,
                 checkpoint: Optional[Callable[[int], None]] = None):
        self._status, self._log, self._preview, self._stop = status, log, preview, should_stop
        self._pause, self._ckpt_req, self._ckpt = should_pause, take_checkpoint_request, checkpoint

    def status(self, **fields) -> None:
        if self._status:
            self._status(**fields)

    def log(self, msg: str) -> None:
        if self._log:
            self._log(msg)

    def preview(self, step: int) -> None:
        if self._preview:
            self._preview(step)

    def should_stop(self) -> bool:
        return bool(self._stop and self._stop())

    def should_pause(self) -> bool:
        return bool(self._pause and self._pause())

    def checkpoint_requested(self) -> bool:
        """True once per request (the request is consumed)."""
        return bool(self._ckpt_req and self._ckpt_req())

    def checkpoint(self, step: int) -> None:
        if self._ckpt:
            self._ckpt(step)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _import_backend():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise TrainingError("PyTorch is not installed; run install.sh") from e
    try:
        import gsplat  # noqa: F401
        from gsplat import rasterization  # noqa: F401
        from gsplat.strategy import DefaultStrategy, MCMCStrategy  # noqa: F401
    except ImportError as e:
        raise TrainingError(f"gsplat is not installed or failed to import: {e}") from e
    import torch
    return torch


class TransientMasker:
    """Robust training against moving objects — same rule as the Mac Metal trainer.

    After max(transientStart, 30 % of training), pixels whose error exceeds
    max(transientSensitivity × the image's median error, 0.10) are outliers when they are
    (a) well covered by the current model (alpha > 0.95: a moving object hides reconstructed
    background, whereas high error on thin coverage means "not learned yet") and (b) part of a
    solid region (≥ 7 of 9 neighbours are outliers) and (c) not in a 16×16 patch that is ≥ 60 %
    inliers (hard texture). Outlier pixels get gt := pred.detach(), removing them from L1 and SSIM.
    The threshold is an EMA over steps (0.95) so one bad view can't swing it.
    """

    def __init__(self, torch, cfg):
        self.torch = torch
        self.cfg = cfg
        self.threshold = None
        self.start = max(cfg.transientStart, int(cfg.iterations * 0.3))
        self.fraction = 0.0

    def apply(self, pred, gt, alpha, step):
        torch = self.torch
        F = torch.nn.functional
        with torch.no_grad():
            r = torch.abs(pred - gt).mean(dim=-1)                       # H, W
            covered = alpha[..., 0] > 0.95 if alpha.dim() == 3 else alpha > 0.95
            flat = r.flatten()
            sample = flat[torch.randint(0, flat.numel(), (min(flat.numel(), 65536),), device=flat.device)]
            median = float(sample.median())
            t_new = max(median * self.cfg.transientSensitivity, 0.10)
            self.threshold = t_new if self.threshold is None else self.threshold * 0.95 + t_new * 0.05
            if step < self.start:
                return gt
            out = ((r > self.threshold) & covered).float()[None, None]   # 1,1,H,W
            # solid regions only (≥ 7 of 9 neighbours), plus a 16×16 patch vote that keeps outliers in
            # mostly-good patches (hard texture). A looser rule lost 7 dB on textured scenes in testing.
            solid = F.avg_pool2d(out, 3, stride=1, padding=1) * 9.0 >= 6.5
            inlier = ~solid
            H, W = r.shape
            ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
            inl = F.pad(inlier.float(), (0, pw, 0, ph), value=1.0)
            frac = F.avg_pool2d(inl, 16, stride=16)
            patch_ok = F.interpolate((frac >= 0.6).float(), scale_factor=16, mode="nearest")[..., :H, :W] > 0.5
            keep = (inlier | patch_ok)[0, 0]
            self.fraction = 0.9 * self.fraction + 0.1 * float(1.0 - keep.float().mean())
            if bool(keep.all()):
                return gt
            return torch.where(keep[..., None], gt, pred.detach())


def ssim_split(torch, corr, raw, gt, size: int = 11, sigma: float = 1.5):
    """SSIM whose luminance term compares the colour-corrected render with the photo, while contrast and
    structure compare the *uncorrected* render (all [1,3,H,W]). Per-photo colour correction can then fix
    brightness and tint, but not paper over missing or wrong detail (decoupled appearance)."""
    import torch.nn.functional as F

    coords = torch.arange(size, dtype=raw.dtype, device=raw.device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()
    conv = lambda x: F.conv2d(x, w, groups=3)  # noqa: E731
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_c, mu_r, mu_g = conv(corr), conv(raw), conv(gt)
    lum = (2 * mu_c * mu_g + c1) / (mu_c * mu_c + mu_g * mu_g + c1)
    s_r = conv(raw * raw) - mu_r * mu_r
    s_g = conv(gt * gt) - mu_g * mu_g
    s_rg = conv(raw * gt) - mu_r * mu_g
    cs = (2 * s_rg + c2) / (s_r + s_g + c2)
    return (lum * cs).mean()


def progressive_scale(step: int, densify_stop: int) -> float:
    """Image scale for coarse-to-fine training: ¼ → ½ → ¾ → full resolution over the first half of the
    densification window (DashGaussian / FastGS style: early steps only need coarse structure, and run several
    times faster at low resolution)."""
    half = max(densify_stop, 1) / 2
    if step < half / 3:
        return 0.25
    if step < 2 * half / 3:
        return 0.5
    if step < half:
        return 0.75
    return 1.0


def depth_normals(torch, depth, K, k: int = 1, rel_tol: float = 0.05):
    """Camera-space normals of a depth map (H, W) by central differences over ``k`` pixels, and where they are
    valid: all four neighbours present and on one surface (within ``rel_tol`` of the centre). Same construction
    as the Mac's normal_loss kernel, so both see the same normals: cross(right − left, down − up)."""
    H, W = depth.shape
    dev = depth.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=depth.dtype), torch.arange(W, device=dev, dtype=depth.dtype),
                            indexing="ij")
    X = torch.stack(((xs - K[0, 2]) / K[0, 0] * depth, (ys - K[1, 2]) / K[1, 1] * depth, depth), -1)
    n = torch.zeros(H, W, 3, device=dev, dtype=depth.dtype)
    valid = torch.zeros(H, W, dtype=torch.bool, device=dev)
    if H <= 2 * k or W <= 2 * k:
        return n, valid
    c = depth[k:-k, k:-k]
    l, r = depth[k:-k, :-2 * k], depth[k:-k, 2 * k:]
    u, d = depth[:-2 * k, k:-k], depth[2 * k:, k:-k]
    e1 = X[k:-k, 2 * k:] - X[k:-k, :-2 * k]
    e2 = X[2 * k:, k:-k] - X[:-2 * k, k:-k]
    m = torch.cross(e1, e2, dim=-1)
    ln = m.norm(dim=-1, keepdim=True)
    ok = (c > 0) & (l > 0) & (r > 0) & (u > 0) & (d > 0) & (ln[..., 0] > 1e-12)
    for nb in (l, r, u, d):
        ok = ok & (torch.abs(nb - c) < rel_tol * c.clamp_min(1e-9))
    n[k:-k, k:-k] = m / ln.clamp_min(1e-12)
    valid[k:-k, k:-k] = ok
    return n, valid


def normal_consistency_loss(torch, rendered_depth, alpha, sensor_depth, K, sensor_width: int):
    """(1−c)/2 + sqrt((1−c)/2) between rendered-depth normals and sensor-depth normals, summed over pixels where
    both are valid and the render is opaque, divided by the pixel count (as on the Mac)."""
    W = rendered_depth.shape[1]
    k = max(1, int(round(W / max(int(sensor_width or W), 1))))   # one sensor pixel either side
    nt, vt = depth_normals(torch, sensor_depth, K, k=k)
    nr, vr = depth_normals(torch, rendered_depth, K, k=1, rel_tol=0.10)
    valid = vt & vr & (alpha > 0.9)
    if not valid.any():
        return None
    c = (nr * nt.detach()).sum(-1).clamp(-1, 1)
    h = (0.5 * (1 - c)).clamp_min(0)
    per = h + torch.sqrt(h + 1e-4)
    return per[valid].sum() / rendered_depth.numel()


def downscale_view(torch, img, alpha, depth, K, scale: float):
    """Image (H,W,3), alpha (H,W,1)|None, depth (H,W)|None and intrinsics K scaled by ``scale``."""
    import torch.nn.functional as F

    h, w = img.shape[0], img.shape[1]
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    sy, sx = nh / h, nw / w
    img = F.interpolate(img.permute(2, 0, 1)[None], size=(nh, nw), mode="area")[0].permute(1, 2, 0).contiguous()
    if alpha is not None:
        alpha = F.interpolate(alpha.permute(2, 0, 1)[None], size=(nh, nw), mode="area")[0].permute(1, 2, 0).contiguous()
    if depth is not None:
        depth = F.interpolate(depth[None, None], size=(nh, nw), mode="nearest")[0, 0]
    K = K.clone()
    K[0, 0] *= sx; K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return img, alpha, depth, K, nw, nh


def _ssim_fn(torch):
    """Return ssim(a, b) for [B,3,H,W] tensors in 0..1 (valid padding)."""
    try:
        from fused_ssim import fused_ssim  # type: ignore

        def ssim(a, b):
            return fused_ssim(a, b, padding="valid")

        return ssim, "fused-ssim"
    except Exception:
        pass
    import torch.nn.functional as F

    size, sigma = 11, 1.5
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_cache: Dict = {}

    def ssim(a, b):
        key = (a.device, a.dtype)
        if key not in window_cache:
            w2 = (g[:, None] @ g[None, :]).to(a.device, a.dtype)
            window_cache[key] = w2.expand(3, 1, size, size).contiguous()
        w = window_cache[key]
        mu_a = F.conv2d(a, w, groups=3)
        mu_b = F.conv2d(b, w, groups=3)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        s_a = F.conv2d(a * a, w, groups=3) - mu_a2
        s_b = F.conv2d(b * b, w, groups=3) - mu_b2
        s_ab = F.conv2d(a * b, w, groups=3) - mu_ab
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mu_ab + c1) * (2 * s_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (s_a + s_b + c2))
        return m.mean()

    return ssim, "torch-ssim"


def _knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours (used for initial scales)."""
    n = points.shape[0]
    if n <= 1:
        return np.full(n, 0.01, dtype=np.float32)
    k = min(k, n - 1)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        d, _ = cKDTree(points).query(points, k=k + 1)
        return d[:, 1:].mean(axis=1).astype(np.float32)
    except ImportError:
        pass
    # Fallback: approximate against a random reference subset with torch/numpy.
    rng = np.random.default_rng(0)
    ref = points if n <= 20000 else points[rng.choice(n, 20000, replace=False)]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, 4096):
        chunk = points[i:i + 4096]
        d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        d2.sort(axis=1)
        out[i:i + 4096] = np.sqrt(d2[:, 1:k + 1]).mean(axis=1)
    return out


def _in_box(xyz, lo, hi):
    return np.all((xyz >= np.asarray(lo)) & (xyz <= np.asarray(hi)), axis=1)



def spread_clone_groups(torch, params) -> int:
    """Long-axis splitting (ImprovedGS / IGS+) on top of gsplat's MCMC strategy, without touching its code.

    MCMC relocation / growth clones a splat by writing copies with *exactly* the same position (and the
    shrunken scale / reduced opacity) into other slots. Right after a refinement step those clone groups are
    therefore just sets of identical means. Each group of m members is spread evenly along the source's
    longest axis (spacing 2σ/m, centred on the source) and only that axis is shortened by √m, as on the Mac.
    Returns the number of splats moved.
    """
    with torch.no_grad():
        means = params["means"].data
        _, inv, counts = torch.unique(means, dim=0, return_inverse=True, return_counts=True)
        m = counts[inv]
        members = (m > 1).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            return 0
        # rank of each member within its group
        order = members[torch.argsort(inv[members], stable=True)]
        g = inv[order]
        first = torch.ones_like(g, dtype=torch.bool)
        first[1:] = g[1:] != g[:-1]
        start = torch.cummax(torch.where(first, torch.arange(len(g), device=g.device), torch.zeros_like(g)), 0).values
        rank = (torch.arange(len(g), device=g.device) - start).float()
        mm = m[order].float()
        ls = params["scales"].data[order]
        ax = torch.argmax(ls, dim=1)
        sigma = torch.exp(ls.gather(1, ax[:, None]))[:, 0]
        q = torch.nn.functional.normalize(params["quats"].data[order], dim=1)   # w, x, y, z
        w, x, y, z = q.unbind(1)
        cols = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1),
            torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1),
            torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1),
        ], 1)                                                                     # [n, axis, xyz]
        u = cols[torch.arange(len(order), device=q.device), ax]
        t = (rank - (mm - 1) / 2) * 2 * sigma / mm
        means[order] = means[order] + u * t[:, None]
        new_ls = ls.clone()
        new_ls.scatter_(1, ax[:, None], (ls.gather(1, ax[:, None])[:, 0] - 0.5 * torch.log(mm))[:, None])
        params["scales"].data[order] = new_ls
        return int(len(order))


def median_nearest_view_gap(cams, scene_extent: float) -> float:
    """Median over photos of the gap to the most similar other photo: view-direction angle plus the camera
    offset as an angle at scene scale (degrees). ~5 = dense capture, >15 = sparse (same as the Mac)."""
    import numpy as np

    if len(cams) < 3:
        return 0.0
    c2w = np.stack([np.asarray(c.camera_to_world, dtype=np.float64) for c in cams])
    pos, fwd = c2w[:, :3, 3], c2w[:, :3, 2]
    fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    ext = max(float(scene_extent), 1e-3)
    idx = range(0, len(cams), max(1, len(cams) // 400))
    gaps = []
    for i in idx:
        ang = np.degrees(np.arccos(np.clip(fwd @ fwd[i], -1, 1)))
        off = np.degrees(np.linalg.norm(pos - pos[i], axis=1) / ext)
        g = ang + off
        g[i] = np.inf
        gaps.append(float(g.min()))
    return float(np.median(gaps))


class PoseRefiner:
    """Per-photo camera correction learned during training (gsplat ``CameraOptModule`` style).

    Each training photo gets 9 numbers — a translation and a 6D rotation residual — starting at zero
    (identity). The corrected world-to-camera is ``inv(c2w @ ΔT)``; gsplat's rasteriser provides the
    gradients w.r.t. ``viewmats``. A sparse Adam only updates photos that were drawn this step, and a
    small L2 pull keeps corrections minimal. Held-out photos are evaluated with their original poses.
    """

    def __init__(self, torch, n_views: int, cfg, dev):
        self.torch = torch
        self.embed = torch.nn.Embedding(n_views, 9, sparse=True).to(dev)
        torch.nn.init.zeros_(self.embed.weight)
        self.opt = torch.optim.SparseAdam(list(self.embed.parameters()), lr=cfg.poseRefinementLR)
        self.reg = cfg.poseRefinementReg
        self.identity6 = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=dev)
        self._last = None

    @staticmethod
    def _rot6d(torch, d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)

    def apply(self, i: int, w2c):
        torch = self.torch
        idx = torch.tensor([i], device=w2c.device)
        d = self.embed(idx)[0]
        self._last = d
        R = self._rot6d(torch, d[3:] + self.identity6)
        dT = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
        dT = torch.cat([torch.cat([R, d[:3, None]], 1), dT[3:4]], 0)
        c2w = torch.linalg.inv(w2c)
        return torch.linalg.inv(c2w @ dT)

    def reg_loss(self):
        """Small L2 pull towards no correction (add to the loss before backward)."""
        if self._last is None or self.reg <= 0:
            return 0.0
        return self.reg * (self._last ** 2).sum()

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)


def exposure_locked(cameras) -> bool:
    """Every photo has a recorded exposure and they're all within 10 % of each other."""
    e = [getattr(c, "exposure", None) or 0.0 for c in cameras]
    return bool(e) and min(e) > 0 and max(e) / min(e) <= 1.1


class AppearanceCorrection:
    """Per-photo colour correction: corrected = M · rendered + b (3×3 + 3 per photo).

    Mirrors the Mac Metal trainer: seeded from recorded exposure (gain = (E/E_median)^(1/2.2),
    clamped to [1/3, 3]), regularised towards that seed, and each photo's parameters are only
    updated on steps where that photo is drawn (one tiny Adam per photo, so idle photos don't drift).
    """

    def __init__(self, torch, cfg, views, dev):
        self.torch = torch
        exps = sorted(v.camera.exposure for v in views if getattr(v.camera, "exposure", None))
        ref = exps[len(exps) // 2] if exps else None
        self.params, self.init, self.opts = [], [], []
        for v in views:
            gain = 1.0
            e = getattr(v.camera, "exposure", None)
            if ref and e:
                gain = min(3.0, max(1.0 / 3.0, (e / ref) ** (1 / 2.2)))
            p0 = torch.cat([torch.eye(3, device=dev).flatten() * gain, torch.zeros(3, device=dev)])
            p = p0.clone().requires_grad_(True)
            self.params.append(p)
            self.init.append(p0)
            self.opts.append(torch.optim.Adam([p], lr=cfg.appearanceLR, eps=1e-8))
        self.reg = cfg.appearanceReg

    def apply(self, i, rgb):
        p = self.params[i]
        M, b = p[:9].view(3, 3), p[9:]
        return rgb @ M.T + b

    def step(self, i):
        p = self.params[i]
        if p.grad is None:
            return
        with self.torch.no_grad():
            if not self.torch.isfinite(p.grad).all():
                p.grad = None
                return
            p.grad.add_(self.reg * (p - self.init[i]))
        self.opts[i].step()
        self.opts[i].zero_grad(set_to_none=True)

class BilateralGrid:
    """Per-photo *local* colour correction: a coarse 3-D grid of colour transforms per photo.

    The grid spans the image (x, y) and brightness (the rendered pixel's luminance): each cell holds a 3×4 affine
    colour transform, and every pixel gets the trilinear blend of the cells around (x, y, its luminance). That lets
    a photo say "this corner is in shadow here" or "highlights are warmer" without the splats having to bake it in —
    e.g. a drone flight where the sun moved and tree shadows swung round over 90 minutes, which a single 3×3 per
    photo (AppearanceCorrection) cannot express, so the trainer smeared the shadows into the ground.

    Only used in the loss. Each grid starts as "no change", is kept smooth (total variation), and the average over
    all photos is pinned to "no change" (every `pin_every` steps the mean deviation is subtracted from every grid),
    so the splats hold the average lighting and the export needs nothing baked in. Held-out photos are scored with
    no correction at all.
    """

    def __init__(self, torch, cfg, n_views: int, dev, shape=(8, 16, 16)):
        self.torch = torch
        d, h, w = shape
        ident = torch.zeros(12, d, h, w, device=dev)
        for r in range(3):
            ident[r * 4 + r] = 1.0                      # rows of [A | b]: A = I, b = 0
        self.ident = ident
        self.grids = [ident.clone().requires_grad_(True) for _ in range(n_views)]
        self.opts = [torch.optim.Adam([g], lr=cfg.bilateralGridLR, eps=1e-15) for g in self.grids]
        self.tv_weight = cfg.bilateralGridTV
        self.lum = torch.tensor([0.299, 0.587, 0.114], device=dev)
        self._coords = {}

    def _xy(self, h, w, dev):
        key = (h, w)
        if key not in self._coords:
            ys = (self.torch.arange(h, device=dev, dtype=self.torch.float32) + 0.5) / h * 2 - 1
            xs = (self.torch.arange(w, device=dev, dtype=self.torch.float32) + 0.5) / w * 2 - 1
            gy, gx = self.torch.meshgrid(ys, xs, indexing="ij")
            self._coords = {key: (gx, gy)}               # one resolution cached (views share sizes)
        return self._coords[key]

    def apply(self, i: int, rgb):
        torch = self.torch
        h, w, _ = rgb.shape
        gx, gy = self._xy(h, w, rgb.device)
        gz = (rgb.detach() @ self.lum).clamp(0, 1) * 2 - 1
        grid = torch.stack([gx, gy, gz], dim=-1)[None, None]          # (1, 1, H, W, 3): x→W, y→H, z→D
        coef = torch.nn.functional.grid_sample(self.grids[i][None], grid, mode="bilinear",
                                               padding_mode="border", align_corners=True)[0, :, 0]   # (12, H, W)
        A = coef.permute(1, 2, 0).reshape(h, w, 3, 4)
        return (A[..., :3] @ rgb[..., None])[..., 0] + A[..., 3]

    def tv_loss(self, i: int):
        g = self.grids[i]
        tv = 0.0
        for dim in (1, 2, 3):
            n = g.shape[dim]
            if n > 1:
                tv = tv + (g.narrow(dim, 1, n - 1) - g.narrow(dim, 0, n - 1)).pow(2).mean()
        return self.tv_weight * tv

    def step(self, i: int):
        g = self.grids[i]
        if g.grad is None:
            return
        if not self.torch.isfinite(g.grad).all():
            g.grad = None
            return
        self.opts[i].step()
        self.opts[i].zero_grad(set_to_none=True)

    def pin_mean(self):
        """Shift every grid so the average over photos is exactly "no change"."""
        with self.torch.no_grad():
            dev_mean = self.torch.stack([g.detach() for g in self.grids]).mean(0) - self.ident
            for g in self.grids:
                g.sub_(dev_mean)


def init_points(ds: DatasetInfo, cfg: TrainingConfig, region_box, log) -> tuple:
    """Return (xyz [N,3], rgb [N,3]) initial points as numpy arrays."""
    rng = np.random.default_rng(cfg.seed)
    xyz = rgb = None
    if ds.points:
        try:
            xyz, rgb = read_point_cloud(ds.resolve(ds.points))
            if region_box is not None:
                keep = _in_box(xyz, *region_box)
                xyz = xyz[keep]
                rgb = rgb[keep] if rgb is not None else None
                log(f"region: kept {len(xyz)} initial points inside the expanded box")
            if len(xyz) == 0:
                log("point cloud empty (after region filter); falling back to random init")
                xyz = None
            else:
                log(f"initialising from {ds.points}: {len(xyz)} points")
                if rgb is None:
                    rgb = rng.random((len(xyz), 3), dtype=np.float32)
        except Exception as e:  # malformed PLY -> random init
            log(f"failed to read point cloud ({e}); falling back to random init")
            xyz = None
    if xyz is None:
        n = cfg.randomInitCount
        if region_box is not None:
            lo, hi = (np.asarray(b, dtype=np.float32) for b in region_box)
        else:
            c = ds.scene_center.astype(np.float32)
            half = np.float32(ds.scene_extent * 1.5)
            lo, hi = c - half, c + half
        xyz = (lo + rng.random((n, 3), dtype=np.float32) * (hi - lo)).astype(np.float32)
        rgb = rng.random((n, 3), dtype=np.float32)
        log(f"random init: {n} gaussians in box {lo.tolist()} .. {hi.tolist()}")
    if len(xyz) > cfg.maxGaussians:
        idx = rng.choice(len(xyz), cfg.maxGaussians, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        log(f"subsampled initial points to maxGaussians={cfg.maxGaussians}")
    if cfg.backgroundShell > 0 and region_box is None:
        shell = background_shell(ds, cfg.backgroundShell)
        xyz = np.concatenate([xyz, shell])
        rgb = np.concatenate([rgb, np.tile(rgb.mean(0, keepdims=True), (len(shell), 1))])
        log(f"background shell: {len(shell)} splats at 3× the scene size")
    return xyz.astype(np.float32), rgb.astype(np.float32)


def background_shell(ds: DatasetInfo, n: int) -> np.ndarray:
    """``n`` points evenly over a sphere 3 scene sizes out around the cameras' centre (Fibonacci lattice) —
    a place for distant land / sky beyond the capture instead of hazy edge splats or floaters (as on the Mac)."""
    cams = np.array([np.asarray(c.camera_to_world)[:3, 3] for c in ds.cameras], dtype=np.float64) if ds.cameras else None
    centre = cams.mean(0) if cams is not None and len(cams) else np.asarray(ds.scene_center, dtype=np.float64)
    r = 3.0 * float(ds.scene_extent)
    k = np.arange(n, dtype=np.float64)
    y = 1 - 2 * (k + 0.5) / n
    rad = np.sqrt(np.maximum(0, 1 - y * y))
    th = np.pi * (3 - np.sqrt(5)) * k
    return (centre + r * np.stack([rad * np.cos(th), y, rad * np.sin(th)], 1)).astype(np.float32)


def export_ply(splats, path: Path, region_box=None, up=None) -> int:
    import torch

    with torch.no_grad():
        means = splats["means"].detach()
        keep = None
        if region_box is not None:
            lo = torch.tensor(region_box[0], device=means.device, dtype=means.dtype)
            hi = torch.tensor(region_box[1], device=means.device, dtype=means.dtype)
            keep = ((means >= lo) & (means <= hi)).all(dim=1)

        def get(name):
            t = splats[name].detach()
            if keep is not None:
                t = t[keep]
            return t.float().cpu().numpy()

        means_np = get("means")
        write_gaussian_ply(path, means_np, get("sh0"),
                           get("shN") if splats["shN"].shape[1] > 0 else None,
                           get("opacities"), get("scales"), get("quats"), up=up)
        return int(means_np.shape[0])


PREVIEW_MAX_SPLATS = 400_000
SH_C0 = 0.28209479177387814


def export_preview_splat(splats, path: Path, max_splats: int = PREVIEW_MAX_SPLATS) -> int:
    """Compact live preview in antimatter15 ``.splat`` layout (32 bytes/splat, no SH): the
    ``max_splats`` most visible splats (opacity × volume), ≈13 MB for 400 k instead of a
    ~350 MB full PLY, so the Mac can poll it over Wi-Fi while training runs."""
    import numpy as np
    import torch

    with torch.no_grad():
        op = torch.sigmoid(splats["opacities"].detach().float().flatten())
        scales = torch.exp(splats["scales"].detach().float())
        score = op * scales.prod(dim=1)
        n = int(score.shape[0])
        idx = torch.topk(score, min(max_splats, n)).indices if n > max_splats else torch.argsort(score, descending=True)
        means = splats["means"].detach().float()[idx].cpu().numpy()
        sc = scales[idx].cpu().numpy()
        rgb = (0.5 + SH_C0 * splats["sh0"].detach().float()[idx].reshape(-1, 3)).clamp(0, 1).cpu().numpy()
        a = op[idx].cpu().numpy()
        q = splats["quats"].detach().float()[idx]
        q = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    rec = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = means, sc
    rec["c"][:, :3] = np.round(rgb * 255)
    rec["c"][:, 3] = np.round(a * 255)
    rec["r"] = np.clip(np.round(q * 128 + 128), 0, 255)
    tmp = Path(str(path) + ".tmp")
    rec.tofile(tmp)
    tmp.replace(path)  # atomic: the server never serves a half-written preview
    return len(idx)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def train(dataset_dir, cfg: TrainingConfig, out_path, device: str = "cuda:0",
          preview_path=None, progress: Optional[Progress] = None,
          resume_path=None, checkpoint_path=None) -> Dict:
    """Train a gaussian splat and write ``out_path`` (3DGS PLY). Returns stats.

    ``resume_path``: where a pause saves the full state (params, optimizer moments, strategy state, RNGs, step;
    ``TrainingPaused`` is raised after saving) and where the next call continues from if the file exists.
    ``checkpoint_path``: where an on-demand checkpoint (and every pause) writes the current splats as a PLY."""
    progress = progress or Progress()
    log = progress.log
    t_start = time.time()
    progress.status(state="loading", message="loading dataset", totalSteps=cfg.iterations)

    torch = _import_backend()
    import gsplat
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    if not torch.cuda.is_available():
        raise TrainingError("CUDA is not available to PyTorch")
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log(f"gsplat {getattr(gsplat, '__version__', '?')}, torch {torch.__version__}, device {dev}"
        f" ({torch.cuda.get_device_name(dev)})")
    _check_gsplat_kernels(torch, gsplat)

    ds = parse_dataset(dataset_dir)
    region_box = cfg.region.expanded() if cfg.region is not None else None
    cams = select_cameras(ds, cfg.cameraIds, region_box)
    if cfg.cameraIds is not None:
        log(f"cameraIds: {len(cams)} of {len(ds.cameras)} cameras selected")
    if region_box is not None:
        log(f"region {cfg.region.to_dict()}: expanded box {region_box}; {len(cams)} cameras see it")
    train_cams = [c for c in cams if not c.holdout]
    eval_cams = [c for c in cams if c.holdout]
    if not train_cams:
        raise TrainingError("no training cameras left after cameraIds/region selection")
    has_depth = any(c.depth for c in train_cams)
    depth_weight = cfg.effective_depth_weight(has_depth)

    from .dataset import resize_nearest, view_header
    from .viewstore import ViewStore, available_ram, choose_mode, view_bytes

    all_cams = train_cams + eval_cams
    # No sensor depth: predicted (monocular) depth, fitted to the scene, for the depth-normal loss.
    priors = None
    if not has_depth and cfg.normalWeight > 0 and cfg.depthPriors in ("auto", "on"):
        from . import depthprior

        if depthprior.available():
            pts = None
            if ds.points:
                try:
                    pts, _ = read_point_cloud(ds.resolve(ds.points))
                except Exception as e:  # noqa: BLE001
                    log(f"depth priors: could not read the dataset's points ({e})")
            priors = depthprior.DepthPriors(ds.root / "depth-prior", log)
            try:
                got = priors.compute(ds, all_cams, pts, device=str(dev), should_stop=progress.should_stop,
                                     status=lambda m: progress.status(message=m))
                log(f"depth priors: {got} of {len(all_cams)} photos have a predicted depth map")
            except Exception as e:  # noqa: BLE001 — a missing model download etc. must not fail the training
                log(f"depth priors unavailable ({type(e).__name__}: {e}); training without them")
                priors = None
        elif cfg.depthPriors == "on":
            log("depth priors: the depth model isn't installed on this worker (install the processing models)")

    def with_prior(v: LoadedView) -> LoadedView:
        if priors is not None and v.depth is None and v.camera.id in priors.maps:
            v.depth = resize_nearest(priors.maps[v.camera.id], v.width, v.height).astype(np.float32)
        return v
    headers = [view_header(ds, c, cfg.maxImageSize) for c in all_cams]
    est = sum(view_bytes(v) for v in headers)
    ram = available_ram()
    mode = choose_mode(est, ram)
    load = lambda h: with_prior(load_view(ds, h.camera, cfg.maxImageSize, load_depth_maps=depth_weight > 0 or cfg.normalWeight > 0))  # noqa: E731
    if mode == "disk":
        views: List[LoadedView] = headers
    else:
        views = []
        for i, c in enumerate(all_cams):
            if progress.should_stop():
                raise TrainingCancelled()
            views.append(with_prior(load_view(ds, c, cfg.maxImageSize, load_depth_maps=depth_weight > 0 or cfg.normalWeight > 0)))
            if i % 20 == 0:
                progress.status(message=f"loading images {i + 1}/{len(all_cams)}")
    train_views = views[:len(train_cams)]
    eval_views = views[len(train_cams):]
    budget = int(0.35 * ram)
    store = ViewStore(load, views, mode, budget)
    import atexit

    atexit.register(store.close)
    gib = 1024 ** 3
    log(f"{len(train_views)} train / {len(eval_views)} holdout images (max side {cfg.maxImageSize}), "
        f"depth supervision weight {depth_weight}; pixels ≈ {est / gib:.1f} GB, {ram / gib:.0f} GB RAM available → "
        + {"gpu": "kept on the GPU", "ram": "kept in RAM",
           "disk": f"read from disk as needed (cache {budget / gib:.0f} GB, background decoding)"}[mode])

    # uint8 pixels: resident on the GPU (gpu mode) or copied up per step from RAM; converted to float per step.
    gpu_cache: Dict[int, tuple] = {}
    pose_cache: Dict[int, tuple] = {}
    masks: Dict[int, object] = {}   # id(view) → its ignore mask for the current step (set by to_dev)
    skies: Dict[int, object] = {}   # id(view) → its sky mask (255 = sky) for the current step

    def resized(m, like):
        if m is None or (m.shape[0] == like.shape[0] and m.shape[1] == like.shape[1]):
            return m
        return torch.nn.functional.interpolate(m.permute(2, 0, 1)[None], size=like.shape[:2], mode="area")[0].permute(1, 2, 0)

    from .skybox import Skybox
    use_sky = cfg.skybox == "on" or (cfg.skybox == "auto" and any(c.sky for c in train_cams))
    skybox = None
    if use_sky:
        skybox = Skybox(torch, ds.up_vector, height=cfg.skyboxResolution, device=dev)
        log(f"learned skybox {2 * cfg.skyboxResolution}×{cfg.skyboxResolution} at infinity; "
            f"{sum(1 for c in train_cams if c.sky)} photos have a sky mask (splats kept out, weight {cfg.skyAlphaWeight})")

    def to_dev(v: LoadedView):
        key = id(v)
        if key not in pose_cache:
            pose_cache[key] = (torch.from_numpy(v.K).to(dev), torch.from_numpy(v.world_to_camera).to(dev))
        K, w2c = pose_cache[key]
        if mode == "gpu":
            if key not in gpu_cache:
                gpu_cache[key] = tuple(torch.from_numpy(a).to(dev) if a is not None else None
                                       for a in (v.image, v.alpha, v.depth, v.mask, v.sky))
            img8, a8, d, m8, s8 = gpu_cache[key]
        else:
            im, al, dp, mk, sk = store.get(v)
            img8 = torch.from_numpy(im).to(dev)
            a8 = torch.from_numpy(al).to(dev) if al is not None else None
            d = torch.from_numpy(dp).to(dev) if dp is not None else None
            m8 = torch.from_numpy(mk).to(dev) if mk is not None else None
            s8 = torch.from_numpy(sk).to(dev) if sk is not None else None
        skies[key] = s8.float()[..., None] / 255.0 if s8 is not None else None
        img = img8.float() / 255.0
        alpha = a8.float()[..., None] / 255.0 if a8 is not None else None
        masks[key] = m8.float()[..., None] / 255.0 if m8 is not None else None
        return img, alpha, d, K, w2c

    # ---------------------------------------------------------------- init
    resume = None
    if resume_path is not None and Path(resume_path).is_file():
        # tensors and plain containers only (see save_state), so the safe loader suffices
        resume = torch.load(str(resume_path), map_location="cpu", weights_only=True)
        if resume.get("version") != RESUME_VERSION:
            raise TrainingError("the saved state of this paused job is from another worker version; "
                                "it can't be resumed")
        if resume.get("views") != [len(train_views), len(eval_views)]:
            raise TrainingError("the dataset changed since this job was paused (different photos); "
                                "it can't be resumed")
        t_start -= float(resume.get("elapsed") or 0.0)
        log(f"resuming from the state saved at step {resume['step']}")
    scene_scale = float(ds.scene_extent)
    if resume is None:
        xyz, rgb = init_points(ds, cfg, region_box, log)
        n = xyz.shape[0]
        dist = _knn_mean_dist(xyz)
        dist = np.clip(dist, 1e-7, None)

        num_sh = (cfg.shDegree + 1) ** 2
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(t(xyz)),
            "scales": torch.nn.Parameter(t(np.log(dist))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.rand((n, 4), device=dev)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), cfg.initialOpacity, device=dev))),
            "sh0": torch.nn.Parameter(((t(rgb) - 0.5) / SH_C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros((n, num_sh - 1, 3), device=dev)),
        })
    else:
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(dev).contiguous())
                                         for k, v in resume["params"].items()})
        n = int(params["means"].shape[0])
    lrs = {
        "means": cfg.lrPosition * scene_scale,
        "scales": cfg.lrScale,
        "quats": cfg.lrRotation,
        "opacities": cfg.lrOpacity,
        "sh0": cfg.lrColor,
        "shN": cfg.lrSHRest,
    }
    if cfg.optimizerPrecision == "fp32":
        optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                      for k, lr in lrs.items()}
    else:   # reduced-precision Adam moments (same update rule) so more splats fit in GPU memory
        from .compact_adam import CompactAdam, moments_for
        optimizers = {k: CompactAdam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15,
                                     moments=moments_for(k, cfg.optimizerPrecision), seed=cfg.seed)
                      for k, lr in lrs.items()}
    from .compact_adam import estimate_bytes_per_splat
    mem = estimate_bytes_per_splat(cfg.shDegree, cfg.optimizerPrecision)
    log(f"optimizerPrecision={cfg.optimizerPrecision}"
        + (" (int8 SH-rest moments, bf16 others)" if cfg.optimizerPrecision == "compact" else "")
        + f": ~{mem['total']} B/splat for params+grads+Adam state "
          f"(state {mem['optimizerState']} B; fp32 Adam would be "
          f"{estimate_bytes_per_splat(cfg.shDegree, 'fp32')['total']} B)")
    gamma = (cfg.lrPositionFinal / cfg.lrPosition) ** (1.0 / cfg.iterations)
    pos_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=gamma)

    densify_stop = cfg.effective_densify_stop()
    if cfg.strategy == "mcmc":
        strategy = MCMCStrategy(cap_max=cfg.maxGaussians, noise_lr=cfg.noiseLR,
                                refine_start_iter=cfg.densifyStart, refine_stop_iter=densify_stop,
                                refine_every=cfg.densifyInterval, min_opacity=cfg.pruneOpacity,
                                verbose=False)
        strategy_state = strategy.initialize_state()
        absgrad = False
    else:
        strategy = DefaultStrategy(prune_opa=cfg.pruneOpacity,
                                   grow_grad2d=cfg.effective_grad_threshold(),
                                   refine_start_iter=cfg.densifyStart, refine_stop_iter=densify_stop,
                                   reset_every=cfg.opacityResetInterval,
                                   refine_every=cfg.densifyInterval,
                                   absgrad=cfg.absGrad, verbose=False)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        absgrad = cfg.absGrad
    strategy.check_sanity(params, optimizers)
    grow_threshold = getattr(strategy, "grow_grad2d", None)
    cap_logged = False

    ssim, ssim_name = _ssim_fn(torch)
    opacity_reg = cfg.effective_opacity_reg()
    scale_reg = cfg.effective_scale_reg()
    rasterize_mode = "antialiased" if cfg.antialiased else "classic"
    log(f"strategy={cfg.strategy} gaussians={n} sceneExtent={scene_scale:.3f} ssim={ssim_name} "
        f"rasterize_mode={rasterize_mode} densify {cfg.densifyStart}..{densify_stop}/{cfg.densifyInterval} "
        f"opacityReg={opacity_reg} scaleReg={scale_reg}")

    if cfg.background == "white":
        fixed_bg = torch.ones(3, device=dev)
    elif cfg.background == "black":
        fixed_bg = torch.zeros(3, device=dev)
    else:
        fixed_bg = None
    eval_bg = fixed_bg if fixed_bg is not None else torch.zeros(3, device=dev)

    def render(w2c, K, width, height, sh_degree, want_depth):
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=w2c[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=1e10,
            sh_degree=sh_degree,
            packed=False,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            render_mode="RGB+ED" if want_depth else "RGB",
        )
        return renders, alphas, info

    def evaluate(sh_degree):
        if not eval_views:
            return None
        psnrs = []
        with torch.no_grad():
            for k, v in enumerate(eval_views):
                img, alpha, _, K, w2c = to_dev(v)
                renders, alphas, _ = render(w2c, K, v.width, v.height, sh_degree, False)
                ebg = skybox.sample(skybox.directions(K, w2c, v.width, v.height)) if skybox is not None else eval_bg
                pred = renders[0, ..., :3] + (1 - alphas[0]) * ebg
                if appearance is not None:  # holdout photos keep their exposure-seeded correction
                    pred = appearance.apply(len(train_views) + k, pred)
                gt = img if alpha is None else img * alpha + (1 - alpha) * ebg
                ignore = masks.get(id(v))
                if ignore is not None:   # score only the pixels the model is meant to explain
                    gt = gt * ignore + pred.clamp(0, 1) * (1 - ignore)
                mse = torch.mean((pred.clamp(0, 1) - gt) ** 2).item()
                psnrs.append(-10 * math.log10(max(mse, 1e-10)))
        return float(np.mean(psnrs))

    # ---------------------------------------------------------------- loop
    progress.status(state="training", message="training", step=0, numGaussians=n)
    order: List[int] = []
    rng = random.Random(cfg.seed)
    psnr_ema = None
    last_eval_psnr = None
    transient = TransientMasker(torch, cfg)
    view_gap = median_nearest_view_gap([v.camera for v in train_views], scene_scale)
    transient_on = cfg.transientMasking and view_gap <= cfg.transientMaxViewGap
    if cfg.transientMasking and not transient_on:
        log(f"sparse coverage (photos ~{view_gap:.0f}° apart): not ignoring moving objects — "
            "real scenery would be mistaken for them")
    loss_ema = None
    preview_step = None
    train_start = time.time()
    last_report = 0.0

    use_appearance = cfg.appearanceCorrection
    if use_appearance and exposure_locked([v.camera for v in train_views + eval_views]):
        # locked exposure (SplatWork Capture): nothing to correct; the extra freedom cost 1 dB held-out on the Mac
        use_appearance = False
        log("exposure is the same in every photo: per-photo colour correction off")
    bilagrid = None
    if cfg.bilateralGrid:
        # local per-photo correction replaces the global 3×3 one (it can express everything that does)
        use_appearance = False
        bilagrid = BilateralGrid(torch, cfg, len(train_views), dev)
        log(f"per-photo local colour correction: {len(train_views)} grids of 16×16×8")
    appearance = AppearanceCorrection(torch, cfg, train_views + eval_views, dev) if use_appearance else None
    pose_refiner = PoseRefiner(torch, len(train_views), cfg, dev) if cfg.poseRefinement else None

    g_up = ds.raw.get("gravityUp") if isinstance(ds.raw, dict) else None
    scene_up = g_up if isinstance(g_up, list) and len(g_up) == 3 else ds.up_vector

    def opt_states(opts):
        return [o.state_dict() for o in opts]

    def save_state(step: int) -> None:
        """Everything the loop needs to carry on exactly where it stopped (pause, then resume)."""
        state = {
            "version": RESUME_VERSION, "step": step, "views": [len(train_views), len(eval_views)],
            "params": {k: v.detach() for k, v in params.items()},
            "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
            "pos_sched": pos_sched.state_dict(),
            "strategy_state": strategy_state,
            "order": list(order), "rng": rng.getstate(),
            "py_random": random.getstate(), "np_random": _np_state_out(torch),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
            "ema": {"psnr": psnr_ema, "loss": loss_ema, "eval": last_eval_psnr, "preview": preview_step},
            "elapsed": time.time() - t_start, "cap_logged": cap_logged,
            "transient": [transient.threshold, transient.fraction],
            "appearance": None if appearance is None else
            {"params": [p.detach() for p in appearance.params], "opts": opt_states(appearance.opts)},
            "bilagrid": None if bilagrid is None else
            {"grids": [g.detach() for g in bilagrid.grids], "opts": opt_states(bilagrid.opts)},
            "pose": None if pose_refiner is None else
            {"embed": pose_refiner.embed.state_dict(), "opt": pose_refiner.opt.state_dict()},
            "skybox": None if skybox is None else {"tex": skybox.tex.detach(), "opt": skybox.opt.state_dict()},
        }
        dst = Path(resume_path)
        tmp = dst.with_name(dst.name + ".tmp")
        torch.save(state, str(tmp))
        tmp.replace(dst)   # atomic: a crash mid-save leaves the previous state (or none), never half a file

    def write_checkpoint(step: int) -> None:
        dst = Path(checkpoint_path)
        tmp = dst.with_name(dst.name + ".tmp")
        n_out = export_ply(params, tmp, up=scene_up)
        tmp.replace(dst)
        log(f"step {step}: checkpoint written ({n_out} splats)")
        progress.checkpoint(step)

    start_step = 0
    if resume is not None:
        start_step = int(resume["step"])
        for k, o in optimizers.items():
            o.load_state_dict(resume["optimizers"][k])
        if cfg.optimizerPrecision != "fp32":
            for o in optimizers.values():
                o.restore_dtypes()   # load_state_dict casts the int8/bf16 moments to the params' fp32
        pos_sched.load_state_dict(resume["pos_sched"])
        strategy_state = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in resume["strategy_state"].items()}
        order = list(resume["order"])
        rng.setstate(resume["rng"])
        random.setstate(resume["py_random"])
        np.random.set_state(_np_state_in(resume["np_random"]))
        torch.set_rng_state(resume["torch_rng"])
        if resume.get("cuda_rng") is not None and dev.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"], dev)
        ema = resume.get("ema") or {}
        psnr_ema, loss_ema = ema.get("psnr"), ema.get("loss")
        last_eval_psnr, preview_step = ema.get("eval"), ema.get("preview")
        cap_logged = bool(resume.get("cap_logged"))
        transient.threshold, transient.fraction = resume["transient"]
        with torch.no_grad():
            if appearance is not None and resume.get("appearance"):
                for p_, v_ in zip(appearance.params, resume["appearance"]["params"]):
                    p_.copy_(v_)
                for o, st in zip(appearance.opts, resume["appearance"]["opts"]):
                    o.load_state_dict(st)
            if bilagrid is not None and resume.get("bilagrid"):
                for g_, v_ in zip(bilagrid.grids, resume["bilagrid"]["grids"]):
                    g_.copy_(v_)
                for o, st in zip(bilagrid.opts, resume["bilagrid"]["opts"]):
                    o.load_state_dict(st)
            if pose_refiner is not None and resume.get("pose"):
                pose_refiner.embed.load_state_dict(resume["pose"]["embed"])
                pose_refiner.opt.load_state_dict(resume["pose"]["opt"])
            if skybox is not None and resume.get("skybox"):
                skybox.tex.copy_(resume["skybox"]["tex"])
                skybox.opt.load_state_dict(resume["skybox"]["opt"])
        resume = None   # free the CPU copy
        progress.status(step=start_step, numGaussians=n, psnr=psnr_ema, evalPsnr=last_eval_psnr, loss=loss_ema)

    for step in range(start_step, cfg.iterations):
        if progress.should_stop():
            raise TrainingCancelled()
        if checkpoint_path is not None and progress.checkpoint_requested():
            write_checkpoint(step)
        if resume_path is not None and progress.should_pause():
            if checkpoint_path is not None:
                write_checkpoint(step)   # a paused job always has a downloadable checkpoint
            progress.status(message="saving the training state")
            save_state(step)
            log(f"paused at step {step}: training state saved")
            raise TrainingPaused(step)
        if not order:
            order = list(range(len(train_views)))
            rng.shuffle(order)
        view_idx = order.pop()
        v = train_views[view_idx]
        if mode == "disk":   # decode the next few views of the shuffled order while this step runs
            store.prefetch([train_views[i] for i in order[-6:]])
        img, alpha, depth_gt, K, w2c = to_dev(v)
        sh_degree = min(step // cfg.shDegreeInterval, cfg.shDegree)
        use_depth = (depth_weight > 0 or cfg.normalWeight > 0) and depth_gt is not None

        width, height = v.width, v.height
        if cfg.progressiveResolution:
            sc = progressive_scale(step, densify_stop)
            if sc < 1.0:
                img, alpha, depth_gt, K, width, height = downscale_view(torch, img, alpha, depth_gt, K, sc)
        if pose_refiner is not None and step >= cfg.poseRefinementStart:
            w2c = pose_refiner.apply(view_idx, w2c)
        renders, alphas, info = render(w2c, K, width, height, sh_degree, use_depth)
        if skybox is not None:      # the backdrop is the learned sky at infinity, per pixel direction
            bg = skybox.sample(skybox.directions(K, w2c, width, height))
        elif fixed_bg is None:
            bg = torch.rand(3, device=dev)
        else:
            bg = fixed_bg
        pred = renders[0, ..., :3] + (1.0 - alphas[0]) * bg
        pred_raw = pred
        if appearance is not None:
            pred = appearance.apply(view_idx, pred)
        if bilagrid is not None:
            pred = bilagrid.apply(view_idx, pred)
        gt = img if alpha is None else img * alpha + (1.0 - alpha) * bg

        if isinstance(strategy, DefaultStrategy):
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        if transient_on:
            gt = transient.apply(pred, gt, alphas[0], step)
        ignore = masks.get(id(v))
        if ignore is not None:
            # masked-out pixels (moving objects, the rig, …) carry no error and no gradient
            if ignore.shape[0] != pred.shape[0] or ignore.shape[1] != pred.shape[1]:
                ignore = torch.nn.functional.interpolate(ignore.permute(2, 0, 1)[None], size=pred.shape[:2],
                                                         mode="area")[0].permute(1, 2, 0)
            gt = gt * ignore + pred.detach() * (1.0 - ignore)
        l1 = torch.abs(pred - gt).mean()
        if cfg.decoupledSSIM and pred is not pred_raw:
            ssim_val = ssim_split(torch, pred.permute(2, 0, 1)[None], pred_raw.permute(2, 0, 1)[None],
                                  gt.permute(2, 0, 1)[None])
        else:
            ssim_val = ssim(pred.permute(2, 0, 1)[None].contiguous(), gt.permute(2, 0, 1)[None].contiguous())
        loss = (1.0 - cfg.ssimWeight) * l1 + cfg.ssimWeight * (1.0 - ssim_val)
        if use_depth:
            ed = renders[0, ..., 3]
            valid = (depth_gt > 0) & torch.isfinite(depth_gt)
            dw = depth_weight if v.camera.depth else cfg.monoDepthWeight   # predicted maps: their own weight
            if dw > 0 and valid.any():
                derr = torch.abs(ed - depth_gt)
                if cfg.depthEdgeAware:
                    # DN-Splatter: trust depth less at image edges (exp(-|∇I|) of the photo, as on the Mac)
                    g = (img[..., :3] * torch.tensor([0.299, 0.587, 0.114], device=dev)).sum(-1)
                    gx = torch.zeros_like(g); gy = torch.zeros_like(g)
                    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
                    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
                    derr = derr * torch.exp(-10.0 * torch.sqrt(gx * gx + gy * gy))
                loss = loss + dw * derr[valid].sum() / valid.sum().clamp_min(1)
            if cfg.normalWeight > 0 and step >= cfg.normalStart:
                sw = v.camera.depth_width or (priors.maps[v.camera.id].shape[1] if priors is not None and v.camera.id in priors.maps else 0)
                nl = normal_consistency_loss(torch, ed, alphas[0, ..., 0], depth_gt, K, sw)
                if nl is not None:
                    loss = loss + cfg.normalWeight * nl
        if opacity_reg > 0:
            loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
        if scale_reg > 0:
            loss = loss + scale_reg * torch.exp(params["scales"]).mean()
        if cfg.flatnessReg > 0:
            # flat discs: the smallest axis's share of the splat's size (s_min² / Σ s²), as on the Mac
            e2 = torch.exp(2.0 * params["scales"])
            loss = loss + cfg.flatnessReg * (e2.min(dim=1).values / e2.sum(dim=1).clamp_min(1e-30)).mean()
        if cfg.anisotropyReg > 0:
            # Penalise needle splats (longest / shortest axis above the limit), as on the Mac.
            # Needle penalty: longest / MIDDLE axis (flat discs — two long axes, one tiny — are the normal
            # surface shape and must not be penalised; longest / shortest squashed them into faint strokes).
            # Only the long axis shrinks (middle detached).
            ls = params["scales"]
            srt = torch.sort(ls, dim=1).values
            ratio = torch.exp(srt[:, 2] - srt[:, 1].detach())
            loss = loss + cfg.anisotropyReg * torch.clamp(ratio - cfg.anisotropyMaxRatio, min=0).mean()

        if bilagrid is not None:
            loss = loss + bilagrid.tv_loss(view_idx)
        if pose_refiner is not None and step >= cfg.poseRefinementStart:
            loss = loss + pose_refiner.reg_loss()
        sky_m = resized(skies.get(id(v)), pred)
        if skybox is not None and sky_m is not None and cfg.skyAlphaWeight > 0:
            # splats don't belong in the sky: push their coverage there towards zero (the skybox paints it)
            loss = loss + cfg.skyAlphaWeight * (alphas[0] * sky_m).sum() / sky_m.sum().clamp(min=1.0)
        loss.backward()

        # Block training does not kill gaussians that leave the region: the block's cameras also see
        # neighbouring blocks and background, and pruning them every pass made training fight itself
        # (MCMC kept regrowing what was pruned; PSNR fell). The region seeds the initial points and
        # picks cameras; the Mac's BlockMerger crops each block to the area it owns when merging.

        # ADC safety cap: stop growing once at maxGaussians (pruning continues).
        if isinstance(strategy, DefaultStrategy) and grow_threshold is not None:
            at_cap = params["means"].shape[0] >= cfg.maxGaussians
            strategy.grow_grad2d = float("inf") if at_cap else grow_threshold
            if at_cap and not cap_logged:
                log(f"step {step}: reached maxGaussians={cfg.maxGaussians}; growth paused")
                cap_logged = True

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if skybox is not None:
            skybox.step()
        pos_sched.step()
        if appearance is not None:
            appearance.step(view_idx)
        if bilagrid is not None:
            bilagrid.step(view_idx)
            if (step + 1) % 50 == 0:
                bilagrid.pin_mean()
        if pose_refiner is not None and step >= cfg.poseRefinementStart:
            pose_refiner.step()

        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)
        else:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                        lr=pos_sched.get_last_lr()[0])
            if (cfg.longAxisSplit and cfg.densifyStart < step < densify_stop
                    and step % cfg.densifyInterval == 0):
                spread_clone_groups(torch, params)
        if cfg.optimizerPrecision != "fp32":
            # gsplat's torch.cat with fp32 zeros promotes compact state to fp32 when it adds splats; cast it back
            # now so the promoted copy doesn't occupy GPU memory through the next forward/backward
            for opt in optimizers.values():
                opt.restore_dtypes()

        # --------------------------------------------------- bookkeeping
        cur = step + 1
        loss_v = float(loss.item()) if (cur % 10 == 0 or cur == cfg.iterations) else None
        if loss_v is not None:
            loss_ema = loss_v if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_v
            with torch.no_grad():
                mse = torch.mean((pred.detach().clamp(0, 1) - gt) ** 2).item()
            p = -10 * math.log10(max(mse, 1e-10))
            psnr_ema = p if psnr_ema is None else 0.9 * psnr_ema + 0.1 * p

        if cfg.evalInterval and eval_views and (cur % cfg.evalInterval == 0 or cur == cfg.iterations):
            last_eval_psnr = evaluate(sh_degree)
            log(f"step {cur}: holdout PSNR {last_eval_psnr:.2f} dB")

        if cfg.previewInterval and preview_path is not None and cur % cfg.previewInterval == 0 \
                and cur < cfg.iterations:
            export_preview_splat(params, Path(preview_path))
            preview_step = cur
            progress.preview(cur)

        now = time.time()
        if now - last_report > 1.0 or cur == cfg.iterations:
            last_report = now
            elapsed = now - t_start
            rate = (now - train_start) / max(1, cur - start_step)
            ngs = int(params["means"].shape[0])
            progress.status(step=cur, loss=loss_ema,
                            psnr=psnr_ema, evalPsnr=last_eval_psnr,   # train and held-out kept apart
                            numGaussians=ngs, elapsedSeconds=elapsed,
                            etaSeconds=rate * (cfg.iterations - cur), message="training")
            if cur % 100 == 0 or cur == cfg.iterations:
                log(f"step {cur}/{cfg.iterations} loss {loss_ema or 0:.4f} psnr {psnr_ema or 0:.2f} "
                    f"gaussians {ngs} sh {sh_degree}")

    final_n = export_ply(params, Path(out_path), up=scene_up)
    if resume_path is not None:
        Path(resume_path).unlink(missing_ok=True)   # finished: the saved state is no longer needed
    if skybox is not None:
        log(f"skybox saved: {skybox.save(Path(out_path)).name}")  # uncropped: the Mac merger crops with open outer faces
    elapsed = time.time() - t_start
    stats = {"numGaussians": final_n, "psnr": last_eval_psnr if last_eval_psnr is not None else psnr_ema,
             "loss": loss_ema, "elapsedSeconds": elapsed, "previewStep": preview_step}
    log(f"finished in {elapsed:.1f}s, {final_n} gaussians written to {Path(out_path).name}")
    return stats


# ---------------------------------------------------------------------------
# subprocess entry (used by the server; one process per job so that GPU
# memory is returned to the driver and crashes do not take down the API)
# ---------------------------------------------------------------------------

def subprocess_entry(dataset_dir: str, config: dict, job_dir: str, device: str,
                     queue, cancel_event, pause_event=None, checkpoint_event=None) -> None:
    job = Path(job_dir)

    def take_checkpoint_request() -> bool:
        if checkpoint_event is not None and checkpoint_event.is_set():
            checkpoint_event.clear()
            return True
        return False

    progress = Progress(
        status=lambda **f: queue.put(("status", f)),
        log=lambda m: queue.put(("log", m)),
        preview=lambda s: queue.put(("preview", s)),
        should_stop=cancel_event.is_set,
        should_pause=pause_event.is_set if pause_event is not None else None,
        take_checkpoint_request=take_checkpoint_request,
        checkpoint=lambda s: queue.put(("checkpoint", s)),
    )
    try:
        cfg = TrainingConfig.from_dict(config)
        stats = train(dataset_dir, cfg, job / "result.ply", device=device,
                      preview_path=job / "preview.splat", progress=progress,
                      resume_path=job / "resume.pt" if pause_event is not None else None,
                      checkpoint_path=job / "checkpoint.ply")
        queue.put(("done", stats))
    except TrainingPaused as e:
        queue.put(("paused", e.step))
    except TrainingCancelled:
        queue.put(("cancelled", None))
    except BaseException as e:  # noqa: BLE001
        msg = str(e) or e.__class__.__name__
        if e.__class__.__name__ == "OutOfMemoryError" or "out of memory" in msg.lower():
            msg = f"CUDA out of memory: lower maxGaussians or maxImageSize ({msg.splitlines()[0]})"
        queue.put(("log", traceback.format_exc()))
        queue.put(("error", msg))
