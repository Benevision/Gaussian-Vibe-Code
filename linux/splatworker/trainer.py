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


class TrainingError(RuntimeError):
    pass


class Progress:
    """Callback sink. Subclass or pass callables; all methods are optional."""

    def __init__(self, status: Optional[Callable[..., None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 preview: Optional[Callable[[int], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self._status, self._log, self._preview, self._stop = status, log, preview, should_stop

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
    return xyz.astype(np.float32), rgb.astype(np.float32)


def export_ply(splats, path: Path, region_box=None) -> int:
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
                           get("opacities"), get("scales"), get("quats"))
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
          preview_path=None, progress: Optional[Progress] = None) -> Dict:
    """Train a gaussian splat and write ``out_path`` (3DGS PLY). Returns stats."""
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

    views: List[LoadedView] = []
    for i, c in enumerate(train_cams + eval_cams):
        if progress.should_stop():
            raise TrainingCancelled()
        views.append(load_view(ds, c, cfg.maxImageSize, load_depth_maps=depth_weight > 0))
        if i % 20 == 0:
            progress.status(message=f"loading images {i + 1}/{len(train_cams) + len(eval_cams)}")
    train_views = views[:len(train_cams)]
    eval_views = views[len(train_cams):]
    log(f"loaded {len(train_views)} train / {len(eval_views)} holdout images "
        f"(max side {cfg.maxImageSize}), depth supervision weight {depth_weight}")

    # Keep uint8 images resident on the GPU when they fit in a modest budget,
    # otherwise in pinned host memory; converted to float per step.
    total_bytes = sum(v.image.nbytes + (v.depth.nbytes if v.depth is not None else 0) for v in views)
    resident = total_bytes < 3 * 1024 ** 3
    cache: Dict[int, tuple] = {}

    def _put(arr):
        t_ = torch.from_numpy(arr)
        return t_.to(dev) if resident else t_.pin_memory()

    def to_dev(v: LoadedView):
        key = id(v)
        if key not in cache:
            cache[key] = (_put(v.image), _put(v.alpha) if v.alpha is not None else None,
                          _put(v.depth) if v.depth is not None else None,
                          torch.from_numpy(v.K).to(dev), torch.from_numpy(v.world_to_camera).to(dev))
        img8, a8, d, K, w2c = cache[key]
        img = img8.to(dev, non_blocking=True).float() / 255.0
        alpha = a8.to(dev, non_blocking=True).float()[..., None] / 255.0 if a8 is not None else None
        depth = d.to(dev, non_blocking=True) if d is not None else None
        return img, alpha, depth, K, w2c

    # ---------------------------------------------------------------- init
    xyz, rgb = init_points(ds, cfg, region_box, log)
    n = xyz.shape[0]
    dist = _knn_mean_dist(xyz)
    dist = np.clip(dist, 1e-7, None)
    scene_scale = float(ds.scene_extent)

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
    lrs = {
        "means": cfg.lrPosition * scene_scale,
        "scales": cfg.lrScale,
        "quats": cfg.lrRotation,
        "opacities": cfg.lrOpacity,
        "sh0": cfg.lrColor,
        "shN": cfg.lrSHRest,
    }
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
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
                pred = renders[0, ..., :3] + (1 - alphas[0]) * eval_bg
                if appearance is not None:  # holdout photos keep their exposure-seeded correction
                    pred = appearance.apply(len(train_views) + k, pred)
                gt = img if alpha is None else img * alpha + (1 - alpha) * eval_bg
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
    loss_ema = None
    preview_step = None
    train_start = time.time()
    last_report = 0.0

    appearance = AppearanceCorrection(torch, cfg, train_views + eval_views, dev) if cfg.appearanceCorrection else None

    for step in range(cfg.iterations):
        if progress.should_stop():
            raise TrainingCancelled()
        if not order:
            order = list(range(len(train_views)))
            rng.shuffle(order)
        view_idx = order.pop()
        v = train_views[view_idx]
        img, alpha, depth_gt, K, w2c = to_dev(v)
        sh_degree = min(step // cfg.shDegreeInterval, cfg.shDegree)
        use_depth = depth_weight > 0 and depth_gt is not None

        renders, alphas, info = render(w2c, K, v.width, v.height, sh_degree, use_depth)
        if fixed_bg is None:
            bg = torch.rand(3, device=dev)
        else:
            bg = fixed_bg
        pred = renders[0, ..., :3] + (1.0 - alphas[0]) * bg
        if appearance is not None:
            pred = appearance.apply(view_idx, pred)
        gt = img if alpha is None else img * alpha + (1.0 - alpha) * bg

        if isinstance(strategy, DefaultStrategy):
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        if cfg.transientMasking:
            gt = transient.apply(pred, gt, alphas[0], step)
        l1 = torch.abs(pred - gt).mean()
        ssim_val = ssim(pred.permute(2, 0, 1)[None].contiguous(), gt.permute(2, 0, 1)[None].contiguous())
        loss = (1.0 - cfg.ssimWeight) * l1 + cfg.ssimWeight * (1.0 - ssim_val)
        if use_depth:
            ed = renders[0, ..., 3]
            valid = (depth_gt > 0) & torch.isfinite(depth_gt)
            if valid.any():
                loss = loss + depth_weight * torch.abs(ed[valid] - depth_gt[valid]).mean()
        if opacity_reg > 0:
            loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
        if scale_reg > 0:
            loss = loss + scale_reg * torch.exp(params["scales"]).mean()

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
        pos_sched.step()
        if appearance is not None:
            appearance.step(view_idx)

        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)
        else:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                        lr=pos_sched.get_last_lr()[0])

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
            rate = (now - train_start) / cur
            ngs = int(params["means"].shape[0])
            progress.status(step=cur, loss=loss_ema,
                            psnr=last_eval_psnr if last_eval_psnr is not None else psnr_ema,
                            numGaussians=ngs, elapsedSeconds=elapsed,
                            etaSeconds=rate * (cfg.iterations - cur), message="training")
            if cur % 100 == 0 or cur == cfg.iterations:
                log(f"step {cur}/{cfg.iterations} loss {loss_ema or 0:.4f} psnr {psnr_ema or 0:.2f} "
                    f"gaussians {ngs} sh {sh_degree}")

    final_n = export_ply(params, Path(out_path))  # uncropped: the Mac merger crops with open outer faces
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
                     queue, cancel_event) -> None:
    job = Path(job_dir)
    progress = Progress(
        status=lambda **f: queue.put(("status", f)),
        log=lambda m: queue.put(("log", m)),
        preview=lambda s: queue.put(("preview", s)),
        should_stop=cancel_event.is_set,
    )
    try:
        cfg = TrainingConfig.from_dict(config)
        stats = train(dataset_dir, cfg, job / "result.ply", device=device,
                      preview_path=job / "preview.splat", progress=progress)
        queue.put(("done", stats))
    except TrainingCancelled:
        queue.put(("cancelled", None))
    except BaseException as e:  # noqa: BLE001
        msg = str(e) or e.__class__.__name__
        if e.__class__.__name__ == "OutOfMemoryError" or "out of memory" in msg.lower():
            msg = f"CUDA out of memory: lower maxGaussians or maxImageSize ({msg.splitlines()[0]})"
        queue.put(("log", traceback.format_exc()))
        queue.put(("error", msg))
