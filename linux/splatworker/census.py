"""Contribution census for Auto clean (job kind ``census``): how much each splat adds to each photo.

For every camera of the dataset the splats are rendered with gsplat using a single colour channel that is
1 for every splat. The image sum is then Σ_pixels Σ_i w_i(p), so its gradient with respect to splat i's colour
is splat i's summed blending weight over that photo (in pixels at the photo's native resolution: the render may
be smaller and is scaled back). Per splat the job records

* ``peak``  — the largest per-photo weight sum (a splat that adds 2 % to one pixel has 0.02), and
* ``views`` — in how many photos that sum is ≥ :data:`SEEN` (0.02).

The Studio turns this into an Auto clean plan (:mod:`splatworker.studio.autoclean`, same rules as the Mac).
Output: ``census.npz`` in the job directory (``peak`` float32, ``views`` uint32), served at
``GET /v1/jobs/{id}/census.npz``.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Dict

import numpy as np

SEEN = 0.02
MAX_SIDE = 1000


def run_census(dataset_dir: Path, ply_path: Path, out_path: Path, device: str = "cuda:0",
               status: Callable[..., None] = lambda **k: None, log: Callable[[str], None] = print,
               should_stop: Callable[[], bool] = lambda: False, max_side: int = MAX_SIDE) -> Dict:
    import torch
    from gsplat import rasterization

    from .dataset import parse_dataset
    from .studio.formats import load

    ds = parse_dataset(dataset_dir, check_files=False)
    g = load(ply_path, with_skybox=False)
    n = g.count
    log(f"census of {n} splats over {len(ds.cameras)} photos")
    dev = torch.device(device)
    means = torch.tensor(g.positions, device=dev)
    quats = torch.tensor(g.normalized_rotations(), device=dev)
    scales = torch.tensor(np.exp(g.log_scales), device=dev)
    opac = torch.tensor(g.opacities, device=dev)
    peak = torch.zeros(n, device=dev)
    views = torch.zeros(n, dtype=torch.int32, device=dev)
    mode = "antialiased" if g.antialiased else "classic"
    for k, cam in enumerate(ds.cameras):
        if should_stop():
            raise KeyboardInterrupt("cancelled")
        s = min(1.0, max_side / max(cam.width, cam.height))
        w, h = max(1, int(round(cam.width * s))), max(1, int(round(cam.height * s)))
        K = torch.tensor([[cam.fx * s, 0, cam.cx * s], [0, cam.fy * s, cam.cy * s], [0, 0, 1]],
                         dtype=torch.float32, device=dev)
        w2c = torch.tensor(np.linalg.inv(np.asarray(cam.camera_to_world, np.float64)), dtype=torch.float32,
                           device=dev)
        colors = torch.ones((n, 1), device=dev, requires_grad=True)
        img, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                                  viewmats=w2c[None], Ks=K[None], width=w, height=h, near_plane=0.01,
                                  far_plane=1e10, sh_degree=None, packed=False, rasterize_mode=mode)
        img.sum().backward()
        wsum = colors.grad[:, 0] * ((cam.width * cam.height) / float(w * h))
        peak = torch.maximum(peak, wsum)
        views += (wsum >= SEEN).int()
        status(step=k + 1, totalSteps=len(ds.cameras), message=f"census: photo {k + 1}/{len(ds.cameras)}",
               state="running")
    np.savez(out_path, peak=peak.cpu().numpy().astype(np.float32), views=views.cpu().numpy().astype(np.uint32))
    seen = int((peak >= SEEN).sum().item())
    log(f"census done: {seen} of {n} splats are visible in at least one photo")
    return {"splats": n, "visible": seen}


def subprocess_entry(dataset_dir: str, spec: dict, job_dir: str, device_index, queue, cancel_event) -> None:
    import os

    if device_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    try:
        stats = run_census(Path(dataset_dir), Path(spec["plyPath"]), Path(job_dir) / "census.npz",
                           device="cuda:0", status=lambda **f: queue.put(("status", f)),
                           log=lambda m: queue.put(("log", m)), should_stop=cancel_event.is_set,
                           max_side=int(spec.get("maxImageSize") or MAX_SIDE))
        queue.put(("done", stats))
    except KeyboardInterrupt:
        queue.put(("cancelled", None))
    except BaseException as e:  # noqa: BLE001
        queue.put(("log", traceback.format_exc()))
        queue.put(("error", str(e) or e.__class__.__name__))

