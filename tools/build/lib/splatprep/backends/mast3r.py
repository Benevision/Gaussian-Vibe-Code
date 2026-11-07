"""``mast3r`` — thin adapter around the official naver/mast3r repo (dense matching + sparse global alignment).

Install (not pip-packaged)::

    git clone --recursive https://github.com/naver/mast3r ~/src/mast3r
    pip install -r ~/src/mast3r/requirements.txt -r ~/src/mast3r/dust3r/requirements.txt
    export SPLATPREP_MAST3R_PATH=~/src/mast3r

Weights (~2.6 GB) ``naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`` are
downloaded from Hugging Face on first use, or set ``SPLATPREP_MAST3R_WEIGHTS``
to a local checkpoint.  Licence: CC BY-NC-SA 4.0 (non-commercial).

Best for few (2–50) images; above ~50 a sliding-window scene graph is used.
"""

from __future__ import annotations

import os

import numpy as np

from ..device import empty_cache, has_module
from ..images import image_size
from .base import Availability, Backend, SfMContext, SfMOutput
from .dense_common import add_repo_paths, build_reconstruction, map_intrinsics_to_original, subsample_points

DEFAULT_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


class MASt3RBackend(Backend):
    id = "mast3r"
    display_name = "MASt3R (dense, few-image)"
    kind = "dense"
    install_hint = "clone naver/mast3r and set SPLATPREP_MAST3R_PATH (see README)"

    def check(self) -> Availability:
        add_repo_paths("SPLATPREP_MAST3R_PATH", ("", "dust3r"))
        missing = [m for m in ("torch", "mast3r", "dust3r") if not has_module(m)]
        if missing:
            return Availability(False, False, f"missing: {', '.join(missing)} — {self.install_hint}")
        return Availability(True, True)

    def run(self, ctx: SfMContext) -> SfMOutput:
        self.require(ctx)
        from mast3r.model import AsymmetricMASt3R  # type: ignore
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # type: ignore
        from dust3r.image_pairs import make_pairs  # type: ignore
        from dust3r.utils.image import load_images  # type: ignore

        n = len(ctx.names)
        size = 512
        files = [str(p) for p in ctx.paths]
        ctx.progress.update("features", 0.05, "loading MASt3R")
        model = AsymmetricMASt3R.from_pretrained(os.environ.get("SPLATPREP_MAST3R_WEIGHTS", DEFAULT_WEIGHTS)).to(ctx.device).eval()
        imgs = load_images(files, size=size, verbose=False)
        if n <= 30:
            graph = "complete"
        else:
            graph = f"swin-{min(10, max(3, int(ctx.opt('sequentialOverlap', 10))))}"
        pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
        ctx.progress.update("features", 1.0, f"{len(pairs)} image pairs ({graph})")
        ctx.progress.update("matching", 0.05, "MASt3R inference + sparse global alignment")
        cache = ctx.work_dir / "mast3r_cache"
        cache.mkdir(parents=True, exist_ok=True)
        scene = sparse_global_alignment(
            files,
            pairs,
            str(cache),
            model,
            lr1=0.07,
            niter1=500,
            lr2=0.014,
            niter2=200,
            device=ctx.device,
            opt_depth=True,
            shared_intrinsics=bool(ctx.opt("singleCamera", True)),
            matching_conf_thr=5.0,
        )
        ctx.progress.update("matching", 1.0, "alignment done")
        c2w = scene.get_im_poses().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
        pps = scene.get_principal_points().detach().cpu().numpy().reshape(-1, 2)
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
        sizes = [image_size(p) for p in ctx.paths]
        intr, xyz_all, rgb_all = [], [], []
        for i in range(n):
            W0, H0 = sizes[i]
            s = size / max(W0, H0)
            W1, H1 = round(W0 * s), round(H0 * s)
            cx, cy = W1 // 2, H1 // 2
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if W1 == H1:
                halfh = 3 * halfw / 4
            off_x, off_y = cx - halfw, cy - halfh
            intr.append(map_intrinsics_to_original([focals[i], focals[i], pps[i, 0], pps[i, 1]], W1 / W0, H1 / H0, off_x, off_y))
            img = np.asarray(scene.imgs[i])
            p = pts3d[i].detach().cpu().numpy().reshape(-1, 3)
            c = confs[i].detach().cpu().numpy().reshape(-1)
            keep = c > 1.5
            xyz_all.append(p[keep])
            rgb_all.append((np.clip(img.reshape(-1, 3)[keep], 0, 1) * 255).astype(np.uint8))
        xyz, rgb = subsample_points(np.concatenate(xyz_all), np.concatenate(rgb_all), int(ctx.opt("maxPoints", 300000)))
        rec = build_reconstruction(ctx.names, sizes, c2w, np.stack(intr), xyz, rgb, bool(ctx.opt("singleCamera", True)))
        del model, scene
        empty_cache()
        return SfMOutput(rec, None, {"backend": self.id, "sceneGraph": graph, "numPairs": len(pairs), "mapper": "mast3r_sparse_ga"})
