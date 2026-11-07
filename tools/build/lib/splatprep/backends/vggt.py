"""``vggt`` — feed-forward poses + depth with facebookresearch/vggt (VGGT-1B).

Install::

    pip install "vggt @ git+https://github.com/facebookresearch/vggt.git"

Weights (~5 GB fp32) ``facebook/VGGT-1B`` from Hugging Face on first use
(override with ``SPLATPREP_VGGT_WEIGHTS``, e.g. ``facebook/VGGT-1B-Commercial``
once access is granted).  Images are processed at 518 px width ("crop" mode).
Memory grows with the number of frames; ``SPLATPREP_VGGT_MAX_FRAMES`` (default
150) guards against OOM.  No bundle adjustment is run here.
"""

from __future__ import annotations

import os

import numpy as np

from ..device import empty_cache, has_module
from ..errors import SplatPrepError
from ..geometry import invert_pose
from ..images import image_size
from .base import Availability, Backend, SfMContext, SfMOutput
from .dense_common import build_reconstruction, map_intrinsics_to_original, subsample_points

DEFAULT_WEIGHTS = "facebook/VGGT-1B"


class VGGTBackend(Backend):
    id = "vggt"
    display_name = "VGGT (feed-forward)"
    kind = "dense"
    install_hint = "pip install 'splatprep[vggt]'"

    def check(self) -> Availability:
        missing = [m for m in ("torch", "vggt") if not has_module(m)]
        if missing:
            return Availability(False, False, f"missing: {', '.join(missing)} — {self.install_hint}")
        return Availability(True, True)

    def run(self, ctx: SfMContext) -> SfMOutput:
        self.require(ctx)
        import torch
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore

        n = len(ctx.names)
        max_frames = int(os.environ.get("SPLATPREP_VGGT_MAX_FRAMES", "150"))
        if n > max_frames:
            raise SplatPrepError(f"vggt: {n} images exceeds SPLATPREP_VGGT_MAX_FRAMES={max_frames}; use a sparse backend")
        dev = torch.device(ctx.device)
        ctx.progress.update("features", 0.05, "loading VGGT-1B")
        model = VGGT.from_pretrained(os.environ.get("SPLATPREP_VGGT_WEIGHTS", DEFAULT_WEIGHTS)).to(dev).eval()
        images = load_and_preprocess_images([str(p) for p in ctx.paths], mode="crop").to(dev)
        ctx.progress.update("features", 1.0, f"{n} frames preprocessed {tuple(images.shape[-2:])}")
        if ctx.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            amp = torch.autocast("cuda", dtype=dtype)
        else:
            import contextlib

            amp = contextlib.nullcontext()
        ctx.progress.update("matching", 0.1, "VGGT forward pass")
        with torch.no_grad(), amp:
            pred = model(images)
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        extr = extr.squeeze(0).float().cpu().numpy()  # S x 3 x 4 (w2c, OpenCV)
        intr = intr.squeeze(0).float().cpu().numpy()  # S x 3 x 3
        depth = pred["depth"].squeeze(0).float().cpu().numpy()
        conf = pred["depth_conf"].squeeze(0).float().cpu().numpy()
        pts = unproject_depth_map_to_point_map(depth, extr, intr)  # S x H x W x 3
        ctx.progress.update("matching", 1.0, "poses predicted")
        Hp, Wp = images.shape[-2:]
        sizes = [image_size(p) for p in ctx.paths]
        c2ws, Ks, xyz_all, rgb_all = [], [], [], []
        imgs_np = images.float().cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(n):
            W0, H0 = sizes[i]
            s = 518.0 / W0
            new_h = round(H0 * s / 14) * 14
            off_y = (new_h - 518) / 2 if new_h > 518 else 0.0
            K = intr[i]
            Ks.append(map_intrinsics_to_original([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 518.0 / W0, new_h / H0, 0.0, off_y))
            T = np.eye(4)
            T[:3, :4] = extr[i]
            c2ws.append(invert_pose(T))
            c = conf[i].reshape(-1)
            thr = np.percentile(c, 50)
            keep = c > max(thr, 1.0)
            xyz_all.append(pts[i].reshape(-1, 3)[keep])
            rgb_all.append((np.clip(imgs_np[i].reshape(-1, 3)[keep], 0, 1) * 255).astype(np.uint8))
        xyz, rgb = subsample_points(np.concatenate(xyz_all), np.concatenate(rgb_all), int(ctx.opt("maxPoints", 300000)))
        rec = build_reconstruction(ctx.names, sizes, np.stack(c2ws), np.stack(Ks), xyz, rgb, bool(ctx.opt("singleCamera", True)))
        del model, pred
        empty_cache()
        return SfMOutput(rec, None, {"backend": self.id, "processedSize": [int(Wp), int(Hp)], "mapper": "vggt_feedforward"})
