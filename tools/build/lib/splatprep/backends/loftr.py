"""``loftr`` — detector-free dense matcher via kornia (``kornia.feature.LoFTR``, outdoor weights).

Install: ``pip install kornia torch``.  Weights (~45 MB) are fetched by kornia on
first use.  Matches are produced per pair; keypoints are then merged per image
by snapping match coordinates to a half-pixel grid (hloc-style) so that tracks
can form across pairs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..device import empty_cache
from ..images import load_gray
from ..pairs import Pair
from .base import FeatureMatchBackend, SfMContext


class LoFTRBackend(FeatureMatchBackend):
    id = "loftr"
    display_name = "LoFTR (kornia)"
    python_modules = ("torch", "kornia")
    install_hint = "pip install kornia"
    detector_free = True
    max_side = 840

    def extract_and_match(self, ctx: SfMContext, pairs: List[Pair]):
        import torch
        import kornia.feature as KF  # type: ignore

        torch.set_grad_enabled(False)
        dev = torch.device(ctx.device)
        model = KF.LoFTR(pretrained="outdoor").eval().to(dev)
        side = min(int(ctx.opt("resizeMax", 1600)), self.max_side)
        cache: Dict[int, Tuple[object, float, float]] = {}

        def load(i: int):
            if i not in cache:
                g, s = load_gray(ctx.paths[i], side)
                h, w = g.shape
                h8, w8 = h - h % 8, w - w % 8
                t = torch.from_numpy(g[:h8, :w8].astype(np.float32) / 255.0)[None, None].to(dev)
                cache[i] = (t, s, s)
                if len(cache) > 64:
                    cache.pop(next(iter(cache)))
            return cache[i]

        n = len(ctx.names)
        grids: List[Dict[Tuple[int, int], int]] = [dict() for _ in range(n)]
        kps: List[List[Tuple[float, float]]] = [[] for _ in range(n)]
        max_kp = int(ctx.opt("maxKeypoints", 4096))

        def kp_index(i: int, x: float, y: float) -> int:
            key = (int(round(x * 2)), int(round(y * 2)))
            idx = grids[i].get(key)
            if idx is None:
                idx = len(kps[i])
                grids[i][key] = idx
                kps[i].append((key[0] / 2.0, key[1] / 2.0))
            return idx

        matches: Dict[Pair, np.ndarray] = {}
        for k, (i, j) in enumerate(pairs):
            t0, s0, _ = load(i)
            t1, s1, _ = load(j)
            out = model({"image0": t0, "image1": t1})
            p0 = out["keypoints0"].cpu().numpy()
            p1 = out["keypoints1"].cpu().numpy()
            conf = out["confidence"].cpu().numpy()
            order = np.argsort(-conf)[:max_kp]
            p0 = (p0[order] + 0.5) / s0 - 0.5
            p1 = (p1[order] + 0.5) / s1 - 0.5
            m = [(kp_index(i, *a), kp_index(j, *b)) for a, b in zip(p0, p1)]
            matches[(i, j)] = np.unique(np.array(m, int).reshape(-1, 2), axis=0)
            ctx.progress.update("matching", (k + 1) / max(1, len(pairs)), f"LoFTR {k + 1}/{len(pairs)} pairs")
        ctx.progress.update("features", 1.0, "detector-free: keypoints derived from matches")
        del model
        empty_cache()
        return [np.array(k, float).reshape(-1, 2) for k in kps], matches
