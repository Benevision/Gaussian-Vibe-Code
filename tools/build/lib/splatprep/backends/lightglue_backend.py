"""LightGlue-family backends (cvg/LightGlue): sift / superpoint / disk / aliked + LightGlue.

Install: ``pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"`` (pulls torch, kornia).
Weights are downloaded by LightGlue to the torch hub cache on first use
(~5 MB SuperPoint, ~4 MB DISK, ~2 MB ALIKED, ~45 MB per LightGlue matcher).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..device import empty_cache
from .base import FeatureMatchBackend, SfMContext

_EXTRACTORS = {"sift": "SIFT", "superpoint": "SuperPoint", "disk": "DISK", "aliked": "ALIKED"}


class LightGlueBackend(FeatureMatchBackend):
    python_modules = ("torch", "lightglue")
    install_hint = "pip install 'splatprep[lightglue]'"

    def __init__(self, feature: str):
        self.feature = feature
        self.id = f"{feature}_lightglue"
        self.display_name = f"{_EXTRACTORS[feature]} + LightGlue"

    def check(self):
        a = super().check()
        if a.available and self.feature == "sift" and not (self._has("pycolmap") or self._has("cv2")):
            from .base import Availability

            return Availability(False, False, "LightGlue SIFT needs opencv or pycolmap")
        return a

    @staticmethod
    def _has(m: str) -> bool:
        from ..device import has_module

        return has_module(m)

    def setup(self, ctx: SfMContext) -> None:
        from ..isolation import block_pycolmap
        block_pycolmap()  # lightglue imports pycolmap opportunistically; see isolation.py
        import torch
        import lightglue  # type: ignore

        torch.set_grad_enabled(False)
        self.torch = torch
        self.dev = torch.device(ctx.device)
        Extractor = getattr(lightglue, _EXTRACTORS[self.feature])
        kw: Dict[str, Any] = {"max_num_keypoints": int(ctx.opt("maxKeypoints", 4096))}
        if self.feature == "sift":
            kw["backend"] = "opencv"  # never pycolmap in a torch process (duplicate libomp)
            # LightGlue's DoG non-max suppression (filter_dog_point) raises "array is not
            # broadcastable" on some OpenCV keypoint sets; OpenCV SIFT already does its own NMS.
            kw["nms_radius"] = 0
        self.extractor = Extractor(**kw).eval().to(self.dev)
        mconf: Dict[str, Any] = {"features": self.feature}
        if ctx.device == "mps":
            mconf["flash"] = False
        self.matcher = lightglue.LightGlue(**mconf).eval().to(self.dev)
        self.resize = int(ctx.opt("resizeMax", 1600))

    def teardown(self) -> None:
        self.extractor = None
        self.matcher = None
        empty_cache()

    def extract(self, ctx: SfMContext, index: int, path: Path) -> Dict[str, Any]:
        from lightglue.utils import load_image  # type: ignore

        img = load_image(path).to(self.dev)
        feats = self.extractor.extract(img, resize=self.resize)
        # keep on CPU between extraction and matching to bound GPU memory
        return {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in feats.items()}

    def keypoints_of(self, feats) -> np.ndarray:
        return feats["keypoints"][0].numpy().astype(np.float64)

    def filter_features(self, feats, keep: np.ndarray):
        n = feats["keypoints"].shape[1]
        idx = self.torch.from_numpy(np.nonzero(keep)[0])
        out = {}
        for k, v in feats.items():
            if k != "image_size" and hasattr(v, "shape") and v.ndim >= 2 and v.shape[1] == n:
                out[k] = v[:, idx]
            else:
                out[k] = v
        return out

    def match(self, ctx: SfMContext, f0, f1) -> np.ndarray:
        if f0["keypoints"].shape[1] == 0 or f1["keypoints"].shape[1] == 0:
            return np.zeros((0, 2), int)
        d0 = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in f0.items()}
        d1 = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in f1.items()}
        out = self.matcher({"image0": d0, "image1": d1})
        m = out["matches"][0] if isinstance(out["matches"], (list, tuple)) else out["matches"]
        if m.ndim == 3:
            m = m[0]
        return m.cpu().numpy().astype(int)
