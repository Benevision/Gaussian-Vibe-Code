"""``opencv_sift`` — dependency-light baseline: OpenCV SIFT (RootSIFT) + ratio test + mutual check.

Full SfM needs a mapper (pycolmap or the colmap/glomap CLI); without one the
backend is *matching-only* and the benchmark reports matching metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..images import load_gray
from .base import FeatureMatchBackend, SfMContext


@dataclass
class _Feats:
    kp: np.ndarray  # Nx2, original-image pixels
    desc: np.ndarray  # Nx128 float32 (RootSIFT)


class OpenCVSiftBackend(FeatureMatchBackend):
    id = "opencv_sift"
    display_name = "OpenCV SIFT + ratio test"
    python_modules = ("cv2",)
    install_hint = "pip install opencv-python-headless"
    ratio = 0.8

    def setup(self, ctx: SfMContext) -> None:
        import cv2

        self._sift = cv2.SIFT_create(nfeatures=int(ctx.opt("maxKeypoints", 4096)), contrastThreshold=0.02)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, ctx: SfMContext, index: int, path: Path) -> _Feats:
        gray, s = load_gray(path, int(ctx.opt("resizeMax", 1600)))
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or not kps:
            return _Feats(np.zeros((0, 2)), np.zeros((0, 128), np.float32))
        pts = np.array([k.pt for k in kps], np.float64)
        # cv2 keypoints at resized resolution -> original pixels (pixel-centre aware)
        pts = (pts + 0.5) / s - 0.5
        desc = desc.astype(np.float32)
        desc /= np.abs(desc).sum(1, keepdims=True) + 1e-8
        desc = np.sqrt(desc)
        return _Feats(pts, desc)

    def keypoints_of(self, feats: _Feats) -> np.ndarray:
        return feats.kp

    def filter_features(self, feats: _Feats, keep: np.ndarray) -> _Feats:
        return _Feats(feats.kp[keep], feats.desc[keep])

    def match(self, ctx: SfMContext, f0: _Feats, f1: _Feats) -> np.ndarray:
        if len(f0.desc) < 2 or len(f1.desc) < 2:
            return np.zeros((0, 2), int)
        knn01 = self._bf.knnMatch(f0.desc, f1.desc, k=2)
        knn10 = self._bf.knnMatch(f1.desc, f0.desc, k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in knn10 if m}
        out = []
        for m in knn01:
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx:
                out.append((m[0].queryIdx, m[0].trainIdx))
        return np.array(out, int).reshape(-1, 2)

    def teardown(self) -> None:
        self._sift = None
        self._bf = None
