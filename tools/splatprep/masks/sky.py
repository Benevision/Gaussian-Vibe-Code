"""Sky segmentation.

1. SegFormer (ADE20K, class 'sky') through ``transformers`` when installed.
   Model id via ``SPLATPREP_SKY_MODEL`` (default ``nvidia/segformer-b0-finetuned-ade-512-512``,
   ~15 MB).  NOTE: NVIDIA's SegFormer weights carry a non-commercial research licence.
2. Dependency-free heuristic fallback: bright / bluish / low-gradient pixels connected
   to the top image border.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..device import has_module
from ..images import load_rgb


class SkySegmenter:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        if has_module("torch") and has_module("transformers") and os.environ.get("SPLATPREP_SKY_HEURISTIC") != "1":
            try:
                from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation  # type: ignore

                mid = os.environ.get("SPLATPREP_SKY_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512")
                self.processor = AutoImageProcessor.from_pretrained(mid)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(mid).eval().to(device)
                labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
                self.sky_id = labels.get("sky", 2)
                self.method = f"segformer:{mid}"
            except Exception as e:  # weights not downloadable etc.
                self.model = None
                self.method = f"heuristic (segformer failed: {e})"
        if self.model is None and not hasattr(self, "method"):
            self.method = "heuristic"

    def __call__(self, path: Path) -> np.ndarray:
        """Boolean HxW, True = sky."""
        rgb, _ = load_rgb(path)
        if self.model is not None:
            import torch

            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            return (logits.argmax(1)[0] == self.sky_id).cpu().numpy()
        return heuristic_sky(rgb)


def heuristic_sky(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    H, W = rgb.shape[:2]
    s = 256 / max(H, W)
    small = np.asarray(Image.fromarray(rgb).resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR), np.float32) / 255.0
    r, b = small[..., 0], small[..., 2]
    lum = small.mean(2)
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    cand = (grad < 0.02) & ((b >= r - 0.02) | (lum > 0.75)) & (lum > 0.35)
    # flood fill from the top border through candidate pixels
    sky = np.zeros_like(cand)
    stack = [(0, x) for x in range(cand.shape[1]) if cand[0, x]]
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= cand.shape[0] or x >= cand.shape[1] or sky[y, x] or not cand[y, x]:
            continue
        sky[y, x] = True
        stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return np.asarray(Image.fromarray(sky.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
