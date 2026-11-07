"""Person segmentation with torchvision DeepLabV3-MobileNetV3 (VOC 'person' class).

Install: ``pip install torch torchvision``; weights (~42 MB) are downloaded by
torchvision on first use (BSD-3 code; weights trained on COCO/VOC subsets).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VOC_PERSON = 15


class PersonSegmenter:
    method = "torchvision deeplabv3_mobilenet_v3_large"

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

        self.torch = torch
        self.device = torch.device(device)
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(self.device)
        self.tf = weights.transforms()

    def __call__(self, path: Path) -> np.ndarray:
        """Boolean HxW array, True = person."""
        from PIL import Image, ImageOps

        torch = self.torch
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            x = self.tf(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)["out"]
            out = torch.nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            person = out.argmax(1)[0] == VOC_PERSON
        m = person.cpu().numpy()
        # dilate a little: people's edges / shadows are the worst floaters
        try:
            import cv2

            m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        except Exception:
            pass
        return m
