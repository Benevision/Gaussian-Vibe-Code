"""Open-vocabulary detection of transient-object classes.

Detector: OWLv2 (``google/owlv2-base-patch16-ensemble``, Apache-2.0) through ``transformers``.
Chosen over Grounding DINO because it is a plain ViT + linear heads (every op has an MPS
kernel; Grounding DINO's multi-scale deformable attention has no MPS kernel and runs a slow
pure-PyTorch path), each class query is scored independently with its own sigmoid (adding or
removing classes does not change other classes' scores, no 256-token prompt limit / phrase
mapping), and both are Apache-2.0.  Override with ``SPLATPREP_DYNAMIC_DETECTOR`` (any
transformers OWLv2/OWL-ViT checkpoint).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_DETECTOR = "google/owlv2-base-patch16-ensemble"


@dataclass
class Detection:
    box: List[float]  # x0 y0 x1 y1, original pixels
    score: float
    label: str

    def to_json(self) -> dict:
        return {"label": self.label, "score": round(float(self.score), 4), "box": [round(float(v), 1) for v in self.box]}


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _containment(small, big) -> float:
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = (small[2] - small[0]) * (small[3] - small[1])
    return ix * iy / a if a > 0 else 0.0


def filter_detections(dets: Sequence[Detection], width: int, height: int, nms_iou: float = 0.5,
                      contain: float = 0.9, max_area: float = 0.9, min_side: float = 8.0) -> List[Detection]:
    """Clip, drop degenerate / near-full-frame boxes, class-agnostic NMS, and drop boxes that
    lie (almost) entirely inside a higher-scoring kept box (SAM 2 on the big box covers them)."""
    keep: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        x0, y0, x1, y1 = d.box
        b = [max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)]
        if b[2] - b[0] < min_side or b[3] - b[1] < min_side:
            continue
        if (b[2] - b[0]) * (b[3] - b[1]) > max_area * width * height:
            continue
        if any(box_iou(b, k.box) > nms_iou or _containment(b, k.box) > contain for k in keep):
            continue
        keep.append(Detection(b, d.score, d.label))
    return keep


class OwlDetector:
    def __init__(self, device: str = "cpu", model_id: Optional[str] = None):
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection, OwlViTForObjectDetection  # type: ignore

        self.torch = torch
        self.model_id = model_id or os.environ.get("SPLATPREP_DYNAMIC_DETECTOR", DEFAULT_DETECTOR)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        cls = OwlViTForObjectDetection if "owlvit" in self.model_id.lower() else Owlv2ForObjectDetection
        self.device = device
        self.model = cls.from_pretrained(self.model_id).eval().to(device)
        self.method = f"owlv2:{self.model_id}"
        # OWLv2 pads to a square (bottom/right) before resizing; OWL-ViT resizes directly
        self.pads_square = cls is Owlv2ForObjectDetection

    def __call__(self, rgbs: Sequence[np.ndarray], classes: Sequence[str], threshold: float) -> List[List[Detection]]:
        torch = self.torch
        from PIL import Image

        ims = [Image.fromarray(r) for r in rgbs]
        inp = self.processor(text=[list(classes)] * len(ims), images=ims, return_tensors="pt")
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.inference_mode():
            out = self.model(**inp)
        sizes = []
        for r in rgbs:
            H, W = r.shape[:2]
            sizes.append((max(H, W), max(H, W)) if self.pads_square else (H, W))
        post = getattr(self.processor, "image_processor", self.processor)
        res = post.post_process_object_detection(out, threshold=float(threshold), target_sizes=torch.tensor(sizes))
        dets = []
        for r, rgb in zip(res, rgbs):
            H, W = rgb.shape[:2]
            ds = [Detection([float(v) for v in b.tolist()], float(s), classes[int(l)])
                  for s, l, b in zip(r["scores"].cpu(), r["labels"].cpu(), r["boxes"].cpu())]
            dets.append(filter_detections(ds, W, H))
        return dets
