"""SAM 2 object masks, propagated through the image sequence with the video predictor.

Install: ``pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git" huggingface_hub``.
Checkpoint via Hugging Face (``SPLATPREP_SAM2_MODEL``, default
``facebook/sam2.1-hiera-small`` ~185 MB; ``-large`` ~900 MB).  Apache-2.0.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..images import load_rgb

DEFAULT_MODEL = "facebook/sam2.1-hiera-small"


def _auto_prompt(first_rgb: np.ndarray, device: str) -> dict:
    """Pick the most salient central object with SAM2AutomaticMaskGenerator; fall back to a centre point."""
    H, W = first_rgb.shape[:2]
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore

        gen = SAM2AutomaticMaskGenerator.from_pretrained(os.environ.get("SPLATPREP_SAM2_MODEL", DEFAULT_MODEL), device=device)
        masks = gen.generate(first_rgb)
        best, best_score = None, -1.0
        for m in masks:
            seg = m["segmentation"]
            area = seg.mean()
            if area < 0.02 or area > 0.8:
                continue
            ys, xs = np.nonzero(seg)
            d = np.hypot(xs.mean() / W - 0.5, ys.mean() / H - 0.5)
            touches = seg[0].any() + seg[-1].any() + seg[:, 0].any() + seg[:, -1].any()
            score = m.get("predicted_iou", 1.0) * (1 - d) * (1 - 0.2 * touches) * np.sqrt(area)
            if score > best_score:
                best, best_score = m, score
        if best is not None:
            x, y, w, h = best["bbox"]
            return {"boxes": [[x, y, x + w, y + h]], "points": []}
    except Exception:
        pass
    return {"points": [[W / 2, H / 2, 1]], "boxes": []}


def segment_sequence(
    paths: Sequence[Path],
    device: str,
    prompt: Optional[dict],
    progress: Callable[[float, str], None],
) -> Tuple[List[np.ndarray], dict]:
    import torch

    model_id = os.environ.get("SPLATPREP_SAM2_MODEL", DEFAULT_MODEL)
    first, _ = load_rgb(paths[0])
    H0, W0 = first.shape[:2]
    auto = not prompt or not (prompt.get("points") or prompt.get("boxes"))
    if auto:
        progress(0.02, "SAM 2: choosing object automatically")
        prompt = _auto_prompt(first, device)
    points = np.array([[p[0], p[1]] for p in prompt.get("points", [])], np.float32).reshape(-1, 2)
    labels = np.array([int(p[2]) if len(p) > 2 else 1 for p in prompt.get("points", [])], np.int32)
    boxes = [np.array(b, np.float32) for b in prompt.get("boxes", [])]
    info = {"method": f"sam2 video predictor ({model_id})", "prompt": prompt, "autoPrompt": auto}

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore

        predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    except Exception as e:  # pragma: no cover - fall back to per-image predictor
        return _per_image(paths, device, model_id, points, labels, boxes, progress, info, str(e))

    tmp = Path(tempfile.mkdtemp(prefix="splatprep_sam2_"))
    try:
        from PIL import Image

        sizes = []
        for i, p in enumerate(paths):
            rgb, _ = load_rgb(p)
            sizes.append(rgb.shape[:2])
            Image.fromarray(rgb).save(tmp / f"{i:05d}.jpg", quality=95)
            progress(0.05 + 0.15 * (i + 1) / len(paths), f"SAM 2: staging frame {i + 1}/{len(paths)}")
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(tmp))
            obj = 1
            if boxes:
                for k, b in enumerate(boxes):
                    kw = {"box": b}
                    if k == 0 and len(points):
                        kw.update(points=points, labels=labels)
                    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=obj + k, **kw)
            else:
                predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=obj, points=points, labels=labels)
            out: List[Optional[np.ndarray]] = [None] * len(paths)
            done = 0
            for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                m = (logits > 0.0).any(0).squeeze(0).cpu().numpy() if len(obj_ids) else np.zeros(sizes[fidx], bool)
                out[fidx] = m
                done += 1
                progress(0.2 + 0.8 * done / len(paths), f"SAM 2: propagated {done}/{len(paths)}")
        res = [o if o is not None else np.zeros(sizes[i], bool) for i, o in enumerate(out)]
        return res, info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _per_image(paths, device, model_id, points, labels, boxes, progress, info, why):  # pragma: no cover
    """Fallback: same prompt (scaled to each image) with the image predictor."""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    pred = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    first, _ = load_rgb(paths[0])
    H0, W0 = first.shape[:2]
    res = []
    for i, p in enumerate(paths):
        rgb, _ = load_rgb(p)
        sx, sy = rgb.shape[1] / W0, rgb.shape[0] / H0
        with torch.inference_mode():
            pred.set_image(rgb)
            masks, scores, _ = pred.predict(
                point_coords=points * [sx, sy] if len(points) else None,
                point_labels=labels if len(points) else None,
                box=(boxes[0] * [sx, sy, sx, sy]) if boxes else None,
                multimask_output=False,
            )
        res.append(masks[0] > 0)
        progress((i + 1) / len(paths), f"SAM 2 (per image): {i + 1}/{len(paths)}")
    info["method"] = f"sam2 image predictor ({model_id}); video predictor unavailable: {why}"
    return res, info
