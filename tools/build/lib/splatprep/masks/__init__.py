"""Masking (task = "masks"): writes ``masks/<image stem>.png`` (8-bit, 255 = keep).

Semantics of "keep" per model (flip with ``invertMask``):

* ``person``     — 255 everywhere EXCEPT people (transient people are removed from training).
* ``sky``        — 255 everywhere EXCEPT sky.
* ``dynamic``    — 255 everywhere EXCEPT moving-object classes (OWLv2 open-vocabulary
  detection → SAM 2 video tracking, ``dynamic.py``), dilated by ``dilatePixels``.
  ``maskPrompts`` objects are removed as well.
* ``sam2_track`` — 255 everywhere EXCEPT the objects of ``maskPrompts`` (points / boxes on ANY
  frame), each tracked forward and backward through the sequence.
* ``sam2_auto``  — with non-empty ``maskPrompts`` (Mac mask editor): identical to
  ``sam2_track`` (prompted objects tracked forward + backward → 0).  Otherwise 255 on the
  segmented object (object-centric capture), 0 on background: legacy ``maskPrompt`` (first
  image) or, without a prompt, the most salient central object from SAM 2's automatic mask
  generator.

Merge: when ``masks/<stem>.png`` already exists the new mask is combined with logical AND
(min — anything either mask ignores stays ignored) unless ``replaceMasks: true``.

This module must stay importable without torch (``model_status`` imports it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..device import has_module
from ..errors import ModelUnavailable, RequestError

PROMPT_MODELS = ("sam2_track", "sam2_auto", "dynamic")


def mask_model_status() -> Dict[str, dict]:
    torch = has_module("torch")
    sam2 = torch and has_module("sam2")
    tf = torch and has_module("transformers")
    return {
        "person": {
            "available": torch and has_module("torchvision"),
            "sfm": False,
            "reason": "" if (torch and has_module("torchvision")) else "missing torch/torchvision — pip install 'splatprep[masks]'",
        },
        "sky": {
            "available": True,
            "sfm": False,
            "reason": "SegFormer (transformers)" if tf else "heuristic fallback (install transformers for SegFormer)",
        },
        "sam2_auto": {
            "available": sam2,
            "sfm": False,
            "reason": "" if sam2 else "missing sam2 — pip install 'splatprep[sam2]'",
        },
        "sam2_track": {
            "available": sam2,
            "sfm": False,
            "reason": "" if sam2 else "missing sam2 — pip install 'splatprep[sam2]'",
        },
        "dynamic": {
            "available": sam2 and tf,
            "sfm": False,
            "reason": "" if (sam2 and tf) else (
                "missing " + " + ".join(x for x, ok in (("transformers", tf), ("sam2", sam2)) if not ok)
                + " — pip install 'splatprep[masks,sam2]'"),
        },
    }


def mask_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / (Path(name).with_suffix("").as_posix() + ".png")


def merge_keep(keep: np.ndarray, existing: Optional[np.ndarray]) -> np.ndarray:
    """Logical AND of two keep masks (existing may be uint8 0/255 of any size → resized nearest)."""
    keep = np.asarray(keep, bool)
    if existing is None:
        return keep
    ex = np.asarray(existing)
    if ex.ndim == 3:
        ex = ex[..., 0]
    if ex.shape != keep.shape:
        from PIL import Image

        ex = np.asarray(Image.fromarray(ex.astype(np.uint8)).resize((keep.shape[1], keep.shape[0]), Image.NEAREST))
    return keep & (ex > 127)


def save_mask(out_dir: Path, name: str, keep: np.ndarray, invert: bool, merge: bool = False) -> Path:
    """Write ``<out_dir>/<stem>.png`` (255 = keep).  ``merge`` ANDs with an existing file.
    Written atomically (temp file + rename) so readers never see a partial PNG."""
    import os

    from PIL import Image

    keep = np.asarray(keep, bool)
    if invert:
        keep = ~keep
    p = mask_path(out_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    if merge and p.exists():
        try:
            with Image.open(p) as im:
                existing = np.asarray(im.convert("L"))
            keep = merge_keep(keep, existing)
        except Exception:  # unreadable old mask → replace it
            pass
    tmp = p.with_name(f".{p.name}.tmp")
    Image.fromarray(keep.astype(np.uint8) * 255).save(tmp, format="PNG")
    os.replace(tmp, p)
    return p


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0 or not mask.any():
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


class MaskWriter:
    """Turns per-frame OBJECT masks into keep masks and writes them.

    ``object_is_keep`` False (dynamic / sam2_track): keep = NOT dilate(object).
    ``object_is_keep`` True (sam2_auto): keep = dilate(object)."""

    def __init__(self, masks_dir: Path, names: List[str], sizes, dilate_px: int, invert: bool, merge: bool,
                 object_is_keep: bool):
        self.masks_dir, self.names, self.sizes = Path(masks_dir), names, sizes
        self.dilate_px, self.invert, self.merge, self.object_is_keep = dilate_px, invert, merge, object_is_keep
        self.written = 0
        self.masked_pixels = 0.0

    def __call__(self, i: int, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            H, W = self.sizes(i)
            obj = np.zeros((H, W), bool)
        obj = dilate(np.asarray(obj, bool), self.dilate_px)
        self.masked_pixels += float(obj.mean())
        keep = obj if self.object_is_keep else ~obj
        save_mask(self.masks_dir, self.names[i], keep, self.invert, self.merge)
        self.written += 1


def run_masks(
    model: str,
    image_dir: Path,
    names: List[str],
    out_dir: Path,
    device: str,
    progress: Callable[[float, str], None],
    prompt: Optional[dict] = None,
    invert: bool = False,
    request: Optional[dict] = None,
) -> dict:
    """``request`` carries the optional keys (maskPrompts, maskClasses, maskThreshold,
    dilatePixels, replaceMasks, ...); ``prompt`` is the legacy ``maskPrompt``."""
    st = mask_model_status().get(model)
    if st is None:
        raise RequestError(f"unknown maskModel: {model}")
    if not st["available"]:
        raise ModelUnavailable(model, st["reason"])
    from .prompts import mask_options, parse_classes, parse_mask_prompts

    req = dict(request or {})
    if prompt is not None and "maskPrompt" not in req:
        req["maskPrompt"] = prompt
    req["invertMask"] = bool(invert or req.get("invertMask"))
    opts = mask_options(req)
    merge = not opts["replace"]
    masks_dir = Path(out_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(image_dir) / n for n in names]
    if not names:
        raise RequestError(f"no images in {image_dir}")
    info: dict
    if model in ("person", "sky"):
        if model == "person":
            from .person import PersonSegmenter as Seg
        else:
            from .sky import SkySegmenter as Seg  # type: ignore[no-redef]
        seg = Seg(device)
        info = {"method": seg.method}
        for i, (n, p) in enumerate(zip(names, paths)):
            save_mask(masks_dir, n, ~seg(p), opts["invert"], merge)
            progress((i + 1) / len(names), f"{model} mask {i + 1}/{len(names)}")
    else:
        from ..images import image_size
        from .dynamic import run_tracking_job

        prompts = parse_mask_prompts(req, names)
        # sam2_auto + non-empty maskPrompts (the Mac mask editor) = prompted tracking of objects
        # to REMOVE (object → 0), exactly like sam2_track; legacy maskPrompt / no prompt keep the
        # object-centric semantics (object → 255).
        removal = model in ("dynamic", "sam2_track") or (model == "sam2_auto" and bool(req.get("maskPrompts")))
        if model == "sam2_auto" and not prompts:
            from ..images import load_rgb
            from .prompts import FramePrompt
            from .sam2 import _auto_prompt

            progress(0.02, "SAM 2: choosing object automatically")
            ap = _auto_prompt(load_rgb(paths[0])[0], device)
            pts = [[p[0], p[1]] for p in ap.get("points", [])]
            prompts = [FramePrompt(0, 1, pts, [1] * len(pts), [list(b) for b in ap.get("boxes", [])], "auto")]
        if model == "sam2_track" and not prompts:
            raise RequestError("maskModel sam2_track needs maskPrompts (points/boxes on any image)")
        if model != "dynamic" and "maskSequence" not in req:
            opts["sequence"] = "video"  # prompts must be tracked even through unordered photo sets
        dpx = opts["dilate"] if (removal or req.get("dilatePixels") is not None) else 0
        sizes_cache: Dict[int, tuple] = {}

        def size_of(i):
            if i not in sizes_cache:
                w, h = image_size(paths[i])
                sizes_cache[i] = (h, w)
            return sizes_cache[i]

        writer = MaskWriter(masks_dir, names, size_of, dpx, opts["invert"], merge, object_is_keep=not removal)
        classes = parse_classes(req) if model == "dynamic" else None
        det_file = (Path(out_dir) / "dynamic_detections.json") if (classes and opts["saveDetections"]) else None
        info = run_tracking_job(paths, names, device, opts, prompts, writer, progress, classes, det_file)
        info.update({"dilatePixels": dpx, "numUserPrompts": sum(1 for p in prompts if p.source == "user"),
                     "meanMaskedFraction": round(writer.masked_pixels / max(1, writer.written), 5)})
        info["objectsRemoved"] = removal
        if model == "sam2_auto":
            info["autoPrompt"] = any(p.source == "auto" for p in prompts)
    info.update({"maskModel": model, "numMasks": len(names), "inverted": opts["invert"], "merged": merge})
    return info
