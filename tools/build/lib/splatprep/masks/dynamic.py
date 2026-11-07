"""Transient / moving-object masks and prompted SAM 2 tracking.

``maskModel = "dynamic"``: OWLv2 open-vocabulary detection of moving-object classes on every
frame → SAM 2 image predictor turns each box into a mask → ONE SAM 2 video object ("all
detections") is seeded with the first detected frame's masks and tracked through the ordered
sequence (chunked, forward + backward, ``track.py``); whenever a detection on some frame is not
covered (``maskCoverage``) by the tracked mask — a new object appeared or tracking lost one —
propagation stops, ``tracked ∪ detections`` is added as a mask prompt there and propagation
resumes.  Tracking fills frames where the detector missed; the per-frame detection masks are
OR-ed in as well.  Cost is ~one object-frame per frame and sweep, independent of the number of
instances.  The union is dilated by ``dilatePixels`` → 0 in the keep mask.  ``maskPrompts`` are tracked
alongside the detections.  Unordered photo sets (no temporal order) use the SAM 2 image
predictor per image with that image's boxes instead of tracking.

``maskModel = "sam2_track"``: only the user's ``maskPrompts`` (any frame), tracked forward and
backward; objects → 0.  ``sam2_auto`` uses the same engine with object → 255.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..images import load_rgb, looks_like_video
from .prompts import FramePrompt

DEFAULT_SAM2 = "facebook/sam2.1-hiera-small"
Progress = Callable[[float, str], None]
Emit = Callable[[int, Optional[np.ndarray]], None]

_CACHE: Dict[tuple, object] = {}


def sam2_model_id() -> str:
    return os.environ.get("SPLATPREP_SAM2_MODEL", DEFAULT_SAM2)


def load_sam2(kind: str, device: str):
    """Cached SAM 2 predictor: kind = "video" | "image"."""
    key = (kind, device, sam2_model_id())
    if key not in _CACHE:
        if kind == "video":
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore

            _CACHE[key] = SAM2VideoPredictor.from_pretrained(sam2_model_id(), device=device)
        else:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            _CACHE[key] = SAM2ImagePredictor.from_pretrained(sam2_model_id(), device=device)
    return _CACHE[key]


def image_predictor_for(video_pred):
    """SAM2ImagePredictor sharing the video predictor's weights (no second model in memory)."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    return SAM2ImagePredictor(video_pred)


def drop_cached(kind: Optional[str] = None) -> None:
    for k in list(_CACHE):
        if kind is None or k[0] == kind:
            del _CACHE[k]
    from ..device import empty_cache

    empty_cache()


def is_sequence(names: Sequence[str], mode: str) -> bool:
    if mode == "video":
        return True
    if mode == "unordered":
        return False
    return looks_like_video(list(names))


def _loader(p: Path) -> np.ndarray:
    return load_rgb(p)[0]


def detect_all(paths: Sequence[Path], device: str, classes: Sequence[str], threshold: float, stride: int,
               progress: Progress, batch: int = 2):
    from .detect import OwlDetector

    t0 = time.perf_counter()
    det = OwlDetector(device)
    frames = list(range(0, len(paths), max(1, stride)))
    if frames and frames[-1] != len(paths) - 1 and stride > 1:
        frames.append(len(paths) - 1)
    out = {}
    for k in range(0, len(frames), batch):
        fs = frames[k:k + batch]
        rgbs = [_loader(paths[f]) for f in fs]
        for f, d in zip(fs, det(rgbs, classes, threshold)):
            out[f] = d
        n_obj = sum(len(v) for v in out.values())
        progress(min(1.0, (k + len(fs)) / len(frames)), f"detecting {', '.join(classes[:3])}...: frame {fs[-1] + 1}/{len(paths)} ({n_obj} boxes)")
    method = det.method
    del det
    from ..device import empty_cache

    empty_cache()
    return out, method, time.perf_counter() - t0


def run_tracking_job(
    paths: Sequence[Path],
    names: Sequence[str],
    device: str,
    opts: dict,
    user_prompts: Sequence[FramePrompt],
    emit: Emit,
    progress: Progress,
    classes: Optional[Sequence[str]] = None,
    detections_out: Optional[Path] = None,
) -> dict:
    """Compute object masks for every frame and hand them to ``emit(i, mask_or_None)`` in
    order (mask = bool HxW at the original resolution, True = object; None = no object).

    ``classes`` given → run the detector (dynamic); else only ``user_prompts`` are used."""
    n = len(paths)
    info: dict = {"sam2Model": sam2_model_id()}
    seq = is_sequence(names, opts.get("sequence", "auto"))
    info["sequence"] = "video" if seq else "unordered"
    prompts: List[FramePrompt] = list(user_prompts)
    det_boxes: Dict[int, List[List[float]]] = {}
    det_frac = 0.35 if classes else 0.0
    timings = {}
    if classes:
        dets, method, dt = detect_all(paths, device, list(classes), opts["threshold"], opts.get("detectStride", 1),
                                      lambda f, m: progress(det_frac * f, m))
        timings["detectSec"] = round(dt, 2)
        info.update({"detector": method, "classes": list(classes), "threshold": opts["threshold"]})
        n_boxes = sum(len(v) for v in dets.values())
        counts: Dict[str, int] = {}
        for v in dets.values():
            for d in v:
                counts[d.label] = counts.get(d.label, 0) + 1
        info.update({"numDetections": n_boxes, "framesWithDetections": sum(1 for v in dets.values() if v), "detectionsPerClass": counts})
        det_boxes = {f: [list(d.box) for d in v] for f, v in dets.items() if v}
        if not seq:  # per image: each detection is just a box prompt on its own image
            k = max([p.object_id for p in prompts], default=0)
            for f, bs in sorted(det_boxes.items()):
                for b in bs:
                    k += 1
                    prompts.append(FramePrompt(f, k, [], [], [b], "detector"))
            det_boxes = {}
        if detections_out is not None:
            import json

            detections_out.parent.mkdir(parents=True, exist_ok=True)
            detections_out.write_text(json.dumps({
                "format": "splatdetections", "version": 1, "detector": method, "classes": list(classes),
                "threshold": opts["threshold"],
                "frames": {names[f]: [d.to_json() for d in v] for f, v in sorted(dets.items()) if v},
            }, indent=1))
            info["detectionsFile"] = str(detections_out)
    info["numPrompts"] = len(prompts) + sum(len(v) for v in det_boxes.values())
    sub = lambda f, m: progress(det_frac + (1 - det_frac) * f, m)  # noqa: E731
    if not prompts and not det_boxes:
        for i in range(n):
            emit(i, None)
        sub(1.0, "no objects detected — all masks keep everything")
        info.update({"framesMasked": 0, "timings": timings})
        return info

    t0 = time.perf_counter()
    masked = 0
    from .track import set_precision

    info["precision"] = set_precision(device, opts.get("precision", "auto"))
    if seq:
        from .track import FrameStore, track_sequence

        store = FrameStore(paths, opts.get("stageMax", 1024), lambda f, m: sub(0.1 * f, m), loader=_loader)
        try:
            pred = load_sam2("video", device)
            # bound memory-attention cost: attend to the temporally closest K conditioning frames
            pred.max_cond_frames_in_attn = int(opts.get("maxCond", 4))
            dm = None
            if det_boxes:
                from .track import detection_masks

                dm = detection_masks(image_predictor_for(pred), store, det_boxes, lambda f, m: sub(0.1 + 0.15 * f, m))
            init_kw = {"offload_video_to_cpu": True, "offload_state_to_cpu": bool(device == "cuda" if opts.get("offloadState") is None else opts["offloadState"])}
            res = track_sequence(pred, store, prompts, opts.get("chunk", 64), opts.get("overlap", 2),
                                 opts.get("bidirectional", True), lambda f, m: sub(0.25 + 0.7 * f, m), init_kw,
                                 dm, opts.get("coverage", 0.5))
            info.update({"method": f"sam2 video predictor ({sam2_model_id()}), chunked", "chunks": res.chunks,
                         "sweeps": res.sweeps, "trackResolution": list(store.ref_size),
                         "detectorReprompts": res.corrections, "framesTracked": res.frames_processed,
                         "framesPerObject": {str(k): len(v) for k, v in res.frames_per_object.items()}})
            for i in range(n):
                if i in res.masks:
                    m = store.to_original(i, res.masks.get(i))
                    masked += bool(m.any())
                    emit(i, m)
                else:
                    emit(i, None)
                sub(0.95 + 0.05 * (i + 1) / n, f"writing mask {i + 1}/{n}")
        finally:
            store.cleanup()
    else:
        from .track import segment_unordered

        pred = load_sam2("image", device)
        res = segment_unordered(pred, paths, prompts, _loader, lambda f, m: sub(0.95 * f, m))
        info["method"] = f"sam2 image predictor per image ({sam2_model_id()})"
        for i in range(n):
            m = res.get(i)
            masked += bool(m is not None and m.any())
            emit(i, m)
    timings["segmentSec"] = round(time.perf_counter() - t0, 2)
    info.update({"framesMasked": masked, "timings": timings})
    sub(1.0, f"{masked}/{n} frames contain masked objects")
    return info
