"""Chunked, bidirectional SAM 2 tracking of prompted objects through an ordered sequence.

Memory: SAM 2's video predictor keeps every frame of a "video" (1024² float tensors, ~12 MB
each) plus per-frame state.  To scale to 500+ frames the sequence is cut into chunks of
``chunk`` frames that overlap by ``overlap`` frames; an object's masks on the overlap frames
of one chunk are fed as mask prompts (``add_new_mask``) into the next chunk, so tracking
continues across chunk borders while only one chunk is ever resident.

Direction: a *forward* sweep (chunks in order) starts every object at its first prompt and
uses all its prompts as conditioning frames; a *backward* sweep (chunks in reverse, SAM 2
``reverse=True``) starts every object at its EARLIEST prompt frame and covers the frames
before it.  Together each object is tracked from any prompted frame to both ends of the
sequence.  The result is the per-frame union over all objects (``True`` = object).

Everything here talks to the predictor through a small duck-typed interface
(``init_state / add_new_points_or_box / add_new_mask / propagate_in_video / reset_state``)
so it is unit-tested with a fake predictor (tests/test_dynamic_masks.py).
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import FramePrompt

Progress = Callable[[float, str], None]


_AUTOCAST: Optional[Tuple[str, object]] = None  # (device_type, dtype) set by set_precision()


def set_precision(device: str, precision: str = "auto") -> str:
    """Choose SAM 2 compute precision: "auto" = bfloat16 autocast on CUDA / MPS (2-3x faster on
    Apple Silicon; masks are thresholded logits so the difference is negligible), "fp32" = off.
    Returns the effective precision.  Falls back to fp32 when bf16 is not supported."""
    global _AUTOCAST
    _AUTOCAST = None
    if precision not in ("auto", "bf16", "fp16") or device not in ("cuda", "mps", "xpu"):
        return "fp32"
    try:
        import torch

        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        torch.zeros(2, device=device, dtype=dtype).sum().item()
        with torch.autocast(device, dtype=dtype):
            pass
        _AUTOCAST = (device, dtype)
        return "fp16" if precision == "fp16" else "bf16"
    except Exception:
        return "fp32"


def _no_grad():
    """torch.inference_mode() (+ autocast per set_precision) when torch is importable
    (tests run fakes without it)."""
    import contextlib

    try:
        import torch
    except ImportError:  # pragma: no cover
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    if _AUTOCAST is not None:
        stack.enter_context(torch.autocast(_AUTOCAST[0], dtype=_AUTOCAST[1]))
    return stack


def plan_chunks(n: int, chunk: int, overlap: int) -> List[Tuple[int, int]]:
    """``[(start, end), ...]`` (end exclusive) covering ``range(n)``; consecutive chunks share
    exactly ``overlap`` frames (``next.start == prev.end - overlap``).  A short tail is merged
    into the previous chunk when that keeps it within 1.5 × ``chunk``."""
    if n <= 0:
        return []
    chunk = max(int(chunk), 2)
    overlap = int(min(max(overlap, 1), chunk - 1))
    if n <= chunk:
        return [(0, n)]
    out = []
    s = 0
    while True:
        e = min(s + chunk, n)
        out.append((s, e))
        if e >= n:
            break
        s = e - overlap
    if len(out) > 1:
        ps, pe = out[-2]
        ls, le = out[-1]
        if le - ps <= int(1.5 * chunk) and le - ls < chunk // 2:
            out[-2:] = [(ps, le)]
    return out


class PackedMasks:
    """Per-frame boolean masks of one fixed shape, stored bit-packed (1/8 of bool memory)."""

    def __init__(self, n: int, shape: Tuple[int, int]):
        self.n = n
        self.shape = (int(shape[0]), int(shape[1]))
        self._data: Dict[int, np.ndarray] = {}

    def __contains__(self, i: int) -> bool:
        return i in self._data

    def get(self, i: int) -> np.ndarray:
        d = self._data.get(i)
        if d is None:
            return np.zeros(self.shape, bool)
        return np.unpackbits(d, count=self.shape[0] * self.shape[1]).reshape(self.shape).astype(bool)

    def set(self, i: int, m: np.ndarray) -> None:
        m = np.asarray(m, bool)
        assert m.shape == self.shape, (m.shape, self.shape)
        self._data[i] = np.packbits(m.ravel())

    def union(self, i: int, m: np.ndarray) -> None:
        m = np.asarray(m, bool)
        if not m.any():
            return
        if i in self._data:
            m = m | self.get(i)
        self.set(i, m)

    def nonempty_frames(self) -> List[int]:
        return sorted(i for i, d in self._data.items() if d.any())


class FrameStore:
    """Stages an image sequence as ``<index>.jpg`` at one reference size for SAM 2.

    All frames are resized (not cropped) to the reference size = first frame scaled to
    ``stage_max`` on the long side.  SAM 2 itself squashes every frame to 1024², so this is
    lossless with respect to what the model sees; prompts are scaled per frame accordingly.
    """

    def __init__(self, paths: Sequence[Path], stage_max: int = 1024, progress: Optional[Progress] = None,
                 root: Optional[Path] = None, loader=None):
        from PIL import Image

        from ..images import load_rgb

        self.paths = [Path(p) for p in paths]
        self.loader = loader or (lambda p: load_rgb(p)[0])
        self.root = Path(tempfile.mkdtemp(prefix="splatprep_frames_", dir=root))
        self.orig_sizes: List[Tuple[int, int]] = []  # (H, W)
        ref = None
        for i, p in enumerate(self.paths):
            rgb = self.loader(p)
            H, W = rgb.shape[:2]
            self.orig_sizes.append((H, W))
            if ref is None:
                s = min(1.0, stage_max / max(H, W))
                ref = (max(2, round(H * s)), max(2, round(W * s)))
            im = Image.fromarray(rgb)
            if (H, W) != ref:
                im = im.resize((ref[1], ref[0]), Image.BILINEAR)
            im.save(self.root / f"{i:06d}.jpg", quality=95)
            if progress:
                progress((i + 1) / len(self.paths), f"staging frame {i + 1}/{len(self.paths)}")
        self.ref_size: Tuple[int, int] = ref or (2, 2)

    def __len__(self) -> int:
        return len(self.paths)

    def scale_of(self, i: int) -> Tuple[float, float]:
        """(sx, sy) mapping original pixels of frame i to reference pixels."""
        H, W = self.orig_sizes[i]
        return self.ref_size[1] / W, self.ref_size[0] / H

    def chunk_dir(self, s: int, e: int) -> Path:
        d = self.root / f"chunk_{s:06d}_{e:06d}"
        if d.exists():
            return d
        d.mkdir()
        for k, i in enumerate(range(s, e)):
            src = self.root / f"{i:06d}.jpg"
            try:
                os.symlink(src, d / f"{k:05d}.jpg")
            except OSError:
                shutil.copy2(src, d / f"{k:05d}.jpg")
        return d

    def release_chunk(self, s: int, e: int) -> None:
        shutil.rmtree(self.root / f"chunk_{s:06d}_{e:06d}", ignore_errors=True)

    def frame_rgb(self, i: int) -> np.ndarray:
        return self.loader(self.paths[i])

    def to_original(self, i: int, m: np.ndarray) -> np.ndarray:
        """Resize a reference-size bool mask to frame i's original size."""
        H, W = self.orig_sizes[i]
        if m.shape == (H, W):
            return m
        import cv2

        return cv2.resize(m.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------


UNION_IID = 0  # SAM 2 object id of the "all detections" object (dynamic masks)


@dataclass
class _Seed:
    iid: int  # internal SAM 2 object id (one per (objectId, box index))
    frame: int
    points: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    box: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


@dataclass
class TrackResult:
    masks: PackedMasks  # union over objects, reference resolution
    frames_per_object: Dict[int, List[int]] = field(default_factory=dict)  # objectId (0 = detections) → frames
    chunks: List[Tuple[int, int]] = field(default_factory=list)
    sweeps: int = 0
    frames_processed: int = 0
    corrections: int = 0  # detector re-prompts of the detections object


def expand_seeds(prompts: Iterable[FramePrompt], scale_of: Callable[[int], Tuple[float, float]]):
    """FramePrompts (original pixels) → per-internal-object seeds (reference pixels).

    Box k of an object becomes internal sub-object k (SAM 2 accepts one box per object and
    frame); points go with box 0.  Returns (seeds, iid → objectId)."""
    keymap: Dict[Tuple[int, int], int] = {}
    owner: Dict[int, int] = {}
    seeds: List[_Seed] = []

    def iid_for(oid, k):
        if (oid, k) not in keymap:
            keymap[(oid, k)] = len(keymap) + 1
            owner[keymap[(oid, k)]] = oid
        return keymap[(oid, k)]

    for p in prompts:
        if p.empty():
            continue
        sx, sy = scale_of(p.frame)
        q = p.scaled(sx, sy)
        pts = np.asarray(q.points, np.float32).reshape(-1, 2) if q.points else None
        labs = np.asarray(q.labels, np.int32) if q.points else None
        if q.boxes:
            for k, b in enumerate(q.boxes):
                seeds.append(_Seed(iid_for(p.object_id, k), p.frame,
                                   pts if k == 0 else None, labs if k == 0 else None, np.asarray(b, np.float32)))
        else:
            seeds.append(_Seed(iid_for(p.object_id, 0), p.frame, pts, labs, None))
    return seeds, owner


def _first_seeds(seeds: List[_Seed]) -> List[_Seed]:
    first: Dict[int, int] = {}
    for s in seeds:
        first[s.iid] = min(first.get(s.iid, s.frame), s.frame)
    return [s for s in seeds if s.frame == first[s.iid]]


def needs_correction(tracked: np.ndarray, det_masks: Sequence[np.ndarray], coverage: float) -> bool:
    """True when some detection's mask is covered by the tracked mask by less than ``coverage``
    (a new object appeared, or tracking lost one)."""
    for m in det_masks:
        a = int(m.sum())
        if a and (tracked & m).sum() < coverage * a:
            return True
    return False


def _sweep(predictor, store, seeds: List[_Seed], chunks, overlap: int, reverse: bool, result: TrackResult,
           owner: Dict[int, int], progress: Optional[Progress], work: List[int], total_work: int,
           init_kwargs: dict, det_masks: Optional[Dict[int, List[np.ndarray]]] = None, coverage: float = 0.5) -> None:
    """One sweep over the chunks.  ``det_masks`` (frame → per-detection masks, reference size)
    drive the single UNION_IID object: seeded on the first detected frame (or the carried
    masks), and whenever a detection is not yet covered by the tracked mask, the propagation is
    stopped, ``tracked ∪ detections`` is added as a mask prompt on that frame and propagation
    resumes from there (SAM 2 supports new prompts for existing objects after tracking)."""
    det_masks = det_masks or {}
    order = list(reversed(chunks)) if reverse else list(chunks)
    pick = max if reverse else min
    carry: Dict[int, Dict[int, np.ndarray]] = {}
    for ci, (s, e) in enumerate(order):
        is_last = ci == len(order) - 1
        user = [sd for sd in seeds if s <= sd.frame < e]
        carried = {iid: {f: m for f, m in fm.items() if s <= f < e} for iid, fm in carry.items()}
        carried = {iid: fm for iid, fm in carried.items() if fm}
        det_frames = sorted((f for f in det_masks if s <= f < e), reverse=reverse)
        if det_frames and UNION_IID not in carried:
            f0 = det_frames[0]
            user.append(_Seed(UNION_IID, f0, mask=np.logical_or.reduce(det_masks[f0])))
        if not user and not carried:
            carry = {}
            work[0] += e - s
            continue
        user_keys = {(sd.iid, sd.frame) for sd in user}
        start_of: Dict[int, int] = {}
        for sd in user:
            start_of[sd.iid] = pick(start_of.get(sd.iid, sd.frame), sd.frame)
        for iid, fm in carried.items():
            for f in fm:
                start_of[iid] = pick(start_of.get(iid, f), f)
        state = predictor.init_state(video_path=str(store.chunk_dir(s, e)), **init_kwargs)
        try:
            with _no_grad():
                for iid, fm in carried.items():
                    for f, m in sorted(fm.items()):
                        if (iid, f) not in user_keys:
                            predictor.add_new_mask(inference_state=state, frame_idx=f - s, obj_id=iid, mask=m)
                for sd in sorted(user, key=lambda x: x.frame):
                    if sd.mask is not None:
                        predictor.add_new_mask(inference_state=state, frame_idx=sd.frame - s, obj_id=sd.iid, mask=sd.mask)
                        continue
                    kw = {}
                    if sd.points is not None and len(sd.points):
                        kw.update(points=sd.points, labels=sd.labels)
                    if sd.box is not None:
                        kw["box"] = sd.box
                    predictor.add_new_points_or_box(inference_state=state, frame_idx=sd.frame - s, obj_id=sd.iid, **kw)
                start = pick(start_of.values())
                new_carry: Dict[int, Dict[int, np.ndarray]] = {}
                prompted = {f for (i, f) in user_keys if i == UNION_IID} | set(carried.get(UNION_IID, {}))
                counted = set()
                while True:
                    restart = None
                    for fidx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start - s, reverse=reverse):
                        g = s + int(fidx)
                        if len(obj_ids) == 0:
                            continue
                        ms = (logits > 0.0)
                        ms = ms.cpu().numpy() if hasattr(ms, "cpu") else np.asarray(ms)
                        ms = ms.reshape(len(obj_ids), *ms.shape[-2:])
                        in_carry_zone = (g < s + overlap) if reverse else (g >= e - overlap)
                        for iid, m in zip(obj_ids, ms):
                            iid = int(iid)
                            st = start_of.get(iid)
                            if st is None or (g > st if reverse else g < st):
                                continue  # before this object's seed in the sweep direction
                            if iid == UNION_IID and g in det_masks and g not in prompted and \
                                    needs_correction(m, det_masks[g], coverage):
                                restart = (g, m | np.logical_or.reduce(det_masks[g]))
                            if m.any():
                                result.masks.union(g, m)
                                fl = result.frames_per_object.setdefault(owner.get(iid, iid), [])
                                if not fl or fl[-1] != g:
                                    fl.append(g)
                                if in_carry_zone and not is_last:
                                    new_carry.setdefault(iid, {})[g] = m.copy()
                            elif in_carry_zone:
                                new_carry.get(iid, {}).pop(g, None)
                        if g not in counted:
                            counted.add(g)
                            result.frames_processed += 1
                            work[0] += 1
                        if progress:
                            progress(min(1.0, work[0] / max(1, total_work)),
                                     f"SAM 2 {'backward' if reverse else 'forward'}: frame {g + 1}/{len(store)}")
                        if restart is not None:
                            break
                    if restart is None:
                        break
                    g, m = restart
                    predictor.add_new_mask(inference_state=state, frame_idx=g - s, obj_id=UNION_IID, mask=m)
                    prompted.add(g)
                    result.corrections += 1
                    start = g
                carry = new_carry
        finally:
            try:
                predictor.reset_state(state)
            except Exception:
                pass
            del state
            gc.collect()
            from ..device import empty_cache

            empty_cache()
            store.release_chunk(s, e)


def track_sequence(
    predictor,
    store: FrameStore,
    prompts: Sequence[FramePrompt],
    chunk: int = 64,
    overlap: int = 2,
    bidirectional: bool = True,
    progress: Optional[Progress] = None,
    init_kwargs: Optional[dict] = None,
    det_masks: Optional[Dict[int, List[np.ndarray]]] = None,
    coverage: float = 0.5,
) -> TrackResult:
    """Track ``prompts`` (one SAM 2 object per objectId/box) and, if given, the detections
    (``det_masks``: frame → list of reference-size masks, tracked as ONE union object so the
    cost does not grow with the number of detected instances).  Detection masks are also
    OR-ed into the result on their own frames."""
    n = len(store)
    chunks = plan_chunks(n, chunk, overlap)
    overlap = 0 if len(chunks) < 2 else chunks[0][1] - chunks[1][0]
    result = TrackResult(PackedMasks(n, store.ref_size), chunks=chunks)
    seeds, owner = expand_seeds(prompts, store.scale_of)
    det_masks = {f: [m for m in ms if m.any()] for f, ms in (det_masks or {}).items()}
    det_masks = {f: ms for f, ms in det_masks.items() if ms}
    owner[UNION_IID] = 0
    for f, ms in det_masks.items():
        result.masks.union(f, np.logical_or.reduce(ms))
    if not seeds and not det_masks:
        return result
    back = _first_seeds(seeds) if bidirectional else []
    # the backward sweep matters whenever something is first prompted / detected after frame 0
    first_frames = [s.frame for s in back] + ([max(det_masks)] if (bidirectional and det_masks) else [])
    need_back = bool(first_frames) and max(first_frames) > 0
    total = n * (2 if need_back else 1)
    work = [0]
    _sweep(predictor, store, seeds, chunks, overlap, False, result, owner, progress, work, total, init_kwargs or {},
           det_masks, coverage)
    result.sweeps = 1
    if need_back:
        _sweep(predictor, store, back, chunks, overlap, True, result, owner, progress, work, total, init_kwargs or {},
               det_masks if bidirectional else None, coverage)
        result.sweeps = 2
    for oid in result.frames_per_object:
        result.frames_per_object[oid] = sorted(set(result.frames_per_object[oid]))
    return result


def detection_masks(img_pred, store: FrameStore, boxes_per_frame: Dict[int, List[List[float]]],
                    progress: Optional[Progress] = None) -> Dict[int, List[np.ndarray]]:
    """SAM 2 image predictor on the staged (reference-size) frames: one mask per detection box."""
    from PIL import Image

    out: Dict[int, List[np.ndarray]] = {}
    frames = sorted(f for f, b in boxes_per_frame.items() if b)
    for k, f in enumerate(frames):
        rgb = np.asarray(Image.open(store.root / f"{f:06d}.jpg").convert("RGB"))
        sx, sy = store.scale_of(f)
        bx = np.asarray([[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes_per_frame[f]], np.float32)
        with _no_grad():
            img_pred.set_image(rgb)
            masks, _, _ = img_pred.predict(box=bx if len(bx) > 1 else bx[0], multimask_output=False)
        masks = np.asarray(masks).reshape(len(bx), -1, rgb.shape[0], rgb.shape[1])[:, 0] > 0
        out[f] = [m for m in masks]
        if progress:
            progress((k + 1) / max(1, len(frames)), f"SAM 2 detection masks: {k + 1}/{len(frames)}")
    return out


# ---------------------------------------------------------------------------
# single image (unordered sets, maskpreview, maskserver)


def predict_image_prompt(img_pred, prompt: FramePrompt, multimask_single_point: bool = True):
    """Run SAM2ImagePredictor (image already set) for one object prompt (that image's pixels).

    Returns (bool mask HxW, score, low-res logits or None).  Several boxes → union; points go
    with the first box.  A single positive click without a box uses multimask output and keeps
    the best-scoring mask (SAM's recommended handling of ambiguous single clicks)."""
    pts = np.asarray(prompt.points, np.float32).reshape(-1, 2) if prompt.points else None
    labs = np.asarray(prompt.labels, np.int32) if prompt.points else None
    boxes = np.asarray(prompt.boxes, np.float32).reshape(-1, 4) if prompt.boxes else None
    union = None
    scores = []
    with _no_grad():
        if boxes is None:
            multi = multimask_single_point and pts is not None and len(pts) == 1
            masks, sc, _ = img_pred.predict(point_coords=pts, point_labels=labs, multimask_output=multi)
            k = int(np.argmax(sc)) if multi else 0
            union = np.asarray(masks[k]) > 0
            scores.append(float(np.asarray(sc).reshape(-1)[k]))
        else:
            masks, sc, _ = img_pred.predict(point_coords=pts, point_labels=labs, box=boxes[0], multimask_output=False)
            union = np.asarray(masks[0]) > 0
            scores.append(float(np.asarray(sc).reshape(-1)[0]))
            if len(boxes) > 1:
                masks, sc, _ = img_pred.predict(box=boxes[1:], multimask_output=False)
                masks = np.asarray(masks).reshape(len(boxes) - 1, -1, *union.shape)
                for m in masks:
                    union |= m[0] > 0
                scores += [float(v) for v in np.asarray(sc).reshape(-1)]
    return union, float(np.mean(scores)) if scores else 0.0


def segment_unordered(img_pred, store_paths: Sequence[Path], prompts: Sequence[FramePrompt], loader,
                      progress: Optional[Progress] = None) -> Dict[int, np.ndarray]:
    """Per-image SAM 2 (no temporal tracking): {frame: bool mask at original size} for prompted frames."""
    by_frame: Dict[int, List[FramePrompt]] = {}
    for p in prompts:
        by_frame.setdefault(p.frame, []).append(p)
    out = {}
    frames = sorted(by_frame)
    for k, f in enumerate(frames):
        rgb = loader(store_paths[f])
        with _no_grad():
            img_pred.set_image(rgb)
        m = np.zeros(rgb.shape[:2], bool)
        for p in by_frame[f]:
            mm, _ = predict_image_prompt(img_pred, p)
            m |= mm
        out[f] = m
        if progress:
            progress((k + 1) / max(1, len(frames)), f"SAM 2 image: {k + 1}/{len(frames)}")
    return out
