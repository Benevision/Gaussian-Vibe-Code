"""Old home video (VHS, Video8, Hi8, …) → clean training frames (``task: frames`` with ``vhs``).

Digitised tapes are not like phone video. They are usually a capture or a re-encode, often upscaled and
frame-rate converted, with tape artefacts baked in (see docs/dev/VHS.md for what real files contain):

* non-square pixels (PAL 720×576 at 16:15, a 4:3 picture stored as 1890×1068 at 712:945, …);
* repeated frames from frame-rate conversion (a 20 fps tape stored at 60 fps repeats every frame 3×);
* a few combed (interlaced) frames in an otherwise progressive file;
* black borders, a head-switching band at the bottom, tracking-dropout lines;
* blue / black "no signal" screens between recordings;
* many short shots cut together, each with its own colour cast, some zoomed, some only panning;
* burnt-in date / time stamps.

``analyse`` makes one fast pass over the file (decoded at a quarter of the width) and returns the clips with a
verdict each. ``prepare`` picks frames per clip for sharpness and camera movement, cleans them (deinterlace when
combed, crop, square pixels at the real resolution, motion-compensated temporal denoise, per-clip colour) and writes
``images/``, ``masks/`` (burnt-in overlays) and a ``frames.json`` FrameSet, like ``frames.extract_frames``. Each
frame carries its ``clip`` and a ``cameraGroup`` (one camera per clip, or one per frame when the clip zooms), which
the camera solve uses as shared intrinsics.

Request (``task: frames``): ``video``, ``outputDir``, ``videoOptions`` (``targetFrameCount`` caps the total,
``encoding``), and ``vhs``:

* ``analyseOnly`` — only write ``vhs.json`` (the clips) and thumbnails, no frames;
* ``clips`` — ids of the clips to use (default: every clip with verdict ``good``);
* ``maxHeight`` (576) — output height; SD tape holds no more vertical detail than PAL's 576 lines;
* ``denoise`` (true), ``colour`` (true), ``deinterlace`` (true), ``repairDropouts`` (true), ``maskOverlays`` (true);
* ``minClipSeconds`` (4), ``framesPerClip`` (max frames per clip, 150).
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import SplatPrepError

ProgressFn = Callable[[float, str], None]

VHS_DEFAULTS: Dict[str, Any] = {
    "analyseOnly": False, "clips": None, "maxHeight": 576, "denoise": True, "colour": True, "deinterlace": True,
    "maskOverlays": True, "repairDropouts": True, "minClipSeconds": 4.0, "framesPerClip": 150, "totalFrames": 400,
}
THUMB_W = 160            # analysis thumbnail width (square pixels)
DUP_THRESHOLD = 0.6      # mean abs difference (0…255) below which a frame repeats the previous one
NOSIGNAL_STD = 6.0       # a uniform frame (blue / black screen)
COMB_RATIO = 0.7         # adjacent-line / same-field-line difference above which a frame is combed or tape-damaged
                         # (progressive pictures measure 0.5–0.6; a smeared dropout frame on a test tape 0.79)
DROPOUT_LEVEL = 0.9      # a thin bright / dark streak along ≥ 90 % of 31 quarter-width pixels (tracking dropout)
CUT_MIN = 14.0           # mean abs thumbnail difference that can be a cut …
CUT_FACTOR = 4.0         # … when it is also this many times the recent typical difference
JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# Per-frame measurements (pure numpy / OpenCV; unit tested)
# ---------------------------------------------------------------------------

def comb_score(gray: np.ndarray) -> float:
    """Interlacing (combing) score of a full-height gray frame: mean |line − next line| over mean |line − line+2|.

    Progressive pictures are smooth vertically, so neighbouring lines differ less than lines two apart (< 1). In a
    combed frame neighbouring lines come from different instants, so moving edges zig-zag and the ratio rises above 1.
    Only rows with some vertical detail count, so flat sky doesn't dilute it.
    """
    g = gray.astype(np.float32)
    if g.shape[0] < 6:
        return 0.0
    d1 = np.abs(g[1:-1] - g[2:])          # line i vs i+1
    d2 = np.abs(g[:-2] - g[2:])           # line i-1 vs i+1 (same field)
    m = d2 + d1 > 8
    if m.sum() < 64:
        return 0.0
    return float(d1[m].mean() / max(1e-3, d2[m].mean()))


def dropout_score(gray: np.ndarray, run: int = 31) -> float:
    """Tracking-dropout lines: the longest horizontal run of pixels (as a fraction of ``run``, capped at 1) that
    are much brighter or darker than the lines two above *and* two below. A dropout is a thin streak across part
    of one line; an edge (horizon, roof) differs from only one side, texture never lines up over ``run`` pixels."""
    import cv2

    g = gray.astype(np.int16)
    if g.shape[0] < 5 or g.shape[1] < run:
        return 0.0
    a = g[2:-2] - g[:-4]
    b = g[2:-2] - g[4:]
    m = (((a > 30) & (b > 30)) | ((a < -30) & (b < -30))).astype(np.float32)
    k = cv2.blur(m, (run, 1), borderType=cv2.BORDER_CONSTANT)
    return float(k.max())


def dropout_mask(gray: np.ndarray, run: int = 41, level: int = 30) -> np.ndarray:
    """Pixels of tracking-dropout streaks in a full-resolution gray frame (uint8, 255 = streak): runs along a line
    that are much brighter or darker than the lines two above and two below, at least half of ``run`` pixels
    long, grown a little so the soft edge of the streak goes too."""
    import cv2

    g = gray.astype(np.int16)
    h, w = g.shape
    out = np.zeros((h, w), np.uint8)
    if h < 5 or w < run:
        return out
    a = g[2:-2] - g[:-4]
    b = g[2:-2] - g[4:]
    m = (((a > level) & (b > level)) | ((a < -level) & (b < -level))).astype(np.float32)
    dense = cv2.blur(m, (run, 1), borderType=cv2.BORDER_CONSTANT) > 0.5
    streak = (m > 0) & cv2.dilate(dense.astype(np.uint8), np.ones((1, run), np.uint8)).astype(bool)
    out[2:-2][streak] = 255
    # a tape line is ~2 rows of an upscaled frame and its glow spreads a little further: ±0.4 % of the height
    grow = max(1, int(round(h * 0.004)))
    return cv2.dilate(out, np.ones((2 * grow + 1, 7), np.uint8))


def repair_dropouts(rgb: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fill tracking-dropout streaks from the picture around them (they are only a line or two high)."""
    import cv2

    m = dropout_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    n = int(np.count_nonzero(m))
    if n == 0:
        return rgb, 0
    return cv2.inpaint(rgb, m, 5, cv2.INPAINT_TELEA), n


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (same as frames.sharpness), on the central 80 % so borders don't count."""
    h, w = gray.shape
    g = gray[h // 10: h - h // 10, w // 10: w - w // 10].astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def detect_borders(col_mean: np.ndarray, row_mean: np.ndarray, row_unstable: Optional[np.ndarray] = None) -> Dict[str, int]:
    """Black side borders and the head-switching band from time-averaged column / row means.

    Returns pixels to crop from each side (in the units of the profiles). Side borders: columns much darker than
    the picture (at most 8 % per side). Bottom: the head-switching band, where rows go dark, bright or unstable in
    the last 7 % of the height. Top: the same, at most 3 %.
    """
    def dark_edge(p: np.ndarray, limit: float) -> int:
        n = p.size
        ref = float(np.median(p[n // 4: 3 * n // 4]))
        k = 0
        lim = int(n * limit)
        while k < lim and p[k] < 0.6 * ref:
            k += 1
        return k

    def band_edge(p: np.ndarray, unstable: Optional[np.ndarray], limit: float) -> int:
        """Rows from the end of ``p`` that belong to a band: deviating > 8 % from the picture or unstable."""
        n = p.size
        lim = max(1, int(n * limit))
        ref_rows = p[n - 4 * lim: n - lim] if n > 4 * lim else p[: n // 2]
        ref = float(np.median(ref_rows))
        spread = float(np.median(np.abs(np.diff(ref_rows)))) + 1e-3
        bad = np.abs(p[n - lim:] - ref) > max(0.08 * ref, 6 * spread)
        if unstable is not None:
            u = unstable[n - lim:]
            bad |= u > 3 * float(np.median(unstable[n // 4: 3 * n // 4]) + 1e-3)
        idx = np.nonzero(bad)[0]
        if idx.size == 0:
            return 0
        return lim - int(idx.min())          # everything from the first bad row to the bottom

    left = dark_edge(col_mean, 0.08)
    right = dark_edge(col_mean[::-1], 0.08)
    bottom = band_edge(row_mean, row_unstable, 0.07)
    top = dark_edge(row_mean, 0.03)
    # a safety margin: the edge of a border is soft
    pad_x = max(1, int(round(col_mean.size * 0.004)))
    pad_y = max(1, int(round(row_mean.size * 0.004)))
    return {"left": left + (pad_x if left else 0), "right": right + (pad_x if right else 0),
            "top": top + (pad_y if top else 0), "bottom": bottom + (pad_y if bottom else 0)}


def is_cut(d: float, recent: Sequence[float]) -> bool:
    """Scene cut: a jump in the thumbnail difference well above the recent typical frame-to-frame change."""
    typical = float(np.median(recent)) if len(recent) else 0.0
    return d > CUT_MIN and d > CUT_FACTOR * max(typical, 2.0)


def colour_transform(means: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> Dict[str, Any]:
    """Per-clip white balance and levels from per-frame RGB means and luma 1 % / 99.5 % levels.

    Gray-world gains (clamped to ±20 %) take out the clip's colour cast; a gentle levels stretch (at most 1.15×)
    uses the clip's median black and white points. One transform for the whole clip, so frames stay consistent
    with each other (per-image drift is the trainer's appearance correction's job).
    """
    m = np.median(means, axis=0).astype(np.float64)
    gray = float(m.mean())
    gains = np.clip(gray / np.maximum(m, 1.0), 0.8, 1.2)
    lo = float(np.median(lows))
    hi = float(np.median(highs))
    target_lo, target_hi = 8.0, 245.0
    scale = (target_hi - target_lo) / max(hi - lo, 1.0)
    scale = float(np.clip(scale, 1.0, 1.15))
    return {"gains": [float(g) for g in gains], "black": lo, "scale": scale, "offset": target_lo}


def apply_colour(rgb: np.ndarray, ct: Dict[str, Any]) -> np.ndarray:
    x = rgb.astype(np.float32) * np.asarray(ct["gains"], np.float32)
    if ct["scale"] > 1.0:
        x = (x - ct["black"]) * ct["scale"] + ct["offset"]
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def overlay_mask(mean: np.ndarray, std: np.ndarray, grad: np.ndarray) -> Optional[np.ndarray]:
    """Burnt-in overlays (date / time stamps, camcorder icons) in one clip: pixels that keep their value while the
    picture moves and carry sharp edges. Inputs are per-pixel temporal mean, std and mean gradient magnitude over
    the clip's thumbnails. Returns a keep mask (255 = keep, 0 = overlay) or None when nothing was found."""
    import cv2

    if float(np.median(std)) < 12.0:          # the picture itself hardly changes: can't tell overlay from scene
        return None
    cand = ((std < 6.0) & (grad > 25.0)).astype(np.uint8)
    h, w = cand.shape
    my, mx = max(1, h // 25), max(1, w // 25)
    cand[:my] = cand[-my:] = 0
    cand[:, :mx] = cand[:, -mx:] = 0
    cand = cv2.dilate(cand, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.full((h, w), 255, np.uint8)
    found = False
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0004 * h * w or area > 0.06 * h * w:
            continue
        # text: a cluster of edges, not a long thin line (a horizon that stays put in a static stretch)
        if bw > 0.4 * w or bh > 0.25 * h:
            continue
        # an overlay stays put while the picture right around it changes; static scenery doesn't
        r = max(3, bh)
        ring = std[max(0, y - r): y + bh + r, max(0, x - r): x + bw + r].copy()
        ring[min(r, y): min(r, y) + bh, min(r, x): min(r, x) + bw] = np.nan
        if not np.isfinite(ring).any() or float(np.nanmedian(ring)) < 20.0:
            continue
        pad = max(2, int(0.25 * bh))
        keep[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad] = 0
        found = True
    return keep if found else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
    index: int                 # decode index
    t: float
    unique: bool = True        # not a repeat of the previous frame
    nosignal: bool = False
    comb: float = 0.0
    dropout: float = 0.0
    sharp: float = 0.0
    motion: float = 0.0        # median feature motion to the previous unique frame, fraction of the width
    zoom: float = 0.0          # log scale change to the previous unique frame
    parallax: float = -1.0     # % of tracks over ~0.5 s that fit the epipolar geometry but no homography (−1 = unknown)
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lo: float = 0.0
    hi: float = 255.0
    clip: int = -1


@dataclass
class _Segment:
    start: int                 # index into frames
    thumb_sum: Optional[np.ndarray] = None
    thumb_sq: Optional[np.ndarray] = None
    grad_sum: Optional[np.ndarray] = None
    n: int = 0
    thumb: Optional[np.ndarray] = None   # a representative RGB thumbnail
    thumb_score: float = -1.0


class _Tracker:
    """KLT feature tracks over unique thumbnails → motion, zoom and parallax per frame."""

    WINDOW = 10   # unique frames between the two views used for parallax

    def __init__(self):
        self.prev: Optional[np.ndarray] = None
        self.tracks: List[List[Tuple[float, float]]] = []

    def reset(self):
        self.prev = None
        self.tracks = []

    def step(self, g: np.ndarray) -> Tuple[float, float, float]:
        """→ (motion since the last frame as a fraction of the width, log zoom over the window (0 when there is
        parallax: moving towards the scene also scales the picture), parallax over the window: the % of tracks that
        fit the two views' epipolar geometry but not a homography — near 0 when the camera only turns or zooms)."""
        import cv2

        h, w = g.shape
        motion = zoom = 0.0
        parallax = -1.0
        if self.prev is not None and self.tracks:
            p0 = np.float32([t[-1] for t in self.tracks]).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(15, 15), maxLevel=3,
                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            ok = st.reshape(-1).astype(bool)
            p1 = p1.reshape(-1, 2)
            inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
            ok &= inside
            a = p0.reshape(-1, 2)[ok]
            b = p1[ok]
            self.tracks = [t + [tuple(p)] for t, p, k in zip(self.tracks, p1, ok) if k]
            for t in self.tracks:
                if len(t) > self.WINDOW + 1:
                    del t[0]
            if len(a) >= 8:
                motion = float(np.median(np.linalg.norm(b - a, axis=1)) / w)
            long = [t for t in self.tracks if len(t) == self.WINDOW + 1]
            if len(long) >= 20:
                a = np.float32([t[0] for t in long])
                b = np.float32([t[-1] for t in long])
                if np.median(np.linalg.norm(b - a, axis=1)) > 1.0:
                    tol = w / 320.0
                    H, _ = cv2.findHomography(a, b, cv2.RANSAC, 2.0 * tol)
                    F, fin = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0 * tol, 0.99)
                    if H is not None and F is not None and fin is not None and fin.sum() >= 12:
                        fin = fin.reshape(-1).astype(bool)
                        pa = cv2.perspectiveTransform(a[fin].reshape(-1, 1, 2), H).reshape(-1, 2)
                        res = np.linalg.norm(pa - b[fin], axis=1)
                        parallax = float(100.0 * np.mean(res > 2.5 * tol))
                        if parallax < 10.0:
                            zoom = float(0.5 * math.log(max(1e-6, abs(np.linalg.det(H[:2, :2])))))
        if len(self.tracks) < 80:
            mask = np.full(g.shape, 255, np.uint8)
            for t in self.tracks:
                cv2.circle(mask, (int(t[-1][0]), int(t[-1][1])), 5, 0, -1)
            pts = cv2.goodFeaturesToTrack(g, 150 - len(self.tracks), 0.01, 5, mask=mask)
            if pts is not None:
                self.tracks += [[(float(p[0]), float(p[1]))] for p in pts.reshape(-1, 2)]
        self.prev = g
        return motion, zoom, parallax


def _open(video: Path):
    try:
        import av
    except ImportError as e:
        raise SplatPrepError("old home video needs PyAV (pip install av)") from e
    c = av.open(str(video))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    s.codec_context.thread_count = 0
    return c, s


def _fingerprint(video: Path) -> str:
    st = video.stat()
    return f"{video.name}:{st.st_size}:{int(st.st_mtime)}"


def analyse(video: Path, opts: Dict[str, Any], progress: ProgressFn = lambda f, m: None,
            log: Callable[[str], None] = lambda m: None, thumbs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """One pass over the video → file properties, per-frame measurements and clips with verdicts."""
    import cv2
    from .frames import pixel_aspect

    o = {**VHS_DEFAULTS, **{k: v for k, v in (opts or {}).items() if v is not None}}
    container, stream = _open(video)
    cc = stream.codec_context
    W, H = cc.width, cc.height
    par = pixel_aspect(stream)
    dar = W * par / H
    mid_w = max(THUMB_W, W // 4)                       # quarter width, full height: enough for combing and borders
    tw, th = THUMB_W, max(16, int(round(THUMB_W / dar)))
    duration = float(stream.duration * stream.time_base) if stream.duration else (container.duration or 0) / 1e6
    fps = float(stream.average_rate or 0) or 25.0

    frames: List[_Frame] = []
    col_acc = np.zeros(mid_w, np.float64)
    row_acc = np.zeros(H, np.float64)
    row_unstable = np.zeros(H, np.float64)
    n_acc = 0
    n_sig = 0
    segments: List[_Segment] = [_Segment(start=0)]
    tracker = _Tracker()
    prev_thumb: Optional[np.ndarray] = None
    recent: List[float] = []
    last_unique_thumb: Optional[np.ndarray] = None
    import time as _time

    last = 0.0
    try:
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            mid = frame.reformat(width=mid_w, height=H, format="gray").to_ndarray()
            if mid.ndim == 3:
                mid = mid[..., 0]
            g = cv2.resize(mid, (tw, th), interpolation=cv2.INTER_AREA)
            small = frame.reformat(width=48, height=max(8, int(round(48 / dar))), format="rgb24").to_ndarray()
            f = _Frame(index=i, t=t)
            f.nosignal = float(g.std()) < NOSIGNAL_STD
            if prev_thumb is not None:
                d = float(np.abs(g.astype(np.int16) - prev_thumb).mean())
                f.unique = d >= DUP_THRESHOLD
            prev_thumb = g.astype(np.int16)
            frames.append(f)
            if not f.unique:
                continue
            # cut detection against the previous unique frame
            if last_unique_thumb is not None:
                d = float(np.abs(g.astype(np.int16) - last_unique_thumb).mean())
                cut = is_cut(d, recent[-15:]) or (f.nosignal != frames[segments[-1].start].nosignal)
                if cut:
                    segments.append(_Segment(start=len(frames) - 1))
                    tracker.reset()
                    recent.clear()
                else:
                    recent.append(d)
            last_unique_thumb = g.astype(np.int16)
            rows = mid.astype(np.float32).mean(axis=1)
            f.comb = comb_score(mid)
            f.dropout = dropout_score(mid[int(0.04 * H): int(0.93 * H)])   # not the head-switching band
            if f.nosignal:
                continue
            f.sharp = sharpness(mid)
            ys, xs = small.shape[0], small.shape[1]
            core = small[ys // 10: ys - ys // 6, xs // 10: xs - xs // 10].reshape(-1, 3).astype(np.float32)
            f.mean = tuple(float(v) for v in core.mean(0))
            luma = core @ np.float32([0.299, 0.587, 0.114])
            f.lo, f.hi = (float(v) for v in np.percentile(luma, [1, 99.5]))
            g2 = cv2.resize(mid, (2 * tw, 2 * th), interpolation=cv2.INTER_AREA)
            f.motion, f.zoom, f.parallax = tracker.step(g2)
            n_sig += 1
            if n_sig % 4 == 1:          # every 4th picture frame (counting repeats would alias with 2× / 4× cadences)
                col_acc += mid.mean(axis=0)
                row_acc += rows
                # head-switching rows jitter sideways: large row-to-row horizontal change compared with the picture
                row_unstable += np.abs(np.diff(mid.astype(np.float32), axis=1)).mean(axis=1)
                n_acc += 1
            seg = segments[-1]
            gf = g2.astype(np.float32)         # overlays: date-stamp strokes are thin, so the larger thumbnail
            gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
            if seg.thumb_sum is None:
                seg.thumb_sum = np.zeros_like(gf)
                seg.thumb_sq = np.zeros_like(gf)
                seg.grad_sum = np.zeros_like(gf)
            seg.thumb_sum += gf
            seg.thumb_sq += gf * gf
            seg.grad_sum += np.sqrt(gx * gx + gy * gy)
            seg.n += 1
            if f.sharp > seg.thumb_score:
                seg.thumb_score = f.sharp
                seg.thumb = frame.reformat(width=320, height=int(round(320 / dar)), format="rgb24").to_ndarray()
            now = _time.monotonic()
            if now - last > 0.5:
                last = now
                progress(min(0.999, t / max(duration, 1e-3)), f"analysing: {t:.0f} of {duration:.0f} s, {len(segments)} clips so far")
    finally:
        container.close()
    if not frames:
        raise SplatPrepError(f"no frames decoded from {video.name}")

    # borders and the head-switching band (in stored pixels)
    if n_acc:
        b = detect_borders(col_acc / n_acc, row_acc / n_acc, row_unstable / n_acc)
        sx = W / float(mid_w)
        crop = {"left": int(math.ceil(b["left"] * sx)), "right": int(math.ceil(b["right"] * sx)), "top": b["top"], "bottom": b["bottom"]}
    else:
        crop = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    uniq = [f for f in frames if f.unique]
    span = (frames[-1].t - frames[0].t) or 1e-3
    unique_fps = len(uniq) / span
    repeat = len(frames) / max(1, len(uniq))
    combed = sum(1 for f in uniq if f.comb > COMB_RATIO)
    clips = []
    all_sharp = np.median([f.sharp for f in uniq if not f.nosignal] or [1.0])
    merged: List[_Segment] = []
    for k, seg in enumerate(segments):
        end_t = frames[segments[k + 1].start].t if k + 1 < len(segments) else frames[-1].t
        if merged and (end_t - frames[seg.start].t < 1.0) and (frames[seg.start].t - frames[merged[-1].start].t < 1.0):
            continue                                  # glitch burst: fold into the previous short segment
        merged.append(seg)
    segments = merged
    bounds = [s.start for s in segments] + [len(frames)]
    for k, seg in enumerate(segments):
        fr = frames[bounds[k]: bounds[k + 1]]
        for f in fr:
            f.clip = k
        u = [f for f in fr if f.unique]
        sig = [f for f in u if not f.nosignal]
        t0, t1 = fr[0].t, fr[-1].t + 1.0 / max(fps, 1.0)
        c: Dict[str, Any] = {"id": k, "start": round(t0, 3), "end": round(t1, 3), "duration": round(t1 - t0, 3),
                             "frames": len(fr), "uniqueFrames": len(u)}
        if not sig or len(sig) < 0.5 * max(1, len(u)):
            c.update(verdict="noSignal", reason="no picture (blue or black screen)")
            clips.append(c)
            continue
        sharp = float(np.median([f.sharp for f in sig]))
        motion = float(np.sum([f.motion for f in sig]))
        par_v = [f.parallax for f in sig if f.parallax >= 0]
        parallax = float(np.median(par_v)) if par_v else 0.0
        # zoom travelled: window scale changes (only where they are not parallax), each window counted once
        zoom_range = float(sum(abs(f.zoom) for f in sig if abs(f.zoom) > 0.03) / _Tracker.WINDOW)
        combed_c = sum(1 for f in sig if f.comb > COMB_RATIO)
        c.update(sharpness=round(sharp, 2), relativeSharpness=round(sharp / max(all_sharp, 1e-3), 2),
                 cameraTravel=round(motion, 3), parallax=round(parallax, 3), zoomRange=round(zoom_range, 3),
                 zooms=bool(zoom_range > 0.25), combedFrames=combed_c)
        if c["duration"] < float(o["minClipSeconds"]) or len(sig) < 20:
            c.update(verdict="short", reason=f"too short ({c['duration']:.1f} s)")
        elif motion < 0.3:
            c.update(verdict="static", reason="the camera hardly moves")
        elif parallax < 10.0:
            c.update(verdict="rotationOnly", reason="the camera only turns: no depth to recover")
        elif sharp < 0.45 * all_sharp:
            c.update(verdict="blurry", reason="mostly blurred")
        else:
            c.update(verdict="good", reason="the camera moves through the scene")
        c["score"] = round(min(parallax, 40.0) * min(c["duration"], 60.0) * min(sharp / max(all_sharp, 1e-3), 1.5), 2)
        # burnt-in overlays
        if o["maskOverlays"] and seg.n >= 20 and motion >= 0.5:
            mean = seg.thumb_sum / seg.n
            std = np.sqrt(np.maximum(seg.thumb_sq / seg.n - mean * mean, 0.0))
            m = overlay_mask(mean, std, seg.grad_sum / seg.n)
            if m is not None:
                c["overlay"] = True
                if thumbs_dir is not None:
                    thumbs_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(thumbs_dir / f"clip{k:03d}_overlay.png"), m)
                seg_mask = m
                c["_overlayMask"] = seg_mask
        if thumbs_dir is not None and seg.thumb is not None:
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            p = thumbs_dir / f"clip{k:03d}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(seg.thumb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            c["thumbnail"] = f"{thumbs_dir.name}/{p.name}"
        clips.append(c)

    out_h = min(H - crop["top"] - crop["bottom"], int(o["maxHeight"]))
    info = {
        "video": video.name, "fingerprint": _fingerprint(video), "width": W, "height": H, "pixelAspect": round(par, 5),
        "displayAspect": round(dar, 4), "fps": round(fps, 3), "uniqueFps": round(unique_fps, 3),
        "repeatFactor": round(repeat, 2), "duration": round(duration, 3), "framesDecoded": len(frames),
        "combedFrames": combed, "noSignalFrames": sum(1 for f in uniq if f.nosignal),
        "dropoutFrames": sum(1 for f in uniq if f.dropout > DROPOUT_LEVEL), "crop": crop,
        "outputHeight": out_h, "clips": clips,
    }
    progress(1.0, f"{len(clips)} clips, {sum(1 for c in clips if c.get('verdict') == 'good')} usable")
    return {"info": info, "frames": frames}


def summary_json(an: Dict[str, Any]) -> Dict[str, Any]:
    """The analysis without per-frame data / masks (what vhs.json holds and the apps show)."""
    info = dict(an["info"])
    info["clips"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in info["clips"]]
    return info


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(frames: Sequence[_Frame], step: float, max_gap: float = 1.5, min_gap: float = 0.12,
                  edge: float = 0.25) -> List[int]:
    """Pick frames of one clip: a new pick each time the camera has moved ``step`` (fraction of the width) or
    ``max_gap`` seconds have passed with some movement; within each stretch the sharpest frame wins. Repeats,
    no-signal frames are never picked; combed frames only when nothing else is there, and frames with a
    (repaired) dropout line less readily. Returns
    positions in ``frames``."""
    cand = [i for i, f in enumerate(frames) if f.unique and not f.nosignal]
    if not cand:
        return []
    t0, t1 = frames[cand[0]].t + edge, frames[cand[-1]].t - edge
    cand = [i for i in cand if t0 <= frames[i].t <= t1] or cand
    picks: List[int] = []
    k = 0
    while k < len(cand):
        acc = 0.0
        j = k
        start_t = frames[cand[k]].t
        while j + 1 < len(cand):
            acc += frames[cand[j + 1]].motion
            j += 1
            if acc >= step or (frames[cand[j]].t - start_t >= max_gap and acc >= 0.15 * step):
                break
        stretch = cand[k: j + 1]
        if j + 1 >= len(cand) and acc < 0.5 * step and picks:
            break                                   # the tail moved too little to add a view

        def q(i: int) -> float:
            f = frames[i]
            return f.sharp * (0.4 if f.comb > COMB_RATIO else 1.0) * (0.8 if f.dropout > DROPOUT_LEVEL else 1.0)

        best = max(stretch, key=q)
        if not picks or frames[best].t - frames[picks[-1]].t >= min_gap:
            picks.append(best)
        k = j + 1
    return picks


def plan_selection(frames: Sequence[_Frame], max_frames: int) -> List[int]:
    """``select_frames`` with the movement step adapted so the clip yields at most ``max_frames``."""
    lo, hi = 0.01, 0.6
    best = select_frames(frames, lo)
    if len(best) <= max_frames:
        return best
    best = select_frames(frames, hi)
    if len(best) > max_frames:            # long clip, little movement: the time gap alone gives too many; thin out
        keep = np.linspace(0, len(best) - 1, max_frames).round().astype(int)
        return [best[k] for k in sorted(set(keep.tolist()))]
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        p = select_frames(frames, mid)
        if len(p) > max_frames:
            lo = mid
        else:
            hi, best = mid, p
    return best


# ---------------------------------------------------------------------------
# Clean-up
# ---------------------------------------------------------------------------

def deinterlace_field(img: np.ndarray) -> np.ndarray:
    """Rebuild a combed frame from its top field (even lines), interpolating the odd lines."""
    out = img.copy()
    h = img.shape[0]
    even = img[0::2].astype(np.float32)
    nxt = np.concatenate([even[1:], even[-1:]], axis=0)
    interp = 0.5 * (even + nxt)
    out[1::2] = np.clip(interp[: len(range(1, h, 2))] + 0.5, 0, 255).astype(img.dtype)
    return out


def geometry(info: Dict[str, Any]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """(crop x0, y0, x1, y1 in stored pixels), (output width, height) in square pixels."""
    W, H, c = info["width"], info["height"], info["crop"]
    x0, x1 = c["left"], W - c["right"]
    y0, y1 = c["top"], H - c["bottom"]
    out_h = int(info["outputHeight"])
    scale = out_h / float(y1 - y0)
    out_w = int(round((x1 - x0) * info["pixelAspect"] * scale))
    return (x0, y0, x1, y1), (out_w - out_w % 2, out_h - out_h % 2)


def clean_geometry(rgb: np.ndarray, box: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    import cv2

    x0, y0, x1, y1 = box
    c = rgb[y0:y1, x0:x1]
    w, h = size
    interp = cv2.INTER_AREA if (w < c.shape[1] or h < c.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(c, (w, h), interpolation=interp)


def temporal_denoise(center: np.ndarray, neighbours: Sequence[np.ndarray], sigma: float = 10.0) -> Tuple[np.ndarray, int]:
    """Motion-compensated temporal average: each neighbour is aligned to the centre frame with a homography from
    tracked features, then blended per pixel with a weight that falls off where it still differs (people moving,
    parallax), so tape noise averages out while nothing ghosts. Returns (image, neighbours used)."""
    import cv2

    cg = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    h, w = cg.shape
    s = 0.5 if w > 400 else 1.0
    cs = cv2.resize(cg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else cg
    pts = cv2.goodFeaturesToTrack(cs, 400, 0.01, 6)
    acc = center.astype(np.float32)
    wsum = np.ones((h, w), np.float32)
    used = 0
    if pts is None or len(pts) < 20:
        return center, 0
    blur_c = cv2.GaussianBlur(center.astype(np.float32), (5, 5), 0)
    for nb in neighbours:
        ng = cv2.cvtColor(nb, cv2.COLOR_RGB2GRAY)
        ns = cv2.resize(ng, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else ng
        p1, st, _ = cv2.calcOpticalFlowPyrLK(cs, ns, pts, None, winSize=(21, 21), maxLevel=4)
        ok = st.reshape(-1).astype(bool)
        if ok.sum() < 20:
            continue
        a = pts.reshape(-1, 2)[ok] / s
        b = p1.reshape(-1, 2)[ok] / s
        Hm, inl = cv2.findHomography(b, a, cv2.RANSAC, 2.0)       # neighbour → centre
        if Hm is None or inl is None or inl.sum() < 15:
            continue
        warped = cv2.warpPerspective(nb, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.ones((h, w), np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST) > 0
        wf = warped.astype(np.float32)
        d = np.abs(cv2.GaussianBlur(wf, (5, 5), 0) - blur_c).mean(axis=2)
        wt = np.exp(-(d / sigma) ** 2) * valid
        acc += wf * wt[..., None]
        wsum += wt
        used += 1
    return np.clip(acc / wsum[..., None] + 0.5, 0, 255).astype(np.uint8), used


# ---------------------------------------------------------------------------
# prepare: analysis → selection → clean frames → FrameSet
# ---------------------------------------------------------------------------

ANALYSIS_VERSION = 2
_FIELDS = ("index", "t", "unique", "nosignal", "comb", "dropout", "sharp", "motion", "zoom", "parallax", "lo", "hi", "clip")


def save_analysis(an: Dict[str, Any], path: Path) -> None:
    """Per-frame measurements + clips (with overlay masks) → one .npz, so finding clips and preparing frames decode
    the video once."""
    fr = an["frames"]
    arrays = {k: np.array([getattr(f, k) for f in fr]) for k in _FIELDS}
    arrays["mean"] = np.array([f.mean for f in fr], np.float32)
    for c in an["info"]["clips"]:
        if c.get("_overlayMask") is not None:
            arrays[f"overlay{c['id']}"] = c["_overlayMask"]
    info = summary_json(an)
    arrays["info"] = np.frombuffer(json.dumps({"version": ANALYSIS_VERSION, **info}).encode(), np.uint8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_analysis(path: Path, video: Path) -> Optional[Dict[str, Any]]:
    try:
        z = np.load(path)
        info = json.loads(bytes(z["info"]).decode())
    except Exception:
        return None
    if info.pop("version", None) != ANALYSIS_VERSION or info.get("fingerprint") != _fingerprint(video):
        return None
    n = len(z["t"])
    frames = []
    for i in range(n):
        f = _Frame(index=int(z["index"][i]), t=float(z["t"][i]))
        for k in _FIELDS[2:]:
            v = z[k][i]
            setattr(f, k, bool(v) if k in ("unique", "nosignal") else int(v) if k == "clip" else float(v))
        f.mean = tuple(float(x) for x in z["mean"][i])
        frames.append(f)
    for c in info["clips"]:
        if c.get("overlay") and f"overlay{c['id']}" in z.files:
            c["_overlayMask"] = z[f"overlay{c['id']}"]
    return {"info": info, "frames": frames}


def _load_or_analyse(video: Path, out_dir: Path, o: Dict[str, Any], progress: ProgressFn, log) -> Dict[str, Any]:
    cache = out_dir / "vhs-analysis.npz"
    if cache.exists():
        an = load_analysis(cache, video)
        if an is not None:
            an["info"]["outputHeight"] = min(an["info"]["height"] - an["info"]["crop"]["top"] - an["info"]["crop"]["bottom"],
                                             int(o["maxHeight"]))
            log("reusing the analysis of this video from the last run")
            progress(1.0, f"{len(an['info']['clips'])} clips")
            return an
    an = analyse(video, o, progress, log, thumbs_dir=out_dir / "vhs-thumbs")
    try:
        save_analysis(an, cache)
    except OSError as e:
        log(f"could not keep the analysis: {e}")
    return an


def camera_groups(frames: Sequence[_Frame], picks: Sequence[int], clip: int, max_zoom: float = 0.08) -> List[int]:
    """Camera group per pick: one per clip, plus a new one each time the lens has zoomed by more than ``max_zoom``
    (log focal length) since the group began, so zoomed stretches get their own focal length without leaving
    every frame with intrinsics of its own."""
    groups = []
    g = 0
    acc = 0.0
    prev = picks[0] if picks else 0
    for p in picks:
        acc += sum(frames[i].zoom for i in range(prev + 1, p + 1)) / _Tracker.WINDOW
        prev = p
        if abs(acc) > max_zoom:
            g += 1
            acc = 0.0
        groups.append(clip * 1000 + g)
    return groups


def prepare(video: Path, out_dir: Path, vo: Dict[str, Any], vhs_opts: Dict[str, Any], progress: ProgressFn,
            log: Callable[[str], None] = lambda m: None) -> Dict[str, Any]:
    import cv2

    o = {**VHS_DEFAULTS, **{k: v for k, v in (vhs_opts or {}).items() if v is not None}}
    vo = vo or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    an = _load_or_analyse(video, out_dir, o, (lambda f, m: progress(0.45 * f, m)), log)
    info = an["info"]
    frames: List[_Frame] = an["frames"]
    summary = summary_json(an)
    (out_dir / "vhs.json").write_text(json.dumps(summary, indent=1))
    clips = info["clips"]
    log(f"{info['width']}×{info['height']} at pixel aspect {info['pixelAspect']}, {info['uniqueFps']} unique frames/s "
        f"(each shown {info['repeatFactor']}×), {info['combedFrames']} combed, crop {info['crop']}, {len(clips)} clips")
    for c in clips:
        log(f"clip {c['id']}: {c['start']:.1f}–{c['end']:.1f} s {c['verdict']} ({c['reason']})"
            + (f", parallax {c.get('parallax')}‰, zoom {c.get('zoomRange')}" if "parallax" in c else ""))
    if o["analyseOnly"]:
        progress(1.0, f"{len(clips)} clips")
        return {"vhs": summary, "numFrames": 0, "numImages": 0}

    wanted = o["clips"]
    if wanted is None:
        use = [c["id"] for c in clips if c["verdict"] == "good"]
        if not use:
            ok = [c for c in clips if c["verdict"] in ("rotationOnly", "blurry") and c["duration"] >= o["minClipSeconds"]]
            use = [max(ok, key=lambda c: c.get("score", 0))["id"]] if ok else []
    else:
        use = [int(i) for i in wanted]
    use = [i for i in use if 0 <= i < len(clips) and clips[i]["verdict"] != "noSignal"]
    if not use:
        raise SplatPrepError("no usable clip in this video: " + "; ".join(f"clip {c['id']}: {c['reason']}" for c in clips))

    # how many frames per clip: share the total by camera travel, capped per clip
    total = int(vo.get("targetFrameCount") or o["totalFrames"])
    travel = {i: max(clips[i].get("cameraTravel", 0.0), 0.05) for i in use}
    tsum = sum(travel.values())
    by_clip: Dict[int, List[_Frame]] = {}
    for f in frames:
        if f.clip in travel:
            by_clip.setdefault(f.clip, []).append(f)
    selected: Dict[int, int] = {}       # decode index → clip
    cam_group: Dict[int, int] = {}      # decode index → camera group
    for i in use:
        cap = max(12, min(int(o["framesPerClip"]), int(round(total * travel[i] / tsum))))
        picks = plan_selection(by_clip.get(i, []), cap)
        for p, g in zip(picks, camera_groups(by_clip.get(i, []), picks, i)):
            selected[by_clip[i][p].index] = i
            cam_group[by_clip[i][p].index] = g
        clips[i]["selected"] = len(picks)
        clips[i]["cameras"] = len({cam_group[by_clip[i][p].index] for p in picks})
    if not selected:
        raise SplatPrepError("no frames could be picked from the chosen clips")
    log("frames picked: " + ", ".join(f"clip {i}: {clips[i]['selected']}" for i in use))

    # colour per clip
    colour: Dict[int, Dict[str, Any]] = {}
    if o["colour"]:
        for i in use:
            fs = [f for f in by_clip[i] if f.unique and not f.nosignal]
            colour[i] = colour_transform(np.array([f.mean for f in fs]), np.array([f.lo for f in fs]), np.array([f.hi for f in fs]))
            clips[i]["colour"] = {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in colour[i].items()}

    box, size = geometry(info)
    ext = "png" if vo.get("encoding") == "png" else "jpg"
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    overlay_files: Dict[int, Path] = {}
    for i in use:
        m = clips[i].get("_overlayMask")
        if m is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            p = masks_dir / f".overlay{i:03d}.png"
            # thumbnail mask → output geometry (the thumbnail covers the whole stored frame in display aspect)
            full = cv2.resize(m, (info["width"], info["height"]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(p), clean_geometry(full[..., None].repeat(3, 2), box, size)[..., 0])
            overlay_files[i] = p

    # neighbours for denoising: the two unique, clean frames either side in the same clip
    by_index = {f.index: f for f in frames}
    neighbours: Dict[int, List[int]] = {}
    need: set = set(selected)
    if o["denoise"]:
        for i in use:
            u = [f.index for f in by_clip[i] if f.unique and not f.nosignal and f.comb <= COMB_RATIO]
            pos = {ix: k for k, ix in enumerate(u)}
            for s in selected:
                if selected[s] != i or s not in pos:
                    continue
                k = pos[s]
                nb = u[max(0, k - 2): k] + u[k + 1: k + 3]
                neighbours[s] = nb
                need.update(nb)
    last_needed = max(need)

    container, stream = _open(video)
    cache: Dict[int, np.ndarray] = {}
    out_frames: List[Dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) - 2)))
    pending = []
    order = sorted(selected)
    names = {ix: f"{n:06d}.{ext}" for n, ix in enumerate(order)}
    done = 0
    denoised = 0
    deinterlaced = 0
    repaired = 0

    def finish(ix: int):
        nonlocal done, denoised, deinterlaced
        img = cache[ix]
        f = by_index[ix]
        nbs = [cache[n] for n in neighbours.get(ix, []) if n in cache]
        if nbs:
            img, used = temporal_denoise(img, nbs)
            denoised += used > 0
        clip = selected[ix]
        if clip in colour:
            img = apply_colour(img, colour[clip])
        name = names[ix]
        pending.append(pool.submit(_write, images / name, img, ext))
        fr = {"image": f"images/{name}", "width": size[0], "height": size[1], "sharpness": round(f.sharp, 3),
              "source": str(video), "timestamp": round(f.t, 4), "clip": clip,
              "cameraGroup": cam_group[ix]}
        if clip in overlay_files:
            dst = masks_dir / f"{Path(name).stem}.png"
            try:
                os.link(overlay_files[clip], dst)
            except OSError:
                import shutil

                shutil.copyfile(overlay_files[clip], dst)
            fr["mask"] = f"masks/{dst.name}"
        out_frames.append(fr)
        done += 1

    ready_at = {s: max([s] + neighbours.get(s, [])) for s in selected}
    users: Dict[int, set] = {}
    for s in selected:
        for n in [s] + neighbours.get(s, []):
            users.setdefault(n, set()).add(s)
    waiting = sorted(selected, key=lambda s: ready_at[s])
    try:
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                continue
            if i > last_needed:
                break
            if i in need:
                rgb = frame.to_ndarray(format="rgb24")
                f = by_index.get(i)
                if o["deinterlace"] and f is not None and f.comb > COMB_RATIO:
                    rgb = deinterlace_field(rgb)
                    deinterlaced += i in selected
                if o["repairDropouts"]:
                    rgb, n = repair_dropouts(rgb)
                    repaired += bool(n) and i in selected
                cache[i] = clean_geometry(rgb, box, size)
            while waiting and ready_at[waiting[0]] <= i:
                s = waiting.pop(0)
                finish(s)
                for n in [s] + neighbours.get(s, []):
                    users.get(n, set()).discard(s)
                    if not users.get(n):
                        cache.pop(n, None)
            if i % 50 == 0:
                progress(0.45 + 0.55 * min(1.0, i / max(1, last_needed)), f"cleaning frames: {done} of {len(selected)}")
            while len(pending) > 32:
                pending.pop(0).result()
        for s in waiting:
            if s in cache:
                finish(s)
        for p in pending:
            p.result()
    finally:
        pool.shutdown(wait=True)
        container.close()
    for p in overlay_files.values():
        p.unlink(missing_ok=True)
    out_frames.sort(key=lambda fr: fr["timestamp"])
    rejected = [{"item": f"clip {c['id']} ({c['start']:.1f}–{c['end']:.1f} s)",
                 "reason": c["reason"] if c["id"] not in use else ""} for c in clips if c["id"] not in use]
    fs = {"root": ".", "frames": out_frames, "rejected": rejected, "isMultiViewRig": False}
    (out_dir / "frames.json").write_text(json.dumps(fs, indent=1))
    summary = summary_json(an)
    summary["used"] = use
    (out_dir / "vhs.json").write_text(json.dumps(summary, indent=1))
    progress(1.0, f"{len(out_frames)} frames from {len(use)} clip(s)")
    return {"numFrames": len(out_frames), "numImages": len(out_frames), "numDecoded": len(frames),
            "numRejected": len(rejected), "width": size[0], "height": size[1], "clipsUsed": use,
            "denoised": denoised, "deinterlaced": deinterlaced, "dropoutsRepaired": repaired, "vhs": summary}


def _write(path: Path, rgb: np.ndarray, ext: str) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ext == "png":
        cv2.imwrite(str(path), bgr)
    else:
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
