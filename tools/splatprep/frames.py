"""Video → training frames (``task: frames``), a port of the Mac app's frame extraction.

Mirrors ``VideoFrameSelector`` / ``VideoFrameExtractor`` / ``Equirect360`` in the Mac app so a Linux
worker can split videos (including 8K 360° footage) and hand back exactly what the Mac would have
written: ``images/``, optional ``masks/`` and a ``frames.json`` FrameSet.

Request keys (besides ``task`` / ``outputDir``):

* ``video`` — path of the video, relative to the work dir.
* ``videoOptions`` — the Mac's ``VideoExtractionOptions`` (targetFPS, targetFrameCount, maxImageSize,
  pickSharpest, analysisStride, skipSceneCuts, sceneCutThreshold, skipStaticFrames, staticThreshold,
  startTime, endTime, toneMapHDR, encoding).
* ``equirect`` — the Mac's ``Equirect360Options`` for 360° footage; absent / null = ordinary video.

Decoding uses PyAV (bundled FFmpeg, frame-threaded); 360° views are resampled on the GPU with
``grid_sample`` when torch has one, else with OpenCV ``remap``.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .errors import SplatPrepError

ProgressFn = Callable[[float, str], None]

VIDEO_DEFAULTS: Dict[str, Any] = {
    "targetFPS": 2.0, "targetFrameCount": None, "maxImageSize": 1600, "pickSharpest": True,
    "analysisStride": 1, "skipSceneCuts": False, "sceneCutThreshold": 0.3, "skipStaticFrames": True,
    "staticThreshold": 0.01, "startTime": None, "endTime": None, "toneMapHDR": True, "encoding": "jpeg",
}
EQUIRECT_DEFAULTS: Dict[str, Any] = {
    "fovDegrees": 90.0, "outputSize": 1200,
    "rings": [{"pitch": 0, "count": 8}, {"pitch": 35, "count": 4, "yawOffset": 45}, {"pitch": -35, "count": 4, "yawOffset": 45}],
    "includeZenith": False, "includeNadir": False, "maskBelowLatitude": -60.0, "maskAboveLatitude": None, "yawOffset": 0.0,
}
JPEG_QUALITY = 92
THUMB_MAX_WIDTH = 480


# ---------------------------------------------------------------------------
# Metrics (same definitions as ImageMetrics on the Mac)
# ---------------------------------------------------------------------------

def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian, ignoring a 1-px border (0…255 gray)."""
    g = gray.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def thumb32(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    ys = np.minimum(h - 1, np.arange(32) * h // 32)
    xs = np.minimum(w - 1, np.arange(32) * w // 32)
    return gray[ys][:, xs].astype(np.int16)


def difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean() / 255.0)


# ---------------------------------------------------------------------------
# 360° views
# ---------------------------------------------------------------------------

def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera → rig rotation (rig: +X right, +Y down, +Z = equirect centre); = Ry(yaw) · Rx(pitch)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def equirect_views(o: Dict[str, Any]) -> List[Dict[str, Any]]:
    size = int(o["outputSize"])
    f = size / 2 / math.tan(math.radians(float(o["fovDegrees"])) / 2)
    views: List[Dict[str, Any]] = []

    def add(yaw, pitch):
        views.append({"index": len(views), "yaw": yaw, "pitch": pitch, "R": view_rotation(yaw, pitch), "size": size, "focal": f})

    for ring in o["rings"]:
        n = int(ring.get("count", 0))
        for k in range(n):
            add(float(o["yawOffset"]) + float(ring.get("yawOffset", 0)) + k * 360.0 / n, float(ring["pitch"]))
    if o.get("includeZenith"):
        add(float(o["yawOffset"]), 90.0)
    if o.get("includeNadir"):
        add(float(o["yawOffset"]), -90.0)
    return views


def view_lookup(v: Dict[str, Any], W: int, H: int, o: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(n, n, 2) source pixel coordinates (pixel centres at +0.5) and an optional latitude keep-mask."""
    n, f, c = v["size"], v["focal"], v["size"] / 2
    xs = (np.arange(n) + 0.5 - c) / f
    X, Y = np.meshgrid(xs, xs)
    d = np.stack([X, Y, np.ones_like(X)], -1) @ v["R"].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
    coords = np.stack([(lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * H], -1).astype(np.float32)
    lo, hi = o.get("maskBelowLatitude"), o.get("maskAboveLatitude")
    mask = None
    if lo is not None or hi is not None:
        keep = np.ones((n, n), bool)
        if lo is not None:
            keep &= lat >= math.radians(lo)
        if hi is not None:
            keep &= lat <= math.radians(hi)
        mask = keep.astype(np.uint8) * 255
    return coords, mask


class Remapper:
    """Resamples one equirect frame into all views at once (torch GPU, else OpenCV)."""

    def __init__(self, views, W: int, H: int, o: Dict[str, Any], device: str):
        self.views, self.W, self.H = views, W, H
        tables = [view_lookup(v, W, H, o) for v in views]
        self.masks = [m for _, m in tables]
        self.coords = [c for c, _ in tables]
        self.torch = None
        if device in ("cuda", "mps"):
            try:
                import torch

                self.torch = torch
                self.device = torch.device(device)
                g = np.stack(self.coords).copy()
                # circular pad by one column each side: x' = x + 1 over width W + 2
                g[..., 0] = (g[..., 0] + 1) / (W + 2) * 2 - 1
                g[..., 1] = g[..., 1] / H * 2 - 1
                self.grid = torch.from_numpy(g).to(self.device)
            except Exception:  # noqa: BLE001 — fall back to OpenCV
                self.torch = None
        if self.torch is None:
            import cv2

            self.cv2 = cv2
            self.maps = [(c[..., 0] - 0.5, c[..., 1] - 0.5) for c in self.coords]

    @property
    def backend(self) -> str:
        return f"torch-{self.device.type}" if self.torch is not None else "opencv"

    def __call__(self, rgb: np.ndarray) -> List[np.ndarray]:
        if self.torch is not None:
            torch = self.torch
            with torch.no_grad():
                t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                t = torch.cat([t[..., -1:], t, t[..., :1]], dim=3)
                out = torch.nn.functional.grid_sample(t.expand(len(self.views), -1, -1, -1), self.grid,
                                                      mode="bilinear", padding_mode="border", align_corners=False)
                out = out.clamp_(0, 255).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            return list(out)
        cv2 = self.cv2
        return [cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) for mx, my in self.maps]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

#: FFmpeg AVColorTransferCharacteristic values (and names) of HDR curves.
HDR_TRANSFERS = {16: "pq", 18: "hlg", "smpte2084": "pq", "arib-std-b67": "hlg", "arib_std_b67": "hlg"}


def _tone_map(rgb16: np.ndarray, curve: str) -> np.ndarray:
    """Approximate HLG / PQ → SDR (BT.2020 → BT.709, extended Reinhard, sRGB curve)."""
    e = rgb16.astype(np.float32) / 65535.0
    if curve == "hlg":
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        lin = np.where(e <= 0.5, (e * e) / 3.0, (np.exp((e - c) / a) + b) / 12.0)
        lin /= 0.2647  # 75 % HLG signal (reference white) → 1.0
    else:
        m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
        p = np.power(np.clip(e, 0, 1), 1 / m2)
        lin = np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1) * 10000.0 / 203.0
    m = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]], np.float32)
    lin = np.clip(lin @ m.T, 0, None)
    white = 4.0
    lin = lin * (1 + lin / (white * white)) / (1 + lin)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _rotation(frame) -> int:
    r = getattr(frame, "rotation", 0) or 0
    return int(round(r / 90.0)) % 4


def _write_image(path: Path, rgb: np.ndarray, encoding: str) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding == "png":
        ok = cv2.imwrite(str(path), bgr)
    else:
        ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise SplatPrepError(f"could not write {path.name}")


def _focal35(container) -> Optional[float]:
    for md in (container.metadata, *(s.metadata for s in container.streams.video)):
        for k, v in (md or {}).items():
            if "focal_length.35mm_equivalent" in k:
                try:
                    return float(v)
                except ValueError:
                    pass
    return None


def extract_frames(video: Path, out_dir: Path, vo: Dict[str, Any], eq: Optional[Dict[str, Any]], device: str,
                   progress: ProgressFn, log: Callable[[str], None] = lambda m: None) -> Dict[str, Any]:
    try:
        import av
    except ImportError as e:
        raise SplatPrepError("frame extraction needs PyAV (pip install av)") from e
    vo = {**VIDEO_DEFAULTS, **{k: v for k, v in (vo or {}).items() if v is not None}}
    if eq is not None:
        eq = {**EQUIRECT_DEFAULTS, **{k: v for k, v in eq.items() if v is not None}}
    ext = "png" if vo["encoding"] == "png" else "jpg"
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = 0
    cc = stream.codec_context
    duration = float(stream.duration * stream.time_base) if stream.duration else (container.duration or 0) / 1e6
    start = float(vo["startTime"] or 0.0)
    end = min(float(vo["endTime"]) if vo["endTime"] is not None else duration, duration or 1e18)
    span = max(end - start, 1e-3)
    window = span / int(vo["targetFrameCount"]) if vo["targetFrameCount"] else 1.0 / max(float(vo["targetFPS"]), 0.01)
    trc = getattr(cc, "color_trc", None)
    trc = getattr(trc, "value", trc)
    hdr_curve = HDR_TRANSFERS.get(trc if isinstance(trc, int) else str(trc).lower()) if vo["toneMapHDR"] else None
    f35 = _focal35(container)
    if start > 0:
        container.seek(int(start / stream.time_base), stream=stream, backward=True)

    W, H = cc.width, cc.height
    views = remap = None
    view_masks: List[Optional[Path]] = []
    if eq is not None:
        if abs(W / max(H, 1) - 2) > 0.1:
            raise SplatPrepError(f"360° video must be 2:1 equirectangular (got {W}×{H})")
        views = equirect_views(eq)
        remap = Remapper(views, W, H, eq, device)
        for v, m in zip(views, remap.masks):
            if m is None:
                view_masks.append(None)
                continue
            (out_dir / "masks").mkdir(parents=True, exist_ok=True)
            p = out_dir / "masks" / f".view{v['index']:02d}.png"
            import cv2

            cv2.imwrite(str(p), m)
            view_masks.append(p)
        log(f"{len(views)} views per frame, resampling on {remap.backend}")
    progress(0.0, f"{W}×{H} {stream.codec_context.name}, {len(views or [1])} images per frame")

    frames: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    pool = ThreadPoolExecutor(max_workers=max(2, min(16, (os.cpu_count() or 4) - 2)))
    pending = []
    tw = min(THUMB_MAX_WIDTH, W)
    th = max(8, int(round(H * tw / W)))

    def write(frame, idx: int, sharp: float, t: float) -> List[Dict[str, Any]]:
        if hdr_curve:
            rgb = _tone_map(frame.to_ndarray(format="rgb48le"), hdr_curve)
        else:
            rgb = frame.to_ndarray(format="rgb24")
        k = _rotation(frame)
        if k:
            rgb = np.ascontiguousarray(np.rot90(rgb, k))
        out: List[Dict[str, Any]] = []
        if views is None:
            h, w = rgb.shape[:2]
            ms = vo["maxImageSize"]
            if ms and max(w, h) > ms:
                import cv2

                s = ms / max(w, h)
                rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                h, w = rgb.shape[:2]
            name = f"{idx:06d}.{ext}"
            pending.append(pool.submit(_write_image, images / name, rgb, vo["encoding"]))
            fr = {"image": f"images/{name}", "width": w, "height": h, "sharpness": sharp, "source": str(video), "timestamp": t}
            if f35:
                fr["focalPx"] = f35 / 36.0 * max(w, h)
            out.append(fr)
            return out
        for v, img, mp in zip(views, remap(rgb), view_masks):
            name = f"{idx:06d}_v{v['index']:02d}"
            pending.append(pool.submit(_write_image, images / f"{name}.{ext}", img, vo["encoding"]))
            fr = {"image": f"images/{name}.{ext}", "width": v["size"], "height": v["size"], "sharpness": sharp,
                  "source": str(video), "timestamp": t, "focalPx": v["focal"], "cx": v["size"] / 2, "cy": v["size"] / 2,
                  "group": idx, "rotationToRig": [[float(x) for x in row] for row in v["R"]]}
            if mp is not None:
                dst = out_dir / "masks" / f"{name}.png"
                try:
                    os.link(mp, dst)
                except OSError:
                    import shutil

                    shutil.copyfile(mp, dst)
                fr["mask"] = f"masks/{name}.png"
            out.append(fr)
        return out

    best = None  # (frame, sharp, thumb, t)
    state = {"window": -1, "origin": start, "last_thumb": None, "last_kept": None, "out": 0}

    def flush():
        nonlocal best
        if best is None:
            return
        b, best = best, None
        if vo["skipStaticFrames"] and state["last_kept"] is not None and difference(state["last_kept"], b[2]) < float(vo["staticThreshold"]):
            rejected.append({"item": f"t={b[3]:.2f}s", "reason": "static (camera not moving)"})
            return
        frames.extend(write(b[0], state["out"], b[1], b[3]))
        state["out"] += 1
        state["last_kept"] = b[2]
        while len(pending) > 64:
            pending.pop(0).result()

    decoded = 0
    stride = max(1, int(vo["analysisStride"]))
    import time as _time

    last_report = 0.0
    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            decoded += 1
            if t < start - 1e-6:
                continue
            if t > end + 1e-6:
                break
            w = int(math.floor((t - state["origin"]) / window))
            analyse = (decoded % stride == 0 or w != state["window"]) if vo["pickSharpest"] else (w != state["window"])
            if not analyse:
                continue
            gray = frame.reformat(width=tw, height=th, format="gray", interpolation="POINT").to_ndarray()
            if gray.ndim == 3:
                gray = gray[..., 0]
            th32 = thumb32(gray)
            if vo["skipSceneCuts"] and state["last_thumb"] is not None and difference(state["last_thumb"], th32) > float(vo["sceneCutThreshold"]):
                if best is not None:
                    rejected.append({"item": f"t={best[3]:.2f}s", "reason": "scene cut"})
                best = None
                state.update(origin=t, window=0, last_thumb=th32, last_kept=None)
                continue
            state["last_thumb"] = th32
            if w != state["window"]:
                flush()
                state["window"] = w
            if best is None or vo["pickSharpest"]:
                s = sharpness(gray)
                if best is None or s > best[1]:
                    best = (frame, s, th32, t)
            now = _time.monotonic()
            if now - last_report > 0.5:
                last_report = now
                progress(min(1.0, (t - start) / span), f"{state['out']} frames ({decoded} decoded)")
        flush()
        for p in pending:
            p.result()
    finally:
        pool.shutdown(wait=True)
        container.close()
    for p in view_masks:
        if p is not None:
            p.unlink(missing_ok=True)
    if not frames:
        raise SplatPrepError(f"no frames extracted from {video.name}")
    if views is None:
        frames.sort(key=lambda f: f.get("timestamp") or 0)
    fs = {"root": ".", "frames": frames, "rejected": rejected, "isMultiViewRig": views is not None}
    (out_dir / "frames.json").write_text(json.dumps(fs, indent=1))
    progress(1.0, f"{len(frames)} images from {state['out']} frames")
    return {"numFrames": state["out"], "numImages": len(frames), "numDecoded": decoded, "numRejected": len(rejected),
            "width": W, "height": H, "views": len(views or []), "toneMapped": bool(hdr_curve),
            "resampler": remap.backend if remap else None}
