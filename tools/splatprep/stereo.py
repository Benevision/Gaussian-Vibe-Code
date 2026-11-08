"""Stereo footage helpers for ``task: frames`` with ``stereo`` (the Mac's ``StereoVideoIngest``), torch-free.

Request key ``stereo`` (all optional)::

    {"layout": "auto" | "sideBySide" | "topBottom" | "dualStream" | "separateFiles",
     "rightVideo": "right.mp4",        # separateFiles: the right eye, relative to the work dir
     "rightTimeOffset": 0.0,           # seconds, right video vs left
     "swapEyes": false, "anamorphic": null,   # half-width SBS / half-height TB (null = from the aspect ratio)
     "baseline": 0.064,                # metres; overrides calibration
     "horizontalFOV": 90.0}            # one eye, degrees; used when there is no calibration / focal metadata

Output: ``images/<frame>_L.<ext>`` and ``_R`` (flat names so basename matching never collides), each frame entry
with ``group`` (the frame index) and ``eye`` ("left" / "right") plus ``focalPx/cx/cy`` when known, and in
``frames.json`` a ``rig`` (the Mac's ``StereoRig``: ``baseline``, ``leftIntrinsics``, ``rightIntrinsics``,
``horizontalFOV``, ``origin``) with two additions: ``parallel`` (true when the eyes can be treated as a rectified
pair: packed SBS / TB layouts, or no calibration saying otherwise) and ``lens`` ("pinhole" / "fisheye").

Arcturus Vision recordings carry their factory calibration as the container tag ``offlineCal`` (JSON, mm):
per camera Kannala-Brandt (``kb``) intrinsics, the image-circle diameter, a ``sourceCamera``
(``passthrough_left`` / ``passthrough_right``, in stream order) and extrinsic positions. The baseline is the
distance between the positions; the eyes are fisheye and toed out, so they are *not* a rectified pair.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

LAYOUTS = ("auto", "sideBySide", "topBottom", "dualStream", "separateFiles")


def detect_layout(width: int, height: int, n_streams: int, second_same_size: bool, name: str,
                  opts: Dict[str, Any], has_right_video: bool = False) -> Tuple[str, bool, str]:
    """(layout, anamorphic, reason) — the Mac's ``StereoVideoIngest.detect`` (MV-HEVC is Apple-only)."""
    layout = opts.get("layout") or "auto"
    ana = opts.get("anamorphic")
    aspect = width / max(height, 1)
    n = name.lower()
    if layout != "auto":
        return layout, bool(ana) if ana is not None else False, "chosen"
    if has_right_video:
        return "separateFiles", False, "a right-eye video was given"
    if n_streams >= 2 and second_same_size:
        return "dualStream", False, "two video streams of equal size"
    if any(k in n for k in ("_tb", "top-bottom", "topbottom", "over-under")) or aspect < 1.05:
        return "topBottom", bool(ana) if ana is not None else aspect >= 1.5, f"aspect {aspect:.2f} → top-bottom"
    if aspect >= 2.6 or "_sbs" in n or "side-by-side" in n or "arcturus" in n:
        return "sideBySide", bool(ana) if ana is not None else aspect < 2.6, f"aspect {aspect:.2f} → side-by-side"
    if abs(aspect - 2) < 0.05:
        return "sideBySide", False, "2:1 frame of two square eyes (assumed side-by-side)"
    raise ValueError(f"cannot detect the stereo layout (aspect {aspect:.2f}): choose it")


def split_eyes(rgb: np.ndarray, layout: str, anamorphic: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Left / right halves of a packed frame (un-squeezed when anamorphic)."""
    h, w = rgb.shape[:2]
    if layout == "sideBySide":
        a, b = rgb[:, : w // 2], rgb[:, w // 2: (w // 2) * 2]
        size = (2 * (w // 2), h) if anamorphic else None
    elif layout == "topBottom":
        a, b = rgb[: h // 2], rgb[h // 2: (h // 2) * 2]
        size = (w, 2 * (h // 2)) if anamorphic else None
    else:
        raise ValueError(f"not a packed layout: {layout}")
    if size is not None:
        import cv2

        a = cv2.resize(a, size, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, size, interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def eye_crop(layout: str, w: int, h: int) -> Optional[Tuple[slice, slice]]:
    """(rows, cols) of the left eye in a packed frame of size w×h (for sharpness), None otherwise."""
    if layout == "sideBySide":
        return slice(0, h), slice(0, w // 2)
    if layout == "topBottom":
        return slice(0, h // 2), slice(0, w)
    return None


def parse_offline_cal(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Arcturus ``offlineCal`` → {"baseline" (m), "cameras": [{"stream", "eye", "intrinsics", "model",
    "distortion", "circleDeg", "width", "height"}, …], "parallel", "lens"}; None if absent / unusable."""
    if not text:
        return None
    try:
        cal = json.loads(text)
        cams = cal.get("cameras") or []
    except (ValueError, AttributeError):
        return None
    if len(cams) < 2:
        return None
    unit = 0.001 if str(cal.get("__units", "mm")).lower() == "mm" else 1.0
    out = []
    for i, c in enumerate(cams[:2]):
        intr = (c.get("intrinsics") or [{}])[0]
        src = str(c.get("sourceCamera", "")).lower()
        eye = "left" if "left" in src else "right" if "right" in src else ("left" if i == 0 else "right")
        try:
            k = [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])]
        except (KeyError, TypeError, ValueError):
            k = None
        out.append({"stream": i, "eye": eye, "intrinsics": k, "model": intr.get("cameraModel"),
                    "distortion": [float(intr.get(f"k{j}", 0.0)) for j in range(1, 5)],
                    "circleDeg": c.get("imageCircleDiameterInDeg"), "width": c.get("width"),
                    "height": c.get("height"), "position": ((c.get("extrinsics") or {}).get("position"))})
    if out[0]["eye"] == out[1]["eye"]:
        out[0]["eye"], out[1]["eye"] = "left", "right"
    base = None
    p0, p1 = out[0]["position"], out[1]["position"]
    if p0 and p1 and len(p0) == 3 and len(p1) == 3:
        base = float(np.linalg.norm(np.subtract(p0, p1))) * unit
    fisheye = any((c["model"] or "").lower() in ("kb", "kb_polar_spline", "fisheye", "opencv_fisheye") for c in out)
    return {"baseline": base, "cameras": out, "parallel": False, "lens": "fisheye" if fisheye else "pinhole",
            "origin": "calibration (offlineCal)"}


def kb_radius(theta: float, k) -> float:
    """Kannala-Brandt: r / f for an incidence angle theta (radians)."""
    t2 = theta * theta
    return theta * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4)


def circle_mask(w: int, h: int, cam: Dict[str, Any], scale: float = 1.0) -> Optional[np.ndarray]:
    """Keep-mask (uint8 255 keep) of a fisheye eye's image circle, for the written size; None when the whole
    frame is inside the circle or the calibration doesn't say."""
    k = cam.get("intrinsics")
    deg = cam.get("circleDeg")
    if not k or not deg:
        return None
    fx, fy, cx, cy = (v * scale for v in k)
    r = kb_radius(math.radians(float(deg)) / 2, cam.get("distortion") or [0, 0, 0, 0])
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx + 0.5 - cx) / (fx * r)) ** 2 + ((yy + 0.5 - cy) / (fy * r)) ** 2 <= 1.0
    if inside.all():
        return None
    return inside.astype(np.uint8) * 255


def focal_for(rig: Dict[str, Any], eye: str, width: int, height: int, f35: Optional[float]):
    """(fx, cx, cy) of an eye image of the written size — the Mac's ``StereoVideoIngest.focal``."""
    k = rig.get("leftIntrinsics") if eye == "left" else (rig.get("rightIntrinsics") or rig.get("leftIntrinsics"))
    calib_w = rig.get("calibrationWidth")
    if k and len(k) >= 4:
        s = width / float(calib_w or max(k[2] * 2, 1))
        return k[0] * s, k[2] * s, k[3] * s
    fov = rig.get("horizontalFOV")
    if fov and fov > 1:
        return width / 2 / math.tan(math.radians(fov) / 2), width / 2, height / 2
    if f35:
        return f35 / 36.0 * max(width, height), width / 2, height / 2
    return None
