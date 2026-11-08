"""360° equirectangular photos → perspective views, in the GUI process (numpy + Pillow, no GPU).

The same geometry as the Mac's ``Equirect360`` and splatprep's ``frames.py`` (360° video on the worker): a rig
frame with +X right, +Y down, +Z = the centre of the equirect image; each view's rotation is camera → rig,
``Ry(yaw) · Rx(pitch)``; square views with focal ``size / 2 / tan(fov / 2)``. Output layout matches the video
path (``images/<photo>_vNN.jpg``, nadir keep-masks, ``frames.json`` with ``group`` + ``rotationToRig``) so
:func:`importers.rig_from_frames` hands the camera solver one rig per photo.

Photos are processed one at a time and view by view (an 8K equirect is ~100 MB as RGB; a lookup table for a
1600 px view is 20 MB), so memory stays bounded however many photos the folder has.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

Progress = Callable[[float, str], None]

DEFAULTS: Dict[str, Any] = {
    "fovDegrees": 90.0, "outputSize": 1200,
    "rings": [{"pitch": 0, "count": 8}, {"pitch": 35, "count": 4, "yawOffset": 45},
              {"pitch": -35, "count": 4, "yawOffset": 45}],
    "includeZenith": False, "includeNadir": False, "maskBelowLatitude": -60.0, "maskAboveLatitude": None,
    "yawOffset": 0.0,
}


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera → rig rotation = Ry(yaw) · Rx(pitch) (yaw turns right, pitch + = up)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def views(options: Dict[str, Any]) -> List[Dict[str, Any]]:
    o = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
    size = int(o["outputSize"])
    f = size / 2 / math.tan(math.radians(float(o["fovDegrees"])) / 2)
    out: List[Dict[str, Any]] = []

    def add(yaw: float, pitch: float) -> None:
        out.append({"index": len(out), "yaw": yaw, "pitch": pitch, "R": view_rotation(yaw, pitch), "size": size,
                    "focal": f})

    for ring in o["rings"]:
        n = int(ring.get("count", 0))
        for k in range(n):
            add(float(o["yawOffset"]) + float(ring.get("yawOffset", 0)) + k * 360.0 / n, float(ring["pitch"]))
    if o.get("includeZenith"):
        add(float(o["yawOffset"]), 90.0)
    if o.get("includeNadir"):
        add(float(o["yawOffset"]), -90.0)
    return out


def lookup(v: Dict[str, Any], W: int, H: int, options: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(n, n, 2) equirect pixel coordinates (pixel centres at +0.5) of each view pixel, and the latitude keep-mask
    (uint8 255 = keep) when ``maskBelowLatitude`` / ``maskAboveLatitude`` is set."""
    o = {**DEFAULTS, **{k: val for k, val in (options or {}).items() if k in DEFAULTS}}
    n, f, c = v["size"], v["focal"], v["size"] / 2
    xs = ((np.arange(n, dtype=np.float64) + 0.5 - c) / f)
    X, Y = np.meshgrid(xs, xs)
    d = np.stack([X, Y, np.ones_like(X)], -1) @ np.asarray(v["R"]).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
    coords = np.stack([(lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * H], -1).astype(np.float32)
    lo, hi = o.get("maskBelowLatitude"), o.get("maskAboveLatitude")
    mask = None
    if lo is not None or hi is not None:
        keep = np.ones((n, n), bool)
        if lo is not None:
            keep &= lat >= math.radians(float(lo))
        if hi is not None:
            keep &= lat <= math.radians(float(hi))
        mask = keep.astype(np.uint8) * 255
    return coords, mask


def remap(rgb: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Bilinear sampling of ``rgb`` (H, W, C uint8) at ``coords`` (pixel-centre convention); x wraps around
    (the equirect seam), y is clamped."""
    H, W = rgb.shape[:2]
    x = coords[..., 0] - 0.5
    y = np.clip(coords[..., 1] - 0.5, 0, H - 1)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)
    x0 = x0.astype(np.int64) % W
    x1 = (x0 + 1) % W
    y0 = y0.astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    top = rgb[y0, x0].astype(np.float32) * (1 - fx) + rgb[y0, x1].astype(np.float32) * fx
    bot = rgb[y1, x0].astype(np.float32) * (1 - fx) + rgb[y1, x1].astype(np.float32) * fx
    out = top * (1 - fy) + bot * fy
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _sharpness(rgb: np.ndarray) -> float:
    from .importers import sharpness

    step = max(1, max(rgb.shape[:2]) // 512)
    return sharpness(rgb[::step, ::step].mean(-1))


def split_photos(photos: List[Path], dataset_dir: Path, options: Dict[str, Any], mask_dir: Optional[Path] = None,
                 progress: Progress = lambda f, m="": None) -> Dict[str, Any]:
    """Every 2:1 photo → views in ``dataset_dir/images`` (+ keep-masks in ``mask_dir``) and ``frames.json``.

    Returns {"photos", "images", "rejected": [{item, reason}]}. Photos that are not 2:1 are skipped (as on the
    Mac). ``group`` is the photo's index, so the solver sees each photo's views as one rig position."""
    from PIL import Image, ImageOps

    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, 16384 * 8192)
    dataset_dir = Path(dataset_dir)
    images = dataset_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    vs = views(options)
    frames: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    used = 0
    for i, src in enumerate(photos):
        try:
            with Image.open(src) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
        except Exception as e:  # noqa: BLE001 — unreadable photos are reported, not fatal
            rejected.append({"item": Path(src).name, "reason": f"unreadable: {e}"})
            continue
        W, H = im.size
        if abs(W / max(H, 1) - 2) > 0.1:
            rejected.append({"item": Path(src).name, "reason": "not 2:1 equirectangular"})
            im.close()
            continue
        rgb = np.asarray(im)
        im.close()
        sharp = _sharpness(rgb)
        for v in vs:
            coords, keep = lookup(v, W, H, options)
            view = remap(rgb, coords)
            del coords
            name = f"{i:06d}_v{v['index']:02d}"
            tmp = images / f".{name}.tmp.jpg"
            Image.fromarray(view).save(tmp, quality=92)
            os.replace(tmp, images / f"{name}.jpg")
            fr = {"image": f"images/{name}.jpg", "width": v["size"], "height": v["size"], "sharpness": sharp,
                  "source": str(src), "focalPx": v["focal"], "cx": v["size"] / 2, "cy": v["size"] / 2,
                  "group": i, "rotationToRig": [[float(x) for x in row] for row in v["R"]]}
            if keep is not None and mask_dir is not None and not keep.all():
                Path(mask_dir).mkdir(parents=True, exist_ok=True)
                Image.fromarray(keep).save(Path(mask_dir) / f"{name}.png")
                fr["mask"] = f"masks/{name}.png"
            frames.append(fr)
        del rgb
        used += 1
        progress((i + 1) / len(photos), f"{i + 1}/{len(photos)} photos, {len(frames)} views")
    if not frames:
        raise ValueError("no usable 360° photos (they must be 2:1 equirectangular)")
    fs = {"root": ".", "frames": frames, "rejected": rejected, "isMultiViewRig": True}
    (dataset_dir / "frames.json").write_text(json.dumps(fs, indent=1))
    return {"photos": used, "images": len(frames), "rejected": rejected}
