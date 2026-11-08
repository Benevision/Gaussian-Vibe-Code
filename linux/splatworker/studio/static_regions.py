"""Mask fixed parts of the frame (``mask.staticRegions``), as the Mac's ``StaticRegions``.

Parts of the picture that never change while the camera moves — a lens rim or vignette, a burned-in timestamp or
logo, the drone's propellers or landing gear, a selfie stick, a car bonnet — would be baked into the scene as a
smear that follows the camera. Per camera (each 360° view direction and each stereo eye is a camera of its own)
up to 48 frames are sampled at 192 px wide. A pixel is static when its whole 3×3 neighbourhood barely varies over
time AND it either has structure in the average image (edges of an overlay / the rig) or is nearly black (lens
rim). Plain sky also barely varies but has no structure, so it stays. If more than a quarter of the frame looks
static the camera hardly moves, and nothing is masked for that camera.

The result is a mask layer of its own, ``splatprep-masks/static/masks/<stem>.png`` (255 = keep), stacked with the
others by :func:`importers.stack_masks`. Only small grey thumbnails are held in memory.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

LAYER = "splatprep-masks/static/masks"
SAMPLE_WIDTH = 192
MAX_SAMPLES = 48
Progress = Callable[[float, str], None]


def _frames_meta(ds: Path) -> Dict[str, dict]:
    f = ds / "frames.json"
    if not f.is_file():
        return {}
    try:
        return {Path(fr["image"]).name: fr for fr in json.loads(f.read_text(encoding="utf-8")).get("frames", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def camera_key(name: str, size: Tuple[int, int], meta: Optional[dict]) -> str:
    """The 360° view direction (``…_v05``), the stereo eye, or the frame size."""
    m = re.search(r"_v\d+$", Path(name).stem)
    if m and (meta is None or meta.get("rotationToRig") is not None):
        return m.group(0)
    eye = (meta or {}).get("eye")
    if eye:
        return f"eye {eye}"
    return f"{size[0]}x{size[1]}"


def _gray_small(path: Path, w: int, h: int) -> Optional[np.ndarray]:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.draft("L", (w * 2, h * 2))
            return np.asarray(im.convert("L").resize((w, h), Image.BILINEAR), np.float64)
    except Exception:  # noqa: BLE001 — an unreadable frame is just not sampled
        return None


def detect(paths: List[Path], size: Tuple[int, int]) -> Optional[Tuple[np.ndarray, float]]:
    """(low-resolution keep mask, uint8 255 keep / 0 static; static fraction) or None when nothing is static."""
    if len(paths) < 12:
        return None
    W, H = size
    w = SAMPLE_WIDTH
    h = max(8, int(round(SAMPLE_WIDTH * H / max(W, 1))))
    step = max(1, len(paths) // MAX_SAMPLES)
    s = np.zeros((h, w))
    sq = np.zeros((h, w))
    used = 0
    for p in paths[::step]:
        g = _gray_small(p, w, h)
        if g is None:
            continue
        s += g
        sq += g * g
        used += 1
    if used < 10:
        return None
    mean = s / used
    std = np.sqrt(np.maximum(0.0, sq / used - mean * mean))
    grad = np.zeros_like(mean)
    grad[1:-1, 1:-1] = (np.abs(mean[1:-1, 2:] - mean[1:-1, :-2]) + np.abs(mean[2:, 1:-1] - mean[:-2, 1:-1]))
    still = std < 4
    all_still = np.zeros_like(still)
    core = np.ones((h - 2, w - 2), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            core &= still[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
    all_still[1:-1, 1:-1] = core
    stat = all_still & ((mean < 12) | (grad > 20))
    stat[0, :] = stat[-1, :] = False
    stat[:, 0] = stat[:, -1] = False
    grown = stat.copy()                              # grow by 3 px at sample resolution (covers soft edges)
    ys, xs = np.nonzero(stat)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            grown[yy[ok], xx[ok]] = True
    frac = float(grown.mean())
    if frac < 0.002:
        return None
    return np.where(grown, 0, 255).astype(np.uint8), frac


def apply(dataset_dir: Path, dry_run: bool = False, progress: Progress = lambda f, m="": None) -> List[Dict]:
    """Detect per camera and write the ``static`` layer. Returns one report per camera
    ({camera, frames, staticFraction, applied})."""
    from PIL import Image

    from .importers import list_images

    ds = Path(dataset_dir)
    layer = ds / LAYER
    if not dry_run:
        shutil.rmtree(layer.parent, ignore_errors=True)
    meta = _frames_meta(ds)
    groups: Dict[str, List[Path]] = {}
    sizes: Dict[str, Tuple[int, int]] = {}
    for p in list_images(ds / "images"):
        try:
            with Image.open(p) as im:
                size = im.size
        except Exception:  # noqa: BLE001
            continue
        k = camera_key(p.name, size, meta.get(p.name))
        groups.setdefault(k, []).append(p)
        sizes.setdefault(k, size)
    reports: List[Dict] = []
    for n, key in enumerate(sorted(groups)):
        paths = groups[key]
        progress(n / max(len(groups), 1), f"looking for fixed parts: {key}")
        r = detect(paths, sizes[key])
        if r is None:
            reports.append({"camera": key, "frames": len(paths), "staticFraction": 0.0, "applied": False})
            continue
        keep, frac = r
        ok = frac <= 0.25                 # more = the camera barely moves; masking would remove the scene
        reports.append({"camera": key, "frames": len(paths), "staticFraction": frac, "applied": ok and not dry_run})
        if not ok or dry_run:
            continue
        layer.mkdir(parents=True, exist_ok=True)
        small = Image.fromarray(keep)
        for p in paths:
            with Image.open(p) as im:
                size = im.size
            small.resize(size, Image.NEAREST).save(layer / f"{p.stem}.png")
    progress(1.0, f"{sum(r['applied'] for r in reports)} of {len(reports)} cameras masked")
    return reports
