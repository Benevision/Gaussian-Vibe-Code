"""Image discovery and loading helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}

try:  # pragma: no cover - optional HEIC support
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(images_dir: Path) -> List[str]:
    """Relative POSIX paths of all images below ``images_dir`` (natural sort)."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    names = [
        p.relative_to(images_dir).as_posix()
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return sorted(names, key=natural_key)


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        exif = im.getexif() if hasattr(im, "getexif") else {}
        if exif and exif.get(0x0112, 1) in (5, 6, 7, 8):
            w, h = h, w
        return w, h


def load_rgb(path: Path, resize_max: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Load as uint8 HxWx3 (EXIF-transposed).  Returns (image, scale) with scale = new/orig."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = 1.0
        if resize_max and max(im.size) > resize_max:
            scale = resize_max / max(im.size)
            im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))), Image.BILINEAR)
        return np.asarray(im), scale


def load_gray(path: Path, resize_max: Optional[int] = None) -> Tuple[np.ndarray, float]:
    rgb, s = load_rgb(path, resize_max)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    return gray, s


def load_mask(masks_dir: Optional[Path], name: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Boolean keep-mask for ``name`` (masks/<stem>.png, 255 = keep) resized to ``shape`` (h, w)."""
    if masks_dir is None:
        return None
    from PIL import Image

    stem = Path(name).with_suffix("").as_posix()
    for cand in (Path(masks_dir) / f"{stem}.png", Path(masks_dir) / f"{Path(name).name}.png", Path(masks_dir) / f"{Path(stem).name}.png"):
        if cand.exists():
            with Image.open(cand) as m:
                m = m.convert("L")
                if m.size != (shape[1], shape[0]):
                    m = m.resize((shape[1], shape[0]), Image.NEAREST)
                return np.asarray(m) > 127
    return None


_TRAILING_INT = re.compile(r"^(.*?)(\d+)$")


def looks_like_video(names: List[str]) -> bool:
    """True when names share a prefix and carry a (mostly) consecutive frame number."""
    if len(names) < 3:
        return False
    prefixes, nums = [], []
    for n in names:
        m = _TRAILING_INT.match(Path(n).stem)
        if not m:
            return False
        prefixes.append((str(Path(n).parent), m.group(1)))
        nums.append(int(m.group(2)))
    if len(set(prefixes)) > max(1, len(names) // 50):
        return False
    nums = np.sort(np.asarray(nums))
    d = np.diff(nums)
    if d.size == 0 or np.any(d <= 0):
        return False
    step = np.median(d)
    return bool(np.mean(d == step) >= 0.9)
