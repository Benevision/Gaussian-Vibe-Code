"""Parsing and loading of the prepared training dataset (PROTOCOL.md §2).

Pure numpy + Pillow so it can be tested without torch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional HEIC support
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


class DatasetError(ValueError):
    pass


@dataclass
class CameraInfo:
    id: int
    image: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: np.ndarray  # [4,4] float64, OpenCV convention
    depth: Optional[str] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    holdout: bool = False
    exposure: Optional[float] = None  # shutter s × ISO / 100 (iPhone captures); seeds appearance correction


@dataclass
class DatasetInfo:
    root: Path
    cameras: List[CameraInfo]
    scene_center: np.ndarray
    scene_extent: float
    up_vector: np.ndarray
    points: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def train_cameras(self) -> List[CameraInfo]:
        return [c for c in self.cameras if not c.holdout]

    @property
    def holdout_cameras(self) -> List[CameraInfo]:
        return [c for c in self.cameras if c.holdout]

    @property
    def has_depth(self) -> bool:
        return any(c.depth for c in self.train_cameras)

    def resolve(self, rel: str) -> Path:
        return safe_join(self.root, rel)


def safe_join(root: Path, rel: str) -> Path:
    """Join a dataset-relative path, refusing anything that escapes ``root``."""
    if not isinstance(rel, str) or rel == "" or "\x00" in rel or "\\" in rel:
        raise DatasetError(f"invalid relative path: {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise DatasetError(f"path escapes dataset directory: {rel!r}")
    root = Path(root).resolve()
    full = (root / Path(*p.parts)).resolve()
    if full != root and root not in full.parents:
        raise DatasetError(f"path escapes dataset directory: {rel!r}")
    return full


def _is_root(d: Path) -> bool:
    return (d / "dataset.json").is_file() or (d / "images").is_dir()


def find_upload_root(directory: Path) -> Path:
    """Locate the dataset root in an extracted upload.

    Accepts the bundle contents at the archive root, or a single top-level
    folder (Finder-style zips; ``__MACOSX`` and dot-folders are ignored). A
    root is a directory containing ``dataset.json`` (training) or ``images/``
    (processing uploads, PROTOCOL.md §6); anything else is kept as a partial upload for composing.
    """
    directory = Path(directory)
    if _is_root(directory):
        return directory
    subdirs = [p for p in directory.iterdir()
               if p.is_dir() and not p.name.startswith(".") and p.name != "__MACOSX"]
    candidates = [d for d in subdirs if _is_root(d)]
    if len(candidates) == 1:
        return candidates[0]
    # A partial upload (e.g. just masks/ and priors.json) meant to be composed over files the worker already
    # has (POST /v1/datasets/compose): the single top-level folder, or the archive root.
    visible = [p for p in directory.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    if len(visible) == 1 and visible[0].is_dir():
        return visible[0]
    if visible:
        return directory
    raise DatasetError("archive is empty")


def find_dataset_root(directory: Path) -> Path:
    root = find_upload_root(directory)
    if not (root / "dataset.json").is_file():
        raise DatasetError("dataset.json not found (this upload can only be used for processing jobs)")
    return root


def _num(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise DatasetError(f"{name} must be a finite number")
    return float(v)


def _matrix(v, name: str) -> np.ndarray:
    try:
        m = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise DatasetError(f"{name} must be a 4x4 numeric matrix")
    if m.shape == (16,):
        m = m.reshape(4, 4)  # tolerate flat row-major
    if m.shape == (3, 4):
        m = np.vstack([m, [0, 0, 0, 1]])
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise DatasetError(f"{name} must be a 4x4 numeric matrix (row-major)")
    return m


def parse_dataset(directory, check_files: bool = True) -> DatasetInfo:
    """Parse ``dataset.json`` in ``directory`` (or its single top-level folder)."""
    root = find_dataset_root(Path(directory)).resolve()
    try:
        raw = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DatasetError(f"cannot read dataset.json: {e}")
    if not isinstance(raw, dict):
        raise DatasetError("dataset.json must be a JSON object")
    if raw.get("format", "splatdataset") != "splatdataset":
        raise DatasetError(f"unsupported dataset format {raw.get('format')!r}")
    version = raw.get("version", 1)
    if not isinstance(version, int) or version > 1:
        raise DatasetError(f"unsupported dataset version {version!r}")
    conv = raw.get("convention", "opencv")
    if conv != "opencv":
        raise DatasetError(f"unsupported camera convention {conv!r} (expected 'opencv')")

    cams_raw = raw.get("cameras")
    if not isinstance(cams_raw, list) or not cams_raw:
        raise DatasetError("dataset.json has no cameras")

    cameras: List[CameraInfo] = []
    for i, c in enumerate(cams_raw):
        if not isinstance(c, dict):
            raise DatasetError(f"camera {i} must be an object")
        label = f"camera {i}"
        image = c.get("image")
        if not isinstance(image, str):
            raise DatasetError(f"{label}: missing image")
        safe_join(root, image)
        width, height = c.get("width"), c.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DatasetError(f"{label}: width/height must be positive integers")
        fx, fy = _num(c.get("fx"), f"{label}.fx"), _num(c.get("fy"), f"{label}.fy")
        cx, cy = _num(c.get("cx"), f"{label}.cx"), _num(c.get("cy"), f"{label}.cy")
        if fx <= 0 or fy <= 0:
            raise DatasetError(f"{label}: focal lengths must be positive")
        c2w = _matrix(c.get("cameraToWorld"), f"{label}.cameraToWorld")
        depth = c.get("depth")
        dw = dh = None
        if depth is not None:
            if not isinstance(depth, str):
                raise DatasetError(f"{label}: depth must be a path")
            safe_join(root, depth)
            dw, dh = c.get("depthWidth"), c.get("depthHeight")
            if not isinstance(dw, int) or not isinstance(dh, int) or dw <= 0 or dh <= 0:
                raise DatasetError(f"{label}: depthWidth/depthHeight required with depth")
        cam_id = c.get("id", i)
        cameras.append(CameraInfo(
            id=int(cam_id) if isinstance(cam_id, (int, float)) else i,
            image=image, width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy, camera_to_world=c2w,
            depth=depth, depth_width=dw, depth_height=dh,
            holdout=bool(c.get("holdout", False)),
            exposure=float(c["exposure"]) if isinstance(c.get("exposure"), (int, float)) and c["exposure"] > 0 else None,
        ))

    if not any(not c.holdout for c in cameras):
        raise DatasetError("all cameras are holdout; nothing to train on")

    positions = np.stack([c.camera_to_world[:3, 3] for c in cameras])
    center = raw.get("sceneCenter")
    if isinstance(center, list) and len(center) == 3:
        center = np.array([_num(x, "sceneCenter") for x in center])
    else:
        center = positions.mean(axis=0)
    extent = raw.get("sceneExtent")
    if isinstance(extent, (int, float)) and not isinstance(extent, bool) and math.isfinite(extent) and extent > 0:
        extent = float(extent)
    else:
        extent = float(np.linalg.norm(positions - center, axis=1).max() * 1.1) or 1.0
    up = raw.get("upVector")
    if isinstance(up, list) and len(up) == 3:
        up = np.array([_num(x, "upVector") for x in up])
    else:
        up = np.array([0.0, -1.0, 0.0])

    points = raw.get("points")
    if points is not None:
        if not isinstance(points, str):
            raise DatasetError("points must be a path")
        safe_join(root, points)
    elif (root / "points3d.ply").is_file():
        points = "points3d.ply"

    ds = DatasetInfo(root=root, cameras=cameras, scene_center=center, scene_extent=extent,
                     up_vector=up, points=points, raw=raw)
    if check_files:
        missing = [c.image for c in cameras if not ds.resolve(c.image).is_file()]
        if missing:
            raise DatasetError(f"{len(missing)} image(s) missing, e.g. {missing[0]}")
        for c in cameras:
            if c.depth and not ds.resolve(c.depth).is_file():
                raise DatasetError(f"depth file missing: {c.depth}")
        if points is not None and not ds.resolve(points).is_file():
            ds.points = None if raw.get("points") is None else points
            if ds.points is not None:
                raise DatasetError(f"point cloud missing: {points}")
    return ds


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def target_size(width: int, height: int, max_size: int) -> Tuple[int, int]:
    """Downscale so that the longest side is <= max_size (never upscale)."""
    longest = max(width, height)
    if longest <= max_size:
        return width, height
    s = max_size / float(longest)
    return max(1, int(round(width * s))), max(1, int(round(height * s)))


@dataclass
class LoadedView:
    camera: CameraInfo
    image: Optional[np.ndarray]  # uint8 [H, W, 3]; None for a view whose pixels are loaded on demand (ViewStore)
    alpha: Optional[np.ndarray]  # uint8 [H, W] or None
    K: np.ndarray                # float32 [3, 3] for the *loaded* resolution
    world_to_camera: np.ndarray  # float32 [4, 4]
    depth: Optional[np.ndarray]  # float32 [H, W] (resized to image), 0 = invalid
    size: Optional[tuple] = None  # (width, height) when image is None

    @property
    def width(self) -> int:
        return self.image.shape[1] if self.image is not None else self.size[0]

    @property
    def height(self) -> int:
        return self.image.shape[0] if self.image is not None else self.size[1]


def scaled_intrinsics(cam: CameraInfo, new_w: int, new_h: int) -> np.ndarray:
    sx = new_w / float(cam.width)
    sy = new_h / float(cam.height)
    return np.array([[cam.fx * sx, 0.0, cam.cx * sx],
                     [0.0, cam.fy * sy, cam.cy * sy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def load_depth(path: Path, width: int, height: int) -> np.ndarray:
    """Load a float16 little-endian row-major depth map (metres)."""
    raw = np.fromfile(path, dtype="<f2")
    if raw.size != width * height:
        raise DatasetError(f"{path.name}: expected {width}x{height} float16 values, got {raw.size}")
    d = raw.reshape(height, width).astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


def resize_nearest(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if (w, h) == (new_w, new_h):
        return arr
    ys = np.minimum(((np.arange(new_h) + 0.5) * h / new_h).astype(np.int64), h - 1)
    xs = np.minimum(((np.arange(new_w) + 0.5) * w / new_w).astype(np.int64), w - 1)
    return arr[ys[:, None], xs[None, :]]


def view_header(ds: DatasetInfo, cam: CameraInfo, max_size: int) -> LoadedView:
    """A view without pixels (sizes, intrinsics, pose) — for sets too large to hold in memory."""
    new_w, new_h = target_size(cam.width, cam.height, max_size)
    return LoadedView(camera=cam, image=None, alpha=None, K=scaled_intrinsics(cam, new_w, new_h),
                      world_to_camera=np.linalg.inv(cam.camera_to_world).astype(np.float32), depth=None,
                      size=(new_w, new_h))


def load_view(ds: DatasetInfo, cam: CameraInfo, max_size: int, load_depth_maps: bool = True) -> LoadedView:
    from PIL import Image, ImageOps

    path = ds.resolve(cam.image)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im) if im.format == "JPEG" else im
        im.load()
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        # Intrinsics are expressed in dataset.json width/height pixel units. If
        # the file differs, resize to the declared aspect first.
        if im.size != (cam.width, cam.height):
            im = im.resize((cam.width, cam.height), Image.LANCZOS)
        new_w, new_h = target_size(cam.width, cam.height, max_size)
        if (new_w, new_h) != im.size:
            im = im.resize((new_w, new_h), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    alpha = None
    if has_alpha:
        alpha = np.ascontiguousarray(arr[..., 3])
        arr = arr[..., :3]
    arr = np.ascontiguousarray(arr)

    K = scaled_intrinsics(cam, new_w, new_h)
    w2c = np.linalg.inv(cam.camera_to_world).astype(np.float32)

    depth = None
    if load_depth_maps and cam.depth:
        d = load_depth(ds.resolve(cam.depth), cam.depth_width, cam.depth_height)
        depth = resize_nearest(d, new_w, new_h).astype(np.float32)
    return LoadedView(camera=cam, image=arr, alpha=alpha, K=K, world_to_camera=w2c, depth=depth)


# ---------------------------------------------------------------------------
# Camera selection (cameraIds / region, PROTOCOL.md §7)
# ---------------------------------------------------------------------------

def camera_sees_box(cam: CameraInfo, bmin, bmax, samples: int = 6, near: float = 1e-3) -> bool:
    """True if any part of the axis-aligned box is inside the camera frustum.

    Conservative sampling test: the camera centre inside the box, or any of a
    ``samples^3`` grid of points over the box (corners included) projecting in
    front of the camera and inside the image bounds.
    """
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    c2w = cam.camera_to_world
    centre = c2w[:3, 3]
    if np.all(centre >= bmin) and np.all(centre <= bmax):
        return True
    t = np.linspace(0.0, 1.0, samples)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    pts = bmin + np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) * (bmax - bmin)
    w2c = np.linalg.inv(c2w)
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > near
    if not front.any():
        return False
    pc = pc[front]
    u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx
    v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
    inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return bool(inside.any())


def select_cameras(ds: DatasetInfo, camera_ids=None, region_box=None) -> List[CameraInfo]:
    """Apply ``cameraIds`` and the region frustum test; returns the kept cameras."""
    cams = list(ds.cameras)
    if camera_ids is not None:
        wanted = set(camera_ids)
        cams = [c for c in cams if c.id in wanted]
    if region_box is not None:
        lo, hi = region_box
        cams = [c for c in cams if camera_sees_box(c, lo, hi)]
    return cams
