"""In-memory Gaussian splat scene (numpy structure of arrays) and the learned-skybox sidecar.

Storage follows the INRIA 3DGS PLY (and the Mac app's ``GaussianCloud``) so files round-trip losslessly:

* ``positions``  (N, 3) float32, world units
* ``log_scales`` (N, 3) float32, natural log of the per-axis standard deviation
* ``rotations``  (N, 4) float32 quaternion **w, x, y, z** (not necessarily normalised)
* ``opacity_logits`` (N,) float32, logit of opacity
* ``sh``         (N, K, 3) float32, K = (degree + 1)², coefficient-major; ``sh[:, 0]`` is the DC term

The scene camera convention is OpenCV (x right, y down, z forward). The skybox sidecar is
``<name>.sky.png`` (equirectangular, 2:1, 8-bit sRGB) + ``<name>.sky.json`` (up / forward / right), written by
both trainers (``splatworker/skybox.py``, Mac ``TrainSkybox.swift``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

C0 = 0.28209479177387814


def sh_coeffs(degree: int) -> int:
    return (int(degree) + 1) ** 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


def inv_sigmoid(p):
    p = np.clip(np.asarray(p, np.float32), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


@dataclass
class Skybox:
    """Equirectangular panorama at infinity. ``image`` (H, 2H, 3) uint8; basis vectors are unit length."""
    image: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    png_bytes: Optional[bytes] = None      # original file, re-written unchanged on export

    @property
    def basis(self) -> np.ndarray:
        """3×3 rows (right, up, forward): ``basis @ dir`` gives the sky-frame components."""
        return np.stack([self.right, self.up, self.forward]).astype(np.float32)

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        """Nearest-texel colour (…, 3) uint8 for world directions (…, 3): the viewer shader's math on the CPU
        (used by tests and thumbnails). Longitude atan2(right, forward), left edge = −180°; top row = up."""
        d = np.asarray(dirs, np.float64)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        k = d @ self.basis.T.astype(np.float64)
        lon = np.arctan2(k[..., 0], k[..., 2])
        lat = np.arcsin(np.clip(k[..., 1], -1, 1))
        h, w = self.image.shape[:2]
        u = (lon / (2 * np.pi) + 0.5) * w
        v = (0.5 - lat / np.pi) * h
        x = np.clip(np.floor(u).astype(int), 0, w - 1)
        y = np.clip(np.floor(v).astype(int), 0, h - 1)
        return self.image[y, x]


def sky_paths(ply_path) -> Tuple[Path, Path]:
    """``scene.ply`` → (``scene.sky.png``, ``scene.sky.json``). Compound names keep their stem
    (``scene.compressed.ply`` → ``scene.compressed.sky.png``), matching ``Path.with_suffix`` on the trainers."""
    p = Path(ply_path)
    return p.with_suffix(".sky.png"), p.with_suffix(".sky.json")


def _unit(v) -> np.ndarray:
    a = np.asarray(v, np.float64).reshape(3)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("zero vector")
    return (a / n).astype(np.float32)


def load_skybox(ply_path) -> Optional[Skybox]:
    """The sidecar next to ``ply_path``, or None when there is none (or it is unreadable)."""
    png, meta = sky_paths(ply_path)
    if not png.is_file() or not meta.is_file():
        return None
    try:
        from PIL import Image

        m = json.loads(meta.read_text(encoding="utf-8"))
        raw = png.read_bytes()
        import io

        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8)
        up, fwd = _unit(m["up"]), _unit(m["forward"])
        right = _unit(m["right"]) if "right" in m else _unit(np.cross(up, fwd))
        return Skybox(image=img, up=up, forward=fwd, right=right, png_bytes=raw)
    except Exception:
        return None


def save_skybox(sky: Skybox, ply_path) -> None:
    png, meta = sky_paths(ply_path)
    if sky.png_bytes is not None:
        png.write_bytes(sky.png_bytes)
    else:
        from PIL import Image

        Image.fromarray(np.ascontiguousarray(sky.image)).save(png)
    meta.write_text(json.dumps({
        "format": "equirectangular", "image": png.name,
        "up": [float(x) for x in sky.up], "forward": [float(x) for x in sky.forward],
        "right": [float(x) for x in sky.right],
        "longitude": "atan2(dir·right, dir·forward), left edge = −180°", "latitude": "asin(dir·up), top = +90°"}))


@dataclass
class GaussianCloud:
    positions: np.ndarray
    log_scales: np.ndarray
    rotations: np.ndarray
    opacity_logits: np.ndarray
    sh: np.ndarray
    antialiased: bool = False
    up: Optional[np.ndarray] = None          # ``comment splatstudio_up`` (scene up direction), if recorded
    skybox: Optional[Skybox] = None
    source: Optional[Path] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.opacity_logits)
        self.positions = np.ascontiguousarray(self.positions, np.float32).reshape(n, 3)
        self.log_scales = np.ascontiguousarray(self.log_scales, np.float32).reshape(n, 3)
        self.rotations = np.ascontiguousarray(self.rotations, np.float32).reshape(n, 4)
        self.opacity_logits = np.ascontiguousarray(self.opacity_logits, np.float32).reshape(n)
        self.sh = np.ascontiguousarray(self.sh, np.float32).reshape(n, -1, 3)
        if self.sh.shape[1] not in (1, 4, 9, 16):
            raise ValueError(f"unsupported SH coefficient count {self.sh.shape[1]}")

    @classmethod
    def empty(cls, n: int = 0, degree: int = 0) -> "GaussianCloud":
        rot = np.zeros((n, 4), np.float32)
        rot[:, 0] = 1
        return cls(np.zeros((n, 3)), np.zeros((n, 3)), rot, np.zeros(n), np.zeros((n, sh_coeffs(degree), 3)))

    @property
    def count(self) -> int:
        return int(self.opacity_logits.shape[0])

    def __len__(self) -> int:
        return self.count

    @property
    def sh_degree(self) -> int:
        return {1: 0, 4: 1, 9: 2, 16: 3}[self.sh.shape[1]]

    @property
    def opacities(self) -> np.ndarray:
        return sigmoid(self.opacity_logits)

    @property
    def base_colors(self) -> np.ndarray:
        """DC colour in [0, 1] (no view dependence), (N, 3)."""
        return np.clip(self.sh[:, 0] * C0 + 0.5, 0, 1)

    def with_degree(self, degree: int) -> "GaussianCloud":
        """Same scene with SH truncated / zero-padded to ``degree``."""
        k = sh_coeffs(max(0, min(3, degree)))
        sh = self.sh[:, :k]
        if sh.shape[1] < k:
            sh = np.concatenate([sh, np.zeros((self.count, k - sh.shape[1], 3), np.float32)], 1)
        return self._replace(sh=sh)

    def _replace(self, **kw) -> "GaussianCloud":
        d = dict(positions=self.positions, log_scales=self.log_scales, rotations=self.rotations,
                 opacity_logits=self.opacity_logits, sh=self.sh, antialiased=self.antialiased, up=self.up,
                 skybox=self.skybox, source=self.source, extra=dict(self.extra))
        d.update(kw)
        return GaussianCloud(**d)

    def subset(self, keep: np.ndarray) -> "GaussianCloud":
        """Splats where ``keep`` (bool mask or index array) selects them."""
        keep = np.asarray(keep)
        return self._replace(positions=self.positions[keep], log_scales=self.log_scales[keep],
                             rotations=self.rotations[keep], opacity_logits=self.opacity_logits[keep],
                             sh=self.sh[keep])

    def robust_bounds(self, percentile: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the centres ignoring the outer ``percentile`` on each side."""
        if self.count == 0:
            return np.zeros(3, np.float32), np.zeros(3, np.float32)
        stride = max(1, self.count // 200_000)
        p = self.positions[::stride]
        p = p[np.isfinite(p).all(1)]
        if len(p) == 0:
            return np.zeros(3, np.float32), np.zeros(3, np.float32)
        lo = np.quantile(p, percentile, axis=0)
        hi = np.quantile(p, 1 - percentile, axis=0)
        return lo.astype(np.float32), hi.astype(np.float32)

    def normalized_rotations(self) -> np.ndarray:
        q = self.rotations
        n = np.linalg.norm(q, axis=1, keepdims=True)
        out = np.where(n > 0, q / np.maximum(n, 1e-12), np.array([1, 0, 0, 0], np.float32))
        return out.astype(np.float32)

    def covariances(self, scale: float = 1.0) -> np.ndarray:
        """World covariance upper triangles (N, 6): xx xy xz yy yz zz, Σ = R S Sᵀ Rᵀ."""
        q = self.normalized_rotations()
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        s = np.exp(self.log_scales) * scale
        M = R * s[:, None, :]
        S = M @ np.transpose(M, (0, 2, 1))
        return np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], 1).astype(np.float32)

    def transformed(self, rotation_wxyz, translation=(0, 0, 0), scale: float = 1.0) -> "GaussianCloud":
        """Similarity transform of the whole scene (positions, orientations, scales). Higher SH bands are kept
        as they are (exact only for rotations about the view axis), as on the Mac."""
        r = np.asarray(rotation_wxyz, np.float64)
        r = r / np.linalg.norm(r)
        w, x, y, z = r
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        pos = (self.positions.astype(np.float64) @ R.T) * scale + np.asarray(translation, np.float64)
        q = self.rotations.astype(np.float64)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nq = np.stack([w * qw - x * qx - y * qy - z * qz,
                       w * qx + x * qw + y * qz - z * qy,
                       w * qy - x * qz + y * qw + z * qx,
                       w * qz + x * qy - y * qx + z * qw], 1)
        up = None if self.up is None else (R @ np.asarray(self.up, np.float64)).astype(np.float32)
        return self._replace(positions=pos.astype(np.float32), rotations=nq.astype(np.float32),
                             log_scales=self.log_scales + np.float32(np.log(scale)), up=up)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.positions, self.log_scales, self.rotations, self.opacity_logits, self.sh))
