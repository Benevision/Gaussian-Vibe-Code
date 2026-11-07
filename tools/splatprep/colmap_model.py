"""COLMAP sparse model (cameras / images / points3D) — text + binary IO.

Pure numpy, no pycolmap needed.  Pixel convention is COLMAP's (the centre of
the top-left pixel is (0.5, 0.5)).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import invert_pose, qvec_to_rotmat, rotmat_to_qvec

# id -> (name, num_params)
CAMERA_MODELS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS: Dict[str, int] = {v[0]: k for k, v in CAMERA_MODELS.items()}
FISHEYE_MODELS = {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE", "FOV"}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def model_id(self) -> int:
        return CAMERA_MODEL_IDS[self.model]

    def pinhole(self) -> Tuple[float, float, float, float]:
        """(fx, fy, cx, cy) ignoring distortion."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    def distortion(self) -> np.ndarray:
        n_intr = 3 if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE") else 4
        return np.asarray(self.params[n_intr:], float)

    def has_distortion(self, eps: float = 1e-7) -> bool:
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return False
        if self.model in FISHEYE_MODELS:
            return True  # fisheye projection is not pinhole even with k = 0
        return bool(np.any(np.abs(self.distortion()) > eps))

    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.pinhole()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # world-to-camera rotation (w, x, y, z)
    tvec: np.ndarray  # world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def w2c(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = qvec_to_rotmat(self.qvec)
        T[:3, 3] = self.tvec
        return T

    def c2w(self) -> np.ndarray:
        return invert_pose(self.w2c())

    def set_c2w(self, c2w: np.ndarray) -> None:
        w2c = invert_pose(np.asarray(c2w, float))
        self.qvec = rotmat_to_qvec(w2c[:3, :3])
        self.tvec = w2c[:3, 3].copy()

    def center(self) -> np.ndarray:
        return self.c2w()[:3, 3]


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float = 0.0
    image_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    point2D_idxs: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))


@dataclass
class Reconstruction:
    cameras: Dict[int, Camera] = field(default_factory=dict)
    images: Dict[int, Image] = field(default_factory=dict)
    points3D: Dict[int, Point3D] = field(default_factory=dict)

    # convenience -------------------------------------------------------
    def image_by_name(self) -> Dict[str, Image]:
        return {im.name: im for im in self.images.values()}

    def c2w_by_name(self) -> Dict[str, np.ndarray]:
        return {im.name: im.c2w() for im in self.images.values()}

    def points_xyz_rgb(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.points3D:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        pts = list(self.points3D.values())
        return np.stack([p.xyz for p in pts]), np.stack([p.rgb for p in pts]).astype(np.uint8)

    def transform(self, sim3) -> None:
        """Apply a Sim3 (geometry.Sim3) to all poses and points in place."""
        for im in self.images.values():
            im.set_c2w(sim3.apply_c2w(im.c2w()))
        for p in self.points3D.values():
            p.xyz = sim3.apply_points(p.xyz[None])[0]

    def stats(self) -> dict:
        tracks = [len(p.image_ids) for p in self.points3D.values()]
        errs = [p.error for p in self.points3D.values() if p.error > 0]
        n_obs = sum(tracks)
        return {
            "numCameras": len(self.cameras),
            "numRegistered": len(self.images),
            "numPoints3D": len(self.points3D),
            "numObservations": int(n_obs),
            "meanTrackLength": float(np.mean(tracks)) if tracks and n_obs else None,
            "meanReprojError": float(np.mean(errs)) if errs else self.compute_reprojection_error(),
        }

    def compute_reprojection_error(self) -> Optional[float]:
        """Mean reprojection error over all observations (pinhole + radial terms ignored)."""
        total, count = 0.0, 0
        for im in self.images.values():
            if im.xys.size == 0:
                continue
            cam = self.cameras[im.camera_id]
            mask = im.point3D_ids >= 0
            if not mask.any():
                continue
            ids = im.point3D_ids[mask]
            valid = [i for i, pid in enumerate(ids) if int(pid) in self.points3D]
            if not valid:
                continue
            X = np.stack([self.points3D[int(ids[i])].xyz for i in valid])
            uv = im.xys[mask][valid]
            proj = project_points(cam, im.w2c(), X)
            e = np.linalg.norm(proj - uv, axis=1)
            total += float(e.sum())
            count += len(e)
        return total / count if count else None


def project_points(cam: Camera, w2c: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(Xc[:, 2:3], 1e-9)
    xn = Xc[:, :2] / z
    d = cam.distortion()
    if cam.model in ("SIMPLE_RADIAL", "RADIAL") and d.size:
        r2 = (xn**2).sum(1, keepdims=True)
        k1 = d[0]
        k2 = d[1] if d.size > 1 else 0.0
        xn = xn * (1 + k1 * r2 + k2 * r2 * r2)
    elif cam.model in ("OPENCV", "FULL_OPENCV") and d.size >= 4:
        k1, k2, p1, p2 = d[:4]
        x, y = xn[:, 0:1], xn[:, 1:2]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        xn = np.hstack([xd, yd])
    fx, fy, cx, cy = cam.pinhole()
    return np.stack([fx * xn[:, 0] + cx, fy * xn[:, 1] + cy], axis=1)


# --------------------------------------------------------------------------
# text IO


def read_text_model(path: Path) -> Reconstruction:
    path = Path(path)
    rec = Reconstruction()
    for line in _lines(path / "cameras.txt"):
        el = line.split()
        cid, model, w, h = int(el[0]), el[1], int(el[2]), int(el[3])
        rec.cameras[cid] = Camera(cid, model, w, h, np.array([float(x) for x in el[4:]]))
    lines = _lines(path / "images.txt")
    for i in range(0, len(lines), 2):
        el = lines[i].split()
        iid = int(el[0])
        q = np.array([float(x) for x in el[1:5]])
        t = np.array([float(x) for x in el[5:8]])
        cid = int(el[8])
        name = " ".join(el[9:])
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        xys = np.array([[float(pts[k]), float(pts[k + 1])] for k in range(0, len(pts), 3)]).reshape(-1, 2)
        ids = np.array([int(pts[k + 2]) for k in range(0, len(pts), 3)], np.int64)
        rec.images[iid] = Image(iid, q, t, cid, name, xys, ids)
    if (path / "points3D.txt").exists():
        for line in _lines(path / "points3D.txt"):
            el = line.split()
            pid = int(el[0])
            xyz = np.array([float(x) for x in el[1:4]])
            rgb = np.array([int(x) for x in el[4:7]], np.uint8)
            err = float(el[7])
            tr = [int(x) for x in el[8:]]
            rec.points3D[pid] = Point3D(pid, xyz, rgb, err, np.array(tr[0::2], np.int32), np.array(tr[1::2], np.int32))
    return rec


def _lines(p: Path) -> List[str]:
    """Non-comment lines.  images.txt has an (possibly empty) 2nd line per image, so keep empties there."""
    raw = Path(p).read_text().splitlines()
    out = [l for l in raw if not l.startswith("#")]
    if Path(p).name != "images.txt":
        out = [l for l in out if l.strip()]
    else:
        while out and not out[-1].strip() and len(out) % 2 == 1:
            out.pop()
    return out


def write_text_model(rec: Reconstruction, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in sorted(rec.cameras.values(), key=lambda c: c.id):
            f.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(repr(float(x)) for x in c.params) + "\n")
    with open(path / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in sorted(rec.images.values(), key=lambda i: i.id):
            f.write(
                f"{im.id} "
                + " ".join(repr(float(x)) for x in im.qvec)
                + " "
                + " ".join(repr(float(x)) for x in im.tvec)
                + f" {im.camera_id} {im.name}\n"
            )
            f.write(" ".join(f"{float(x)!r} {float(y)!r} {int(pid)}" for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")
    with open(path / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p in sorted(rec.points3D.values(), key=lambda p: p.id):
            tr = " ".join(f"{int(a)} {int(b)}" for a, b in zip(p.image_ids, p.point2D_idxs))
            f.write(
                f"{p.id} {' '.join(repr(float(x)) for x in p.xyz)} {' '.join(str(int(c)) for c in p.rgb)} {float(p.error)!r} {tr}".rstrip()
                + "\n"
            )


# --------------------------------------------------------------------------
# binary IO


def _read(f, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))


def read_binary_model(path: Path) -> Reconstruction:
    path = Path(path)
    rec = Reconstruction()
    with open(path / "cameras.bin", "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _read(f, "iiQQ")
            name, npar = CAMERA_MODELS[mid]
            params = np.array(_read(f, "d" * npar))
            rec.cameras[cid] = Camera(cid, name, int(w), int(h), params)
    with open(path / "images.bin", "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "idddddddi")
            iid, q, t, cid = vals[0], np.array(vals[1:5]), np.array(vals[5:8]), vals[8]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            (npts,) = _read(f, "Q")
            if npts:
                data = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
                xys = np.stack([data["x"], data["y"]], 1)
                ids = data["id"].astype(np.int64)
            else:
                xys, ids = np.zeros((0, 2)), np.zeros((0,), np.int64)
            rec.images[iid] = Image(iid, q, t, cid, name.decode("utf-8"), xys, ids)
    pfile = path / "points3D.bin"
    if pfile.exists():
        with open(pfile, "rb") as f:
            (n,) = _read(f, "Q")
            for _ in range(n):
                vals = _read(f, "QdddBBBd")
                pid = int(vals[0])
                (tl,) = _read(f, "Q")
                tr = np.frombuffer(f.read(8 * tl), dtype="<i4").reshape(-1, 2) if tl else np.zeros((0, 2), np.int32)
                rec.points3D[pid] = Point3D(
                    pid, np.array(vals[1:4]), np.array(vals[4:7], np.uint8), float(vals[7]), tr[:, 0].copy(), tr[:, 1].copy()
                )
    return rec


def write_binary_model(rec: Reconstruction, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.cameras)))
        for c in sorted(rec.cameras.values(), key=lambda c: c.id):
            f.write(struct.pack("<iiQQ", c.id, c.model_id, c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *[float(x) for x in c.params]))
    with open(path / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.images)))
        for im in sorted(rec.images.values(), key=lambda i: i.id):
            f.write(struct.pack("<i", im.id))
            f.write(struct.pack("<7d", *[float(x) for x in im.qvec], *[float(x) for x in im.tvec]))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im.xys)))
            if len(im.xys):
                arr = np.zeros(len(im.xys), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
                arr["x"], arr["y"], arr["id"] = im.xys[:, 0], im.xys[:, 1], im.point3D_ids
                f.write(arr.tobytes())
    with open(path / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rec.points3D)))
        for p in sorted(rec.points3D.values(), key=lambda p: p.id):
            f.write(struct.pack("<Q3d3Bd", p.id, *[float(x) for x in p.xyz], *[int(c) for c in p.rgb], float(p.error)))
            f.write(struct.pack("<Q", len(p.image_ids)))
            if len(p.image_ids):
                f.write(np.stack([p.image_ids, p.point2D_idxs], 1).astype("<i4").tobytes())


def read_model(path: Path) -> Reconstruction:
    path = Path(path)
    if (path / "cameras.bin").exists():
        return read_binary_model(path)
    if (path / "cameras.txt").exists():
        return read_text_model(path)
    raise FileNotFoundError(f"no COLMAP model in {path}")


def find_models(sparse_root: Path) -> List[Path]:
    """All model directories below ``sparse_root`` (sparse/, sparse/0, sparse/1 …)."""
    sparse_root = Path(sparse_root)
    out = []
    for p in [sparse_root] + sorted(sparse_root.glob("*")):
        if p.is_dir() and ((p / "cameras.bin").exists() or (p / "cameras.txt").exists()):
            out.append(p)
    return out


def read_largest_model(sparse_root: Path) -> Optional[Reconstruction]:
    best = None
    for p in find_models(sparse_root):
        rec = read_model(p)
        if best is None or len(rec.images) > len(best.images):
            best = rec
    return best
