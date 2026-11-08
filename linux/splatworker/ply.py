"""PLY reading (point clouds) and writing (INRIA 3DGS gaussian PLY).

Pure numpy; no torch required.

Gaussian PLY layout (identical to the reference 3DGS ``save_ply``)::

    x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3

* ``f_dc_*``   SH band-0 coefficients (R, G, B)
* ``f_rest_*`` higher SH bands, *channel-major*: f_rest_0..14 = R coeffs 1..15,
  f_rest_15..29 = G, f_rest_30..44 = B (the original code stores features_rest
  as [N, 15, 3] and exports ``transpose(1, 2).flatten``). Bands above the
  trained SH degree are zero-filled so the file always has 45 f_rest columns.
* ``opacity``  logit (pre-sigmoid)
* ``scale_*``  log scale
* ``rot_*``    quaternion w, x, y, z (normalised on export)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

PathLike = Union[str, os.PathLike]

NUM_REST = 45  # 15 coefficients x 3 channels for SH degree 3

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


class PLYError(ValueError):
    pass


def gaussian_property_names() -> list:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(NUM_REST)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def write_gaussian_ply(
    path: PathLike,
    means: np.ndarray,
    sh0: np.ndarray,
    shN: Optional[np.ndarray],
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    quats_wxyz: np.ndarray,
    up=None,
) -> None:
    """Write a standard 3DGS PLY atomically (tmp file + rename).

    ``up`` (the scene's up direction, from the dataset) is recorded as ``comment splatstudio_up x y z`` so
    SplatWork opens the result the right way up (other viewers ignore comments).

    Args:
        means: [N, 3]
        sh0: [N, 3] or [N, 1, 3] band-0 SH coefficients
        shN: [N, K, 3] higher-band SH coefficients (K = (deg+1)^2 - 1), or None
        opacity_logits: [N] or [N, 1]
        log_scales: [N, 3]
        quats_wxyz: [N, 4]
    """
    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    n = means.shape[0]
    f_dc = np.asarray(sh0, dtype=np.float32).reshape(n, 3)
    f_rest = np.zeros((n, NUM_REST), dtype=np.float32)
    if shN is not None:
        shN = np.asarray(shN, dtype=np.float32).reshape(n, -1, 3)
        k = shN.shape[1]
        if k > 15:
            raise PLYError("SH degree > 3 is not supported by the 3DGS PLY format")
        if k > 0:
            # [N, K, 3] -> [N, 3, K]; place each channel's K coeffs in its 15-slot block.
            chan_major = np.transpose(shN, (0, 2, 1))
            for c in range(3):
                f_rest[:, c * 15:c * 15 + k] = chan_major[:, c, :]
    opacity = np.asarray(opacity_logits, dtype=np.float32).reshape(n, 1)
    scales = np.asarray(log_scales, dtype=np.float32).reshape(n, 3)
    quats = np.asarray(quats_wxyz, dtype=np.float32).reshape(n, 4)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(norm > 0, quats / np.maximum(norm, 1e-12), np.array([1, 0, 0, 0], np.float32))
    normals = np.zeros((n, 3), dtype=np.float32)

    data = np.concatenate([means, normals, f_dc, f_rest, opacity, scales, quats], axis=1)
    data = np.ascontiguousarray(data, dtype="<f4")
    names = gaussian_property_names()
    assert data.shape[1] == len(names)

    header = ["ply", "format binary_little_endian 1.0"]
    if up is not None:
        u = [float(x) for x in np.asarray(up, dtype=np.float64).reshape(3)]
        header.append(f"comment splatstudio_up {u[0]:.6g} {u[1]:.6g} {u[2]:.6g}")
    header.append(f"element vertex {n}")
    header += [f"property float {p}" for p in names]
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header_bytes)
        data.tofile(fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Generic reader (vertex element only)
# ---------------------------------------------------------------------------

def read_ply_vertices(path: PathLike) -> Dict[str, np.ndarray]:
    """Read the ``vertex`` element of a PLY file into ``{property: array}``.

    Supports ascii, binary_little_endian and binary_big_endian. Elements that
    precede ``vertex`` must have fixed-size rows. List properties on the
    vertex element are not supported.
    """
    with open(path, "rb") as fh:
        first = fh.readline().strip()
        if first != b"ply":
            raise PLYError("not a PLY file")
        fmt = None
        elements = []  # (name, count, [(prop, dtype)] or None for list)
        while True:
            line = fh.readline()
            if not line:
                raise PLYError("unexpected end of header")
            tokens = line.decode("ascii", "replace").strip().split()
            if not tokens:
                continue
            key = tokens[0]
            if key == "format":
                fmt = tokens[1]
            elif key == "element":
                elements.append([tokens[1], int(tokens[2]), []])
            elif key == "property":
                if not elements:
                    raise PLYError("property before element")
                if tokens[1] == "list":
                    elements[-1][2].append((tokens[-1], None))
                else:
                    t = _PLY_TYPES.get(tokens[1])
                    if t is None:
                        raise PLYError(f"unknown PLY type {tokens[1]}")
                    elements[-1][2].append((tokens[2], t))
            elif key == "end_header":
                break
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise PLYError(f"unsupported PLY format {fmt}")

        endian = ">" if fmt == "binary_big_endian" else "<"
        for name, count, props in elements:
            if any(t is None for _, t in props):
                if name == "vertex":
                    raise PLYError("list properties on vertex are not supported")
                raise PLYError(f"variable-size element '{name}' before vertex is not supported")
            if name == "vertex":
                dtype = np.dtype([(p, endian + t) for p, t in props])
                if fmt == "ascii":
                    rows = []
                    for _ in range(count):
                        rows.append(fh.readline().split())
                    arr = np.array(rows, dtype=np.float64).reshape(count, len(props))
                    return {p: arr[:, i].astype(np.dtype(t)) for i, (p, t) in enumerate(props)}
                raw = fh.read(dtype.itemsize * count)
                if len(raw) < dtype.itemsize * count:
                    raise PLYError("PLY file truncated")
                rec = np.frombuffer(raw, dtype=dtype, count=count)
                return {p: rec[p].astype(rec[p].dtype.newbyteorder("=")) for p, _ in props}
            # skip a fixed-size element that precedes vertex
            if fmt == "ascii":
                for _ in range(count):
                    fh.readline()
            else:
                size = np.dtype([(p, endian + t) for p, t in props]).itemsize
                fh.seek(size * count, os.SEEK_CUR)
    raise PLYError("PLY has no vertex element")


def read_point_cloud(path: PathLike):
    """Return ``(xyz float32 [N,3], rgb float32 [N,3] in 0..1 or None)``."""
    v = read_ply_vertices(path)
    for k in ("x", "y", "z"):
        if k not in v:
            raise PLYError("point cloud has no x/y/z")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    names = [("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")]
    for r, g, b in names:
        if r in v and g in v and b in v:
            cols = np.stack([v[r], v[g], v[b]], axis=1)
            if np.issubdtype(cols.dtype, np.integer):
                maxv = 65535.0 if cols.dtype.itemsize == 2 else 255.0
                rgb = cols.astype(np.float32) / maxv
            else:
                rgb = np.clip(cols.astype(np.float32), 0.0, 1.0)
            break
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz = xyz[finite]
        rgb = rgb[finite] if rgb is not None else None
    return xyz, rgb
