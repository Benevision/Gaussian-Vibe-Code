"""Point cloud PLY IO: binary_little_endian, x y z float, red green blue uchar (PROTOCOL §1/§2)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 128, np.uint8)
    rgb = np.asarray(rgb).reshape(-1, 3)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    arr = np.zeros(len(xyz), _DTYPE)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_points_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read xyz (+rgb) vertices from a binary little-endian or ASCII PLY."""
    data = Path(path).read_bytes()
    end = data.index(b"\n", data.index(b"end_header")) + 1
    header = data[:end].decode("ascii", "replace").splitlines()
    fmt, n, props, in_vertex = "binary_little_endian", 0, [], False
    for line in header:
        el = line.split()
        if not el:
            continue
        if el[0] == "format":
            fmt = el[1]
        elif el[0] == "element":
            in_vertex = el[1] == "vertex"
            if in_vertex:
                n = int(el[2])
        elif el[0] == "property" and in_vertex:
            if el[1] == "list":
                raise ValueError("list properties in vertex element are not supported")
            props.append((el[2], _PLY_TYPES[el[1]]))
    if fmt == "ascii":
        rows = data[end:].decode("ascii").split("\n")[:n]
        vals = np.array([[float(x) for x in r.split()] for r in rows]).reshape(n, len(props))
        names = [p[0] for p in props]
        get = lambda k: vals[:, names.index(k)]
    else:
        arr = np.frombuffer(data, np.dtype(props), count=n, offset=end)
        get = lambda k: arr[k]
        names = [p[0] for p in props]
    xyz = np.stack([get("x"), get("y"), get("z")], 1).astype(np.float64)
    if all(k in names for k in ("red", "green", "blue")):
        rgb = np.stack([get("red"), get("green"), get("blue")], 1).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 128, np.uint8)
    return xyz, rgb
