"""Splat file formats, vectorised with numpy (same conventions as the Mac app's ``Formats/``):

* ``.ply``             INRIA 3DGS PLY (any SH degree; ``comment splatstudio_up`` / ``splatstudio_antialiased``);
                       a plain coloured point cloud opens as small round splats
* ``.compressed.ply``  PlayCanvas / SuperSplat compressed PLY (256-splat chunks, 11-10-11 packing, Morton order;
                       detected from the header whatever the file is called)
* ``.splat``           antimatter15 (32 bytes per splat, no SH)
* ``.spz``             Niantic SPZ, gzip; reads versions 1–3, writes version 2 (the most widely supported)

The learned skybox sidecar (``<name>.sky.png`` + ``.sky.json``) is loaded and written alongside every format.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ply import read_ply_vertices
from .splats import C0, GaussianCloud, inv_sigmoid, load_skybox, save_skybox, sh_coeffs, sigmoid

FORMATS = {
    "ply": "PLY (3DGS standard, lossless)",
    "compressed.ply": "Compressed PLY (PlayCanvas/SuperSplat, ~16 B/splat + SH)",
    "splat": ".splat (web viewers, no SH, ~32 B/splat)",
    "spz": "SPZ (compressed, ~10× smaller)",
}
OPENABLE_SUFFIXES = (".ply", ".splat", ".spz")


class FormatError(ValueError):
    pass


def format_for_path(path) -> Optional[str]:
    name = Path(path).name.lower()
    if name.endswith(".compressed.ply"):
        return "compressed.ply"
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    return ext if ext in FORMATS else None


def _atomic_write(path: Path, write) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------- PLY

def _ply_header(path) -> Tuple[List[str], int]:
    """Header lines (without ``end_header``) and the byte offset of the body."""
    with open(path, "rb") as fh:
        head = fh.read(64 * 1024)
    end = head.find(b"end_header")
    if not head.startswith(b"ply") or end < 0:
        raise FormatError("not a PLY file (no header)")
    body = end + len(b"end_header")
    if head[body:body + 1] == b"\r":
        body += 1
    if head[body:body + 1] == b"\n":
        body += 1
    lines = head[:end].decode("ascii", "replace").replace("\r", "").split("\n")
    return [l.strip() for l in lines if l.strip()], body


def is_compressed_ply(path) -> bool:
    try:
        lines, _ = _ply_header(path)
    except (OSError, FormatError):
        return False
    text = "\n".join(lines)
    return "element chunk" in text and "packed_position" in text


def _comments(lines: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for l in lines:
        t = l.split()
        if len(t) >= 2 and t[0] == "comment":
            out[t[1]] = t[2:]
    return out


def read_ply(path) -> GaussianCloud:
    lines, _ = _ply_header(path)
    comments = _comments(lines)
    v = read_ply_vertices(path)
    n = len(next(iter(v.values()))) if v else 0
    for k in ("x", "y", "z"):
        if k not in v:
            raise FormatError("PLY has no x/y/z")
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    up = None
    if "splatstudio_up" in comments:
        try:
            up = np.array([float(x) for x in comments["splatstudio_up"][:3]], np.float32)
        except ValueError:
            up = None
    antialiased = "splatstudio_antialiased" in comments and comments["splatstudio_antialiased"][:1] != ["0"]
    if "f_dc_0" not in v or "scale_0" not in v:
        return _point_cloud_as_splats(pos, v, up)
    rest = sorted((k for k in v if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    per = len(rest) // 3
    k = 1 + per
    if k not in (1, 4, 9, 16):     # tolerate odd exporters: keep the largest complete degree
        k = max(c for c in (1, 4, 9, 16) if c <= k)
        per = k - 1
    sh = np.zeros((n, k, 3), np.float32)
    for c in range(3):
        sh[:, 0, c] = v[f"f_dc_{c}"]
        for j in range(per):
            sh[:, 1 + j, c] = v[rest[c * (len(rest) // 3) + j]]
    rot = np.stack([v.get(f"rot_{i}", np.zeros(n) if i else np.ones(n)) for i in range(4)], 1)
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], 1)
    opa = v["opacity"] if "opacity" in v else np.full(n, 4.0, np.float32)
    g = GaussianCloud(pos, scales, rot, opa, sh, antialiased=antialiased, up=up)
    return _drop_non_finite(g)


def _point_cloud_as_splats(pos: np.ndarray, v: Dict[str, np.ndarray], up) -> GaussianCloud:
    n = len(pos)
    rgb = np.full((n, 3), 0.7, np.float32)
    for r, gg, b in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if r in v and gg in v and b in v:
            cols = np.stack([v[r], v[gg], v[b]], 1)
            rgb = cols.astype(np.float32) / (255.0 if np.issubdtype(cols.dtype, np.integer) else 1.0)
            break
    finite = np.isfinite(pos).all(1)
    ext = np.ptp(pos[finite], axis=0).max() if finite.any() else 1.0
    radius = max(float(ext) / max(np.cbrt(max(n, 1)) * 4, 1.0), 1e-4)
    rot = np.zeros((n, 4), np.float32)
    rot[:, 0] = 1
    sh = ((np.clip(rgb, 0, 1) - 0.5) / C0)[:, None, :]
    g = GaussianCloud(pos, np.full((n, 3), np.log(radius), np.float32), rot, np.full(n, 4.0, np.float32), sh, up=up)
    g.extra["pointCloud"] = True
    return _drop_non_finite(g)


def _drop_non_finite(g: GaussianCloud) -> GaussianCloud:
    ok = np.isfinite(g.positions).all(1) & np.isfinite(g.log_scales).all(1) & np.isfinite(g.opacity_logits)
    return g if ok.all() else g.subset(ok)


def write_ply(g: GaussianCloud, path, sh_degree: Optional[int] = None) -> None:
    deg = g.sh_degree if sh_degree is None else max(0, min(3, int(sh_degree)))
    g = g.with_degree(deg)
    n, k = g.count, sh_coeffs(deg)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * (k - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rest = np.transpose(g.sh[:, 1:], (0, 2, 1)).reshape(n, -1)       # channel-major, as INRIA save_ply
    data = np.concatenate([g.positions, np.zeros((n, 3), np.float32), g.sh[:, 0], rest,
                           g.opacity_logits[:, None], g.log_scales, g.normalized_rotations()], 1)
    data = np.ascontiguousarray(data, "<f4")
    header = ["ply", "format binary_little_endian 1.0", "comment Generated by SplatWork Node"]
    if g.up is not None:
        u = [float(x) for x in g.up]
        header.append(f"comment splatstudio_up {u[0]:.6g} {u[1]:.6g} {u[2]:.6g}")
    if g.antialiased:
        header.append("comment splatstudio_antialiased 1")
    header.append(f"element vertex {n}")
    header += [f"property float {p}" for p in names]
    header.append("end_header")
    hb = ("\n".join(header) + "\n").encode("ascii")

    def w(fh):
        fh.write(hb)
        data.tofile(fh)
    _atomic_write(Path(path), w)


# -------------------------------------------------------------------------------------------- compressed PLY

CHUNK = 256
_CHUNK_PROPS = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
                "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
                "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"]
_VERTEX_PROPS = ["packed_position", "packed_rotation", "packed_scale", "packed_color"]
_PLY_NP = {"float": "f4", "float32": "f4", "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
           "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2",
           "short": "i2", "int16": "i2", "double": "f8", "float64": "f8"}


def _part1by2(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.uint32) & 0x3FF
    v = (v ^ (v << 16)) & 0xFF0000FF
    v = (v ^ (v << 8)) & 0x0300F00F
    v = (v ^ (v << 4)) & 0x030C30C3
    v = (v ^ (v << 2)) & 0x09249249
    return v


def _morton_codes(p: np.ndarray) -> Optional[np.ndarray]:
    mn, mx = p.min(0), p.max(0)
    ln = mx - mn
    if not np.isfinite(ln).all() or (ln == 0).all():
        return None
    mul = np.where(ln == 0, 0, 1024 / np.where(ln == 0, 1, ln))
    q = np.clip(np.nan_to_num((p - mn) * mul, nan=0.0), 0, 1023).astype(np.uint32)
    return (_part1by2(q[:, 2]) << 2) + (_part1by2(q[:, 1]) << 1) + _part1by2(q[:, 0])


def morton_order(positions: np.ndarray) -> np.ndarray:
    """Indices in Morton (Z-order) of the centres; buckets of > 256 equal codes are refined recursively
    (port of splat-transform's ``sortMortonStrided``)."""
    idx = np.arange(len(positions), dtype=np.int64)

    def sort(ix: np.ndarray) -> np.ndarray:
        if len(ix) == 0:
            return ix
        codes = _morton_codes(positions[ix].astype(np.float64))
        if codes is None:
            return ix
        o = np.argsort(codes, kind="stable")
        ix, codes = ix[o], codes[o]
        uniq, start, counts = np.unique(codes, return_index=True, return_counts=True)
        for s, c in zip(start[counts > CHUNK], counts[counts > CHUNK]):
            ix[s:s + c] = sort(ix[s:s + c])
        return ix
    return sort(idx)


def _unorm(v: np.ndarray, bits: int) -> np.ndarray:
    t = (1 << bits) - 1
    return (v & t).astype(np.float32) / t


def _pack_unorm(v: np.ndarray, bits: int) -> np.ndarray:
    t = float((1 << bits) - 1)
    return np.clip(np.floor(v * t + 0.5), 0, t).astype(np.uint32)


def _norm01(x, lo, hi):
    rng = hi - lo
    out = np.where(rng < 1e-5, 0.0, (x - lo) / np.where(rng < 1e-5, 1, rng))
    return np.where(x <= lo, 0.0, np.where(x >= hi, 1.0, out))


def write_compressed_ply(g: GaussianCloud, path, sh_degree: Optional[int] = None) -> None:
    deg = g.sh_degree if sh_degree is None else max(0, min(3, int(sh_degree)))
    g = g.with_degree(deg)
    n, coeffs = g.count, sh_coeffs(deg) - 1
    nchunks = (n + CHUNK - 1) // CHUNK
    order = morton_order(g.positions)
    pad = nchunks * CHUNK - n
    padded = np.concatenate([order, np.repeat(order[-1:], pad)]) if n and pad else order

    pos = g.positions[padded].astype(np.float64).reshape(nchunks, CHUNK, 3)
    scl = g.log_scales[padded].astype(np.float64).reshape(nchunks, CHUNK, 3)
    col = (g.sh[padded, 0].astype(np.float64) * C0 + 0.5).reshape(nchunks, CHUNK, 3)
    lo_p, hi_p = pos.min(1), pos.max(1)
    lo_s, hi_s = np.clip(scl.min(1), -20, 20), np.clip(scl.max(1), -20, 20)
    lo_c, hi_c = col.min(1), col.max(1)
    chunk = np.concatenate([lo_p, hi_p, lo_s, hi_s, lo_c, hi_c], 1).astype("<f4")

    def per_splat(a):                    # (nchunks, 3) → (n, 3) for the real (unpadded) splats
        return np.repeat(a, CHUNK, 0)[:n]

    pn = _norm01(pos.reshape(-1, 3)[:n], per_splat(lo_p), per_splat(hi_p))
    sn = _norm01(scl.reshape(-1, 3)[:n], per_splat(lo_s), per_splat(hi_s))
    cn = _norm01(col.reshape(-1, 3)[:n], per_splat(lo_c), per_splat(hi_c))
    pack3 = lambda a: (_pack_unorm(a[:, 0], 11) << 21) | (_pack_unorm(a[:, 1], 10) << 11) | _pack_unorm(a[:, 2], 11)  # noqa: E731

    q = g.rotations[order].astype(np.float64)
    ln = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(ln == 0, np.array([0, 0, 0, 1.0]), q / np.where(ln == 0, 1, ln))
    largest = np.argmax(np.abs(q), 1)
    q = np.where(q[np.arange(n), largest][:, None] < 0, -q, q)
    rot = largest.astype(np.uint32)
    norm = np.sqrt(2.0) * 0.5
    for i in range(4):
        sel = largest != i
        rot = np.where(sel, (rot << 10) | _pack_unorm(q[:, i] * norm + 0.5, 10), rot)
    opa = sigmoid(g.opacity_logits[order]).astype(np.float64)
    color = (_pack_unorm(cn[:, 0], 8) << 24) | (_pack_unorm(cn[:, 1], 8) << 16) | (_pack_unorm(cn[:, 2], 8) << 8) \
        | _pack_unorm(opa, 8)
    vertex = np.stack([pack3(pn), rot, pack3(sn), color], 1).astype("<u4")

    header = ["ply", "format binary_little_endian 1.0", "comment Generated by SplatWork Node"]
    if g.antialiased:
        header.append("comment splatstudio_antialiased 1")
    header.append(f"element chunk {nchunks}")
    header += [f"property float {p}" for p in _CHUNK_PROPS]
    header.append(f"element vertex {n}")
    header += [f"property uint {p}" for p in _VERTEX_PROPS]
    shb = None
    if coeffs:
        header.append(f"element sh {n}")
        header += [f"property uchar f_rest_{i}" for i in range(coeffs * 3)]
        rest = np.transpose(g.sh[order, 1:].astype(np.float64), (0, 2, 1)).reshape(n, -1)   # channel-major
        shb = np.clip(np.trunc((rest / 8 + 0.5) * 256), 0, 255).astype(np.uint8)
    header.append("end_header")
    hb = ("\n".join(header) + "\n").encode("ascii")

    def w(fh):
        fh.write(hb)
        chunk.tofile(fh)
        vertex.tofile(fh)
        if shb is not None:
            shb.tofile(fh)
    _atomic_write(Path(path), w)


def read_compressed_ply(path) -> GaussianCloud:
    lines, body = _ply_header(path)
    els: List[list] = []
    antialiased = False
    for l in lines:
        t = l.split()
        if t[0] == "format" and t[1] != "binary_little_endian":
            raise FormatError("compressed PLY must be binary little endian")
        if t[0] == "comment" and len(t) > 1 and t[1] == "splatstudio_antialiased":
            antialiased = True
        if t[0] == "element":
            els.append([t[1], int(t[2]), []])
        if t[0] == "property":
            if t[1] == "list" or not els:
                raise FormatError("unexpected property in compressed PLY")
            els[-1][2].append((t[2], _PLY_NP[t[1]]))
    raw = np.memmap(path, np.uint8, mode="r")
    arrays = {}
    off = body
    for name, count, props in els:
        dt = np.dtype([(p, "<" + t) for p, t in props])
        size = dt.itemsize * count
        if off + size > len(raw):
            raise FormatError("compressed PLY truncated")
        arrays[name] = np.frombuffer(raw[off:off + size].tobytes(), dt, count)
        off += size
    if "chunk" not in arrays or "vertex" not in arrays:
        raise FormatError("not a compressed PLY (no chunk/vertex elements)")
    ch, vx = arrays["chunk"], arrays["vertex"]
    n = len(vx)
    if (n + CHUNK - 1) // CHUNK != len(ch):
        raise FormatError("chunk count does not match vertex count")
    ci = np.arange(n) // CHUNK
    c = lambda k: ch[k][ci].astype(np.float32)  # noqa: E731
    lerp = lambda a, b, t: a * (1 - t) + b * t  # noqa: E731
    pp, pr, ps, pc = (vx[k].astype(np.uint32) for k in _VERTEX_PROPS)
    pos = np.stack([lerp(c("min_x"), c("max_x"), _unorm(pp >> 21, 11)),
                    lerp(c("min_y"), c("max_y"), _unorm(pp >> 11, 10)),
                    lerp(c("min_z"), c("max_z"), _unorm(pp, 11))], 1)
    scl = np.stack([lerp(c("min_scale_x"), c("max_scale_x"), _unorm(ps >> 21, 11)),
                    lerp(c("min_scale_y"), c("max_scale_y"), _unorm(ps >> 11, 10)),
                    lerp(c("min_scale_z"), c("max_scale_z"), _unorm(ps, 11))], 1)
    norm = 1 / (np.sqrt(2.0) * 0.5)
    a = (_unorm(pr >> 20, 10) - 0.5) * norm
    b = (_unorm(pr >> 10, 10) - 0.5) * norm
    cc = (_unorm(pr, 10) - 0.5) * norm
    m = np.sqrt(np.maximum(0, 1 - (a * a + b * b + cc * cc)))
    largest = pr >> 30
    rot = np.select([largest[:, None] == 0, largest[:, None] == 1, largest[:, None] == 2],
                    [np.stack([m, a, b, cc], 1), np.stack([a, m, b, cc], 1), np.stack([a, b, m, cc], 1)],
                    np.stack([a, b, cc, m], 1))
    rgb = np.stack([_unorm(pc >> 24, 8), _unorm(pc >> 16, 8), _unorm(pc >> 8, 8)], 1)
    if "min_r" in ch.dtype.names:
        rgb = np.stack([lerp(c("min_r"), c("max_r"), rgb[:, 0]), lerp(c("min_g"), c("max_g"), rgb[:, 1]),
                        lerp(c("min_b"), c("max_b"), rgb[:, 2])], 1)
    opa = inv_sigmoid(_unorm(pc, 8))
    coeffs = 0
    if "sh" in arrays:
        shel = arrays["sh"]
        if len(shel) != n or len(shel.dtype.names) not in (9, 24, 45):
            raise FormatError("bad sh element")
        coeffs = len(shel.dtype.names) // 3
    sh = np.zeros((n, 1 + coeffs, 3), np.float32)
    sh[:, 0] = (rgb - 0.5) / C0
    if coeffs:
        shel = arrays["sh"]
        for chn in range(3):
            for k in range(coeffs):
                v = shel[f"f_rest_{chn * coeffs + k}"].astype(np.float32)
                nv = np.where(v == 0, 0.0, np.where(v == 255, 1.0, (v + 0.5) / 256))
                sh[:, 1 + k, chn] = (nv - 0.5) * 8
    return GaussianCloud(pos, scl, rot, opa, sh, antialiased=antialiased)


# ----------------------------------------------------------------------------------------------------- .splat

def read_dotsplat(path) -> GaussianCloud:
    raw = np.fromfile(path, np.uint8)
    if len(raw) % 32:
        raise FormatError(".splat size is not a multiple of 32")
    rec = raw.reshape(-1, 32)
    f = rec[:, :24].copy().view("<f4").reshape(-1, 6)
    pos = f[:, :3]
    scl = np.log(np.maximum(f[:, 3:6], 1e-8))
    col = rec[:, 24:28].astype(np.float32) / 255
    rot = (rec[:, 28:32].astype(np.float32) - 128) / 128
    sh = ((col[:, :3] - 0.5) / C0)[:, None, :]
    return _drop_non_finite(GaussianCloud(pos, scl, rot, inv_sigmoid(col[:, 3]), sh))


def write_dotsplat(g: GaussianCloud, path) -> None:
    n = g.count
    importance = np.exp(g.log_scales.sum(1)) * g.opacities
    order = np.argsort(-importance, kind="stable")
    out = np.zeros((n, 32), np.uint8)
    f = np.concatenate([g.positions[order], np.exp(g.log_scales[order])], 1).astype("<f4")
    out[:, :24] = f.view(np.uint8).reshape(n, 24)
    out[:, 24:27] = np.clip(np.floor(g.base_colors[order] * 255 + 0.5), 0, 255)
    out[:, 27] = np.clip(np.floor(g.opacities[order] * 255 + 0.5), 0, 255)
    out[:, 28:32] = np.clip(np.floor(g.normalized_rotations()[order] * 128 + 128), 0, 255)
    _atomic_write(Path(path), lambda fh: out.tofile(fh))


# ------------------------------------------------------------------------------------------------------- .spz

SPZ_MAGIC = 0x5053474E
SPZ_COLOR_SCALE = 0.15
# SPZ stores RUB, splat PLYs RDF: these SH coefficient indices (0-based in the 15 higher ones) flip sign.
_SPZ_FLIP = np.array([1 if j in (0, 1, 3, 6, 8, 10, 11, 13) else 0 for j in range(15)], bool)


def read_spz(path) -> GaussianCloud:
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if len(data) < 16:
        raise FormatError("spz header")
    magic, version, n = np.frombuffer(data[:12], "<u4")
    if magic != SPZ_MAGIC:
        raise FormatError("spz magic")
    if not 1 <= version <= 3:
        raise FormatError(f"spz version {version}")
    n = int(n)
    degree, frac, flags = data[12], data[13], data[14]
    dim = [0, 3, 8, 15][min(degree, 3)]
    pos_b = 6 if version == 1 else 9
    rot_b = 4 if version >= 3 else 3
    need = 16 + n * (pos_b + 1 + 3 + 3 + rot_b + dim * 3)
    if len(data) < need:
        raise FormatError("spz truncated")
    u8 = np.frombuffer(data, np.uint8)
    o = 16
    if version == 1:
        pos = u8[o:o + n * 6].copy().view("<f2").reshape(n, 3).astype(np.float32)
    else:
        b = u8[o:o + n * 9].reshape(n, 3, 3).astype(np.int32)
        v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        pos = v.astype(np.float32) / float(1 << frac)
    o += n * pos_b
    alpha = u8[o:o + n].astype(np.float32) / 255
    o += n
    col = u8[o:o + n * 3].reshape(n, 3).astype(np.float32)
    o += n * 3
    scl = u8[o:o + n * 3].reshape(n, 3).astype(np.float32) / 16 - 10
    o += n * 3
    if version >= 3:
        comp = u8[o:o + n * 4].copy().view("<u4").astype(np.uint64)
        largest = (comp >> 30).astype(np.int64)
        q = np.zeros((n, 4), np.float64)          # x y z w
        mask = (1 << 9) - 1
        rem = comp.copy()
        ss = np.zeros(n)
        for cidx in (3, 2, 1, 0):
            sel = largest != cidx
            mag = (rem & mask).astype(np.float64)
            neg = ((rem >> 9) & 1).astype(bool)
            val = np.sqrt(0.5) * mag / mask
            val = np.where(neg, -val, val)
            q[:, cidx] = np.where(sel, val, q[:, cidx])
            ss += np.where(sel, val * val, 0)
            rem = np.where(sel, rem >> 10, rem)
        q[np.arange(n), largest] = np.sqrt(np.maximum(0, 1 - ss))
    else:
        xyz = u8[o:o + n * 3].reshape(n, 3).astype(np.float64) / 127.5 - 1
        w = np.sqrt(np.maximum(0, 1 - (xyz ** 2).sum(1)))
        q = np.concatenate([xyz, w[:, None]], 1)
    o += n * rot_b
    # RUB → RDF: flip y and z. Position (x, −y, −z); quaternion (w, x, −y, −z).
    pos[:, 1:] *= -1
    rot = np.stack([q[:, 3], q[:, 0], -q[:, 1], -q[:, 2]], 1).astype(np.float32)
    sh = np.zeros((n, 1 + dim, 3), np.float32)
    sh[:, 0] = (col / 255 - 0.5) / SPZ_COLOR_SCALE
    if dim:
        rest = (u8[o:o + n * dim * 3].reshape(n, dim, 3).astype(np.float32) - 128) / 128
        sign = np.where(_SPZ_FLIP[:dim], -1.0, 1.0).astype(np.float32)
        sh[:, 1:] = rest * sign[None, :, None]
    return GaussianCloud(pos, scl, rot, inv_sigmoid(alpha), sh, antialiased=bool(flags & 1))


def write_spz(g: GaussianCloud, path, sh_degree: Optional[int] = None, fractional_bits: int = 12) -> None:
    deg = g.sh_degree if sh_degree is None else max(0, min(3, int(sh_degree)))
    g = g.with_degree(deg)
    n, dim = g.count, [0, 3, 8, 15][deg]
    head = np.zeros(16, np.uint8)
    head[:12] = np.array([SPZ_MAGIC, 2, n], "<u4").view(np.uint8)
    head[12], head[13], head[14] = deg, fractional_bits, 1 if g.antialiased else 0
    p = g.positions.astype(np.float64).copy()
    p[:, 1:] *= -1
    qv = np.clip(np.round(p * (1 << fractional_bits)), -0x800000, 0x7FFFFF).astype(np.int64) & 0xFFFFFF
    pos = np.stack([qv & 0xFF, (qv >> 8) & 0xFF, (qv >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, 9)
    q8 = lambda v: np.clip(np.round(v), 0, 255).astype(np.uint8)  # noqa: E731
    alpha = q8(g.opacities * 255)
    col = q8(g.sh[:, 0] * SPZ_COLOR_SCALE * 255 + 127.5)
    scl = q8((g.log_scales + 10) * 16)
    r = g.rotations.astype(np.float64)
    q = np.stack([r[:, 1], -r[:, 2], -r[:, 3], r[:, 0]], 1)          # RDF → RUB, x y z w
    ln = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(ln > 0, q / np.where(ln > 0, ln, 1), np.array([0, 0, 0, 1.0]))
    q = np.where(q[:, 3:4] < 0, -q, q)
    rot = q8((q[:, :3] + 1) * 127.5)
    parts = [head, pos.ravel(), alpha, col.ravel(), scl.ravel(), rot.ravel()]
    if dim:
        sign = np.where(_SPZ_FLIP[:dim], -1.0, 1.0)
        v = sign[None, :, None] * g.sh[:, 1:].astype(np.float64) * 128 + 128
        bucket = np.where(np.arange(dim) < 3, 8.0, 16.0)[None, :, None]
        v = np.floor((v + bucket / 2) / bucket) * bucket
        parts.append(np.clip(v, 0, 255).astype(np.uint8).ravel())
    blob = gzip.compress(np.concatenate(parts).tobytes(), compresslevel=6)
    _atomic_write(Path(path), lambda fh: fh.write(blob))


# --------------------------------------------------------------------------------------------------- dispatch

def load(path, with_skybox: bool = True) -> GaussianCloud:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ply":
        g = read_compressed_ply(path) if is_compressed_ply(path) else read_ply(path)
    elif ext == ".splat":
        g = read_dotsplat(path)
    elif ext == ".spz":
        g = read_spz(path)
    else:
        raise FormatError(f"unsupported file type {ext or path.name}")
    g.source = path
    if with_skybox:
        g.skybox = load_skybox(path)
    return g


def save(g: GaussianCloud, path, fmt: Optional[str] = None, sh_degree: Optional[int] = None,
         with_skybox: bool = True) -> Path:
    """Write ``g`` as ``fmt`` (default: from the file name) and its skybox sidecar next to it."""
    path = Path(path)
    fmt = fmt or format_for_path(path) or "ply"
    if fmt == "ply":
        write_ply(g, path, sh_degree)
    elif fmt == "compressed.ply":
        write_compressed_ply(g, path, sh_degree)
    elif fmt == "splat":
        write_dotsplat(g, path)
    elif fmt == "spz":
        write_spz(g, path, sh_degree)
    else:
        raise FormatError(f"unknown format {fmt}")
    if with_skybox and g.skybox is not None:
        save_skybox(g.skybox, path)
    return path


def export_name(stem: str, fmt: str) -> str:
    return f"{stem}.{fmt}"
