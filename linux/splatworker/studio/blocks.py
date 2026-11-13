"""Block-parallel training across machines (PROTOCOL.md §7), the Mac's ``BlockTraining.swift`` /
``ComputeScheduler`` in numpy:

* :func:`partition` — K spatial blocks by recursive weighted-median splits along the longer horizontal axis
  (points + camera centres, cameras weigh 50 points), never split vertically; each block gets the cameras that
  see its expanded box. The outer faces stay open so merging drops nothing outside the scene bounds.
* :func:`block_config` — the block's ``region`` + ``cameraIds`` and its share of the splat budget.
* :func:`distribute` — greedy longest-first assignment proportional to throughput; a block only goes to devices
  it fits (else to the fastest one).
* :func:`merge_plys` — the merged scene keeps from every block the splats whose centre the block *owns*
  (un-expanded box, open outer faces); streamed from disk in chunks so memory stays bounded.

The worker itself already honours ``region`` (seeds points, picks cameras with ``camera_sees_box``) and
``cameraIds`` (``splatworker.dataset.select_cameras``); nothing here needs torch.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Block:
    index: int
    min: List[float]
    max: List[float]
    margin: float = 0.15
    open_min: List[bool] = field(default_factory=lambda: [True, True, True])
    open_max: List[bool] = field(default_factory=lambda: [True, True, True])
    camera_ids: List[int] = field(default_factory=list)
    weight: float = 0.0

    @property
    def expanded(self) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = np.asarray(self.min, np.float64), np.asarray(self.max, np.float64)
        s = (hi - lo) * self.margin
        return lo - s, hi + s

    @property
    def region(self) -> Dict[str, Any]:
        return {"min": [float(x) for x in self.min], "max": [float(x) for x in self.max], "margin": float(self.margin)}

    def owns(self, xyz: np.ndarray) -> np.ndarray:
        """Boolean mask: splat centres (N, 3) this block keeps when merging."""
        xyz = np.asarray(xyz)
        keep = np.ones(len(xyz), bool)
        for a in range(3):
            if not self.open_min[a]:
                keep &= xyz[:, a] >= self.min[a]
            if not self.open_max[a]:
                keep &= xyz[:, a] < self.max[a]
        return keep

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "min": self.min, "max": self.max, "margin": self.margin,
                "openMin": self.open_min, "openMax": self.open_max, "cameras": len(self.camera_ids),
                "weight": self.weight}


# --------------------------------------------------------------------------------------------- cameras


@dataclass
class Cam:
    id: int
    c2w: np.ndarray       # 4×4 OpenCV
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    holdout: bool = False

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]


def cameras_from_dataset(doc: Dict[str, Any]) -> List[Cam]:
    out = []
    for c in doc.get("cameras") or []:
        try:
            out.append(Cam(int(c["id"]), np.asarray(c["cameraToWorld"], np.float64).reshape(4, 4),
                           float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]),
                           int(c["width"]), int(c["height"]), bool(c.get("holdout", False))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def axes(up: Sequence[float]) -> Tuple[List[int], int]:
    """The two horizontal axes (most orthogonal to ``up``) and the vertical one."""
    a = np.abs(np.asarray(up, np.float64))
    v = 0 if a[0] >= a[1] and a[0] >= a[2] else (1 if a[1] >= a[2] else 2)
    return [i for i in range(3) if i != v], v


def _project(cams: List[Cam], pts: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per camera, how many of ``pts`` land in front of it inside the image: (C,) int."""
    out = np.zeros(len(cams), np.int64)
    if not len(pts) or not cams:
        return out
    P = np.c_[pts, np.ones(len(pts))]
    for s in range(0, len(cams), chunk):
        cs = cams[s:s + chunk]
        w2c = np.linalg.inv(np.stack([c.c2w for c in cs]))[:, :3, :]        # (c, 3, 4)
        q = np.einsum("cij,pj->cpi", w2c, P)                                # (c, p, 3)
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.array([c.fx for c in cs])[:, None] * q[..., 0] / z + np.array([c.cx for c in cs])[:, None]
            v = np.array([c.fy for c in cs])[:, None] * q[..., 1] / z + np.array([c.cy for c in cs])[:, None]
        W = np.array([c.width for c in cs])[:, None]
        H = np.array([c.height for c in cs])[:, None]
        ok = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        out[s:s + chunk] = ok.sum(1)
    return out


def sees(cams: List[Cam], block: Block, points: np.ndarray, min_visible: float = 0.05,
         min_coverage: float = 0.1) -> np.ndarray:
    """Which cameras see the block (expanded box): centre inside, or ≥ 5 % of the block's points visible, or
    (no points) the projected box covers ≥ 10 % of the image. Boolean (C,)."""
    lo, hi = block.expanded
    centres = np.array([c.center for c in cams]) if cams else np.zeros((0, 3))
    inside = np.all((centres >= lo) & (centres <= hi), axis=1) if len(centres) else np.zeros(0, bool)
    if len(points):
        frac = _project(cams, points) / float(len(points))
        return inside | (frac >= min_visible)
    out = inside.copy()
    corners = np.array([[hi[0] if k & 1 else lo[0], hi[1] if k & 2 else lo[1], hi[2] if k & 4 else lo[2]]
                        for k in range(8)])
    for i, c in enumerate(cams):
        if out[i]:
            continue
        w2c = np.linalg.inv(c.c2w)
        q = corners @ w2c[:3, :3].T + w2c[:3, 3]
        front = q[:, 2] > 0.01
        if not front.any():
            continue
        u = c.fx * q[front, 0] / q[front, 2] + c.cx
        v = c.fy * q[front, 1] / q[front, 2] + c.cy
        W, H = float(c.width), float(c.height)
        if front.sum() < 8:          # partly behind the camera: conservative
            out[i] = u.max() > 0 and u.min() < W and v.max() > 0 and v.min() < H
            continue
        iw = max(0.0, min(u.max(), W) - max(u.min(), 0.0))
        ih = max(0.0, min(v.max(), H) - max(v.min(), 0.0))
        out[i] = iw * ih / (W * H) >= min_coverage
    return out


# --------------------------------------------------------------------------------------------- partition


def partition(doc: Dict[str, Any], points: Optional[np.ndarray], count: int, margin: float = 0.15,
              camera_weight: float = 50.0) -> List[Block]:
    """Split the dataset (``dataset.json`` as a dict, its initial points (N, 3) or None) into ``count`` blocks."""
    cams = [c for c in cameras_from_dataset(doc) if not c.holdout]
    up = doc.get("upVector") if isinstance(doc.get("upVector"), list) and len(doc["upVector"]) == 3 else [0, -1, 0]
    h_axes, v_axis = axes(up)
    pts = np.zeros((0, 3)) if points is None else np.asarray(points, np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    stride = max(1, len(pts) // 200_000)
    sub = pts[::stride]
    centres = np.array([c.center for c in cams]).reshape(-1, 3)
    xyz = np.concatenate([sub, centres]) if len(centres) else sub
    w = np.concatenate([np.ones(len(sub)), np.full(len(centres), camera_weight)])
    if not len(xyz):
        return []
    lo = np.array([np.sort(xyz[:, a])[int((len(xyz) - 1) * 0.01)] for a in range(3)])
    hi = np.array([np.sort(xyz[:, a])[int((len(xyz) - 1) * 0.99)] for a in range(3)])
    if len(centres):
        lo, hi = np.minimum(lo, centres.min(0)), np.maximum(hi, centres.max(0))
    hi = np.maximum(hi, lo + 1e-3)

    done = []
    stack = [(lo.copy(), hi.copy(), max(1, int(count)), np.arange(len(xyz)), [True] * 3, [True] * 3)]
    while stack:
        clo, chi, k, items, olo, ohi = stack.pop()
        if k <= 1 or len(items) < 2:
            done.append((clo, chi, items, olo, ohi))
            continue
        ext = chi - clo
        axis = h_axes[0] if ext[h_axes[0]] >= ext[h_axes[1]] else h_axes[1]
        kl, kr = k // 2, k - k // 2
        order = items[np.argsort(xyz[items, axis], kind="stable")]
        acc = np.cumsum(w[order])
        target = acc[-1] * kl / k
        j = int(np.searchsorted(acc, target))           # first index where acc >= target
        cut = max(1, min(len(order) - 1, j + 1))
        split = (xyz[order[cut - 1], axis] + xyz[order[cut], axis]) / 2
        s = min(max(split, clo[axis] + 1e-4), chi[axis] - 1e-4)
        llo, lhi, rlo, rhi = clo.copy(), chi.copy(), clo.copy(), chi.copy()
        lhi[axis] = s
        rlo[axis] = s
        lo_h, ro_l = list(ohi), list(olo)
        lo_h[axis] = False
        ro_l[axis] = False
        stack.append((rlo, rhi, kr, order[cut:], ro_l, list(ohi)))
        stack.append((llo, lhi, kl, order[:cut], list(olo), lo_h))
    done.sort(key=lambda c: (c[0][h_axes[0]], c[0][h_axes[1]]))

    blocks = []
    for n, (clo, chi, items, olo, ohi) in enumerate(done):
        olo, ohi = list(olo), list(ohi)
        olo[v_axis] = ohi[v_axis] = True                 # never split vertically
        b = Block(n, [float(x) for x in clo], [float(x) for x in chi], margin, olo, ohi,
                  weight=float(w[items].sum()))
        elo, ehi = b.expanded
        bp = pts[np.all((pts >= elo) & (pts <= ehi), axis=1)] if len(pts) else pts
        if len(bp) > 2000:
            bp = bp[(np.arange(2000) * (len(bp) / 2000)).astype(int)]
        seen = sees(cams, b, bp)
        b.camera_ids = [c.id for c, s in zip(cams, seen) if s]
        blocks.append(b)
    return blocks


def block_config(config: Dict[str, Any], block: Block, blocks: Sequence[Block]) -> Dict[str, Any]:
    """The block's job: its region and cameras, and its share of the splat budget (by weight, +25 % for the
    overlap margins, at least 1/(2K) of the total, at least 100 k) so the merged scene stays near the cap."""
    cfg = copy.deepcopy(config)
    cfg["region"] = block.region
    cfg["cameraIds"] = list(block.camera_ids)
    total = sum(max(b.weight, 1.0) for b in blocks)
    share = max(max(block.weight, 1.0) / max(total, 1.0), 0.5 / max(1, len(blocks)))
    cfg["maxGaussians"] = max(100_000, int(int(config.get("maxGaussians") or 1_500_000) * share * 1.25))
    return cfg


def block_cost(b: Block) -> float:
    return float(max(1, len(b.camera_ids))) + b.weight / 10_000


def distribute(blocks: Sequence[Block], throughput: Dict[str, float],
               fits: Optional[Callable[[str, Block], bool]] = None) -> Dict[str, List[Block]]:
    """Device id → its blocks. Longest blocks first, each to the device that would finish it soonest
    (load / throughput); only devices it fits, else the fastest device."""
    devs = [d for d, t in throughput.items()]
    if not devs:
        return {}
    load: Dict[str, float] = {d: 0.0 for d in devs}
    out: Dict[str, List[Block]] = {}
    fastest = max(devs, key=lambda d: throughput[d])
    for b in sorted(blocks, key=block_cost, reverse=True):
        able = [d for d in devs if fits is None or fits(d, b)]
        pool = able or [fastest]
        d = min(pool, key=lambda x: ((load[x] + block_cost(b)) / max(throughput[x], 0.01), devs.index(x)))
        load[d] += block_cost(b)
        out.setdefault(d, []).append(b)
    return out


def block_count(cameras: int, devices: int) -> int:
    """Two blocks per device for balance, at least one per 2,000 photos."""
    return max(2, devices * 2, int(math.ceil(cameras / 2000)))


def plan_blocks(doc: Dict[str, Any], points: Optional[np.ndarray], devices: int,
                fits_somewhere: Callable[[Block, List[Block]], bool], max_blocks: int = 32) -> List[Block]:
    """More blocks until each fits on some device — but only while that shrinks them (when every camera sees
    most of the scene, splitting further keeps nearly all photos in every block)."""
    n = block_count(len([c for c in doc.get("cameras") or [] if not c.get("holdout")]), devices)
    blocks = partition(doc, points, n)
    widest = lambda bs: max((len(b.camera_ids) for b in bs), default=0)  # noqa: E731
    while n < max_blocks and not all(fits_somewhere(b, blocks) for b in blocks):
        nxt_n = n + max(1, n // 2)
        nxt = partition(doc, points, nxt_n)
        if widest(nxt) > 0.85 * widest(blocks):
            break
        n, blocks = nxt_n, nxt
    return blocks


# --------------------------------------------------------------------------------------------- merge

_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
          "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
          "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


class MergeError(ValueError):
    pass


def ply_layout(path: Path) -> Tuple[List[str], np.dtype, int, int]:
    """(header lines before ``element vertex``, vertex dtype, vertex count, body offset) of a binary
    little-endian PLY whose first element is ``vertex`` (what every trainer writes)."""
    with open(path, "rb") as fh:
        head = fh.read(64 * 1024)
    end = head.find(b"end_header")
    if not head.startswith(b"ply") or end < 0:
        raise MergeError(f"{Path(path).name}: not a PLY file")
    body = end + len(b"end_header")
    body += 1 if head[body:body + 1] == b"\r" else 0
    body += 1 if head[body:body + 1] == b"\n" else 0
    lines = [l.strip() for l in head[:end].decode("ascii", "replace").replace("\r", "").split("\n") if l.strip()]
    if "format binary_little_endian 1.0" not in lines:
        raise MergeError(f"{Path(path).name}: only binary little-endian PLYs can be merged")
    pre, fields, count, state = [], [], 0, "pre"
    for l in lines[1:]:
        t = l.split()
        if t[0] == "element":
            if state == "pre" and t[1] == "vertex":
                state, count = "vertex", int(t[2])
                continue
            if state == "vertex":
                break                   # later elements are ignored (their data follows the vertices)
            raise MergeError(f"{Path(path).name}: the first element must be vertex")
        if state == "pre" and t[0] != "format":
            pre.append(l)
        elif state == "vertex" and t[0] == "property":
            if t[1] == "list" or t[1] not in _TYPES:
                raise MergeError(f"{Path(path).name}: unsupported vertex property {l!r}")
            fields.append((t[2], _TYPES[t[1]]))
    if state != "vertex":
        raise MergeError(f"{Path(path).name}: no vertex element")
    return pre, np.dtype(fields), count, body


def merge_plys(parts: Sequence[Tuple[Block, Path]], out_path: Path, chunk: int = 1 << 18) -> int:
    """Merge block results into one PLY: each block contributes the splats it owns. Streamed (``chunk``
    vertices at a time, one block after the other); blocks with different layouts (e.g. a Mac block at a lower
    SH degree) are written in the richest layout, missing columns zero. Returns the splat count."""
    out_path = Path(out_path)
    layouts = [(b, Path(p), *ply_layout(Path(p))) for b, p in parts]
    if not layouts:
        raise MergeError("no blocks to merge")
    ref = max(layouts, key=lambda l: len(l[3].names))
    out_dtype = ref[3]
    body_tmp = out_path.with_name(out_path.name + ".body")
    total = 0
    try:
        with open(body_tmp, "wb") as fh:
            for block, path, _pre, dt, count, offset in layouts:
                if count == 0:
                    continue
                mm = np.memmap(path, dtype=dt, mode="r", offset=offset, shape=(count,))
                for s in range(0, count, chunk):
                    rows = np.asarray(mm[s:s + chunk])
                    xyz = np.stack([rows["x"], rows["y"], rows["z"]], 1)
                    keep = block.owns(xyz) & np.isfinite(xyz).all(1)
                    rows = rows[keep]
                    if dt != out_dtype:
                        conv = np.zeros(len(rows), out_dtype)
                        for name in out_dtype.names:
                            if name in dt.names:
                                conv[name] = rows[name]
                        rows = conv
                    rows.tofile(fh)
                    total += len(rows)
                del mm
        pre = [l for l in ref[2] if not l.startswith("comment Generated")]
        header = ["ply", "format binary_little_endian 1.0",
                  f"comment Generated by SplatWork for Linux (merged from {len(layouts)} blocks)"] + pre
        header.append(f"element vertex {total}")
        rev = {v: k for k, v in _TYPES.items() if not k[-1].isdigit()}
        for name in out_dtype.names:
            header.append(f"property {rev[out_dtype[name].str.replace('|', '')]} {name}")
        header.append("end_header")
        tmp = out_path.with_name(out_path.name + ".tmp")
        with open(tmp, "wb") as out, open(body_tmp, "rb") as body:
            out.write(("\n".join(header) + "\n").encode("ascii"))
            while True:
                buf = body.read(8 << 20)
                if not buf:
                    break
                out.write(buf)
        os.replace(tmp, out_path)
    finally:
        body_tmp.unlink(missing_ok=True)
    return total
