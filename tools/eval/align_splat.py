#!/usr/bin/env python3
"""Move a splat trained elsewhere (e.g. Spirula Studio, Postshot, nerfstudio) into one of our datasets' frames,
so ``SplatStudio --eval`` can score it on the same held-out photos.

The other tool's COLMAP model (``sparse/0/images.bin`` or ``images.txt``) gives its camera centres; ours come
from ``dataset.json``. Cameras are matched by image file name, a similarity transform (scale, rotation,
translation) is fitted robustly (Umeyama + trimming of the worst 10 %), and applied to the splat's positions,
scales and rotations.

    python3 tools/eval/align_splat.py --splat other.ply --colmap <other>/sparse/0 --dataset <our dataset> --out aligned.ply

Note: if the other tool trained on our held-out photos too, its held-out scores are optimistic.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def colmap_centres(model: Path) -> dict:
    out = {}
    b = model / "images.bin"
    if b.is_file():
        with open(b, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                _id, *rest = struct.unpack("<I7dI", f.read(64))
                q, t = np.array(rest[0:4]), np.array(rest[4:7])
                name = b""
                while (c := f.read(1)) != b"\0":
                    name += c
                npts = struct.unpack("<Q", f.read(8))[0]
                f.seek(npts * 24, 1)
                out[Path(name.decode()).name] = -qvec_to_R(q).T @ t
        return out
    lines = [l for l in (model / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        q, t = np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8])))
        out[Path(p[9]).name] = -qvec_to_R(q).T @ t
    return out


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s0 ** 2).sum(1).mean()
    return scale, R, md - scale * R @ ms


def read_ply(path: Path):
    f = open(path, "rb")
    hdr, props = b"", []
    while True:
        l = f.readline(); hdr += l
        s = l.decode().strip()
        if s.startswith("element vertex"):
            n = int(s.split()[2])
        if s.startswith("property"):
            props.append(s.split()[2])
        if s == "end_header":
            break
    a = np.frombuffer(f.read(n * 4 * len(props)), np.float32).reshape(n, len(props)).copy()
    return hdr, props, a


def quat_mul(a, b):  # (w,x,y,z), batched
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def R_to_q(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat", required=True, type=Path)
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    theirs = colmap_centres(a.colmap)
    ds = json.loads((a.dataset / "dataset.json").read_text())
    ours = {Path(c["image"]).name: np.array(c["cameraToWorld"])[:3, 3] for c in ds["cameras"]}
    names = sorted(set(theirs) & set(ours))
    if len(names) < 4:
        raise SystemExit(f"only {len(names)} cameras in common")
    src = np.array([theirs[n] for n in names]); dst = np.array([ours[n] for n in names])
    keep = np.ones(len(names), bool)
    for _ in range(3):
        s, R, t = umeyama(src[keep], dst[keep])
        err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        keep = err <= np.quantile(err, 0.9)
    s, R, t = umeyama(src[keep], dst[keep])
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    extent = float(np.linalg.norm(dst - dst.mean(0), axis=1).max())
    print(f"{len(names)} common cameras; scale {s:.5f}; residual median {np.median(err):.4f} "
          f"(= {100 * np.median(err) / extent:.3f} % of the camera spread), 90th pct {np.quantile(err, 0.9):.4f}")
    hdr, props, v = read_ply(a.splat)
    ix = [props.index(k) for k in ("x", "y", "z")]
    v[:, ix] = (s * (R @ v[:, ix].T)).T + t
    sc = [props.index(f"scale_{i}") for i in range(3)]
    v[:, sc] += np.log(s)
    rq = [props.index(f"rot_{i}") for i in range(4)]
    q = v[:, rq].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    v[:, rq] = quat_mul(np.tile(R_to_q(R), (len(q), 1)), q).astype(np.float32)
    # SH bands above DC rotate with the scene; leaving them unrotated only affects view-dependent colour.
    a.out.write_bytes(hdr + v.tobytes())
    print(f"wrote {a.out} ({len(v)} splats)")


if __name__ == "__main__":
    main()
