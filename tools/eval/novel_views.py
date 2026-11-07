#!/usr/bin/env python3
"""Visual comparison from viewpoints no photo was taken from — where floaters and haze show up.

Takes held-out cameras of a dataset, moves each forward along its view direction by ``--closer`` × the median
depth of the scene points it sees (so a drone photo becomes a lower, zoomed-in look at the ground), writes those
cameras as a render-only dataset, renders every model from them with ``SplatStudio --render`` and lays the
results side by side (no ground truth exists for these views; judge floaters, haze and sharpness by eye).

    python3 tools/eval/novel_views.py --dataset <dataset> --model ours=a.ply --model spirula=b.ply \
        --app mac/.build/release/SplatStudio --out novel_out [--views 6] [--closer 0.6]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_points(path: Path) -> np.ndarray:
    f = open(path, "rb")
    props, n, fmt = [], 0, ""
    while True:
        s = f.readline().decode().strip()
        if s.startswith("format"):
            fmt = s.split()[1]
        if s.startswith("element vertex"):
            n = int(s.split()[2])
        if s.startswith("property"):
            props.append((s.split()[1], s.split()[2]))
        if s == "end_header":
            break
    if fmt != "binary_little_endian":
        raise SystemExit("points PLY must be binary little-endian")
    tmap = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4", "float32": "<f4"}
    dt = np.dtype([(name, tmap[t]) for t, name in props])
    a = np.frombuffer(f.read(n * dt.itemsize), dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, help="name=splat.ply")
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--closer", type=float, default=0.6)
    ap.add_argument("--width", type=int, default=520)
    a = ap.parse_args()
    ds = json.loads((a.dataset / "dataset.json").read_text())
    pts = read_points(a.dataset / ds["points"]) if ds.get("points") else None
    held = [c for c in ds["cameras"] if c.get("holdout")] or ds["cameras"]
    picks = [held[int(i)] for i in np.linspace(0, len(held) - 1, min(a.views, len(held)))]
    cams = []
    for k, c in enumerate(picks):
        M = np.array(c["cameraToWorld"], dtype=np.float64)
        eye, fwd = M[:3, 3], M[:3, 2] / np.linalg.norm(M[:3, 2])
        depth = None
        if pts is not None:
            rel = pts - eye
            z = rel @ fwd
            lateral = np.linalg.norm(rel - np.outer(z, fwd), axis=1)
            sel = (z > 0) & (lateral < 0.25 * z)          # points near the middle of the view
            if sel.sum() > 20:
                depth = float(np.median(z[sel]))
        if depth is None:
            depth = float(np.linalg.norm(np.array(ds.get("sceneCenter", [0, 0, 0])) - eye))
        M2 = M.copy()
        M2[:3, 3] = eye + fwd * depth * a.closer
        c2 = dict(c); c2["id"] = k; c2["cameraToWorld"] = M2.tolist(); c2.pop("holdout", None)
        cams.append(c2)
    a.out.mkdir(parents=True, exist_ok=True)
    nd = a.out / "dataset"
    nd.mkdir(exist_ok=True)
    meta = {k: v for k, v in ds.items() if k not in ("cameras", "points")}
    meta["cameras"] = cams
    (nd / "dataset.json").write_text(json.dumps(meta))
    rows = []
    for k, c in enumerate(cams):
        row = []
        for m in a.model:
            name, _, ply = m.partition("=")
            png = a.out / f"{name}_{k}.png"
            subprocess.run([str(a.app), "--render", ply, str(png), "--dataset", str(nd), "--view", str(k)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if png.is_file():
                im = Image.open(png).convert("RGB")
                im = im.resize((a.width, int(a.width * im.height / im.width)), Image.LANCZOS)
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 8 + 7 * len(name), 16], fill=(0, 0, 0)); d.text((4, 2), name, fill=(255, 255, 255))
                row.append(im)
        if row:
            rows.append((Path(c["image"]).name, row))
    if not rows:
        raise SystemExit("nothing rendered")
    h = max(im.height for _, r in rows for im in r)
    cols = max(len(r) for _, r in rows)
    sheet = Image.new("RGB", ((a.width + 8) * cols + 8, (h + 22) * len(rows) + 8), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for i, (n, r) in enumerate(rows):
        y = 8 + i * (h + 22)
        d.text((8, y), f"{n} moved {int(a.closer * 100)} % closer", fill=(0, 0, 0))
        for j, im in enumerate(r):
            sheet.paste(im, (8 + j * (a.width + 8), y + 14))
    sheet.save(a.out / "novel_sheet.jpg", quality=88)
    print(f"sheet: {a.out / 'novel_sheet.jpg'} ({len(rows)} views × {cols} models)")


if __name__ == "__main__":
    main()
