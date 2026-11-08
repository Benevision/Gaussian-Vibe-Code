"""Stereo footage on this side of the pipeline (GTK-free, numpy only), as the Mac's ``DatasetBuilder`` /
``RigConstraints`` / ``StereoDepth``.

The worker's ``frames`` job splits the footage into ``images/<n>_L`` / ``_R`` and writes ``frames.json`` with the
stereo rig (``tools/splatprep/stereo.py``). Here:

* :func:`prepare_frames` — drops the right eye when it shouldn't train ("Train on both eyes" off);
* :func:`sfm_rig` — the camera solver's rig hint for a parallel pair with a known baseline (left = identity,
  right = centre at +baseline on x), so COLMAP / GLOMAP register both eyes of a frame together, in metres;
* :func:`apply_constraints` — after solving: metric scale = baseline / median eye distance (robust inner 50 %),
  applied to cameras, scene bounds and ``points3d.ply``; then (parallel pairs) every right eye is re-derived from its
  left eye and the rig, which also places right eyes the solver missed;
* :func:`compute_depth` — census block matching on each rectified pair (the Mac's ``StereoDepth``: 5×5 census,
  box aggregation, uniqueness, sub-pixel, left-right check, 3×3 median) → ``depth/<n>.bin`` (float16 metres) for
  the left eye, used by the trainer's depth loss. One pair at a time, ~45 MB of cost volume at 640 px.

Fisheye / toed-out rigs (Arcturus Vision) are not a rectified pair: they get the metric scale only (no rig hint,
no right-eye re-derivation, no block-matching depth).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Progress = Callable[[float, str], None]

DEPTH_DEFAULTS = {"matchWidth": 640, "maxDisparity": 96, "blockRadius": 4, "lrTolerance": 1.0, "uniqueness": 0.08,
                  "minDepth": 0.1, "maxDepth": 50.0}


def load_frames(ds: Path) -> Dict:
    f = Path(ds) / "frames.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def rig_info(ds: Path) -> Dict:
    return load_frames(ds).get("rig") or {}


def pairs(fs: Dict) -> List[Tuple[Dict, Dict]]:
    by: Dict[int, Dict[str, Dict]] = {}
    for fr in fs.get("frames", []):
        if "group" in fr and fr.get("eye") in ("left", "right"):
            by.setdefault(int(fr["group"]), {})[fr["eye"]] = fr
    return [(v["left"], v["right"]) for k, v in sorted(by.items()) if "left" in v and "right" in v]


def prepare_frames(ds: Path, both_eyes: bool = True) -> Dict:
    """After the frames job: drop right-eye images (and their masks) when only the left eye trains."""
    ds = Path(ds)
    fs = load_frames(ds)
    if not fs:
        raise ValueError("the stereo frames job wrote no frames.json")
    rig = fs.get("rig") or {}
    n = len(pairs(fs))
    dropped = 0
    if not both_eyes:
        keep = []
        for fr in fs.get("frames", []):
            if fr.get("eye") == "right":
                stem = Path(fr["image"]).stem
                (ds / fr["image"]).unlink(missing_ok=True)
                for m in (ds / "masks" / f"{stem}.png", ds / "splatprep-masks" / "frames" / "masks" / f"{stem}.png"):
                    m.unlink(missing_ok=True)
                dropped += 1
            else:
                keep.append(fr)
        fs["frames"] = keep
        tmp = ds / "frames.json.tmp"
        tmp.write_text(json.dumps(fs, indent=1))
        tmp.replace(ds / "frames.json")
    return {"pairs": n, "layout": fs.get("stereoLayout"), "baseline": rig.get("baseline"),
            "origin": rig.get("origin"), "parallel": bool(rig.get("parallel")), "lens": rig.get("lens"),
            "droppedRight": dropped}


def photo_layout(w: int, h: int, name: str, layout: str = "auto") -> Tuple[str, bool]:
    """(layout, anamorphic) of a packed stereo photo — the worker's rule for videos (splatprep/stereo.py)."""
    if layout in ("sideBySide", "topBottom"):
        return layout, False
    a = w / max(h, 1)
    n = name.lower()
    if any(k in n for k in ("_tb", "top-bottom", "topbottom", "over-under")) or a < 1.05:
        return "topBottom", a >= 1.5
    if a >= 2.6 or "_sbs" in n or "side-by-side" in n:
        return "sideBySide", a < 2.6
    if abs(a - 2) < 0.05:
        return "sideBySide", False
    raise ValueError(f"{name}: can't tell the stereo layout (aspect {a:.2f}): choose side by side or over / under")


def split_photos(src: Path, ds: Path, layout: str = "auto", swap: bool = False, baseline: Optional[float] = None,
                 fov: Optional[float] = None, max_size: int = 0, progress: Progress = lambda f, m="": None) -> Dict:
    """A folder of side-by-side / over-under stereo photos → ``images/<n>_L|R.jpg`` + ``frames.json`` with the rig
    (the same layout the worker's ``frames`` job writes for stereo video). One photo in memory at a time."""
    import math

    from PIL import Image, ImageOps

    from .importers import list_images

    ds = Path(ds)
    (ds / "images").mkdir(parents=True, exist_ok=True)
    files = list_images(Path(src))
    if not files:
        raise ValueError(f"no photos in {src}")
    frames: List[Dict] = []
    rejected: List[Dict] = []
    used_layout = None
    for i, p in enumerate(files):
        try:
            with Image.open(p) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
            lay, ana = photo_layout(im.width, im.height, p.name, layout)
        except Exception as e:  # noqa: BLE001 — reported, not fatal
            rejected.append({"item": p.name, "reason": str(e)})
            continue
        used_layout = used_layout or lay
        W, H = im.size
        boxes = ((0, 0, W // 2, H), (W // 2, 0, 2 * (W // 2), H)) if lay == "sideBySide" else \
            ((0, 0, W, H // 2), (0, H // 2, W, 2 * (H // 2)))
        eyes = [im.crop(b) for b in boxes]
        im.close()
        if swap:
            eyes.reverse()
        for eye, img, tag in zip(("left", "right"), eyes, ("L", "R")):
            if ana:
                img = img.resize((img.width * 2, img.height) if lay == "sideBySide" else (img.width, img.height * 2),
                                 Image.BICUBIC)
            if max_size and max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
            name = f"{i:06d}_{tag}.jpg"
            img.save(ds / "images" / name, quality=95)
            fr = {"image": f"images/{name}", "width": img.width, "height": img.height, "group": i, "eye": eye,
                  "source": str(p)}
            if fov:
                f = img.width / 2 / math.tan(math.radians(fov) / 2)
                fr.update(focalPx=f, cx=img.width / 2, cy=img.height / 2)
            frames.append(fr)
        progress((i + 1) / len(files), f"{i + 1}/{len(files)} stereo photos")
    if not frames:
        raise ValueError("no usable stereo photos")
    rig = {"baseline": baseline or None, "origin": "user" if baseline else "default", "parallel": True,
           "lens": "pinhole"}
    if fov:
        rig["horizontalFOV"] = float(fov)
    fs = {"root": ".", "frames": frames, "rejected": rejected, "isMultiViewRig": False, "rig": rig,
          "stereoLayout": used_layout}
    (ds / "frames.json").write_text(json.dumps(fs, indent=1))
    return {"pairs": len(frames) // 2, "rejected": rejected, "layout": used_layout}


def sfm_rig(ds: Path) -> Optional[Dict]:
    """``rig`` hint for the camera solve (PROTOCOL.md §4, rigs with a baseline), or None."""
    fs = load_frames(ds)
    rig = fs.get("rig") or {}
    b = rig.get("baseline")
    if not rig.get("parallel") or not b:
        return None
    eye = np.eye(3).tolist()
    groups = [{Path(l["image"]).name: eye, Path(r["image"]).name: {"R": eye, "t": [float(b), 0.0, 0.0]}}
              for l, r in pairs(fs) if (Path(ds) / r["image"]).is_file()]
    return {"kind": "multiview", "groups": groups} if len(groups) >= 2 else None


# ------------------------------------------------------------------------------------- after solving

def _scale_ply(path: Path, s: float) -> bool:
    """Scale x/y/z of a binary little-endian vertex PLY in place."""
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        return False
    header = data[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        return False
    types = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
             "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "float32": "<f4", "float64": "<f8",
             "int32": "<i4", "uint32": "<u4", "int16": "<i2", "uint16": "<u2"}
    n, props, in_vertex = 0, [], False
    for line in header:
        p = line.split()
        if p[:2] == ["element", "vertex"]:
            n, in_vertex = int(p[2]), True
        elif p and p[0] == "element":
            in_vertex = False
        elif in_vertex and p[:1] == ["property"] and len(p) == 3:
            props.append((p[2], types[p[1]]))
    body = end + len(b"end_header\n")
    arr = np.frombuffer(data, np.dtype(props), count=n, offset=body).copy()
    for k in ("x", "y", "z"):
        arr[k] = arr[k] * s
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data[:body] + arr.tobytes() + data[body + arr.nbytes:])
    tmp.replace(path)
    return True


def apply_constraints(ds: Path, enforce: bool = True) -> Dict:
    """Metric scale from the baseline, then right eyes re-derived from left eyes (parallel rigs)."""
    ds = Path(ds)
    fs = load_frames(ds)
    rig = fs.get("rig") or {}
    b = rig.get("baseline")
    path = ds / "dataset.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    cams = doc.get("cameras", [])
    by_name = {Path(c["image"]).name: c for c in cams}
    prs = [(Path(l["image"]).name, Path(r["image"]).name, r) for l, r in pairs(fs)]
    out: Dict = {"scale": None, "enforced": 0, "added": 0}
    if b:
        ratios = []
        for ln, rn, _ in prs:
            if ln in by_name and rn in by_name:
                d = np.linalg.norm(np.asarray(by_name[ln]["cameraToWorld"])[:3, 3]
                                   - np.asarray(by_name[rn]["cameraToWorld"])[:3, 3])
                if d > 1e-9:
                    ratios.append(b / d)
        if len(ratios) >= 3:
            ratios.sort()
            lo, hi = len(ratios) // 4, max(len(ratios) // 4 + 1, len(ratios) - len(ratios) // 4)
            inner = ratios[lo:hi]
            s = float(inner[len(inner) // 2])
            for c in cams:
                m = np.asarray(c["cameraToWorld"], np.float64)
                m[:3, 3] *= s
                c["cameraToWorld"] = m.tolist()
            if "sceneCenter" in doc:
                doc["sceneCenter"] = [float(v) * s for v in doc["sceneCenter"]]
            if "sceneExtent" in doc:
                doc["sceneExtent"] = float(doc["sceneExtent"]) * s
            pts = ds / doc.get("points", "points3d.ply")
            if pts.is_file():
                _scale_ply(pts, s)
            out["scale"] = s
        if enforce and rig.get("parallel"):
            r2l = np.eye(4)
            r2l[0, 3] = float(b) if out["scale"] else 0.0
            if out["scale"]:                    # (unscaled scenes would get a wrong baseline: leave them)
                next_id = max([int(c.get("id", i)) for i, c in enumerate(cams)] + [-1]) + 1
                for ln, rn, rf in prs:
                    left = by_name.get(ln)
                    if left is None:
                        continue
                    c2w = (np.asarray(left["cameraToWorld"], np.float64) @ r2l).tolist()
                    right = by_name.get(rn)
                    if right is not None:
                        right["cameraToWorld"] = c2w
                        out["enforced"] += 1
                    elif (ds / "images" / rn).is_file():
                        new = {k: v for k, v in left.items() if k not in ("mask", "sky", "depth", "depthWidth",
                                                                         "depthHeight", "holdout")}
                        new.update(id=next_id, image=f"images/{rn}", cameraToWorld=c2w)
                        next_id += 1
                        cams.append(new)
                        by_name[rn] = new
                        out["added"] += 1
    tmp = path.with_name("dataset.json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)
    return out


# ------------------------------------------------------------------------------------- stereo depth

def _popcount32(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint16)
    b = x.view(np.uint8).reshape(x.shape + (4,))
    table = np.array([bin(i).count("1") for i in range(256)], np.uint16)
    return table[b].sum(-1, dtype=np.uint16)


def census(g: np.ndarray) -> np.ndarray:
    """5×5 census transform (24 bits), edge-replicated borders."""
    h, w = g.shape
    p = np.pad(g, 2, mode="edge")
    bits = np.zeros((h, w), np.uint32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            bits = (bits << np.uint32(1)) | (p[2 + dy:2 + dy + h, 2 + dx:2 + dx + w] < g).astype(np.uint32)
    return bits


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)² window, truncated at the borders (like the Mac's running sums)."""
    h, w = a.shape
    c = np.zeros((h + 1, w + 1), np.int32)
    c[1:, 1:] = a.astype(np.int32).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    return np.minimum(s, 65535).astype(np.uint16)


def _median3(d: np.ndarray) -> np.ndarray:
    h, w = d.shape
    out = d.copy()
    if h < 3 or w < 3:
        return out
    stack = np.stack([d[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], -1)
    valid = stack > 0
    cnt = valid.sum(-1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(valid, stack, np.nan), -1)
    centre = d[1:-1, 1:-1]
    res = np.where(cnt < 4, 0.0, np.where((centre == 0) & (cnt < 7), 0.0, np.nan_to_num(med)))
    out[1:-1, 1:-1] = res
    return out


def disparity(left: np.ndarray, right: np.ndarray, o: Optional[Dict] = None) -> np.ndarray:
    """Disparity (pixels, 0 = invalid) of the left grey image against the right, both uint8 H×W, rectified."""
    o = {**DEPTH_DEFAULTS, **(o or {})}
    h, w = left.shape
    D = max(4, min(int(o["maxDisparity"]), w // 2))
    r = int(o["blockRadius"])
    cl, cr = census(left), census(right)
    cost = np.empty((D, h, w), np.uint16)
    for d in range(D):
        raw = np.full((h, w), 24, np.uint16)
        if d < w:
            raw[:, d:] = _popcount32(cl[:, d:] ^ cr[:, :w - d])
        c = _box(raw, r)
        c[:, :d] = 65535                      # x − d < 0: no match (the Mac stops at d ≤ x)
        cost[d] = c
    bd = cost.argmin(0)
    two = np.partition(cost, 1, axis=0)[:2].astype(np.float32)
    best, second = two[0], two[1]
    disp = bd.astype(np.float32)
    ok = np.ones((h, w), bool)
    if o["uniqueness"] > 0:
        ok &= ~((second - best) < o["uniqueness"] * best)
    xs = np.arange(w)[None, :].repeat(h, 0)
    ok &= np.minimum(D - 1, xs) >= 1
    inner = (bd > 0) & (bd < np.minimum(D - 1, xs))
    yy, xx = np.nonzero(inner)
    c0 = cost[bd[yy, xx] - 1, yy, xx].astype(np.float32)
    c1 = cost[bd[yy, xx], yy, xx].astype(np.float32)
    c2 = cost[bd[yy, xx] + 1, yy, xx].astype(np.float32)
    den = c0 - 2 * c1 + c2
    sub = np.where(den > 0, 0.5 * (c0 - c2) / np.where(den > 0, den, 1), 0)
    disp[yy, xx] += sub
    disp = np.where(ok, np.maximum(disp, 1e-4), 0).astype(np.float32)
    # right view: xr matches xl = xr + d
    best_r = np.full((h, w), np.iinfo(np.uint16).max, np.uint32)
    disp_r = np.full((h, w), -1, np.int32)
    for d in range(D):
        if d >= w:
            break
        c = cost[d][:, d:].astype(np.uint32)            # cost at xl = xr + d, for xr in [0, w − d)
        better = c < best_r[:, :w - d]
        best_r[:, :w - d][better] = c[better]
        disp_r[:, :w - d][better] = d
    del cost
    xr = np.clip(xs - np.rint(disp).astype(np.int64), 0, w - 1)
    dr = disp_r[np.arange(h)[:, None], xr]
    bad = (disp > 0) & (((xs - np.rint(disp)) < 0) | (dr < 0) | (np.abs(dr - disp) > o["lrTolerance"]))
    disp[bad] = 0
    return _median3(disp)


def depth_map(left_rgb: np.ndarray, right_rgb: np.ndarray, focal: float, baseline: float,
              o: Optional[Dict] = None) -> np.ndarray:
    """Metric depth (float32, 0 = invalid) of the left image at the matching width. ``focal`` is in pixels of
    ``left_rgb``'s width."""
    from PIL import Image

    o = {**DEPTH_DEFAULTS, **(o or {})}
    H, W = left_rgb.shape[:2]
    w = min(int(o["matchWidth"]), W)
    h = max(8, int(round(H * w / W)))

    def grey(a):
        return np.asarray(Image.fromarray(a).convert("L").resize((w, h), Image.BILINEAR))

    disp = disparity(grey(left_rgb), grey(right_rgb), o)
    fb = focal * w / W * baseline
    with np.errstate(divide="ignore"):
        z = np.where(disp > 0, fb / np.maximum(disp, 1e-6), 0).astype(np.float32)
    z[(z < o["minDepth"]) | (z > o["maxDepth"])] = 0
    return z


def compute_depth(ds: Path, progress: Progress = lambda f, m="": None, check: Callable[[], None] = lambda: None,
                  options: Optional[Dict] = None) -> Dict:
    """Depth for every left eye with a registered pair → ``depth/<stem>.bin`` + dataset.json entries."""
    from PIL import Image

    ds = Path(ds)
    fs = load_frames(ds)
    rig = fs.get("rig") or {}
    b = rig.get("baseline")
    if not b or not rig.get("parallel"):
        return {"written": 0, "reason": "needs a known baseline and a rectified (parallel) pair"
                if rig.get("parallel") is False or b else "the stereo baseline is unknown"}
    path = ds / "dataset.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_name = {Path(c["image"]).name: c for c in doc.get("cameras", [])}
    prs = pairs(fs)
    (ds / "depth").mkdir(exist_ok=True)
    written = 0
    valid_frac = []
    for k, (l, r) in enumerate(prs):
        check()
        cam = by_name.get(Path(l["image"]).name)
        rimg = ds / r["image"]
        if cam is None or not rimg.is_file():
            continue
        with Image.open(ds / cam["image"]) as a, Image.open(rimg) as bimg:
            L = np.asarray(a.convert("RGB"))
            R = np.asarray(bimg.convert("RGB").resize(a.size))
        z = depth_map(L, R, float(cam["fx"]) * L.shape[1] / float(cam["width"]), float(b), options)
        del L, R
        stem = Path(cam["image"]).stem
        z.astype("<f2").tofile(ds / "depth" / f"{stem}.bin")
        cam.update(depth=f"depth/{stem}.bin", depthWidth=int(z.shape[1]), depthHeight=int(z.shape[0]))
        valid_frac.append(float((z > 0).mean()))
        written += 1
        progress((k + 1) / max(len(prs), 1), f"{k + 1}/{len(prs)} pairs")
    tmp = path.with_name("dataset.json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"written": written, "validFraction": round(float(np.mean(valid_frac)), 3) if valid_frac else 0.0}
