"""Sources that are converted on this side before the worker sees them (no GPU needed):

* photos folder → ``images/`` (resized to the maximum image size, blurry photos rejected)
* ``.splatcapture`` (iPhone / headset, PROTOCOL.md §1) → ``images/`` + ``priors.json`` (ARKit poses, intrinsics)
* Nerfstudio / Instant-NGP ``transforms.json`` → ``dataset.json`` (OpenGL → OpenCV camera convention)
* COLMAP project (``sparse/0`` + images) → ``dataset.json`` via splatprep's model reader / exporter
* stacked masks: each mask layer (255 = keep) ANDed into ``masks/<stem>.png``, removed areas grown by 4 px
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp")
Progress = Callable[[float, str], None]


def _noop(f: float, m: str = "") -> None:
    pass


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."))


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (same metric as splatprep / the Mac)."""
    g = gray.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def _location(im):
    """(EXIF holding only the GPS block, XMP holding only DJI's height above take-off) of a photo, or Nones.
    Imported copies keep where each photo was taken, and nothing else, so camera solving can check against GPS
    and put the scene in metres (as the Mac's import does)."""
    import re

    from PIL import Image

    try:
        gps = dict(im.getexif().get_ifd(0x8825))
    except Exception:
        gps = {}
    if not (gps.get(2) and gps.get(4)):
        return None, None
    ex = Image.Exif()
    ex.get_ifd(0x8825).update(gps)
    xmp = im.info.get("xmp") or b""
    if isinstance(xmp, str):
        xmp = xmp.encode()
    m = re.search(rb'RelativeAltitude="([-+0-9.]+)"', xmp) or re.search(rb"RelativeAltitude>\s*([-+0-9.]+)\s*<", xmp)
    out_xmp = None
    if m:
        out_xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                   '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:RelativeAltitude="%s"/>'
                   '</rdf:RDF></x:xmpmeta>' % m.group(1).decode()).encode()
    return ex, out_xmp


def import_photos(src: Path, dataset_dir: Path, max_size: int = 1600, blur_rejection: bool = True,
                  blur_threshold: float = 0.35, progress: Progress = _noop, keep_location: bool = True) -> Dict:
    """Copy (and downscale) photos into ``dataset_dir/images`` as JPEG/PNG; returns stats."""
    from PIL import Image, ImageOps

    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    files = list_images(src)
    if not files:
        raise ValueError(f"no photos in {src}")
    out = Path(dataset_dir) / "images"
    out.mkdir(parents=True, exist_ok=True)
    sharp: Dict[str, float] = {}
    written: List[Path] = []
    for k, f in enumerate(files):
        try:
            raw = Image.open(f)
            loc = _location(raw) if keep_location else (None, None)
            im = ImageOps.exif_transpose(raw)
        except Exception as e:
            progress(k / len(files), f"skipped {f.name}: {e}")
            continue
        im = im.convert("RGB")
        if max_size and max(im.size) > max_size:
            s = max_size / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        ext = ".png" if f.suffix.lower() == ".png" else ".jpg"
        dst = out / (f.stem + ext)
        if ext == ".jpg":
            kw = {"quality": 95}
            if loc[0] is not None:
                kw["exif"] = loc[0]
            if loc[1]:
                kw["xmp"] = loc[1]
            im.save(dst, **kw)
        else:
            im.save(dst)
        small = im.convert("L")
        small.thumbnail((512, 512))
        sharp[dst.name] = sharpness(np.asarray(small))
        written.append(dst)
        progress((k + 1) / len(files), f"{k + 1}/{len(files)} photos")
    rejected: List[str] = []
    if blur_rejection and len(written) >= 8:
        med = float(np.median(list(sharp.values())))
        for p in written:
            if sharp[p.name] < blur_threshold * med:
                rejected.append(p.name)
                p.unlink()
    return {"images": len(written) - len(rejected), "rejected": rejected}


def import_photos360(src: Path, dataset_dir: Path, options: Dict, progress: Progress = _noop) -> Dict:
    """Folder of 2:1 equirectangular photos → perspective views + ``frames.json`` (one rig group per photo);
    nadir keep-masks go to the ``splatprep-masks/frames`` layer, like the 360° video path's."""
    from . import equirect

    files = list_images(src)
    if not files:
        raise ValueError(f"no photos in {src}")
    layer = Path(dataset_dir) / "splatprep-masks" / "frames"
    shutil.rmtree(layer, ignore_errors=True)
    return equirect.split_photos(files, dataset_dir, options, layer / "masks", progress)


def import_capture(bundle: Path, dataset_dir: Path, progress: Progress = _noop) -> Dict:
    """``.splatcapture`` → images/ + priors.json. The manifest itself is splatprep's prior format (ARKit poses,
    column-major, converted with c2w · diag(1, −1, −1, 1)); frames without tracking are left out."""
    bundle = Path(bundle)
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    out = Path(dataset_dir) / "images"
    out.mkdir(parents=True, exist_ok=True)
    frames = [f for f in man.get("frames", []) if f.get("trackingState", "normal") != "notAvailable"]
    kept = []
    for k, f in enumerate(frames):
        src = bundle / f["image"]
        if not src.is_file():
            continue
        dst = out / src.name
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        kept.append(dict(f, image=f"images/{dst.name}"))
        if k % 25 == 0:
            progress((k + 1) / max(len(frames), 1), f"{k + 1}/{len(frames)} frames")
    priors = dict(man, frames=kept)
    priors.setdefault("format", "splatcapture")
    priors.setdefault("coordinateSystem", "arkit")
    (Path(dataset_dir) / "priors.json").write_text(json.dumps(priors, indent=1))
    return {"images": len(kept), "hasLiDAR": bool(man.get("hasLiDAR"))}


def import_nerfstudio(transforms: Path, dataset_dir: Path, progress: Progress = _noop) -> Dict:
    """Nerfstudio / Instant-NGP ``transforms.json`` → ``dataset.json`` (images copied/linked)."""
    transforms = Path(transforms)
    if transforms.is_dir():
        transforms = transforms / "transforms.json"
    src_root = transforms.parent
    tf = json.loads(transforms.read_text(encoding="utf-8"))
    ds_dir = Path(dataset_dir)
    (ds_dir / "images").mkdir(parents=True, exist_ok=True)
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    cams = []
    frames = tf.get("frames", [])
    for i, f in enumerate(frames):
        src = (src_root / f["file_path"])
        if not src.suffix and not src.exists():
            for ext in (".png", ".jpg", ".jpeg"):
                if src.with_suffix(ext).exists():
                    src = src.with_suffix(ext)
                    break
        if not src.is_file():
            continue
        w = int(f.get("w", tf.get("w", 0)))
        h = int(f.get("h", tf.get("h", 0)))
        if not w or not h:
            from PIL import Image

            with Image.open(src) as im:
                w, h = im.size
        fx = f.get("fl_x", tf.get("fl_x"))
        fy = f.get("fl_y", tf.get("fl_y", fx))
        if fx is None and "camera_angle_x" in tf:
            fx = fy = 0.5 * w / np.tan(0.5 * float(tf["camera_angle_x"]))
        cx = f.get("cx", tf.get("cx", w / 2))
        cy = f.get("cy", tf.get("cy", h / 2))
        dst = ds_dir / "images" / f"{i:06d}{src.suffix.lower()}"
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        c2w = np.asarray(f["transform_matrix"], np.float64).reshape(4, 4) @ gl_to_cv
        cams.append({"id": len(cams), "image": f"images/{dst.name}", "width": w, "height": h, "fx": float(fx),
                     "fy": float(fy), "cx": float(cx), "cy": float(cy), "cameraToWorld": c2w.tolist(),
                     "holdout": False})
        progress((i + 1) / max(len(frames), 1), f"{i + 1}/{len(frames)} images")
    if not cams:
        raise ValueError("transforms.json has no usable frames")
    pos = np.array([np.asarray(c["cameraToWorld"])[:3, 3] for c in cams])
    center = pos.mean(0)
    extent = float(np.linalg.norm(pos - center, axis=1).max() * 1.1 or 1.0)
    up = -np.mean([np.asarray(c["cameraToWorld"])[:3, 1] for c in cams], 0)
    up = up / (np.linalg.norm(up) or 1)
    doc = {"format": "splatdataset", "version": 1, "convention": "opencv", "sceneCenter": center.tolist(),
           "sceneExtent": extent, "upVector": up.tolist(), "cameras": cams}
    ply = tf.get("ply_file_path")
    if ply and (src_root / ply).is_file():
        shutil.copy2(src_root / ply, ds_dir / "points3d.ply")
        doc["points"] = "points3d.ply"
    (ds_dir / "dataset.json").write_text(json.dumps(doc, indent=1))
    return {"cameras": len(cams)}


def import_colmap(project: Path, dataset_dir: Path, images: Optional[Path] = None, progress: Progress = _noop) -> Dict:
    """COLMAP ``sparse/<n>`` (+ ``images/``) → dataset.json with splatprep's reader / exporter. Distorted lenses
    need OpenCV (in the GPU runtime): such projects are re-solved instead (the caller falls back to an sfm job)."""
    from splatprep.colmap_model import read_largest_model  # type: ignore
    from splatprep.export import export_dataset  # type: ignore

    project = Path(project)
    sparse = project / "sparse" if (project / "sparse").is_dir() else project
    rec = read_largest_model(sparse)
    if rec is None:
        raise ValueError(f"no COLMAP model under {sparse}")
    if any(c.has_distortion() for c in rec.cameras.values()):
        raise NotImplementedError("distorted COLMAP cameras")
    image_dir = Path(images) if images else project / "images"
    work = Path(dataset_dir) / ".colmap-import"
    work.mkdir(parents=True, exist_ok=True)
    stats = export_dataset(rec, image_dir, Path(dataset_dir), work, progress=lambda f, m="": progress(f, m))
    shutil.rmtree(work, ignore_errors=True)
    return stats


# ------------------------------------------------------------------------------------------- mask stacking

MANUAL_LAYER = "splatprep-masks/manual"       # the mask editor's layer: masks/ (255 keep) + restore/ (255 = keep)
PIPELINE_LAYERS = ("frames", "static", "manual")


def manual_restore_dirs(dataset_dir: Path) -> List[Path]:
    d = Path(dataset_dir) / MANUAL_LAYER / "restore"
    return [d] if d.is_dir() else []


def mask_layers(dataset_dir: Path) -> List[Path]:
    """Every mask layer of a dataset (``splatprep-masks/<layer>/masks``), the mask editor's own last."""
    root = Path(dataset_dir) / "splatprep-masks"
    if not root.is_dir():
        return []
    ls = sorted((p / "masks" for p in root.iterdir() if (p / "masks").is_dir()),
                key=lambda p: (p.parent.name == "manual", p.parent.name))
    return ls


def prune_mask_layers(dataset_dir: Path, keep: List[str]) -> None:
    """Drop model layers of an earlier run that this run doesn't make (the editor's ``manual`` layer stays)."""
    root = Path(dataset_dir) / "splatprep-masks"
    if not root.is_dir():
        return
    for p in root.iterdir():
        if p.is_dir() and p.name not in keep and p.name != "manual":
            shutil.rmtree(p, ignore_errors=True)


def stack_masks(dataset_dir: Path, layers: List[Path], dilate: int = 4, out_name: str = "masks",
                restore: Optional[List[Path]] = None, only: Optional[List[str]] = None) -> Dict[str, str]:
    """AND every layer's ``<stem>.png`` (255 = keep) into ``dataset_dir/masks/<stem>.png``; ignored areas grow by
    ``dilate`` px; then ``restore`` layers (255 = keep whatever the others say: the editor's "paint back") are
    ORed in. ``only`` limits the work to those stems (the mask editor re-stacks just the edited images).
    Returns {image stem: relative mask path} for images that have any ignored pixel."""
    from PIL import Image, ImageFilter

    dataset_dir = Path(dataset_dir)
    out = dataset_dir / out_name
    out.mkdir(parents=True, exist_ok=True)
    stems: Dict[str, List[Path]] = {}
    for layer in layers:
        if not Path(layer).is_dir():
            continue
        for p in Path(layer).glob("*.png"):
            if only is None or p.stem in only:
                stems.setdefault(p.stem, []).append(p)
    restores: Dict[str, Path] = {}
    for d in restore or []:
        for p in Path(d).glob("*.png"):
            restores[p.stem] = p
    if only is not None:
        for stem in only:
            if stem not in stems:                  # no layer masks it any more
                (out / f"{stem}.png").unlink(missing_ok=True)
    result: Dict[str, str] = {}
    for stem, files in stems.items():
        keep = None
        for f in files:
            m = np.asarray(Image.open(f).convert("L")) >= 128
            if keep is None:
                keep = m
            elif m.shape == keep.shape:
                keep &= m
            else:
                keep &= np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(keep.shape[::-1])) >= 128
        if keep is None or keep.all():
            (out / f"{stem}.png").unlink(missing_ok=True)
            continue
        img = Image.fromarray(keep.astype(np.uint8) * 255)
        if dilate > 0:
            img = img.filter(ImageFilter.MinFilter(2 * dilate + 1))
        if stem in restores:
            r = Image.open(restores[stem]).convert("L")
            if r.size != img.size:
                r = r.resize(img.size, Image.NEAREST)
            arr = np.asarray(img) >= 128
            arr |= np.asarray(r) >= 128
            if arr.all():
                (out / f"{stem}.png").unlink(missing_ok=True)
                continue
            img = Image.fromarray(arr.astype(np.uint8) * 255)
        dst = out / f"{stem}.png"
        tmp = dst.with_name(dst.name + ".tmp.png")
        img.save(tmp)
        tmp.replace(dst)
        result[stem] = f"{out_name}/{stem}.png"
    return result


def sky_layer(dataset_dir: Path, min_fraction: float = 0.002) -> Dict[str, str]:
    """{image stem: sky mask path} for photos with at least ``min_fraction`` sky (255 = sky)."""
    from PIL import Image

    d = Path(dataset_dir) / "splatprep-sky" / "masks"
    out: Dict[str, str] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.png")):
        m = np.asarray(Image.open(p).convert("L")) >= 128
        if m.mean() > min_fraction:
            out[p.stem] = f"splatprep-sky/masks/{p.name}"
    return out


def finish_dataset(dataset_dir: Path, masks: Dict[str, str], sky: Dict[str, str], holdout_every: int = 0) -> Dict:
    """Record masks / sky / holdout in ``dataset.json``."""
    path = Path(dataset_dir) / "dataset.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    nm = ns = nh = 0
    for i, c in enumerate(doc.get("cameras", [])):
        stem = Path(c["image"]).stem
        if stem in masks:
            c["mask"] = masks[stem]
            nm += 1
        if stem in sky:
            c["sky"] = sky[stem]
            ns += 1
        if holdout_every and holdout_every > 1:
            c["holdout"] = (i % holdout_every == holdout_every - 1)
            nh += int(c["holdout"])
    tmp = path.with_name("dataset.json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"cameras": len(doc.get("cameras", [])), "masked": nm, "sky": ns, "holdout": nh}


def camera_groups_from_frames(dataset_dir: Path) -> Optional[Dict[str, int]]:
    """splatprep ``cameraGroups`` from ``frames.json`` (home video: frames of one clip / zoom setting share a camera)."""
    f = Path(dataset_dir) / "frames.json"
    if not f.is_file():
        return None
    fs = json.loads(f.read_text(encoding="utf-8"))
    g = {Path(fr["image"]).name: int(fr["cameraGroup"]) for fr in fs.get("frames", []) if "cameraGroup" in fr}
    return g or None


def clip_travel(dataset_dir: Path) -> Optional[Dict[str, float]]:
    """splatprep ``clipTravel`` from ``vhs.json`` (lets the solve leave out clips it joined by mistake)."""
    f = Path(dataset_dir) / "vhs.json"
    if not f.is_file():
        return None
    v = json.loads(f.read_text(encoding="utf-8"))
    t = {str(c["id"]): float(c["cameraTravel"]) for c in v.get("clips", []) if c.get("cameraTravel") is not None}
    return t or None


def home_video_report(dataset_dir: Path) -> List[str]:
    """Plain-words lines about the clips splatprep found (``vhs.json``), for the project log."""
    f = Path(dataset_dir) / "vhs.json"
    if not f.is_file():
        return []
    v = json.loads(f.read_text(encoding="utf-8"))
    used = set(v.get("used") or [])
    out = [f"home video: {v.get('width')}×{v.get('height')}, pixel aspect {v.get('pixelAspect')}, "
           f"{v.get('uniqueFps')} unique frames/s, {len(v.get('clips', []))} clips"]
    for c in v.get("clips", []):
        if c.get("verdict") in ("short", "noSignal"):
            continue
        mark = "used" if c["id"] in used else ("recommended, not used" if c["id"] == v.get("recommended") else "not used")
        out.append(f"  clip {c['id'] + 1}: {c['start']:.0f}–{c['end']:.0f} s, {c['reason']} — {mark}"
                   + (f", {c['selected']} frames" if c.get("selected") else ""))
    return out


def rig_from_frames(dataset_dir: Path) -> Optional[Dict]:
    """splatprep ``rig`` hint for 360° views from ``frames.json`` (one group per 360° frame)."""
    f = Path(dataset_dir) / "frames.json"
    if not f.is_file():
        return None
    fs = json.loads(f.read_text(encoding="utf-8"))
    if not fs.get("isMultiViewRig"):
        return None
    groups: Dict[int, Dict[str, list]] = {}
    intr = None
    for fr in fs.get("frames", []):
        if "group" not in fr or "rotationToRig" not in fr:
            continue
        groups.setdefault(int(fr["group"]), {})[Path(fr["image"]).name] = fr["rotationToRig"]
        if intr is None and fr.get("focalPx"):
            intr = [fr["focalPx"], fr["focalPx"], fr.get("cx", fr["width"] / 2), fr.get("cy", fr["height"] / 2)]
    if not groups:
        return None
    rig = {"kind": "multiview", "groups": [groups[k] for k in sorted(groups)]}
    if intr:
        rig["intrinsics"] = intr
    return rig
