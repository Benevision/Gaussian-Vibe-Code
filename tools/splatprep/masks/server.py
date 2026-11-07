"""Single-frame SAM 2 masks for click-to-mask UIs: ``task = "maskpreview"`` and the
long-running ``python -m splatprep maskserver`` JSONL loop.

maskserver protocol (one JSON object per line; stdout carries ONLY reply lines, exactly one
per request line, in order; start-up status and logs go to stderr):

    ← {"id": 7, "image": "/abs/000123.jpg", "points": [[x, y, 1], [x, y, 0]], "boxes": [[x0, y0, x1, y1]]}
    → {"id": 7, "mask": "/abs/path/preview_000123.png", "score": 0.97, "area": 0.081,
       "bbox": [x0, y0, x1, y1], "width": 1600, "height": 1200, "embeddingCached": true, "timeSec": 0.06}
       (PNG: 255 = OBJECT by default; "objectMask": false → keep convention, object = 0)
    ← {"cmd": "warm", "image": "..."}   → {"warmed": "/abs/...jpg", "timeSec": 0.9}
    ← {"cmd": "ping"}                   → {"pong": true}
    ← {"cmd": "quit"}                   → {"bye": true}   (EOF on stdin also quits)
    errors → {"id": 7, "error": "..."}  (the server keeps running; a SAM 2 load failure is
    reported as the error of every request)
    ``--ready`` additionally prints {"ready": true, ...} as the FIRST stdout line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..errors import RequestError
from .prompts import FramePrompt, parse_boxes, parse_points


def _bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class PreviewEngine:
    """SAM 2 image predictor + LRU cache of image embeddings (re-clicks on the same image only
    run the light prompt decoder: tens of ms instead of ~1 s)."""

    _STATE = ("_features", "_orig_hw", "_is_image_set", "_is_batch")

    def __init__(self, device: str, cache_size: int = 4, predictor=None, loader=None):
        from ..images import load_rgb

        t0 = time.perf_counter()
        if predictor is None:
            from .dynamic import load_sam2

            predictor = load_sam2("image", device)
        self.pred = predictor
        self.device = device
        self.load_sec = time.perf_counter() - t0
        self.loader = loader or (lambda p: load_rgb(p)[0])
        self.cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self.cache_size = cache_size
        self.current = None

    def _key(self, path: Path):
        st = path.stat()
        return (str(path.resolve()), st.st_mtime_ns, st.st_size)

    def set_image(self, path: Path):
        """Returns ((H, W), cached: bool)."""
        from .track import _no_grad

        key = self._key(path)
        if key == self.current:
            return self.cache[key]["hw"], True
        if key in self.cache:
            ent = self.cache.pop(key)
            self.cache[key] = ent
            for a in self._STATE:
                setattr(self.pred, a, ent["state"][a])
            self.current = key
            return ent["hw"], True
        rgb = self.loader(path)
        with _no_grad():
            self.pred.set_image(rgb)
        hw = rgb.shape[:2]
        self.cache[key] = {"hw": hw, "state": {a: getattr(self.pred, a, None) for a in self._STATE}}
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        self.current = key
        return hw, False

    def predict(self, path: Path, points, boxes):
        from .track import predict_image_prompt

        t0 = time.perf_counter()
        hw, cached = self.set_image(path)
        pts, labels = parse_points(points)
        bxs = parse_boxes(boxes)
        if not pts and not bxs:
            raise RequestError("request needs points and/or boxes")
        m, score = predict_image_prompt(self.pred, FramePrompt(0, 1, pts, labels, bxs))
        return m, score, hw, cached, time.perf_counter() - t0


def write_preview(m: np.ndarray, out: Path, object_mask: bool = False, invert: bool = False, dilate_px: int = 0) -> Path:
    """Preview PNG.  Default = the keep convention of the final masks (object → 0, 255 = keep);
    ``object_mask`` → 255 on the object; ``invert`` flips the keep mask like ``invertMask``."""
    from PIL import Image

    from . import dilate

    m = dilate(np.asarray(m, bool), dilate_px)
    img = m if object_mask else (m if invert else ~m)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    Image.fromarray(img.astype(np.uint8) * 255).save(tmp, format="PNG")
    os.replace(tmp, out)
    return out


def run_preview_request(req: dict, image_dir: Path, out_dir: Path, device: str, engine: Optional[PreviewEngine] = None) -> dict:
    """task = maskpreview: all ``maskPrompts`` entries on one image (``previewImage`` or the
    first entry's image) → ``out/preview.png``."""
    from ..images import list_images
    from .prompts import parse_mask_prompts

    names = list_images(image_dir)
    prompts = parse_mask_prompts(req, names)
    if not prompts:
        raise RequestError("maskpreview needs maskPrompts (or maskPrompt)")
    from .prompts import resolve_image

    frame = resolve_image(req["previewImage"], names) if req.get("previewImage") else prompts[0].frame
    sel = [p for p in prompts if p.frame == frame]
    if not sel:
        raise RequestError("no maskPrompts on previewImage")
    if engine is None:
        from .track import set_precision

        set_precision(device, req.get("maskPrecision") or "auto")
        engine = PreviewEngine(device)
    t0 = time.perf_counter()
    path = Path(image_dir) / names[frame]
    union = None
    score = []
    cached = False
    for p in sel:
        pts = [[x, y, l] for (x, y), l in zip(p.points, p.labels)]
        m, s, hw, cached, _ = engine.predict(path, pts, p.boxes)
        union = m if union is None else (union | m)
        score.append(s)
    out = write_preview(union, Path(out_dir) / "preview.png", bool(req.get("previewObjectMask")), bool(req.get("invertMask")),
                        int(req.get("dilatePixels") or 0))
    return {"task": "maskpreview", "status": "ok", "image": names[frame], "preview": str(out),
            "score": round(float(np.mean(score)), 4), "area": round(float(union.mean()), 5), "bbox": _bbox(union),
            "width": int(union.shape[1]), "height": int(union.shape[0]), "modelLoadSec": round(engine.load_sec, 3),
            "predictSec": round(time.perf_counter() - t0, 3), "embeddingCached": cached}


class MaskServer:
    """Request handler (I/O-free so it can be unit tested).  ``engine`` None + ``load_error``
    → every request is answered with that error (the server never dies on a load failure)."""

    def __init__(self, engine: Optional[PreviewEngine], images_dir: Optional[Path], out_dir: Path, load_error: str = ""):
        self.engine = engine
        self.load_error = load_error
        self.images_dir = Path(images_dir) if images_dir else None
        self.out_dir = Path(out_dir)

    def _image(self, name) -> Path:
        if not isinstance(name, str) or not name:
            raise RequestError("'image' (path) is required")
        p = Path(name).expanduser()
        if not p.is_absolute():
            if self.images_dir is None:
                raise RequestError(f"relative image path {name!r} needs --images")
            p = self.images_dir / p
        if not p.is_file():
            raise RequestError(f"image not found: {p}")
        return p

    def handle(self, req) -> Optional[dict]:
        """Returns the response object (None = quit)."""
        if not isinstance(req, dict):
            return {"error": "request must be a JSON object"}
        rid = req.get("id")
        base = {"id": rid} if rid is not None else {}
        try:
            cmd = req.get("cmd")
            if cmd == "quit":
                return None
            if cmd == "ping":
                return {**base, "pong": True}
            if self.engine is None:
                raise RuntimeError(self.load_error or "SAM 2 is not loaded")
            if cmd == "warm":
                t0 = time.perf_counter()
                p = self._image(req.get("image"))
                self.engine.set_image(p)
                return {**base, "warmed": str(p), "timeSec": round(time.perf_counter() - t0, 3)}
            if cmd not in (None, "predict"):
                raise RequestError(f"unknown cmd: {cmd!r}")
            p = self._image(req.get("image"))
            m, score, hw, cached, dt = self.engine.predict(p, req.get("points"), req.get("boxes"))
            out = req.get("out")
            out = Path(out).expanduser() if out else self.out_dir / f"preview_{p.stem}.png"
            write_preview(m, out, bool(req.get("objectMask", True)), bool(req.get("invertMask")), int(req.get("dilatePixels") or 0))
            return {**base, "mask": str(out.resolve()), "score": round(score, 4), "area": round(float(m.mean()), 5),
                    "bbox": _bbox(m), "width": int(hw[1]), "height": int(hw[0]), "embeddingCached": cached,
                    "timeSec": round(dt, 3)}
        except Exception as e:  # noqa: BLE001 - the server must survive bad requests
            return {**base, "error": str(e) or e.__class__.__name__}

    def serve(self, lines, write: Callable[[dict], None]) -> int:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as e:
                write({"error": f"bad JSON: {e}"})
                continue
            resp = self.handle(req)
            if resp is None:
                write({"bye": True})
                return 0
            write(resp)
        return 0


def serve_main(images_dir: Optional[str], out_dir: Optional[str], device_pref: Optional[str], writer, stdin=None,
               ready_line: bool = False, precision: str = "auto") -> int:
    """stdout gets exactly one reply line per request line.  The startup status
    (``{"ready": ...}``) goes to stderr, or to stdout too with ``ready_line`` (--ready)."""
    from ..device import select_device
    from .dynamic import sam2_model_id
    from .track import set_precision

    device = select_device(device_pref)
    out = Path(out_dir) if out_dir else Path(os.environ.get("TMPDIR", "/tmp")) / "splatprep_maskserver"
    out.mkdir(parents=True, exist_ok=True)
    engine, err = None, ""
    prec = set_precision(device, precision)
    try:
        engine = PreviewEngine(device)
        status = {"ready": True, "model": sam2_model_id(), "device": device, "precision": prec,
                  "loadSec": round(engine.load_sec, 3), "outDir": str(out.resolve())}
    except Exception as e:  # noqa: BLE001
        err = f"cannot load SAM 2: {e}"
        status = {"ready": False, "error": err}
    sys.stderr.write("maskserver " + json.dumps(status) + "\n")
    sys.stderr.flush()
    if ready_line:
        writer(status)
    return MaskServer(engine, Path(images_dir) if images_dir else None, out, err).serve(stdin or sys.stdin, writer)
