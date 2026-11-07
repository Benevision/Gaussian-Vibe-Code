"""Parsing of mask prompts and the mask-related request keys (pure Python, no torch).

``maskPrompts`` (new, any frame)::

    [{"image": "000123.jpg",                 # file name (relative path, basename or stem)
      "points": [[x, y, label], ...],        # pixels of THAT image, label 1 = object, 0 = background
      "boxes": [[x0, y0, x1, y1], ...],      # pixels of that image
      "objectId": 3}]                        # optional; entries sharing an id are the same object

``maskPrompt`` (legacy): ``{"points": [...], "boxes": [...]}`` on the first image (object 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..errors import RequestError

DEFAULT_DYNAMIC_CLASSES = [
    "person", "dog", "cat", "bird", "car", "truck", "bus", "bicycle", "motorcycle",
    "stroller", "shopping cart", "umbrella", "bag",
]
DEFAULT_THRESHOLD = 0.25
DEFAULT_DILATE = 8


@dataclass
class FramePrompt:
    """Prompt for one object on one frame (coordinates in that frame's original pixels)."""

    frame: int
    object_id: int
    points: List[List[float]] = field(default_factory=list)  # [x, y]
    labels: List[int] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)  # [x0, y0, x1, y1]
    source: str = "user"  # "user" | "detector" | "auto"

    def empty(self) -> bool:
        return not self.points and not self.boxes

    def scaled(self, sx: float, sy: float) -> "FramePrompt":
        return FramePrompt(
            self.frame,
            self.object_id,
            [[p[0] * sx, p[1] * sy] for p in self.points],
            list(self.labels),
            [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in self.boxes],
            self.source,
        )

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "objectId": self.object_id,
            "points": [[float(p[0]), float(p[1]), int(l)] for p, l in zip(self.points, self.labels)],
            "boxes": [[float(v) for v in b] for b in self.boxes],
            "source": self.source,
        }


def _num(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise RequestError(f"{what}: not a number: {v!r}") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise RequestError(f"{what}: not finite: {v!r}")
    return f


def parse_points(raw, what: str = "points"):
    pts, labels = [], []
    for k, p in enumerate(raw or []):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise RequestError(f"{what}[{k}] must be [x, y] or [x, y, label]")
        pts.append([_num(p[0], what), _num(p[1], what)])
        lab = int(_num(p[2], what)) if len(p) > 2 else 1
        if lab not in (0, 1):
            raise RequestError(f"{what}[{k}]: label must be 0 (background) or 1 (object)")
        labels.append(lab)
    return pts, labels


def parse_boxes(raw, what: str = "boxes"):
    out = []
    for k, b in enumerate(raw or []):
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise RequestError(f"{what}[{k}] must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (_num(v, what) for v in b)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise RequestError(f"{what}[{k}] is empty: {list(b)}")
        out.append([x0, y0, x1, y1])
    return out


def resolve_image(name: str, names: Sequence[str]) -> int:
    """Index of ``name`` in ``names``: exact relative path, then basename, then stem."""
    if not isinstance(name, str) or not name:
        raise RequestError("maskPrompts[].image must be a file name")
    if name in names:
        return list(names).index(name)
    base = Path(name).name
    hits = [i for i, n in enumerate(names) if Path(n).name == base]
    if len(hits) == 1:
        return hits[0]
    stem = Path(name).stem
    hits = [i for i, n in enumerate(names) if Path(n).stem == stem]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RequestError(f"maskPrompts image {name!r} is ambiguous ({len(hits)} matches)")
    raise RequestError(f"maskPrompts image not found: {name!r}")


def parse_mask_prompts(req: dict, names: Sequence[str]) -> List[FramePrompt]:
    """``maskPrompts`` (+ legacy ``maskPrompt`` on the first image) → FramePrompts.

    Object ids: explicit ``objectId`` or 1, 2, ... in order of appearance (ids never collide
    with explicit ones).  Several entries with the same id on different frames prompt the same
    object on each of those frames; on the same frame they are merged.
    """
    out: List[FramePrompt] = []
    raw = req.get("maskPrompts")
    if raw is not None and not isinstance(raw, list):
        raise RequestError("maskPrompts must be a list")
    entries = list(raw or [])
    legacy = req.get("maskPrompt")
    if legacy:
        if not isinstance(legacy, dict):
            raise RequestError("maskPrompt must be an object {points, boxes}")
        if not names:
            raise RequestError("maskPrompt given but there are no images")
        entries.insert(0, dict(legacy, image=names[0], objectId=legacy.get("objectId", 1)))
    explicit = {int(_num(e["objectId"], "objectId")) for e in entries if isinstance(e, dict) and e.get("objectId") is not None}
    next_id = 1
    merged: Dict[tuple, FramePrompt] = {}
    for k, e in enumerate(entries):
        if not isinstance(e, dict):
            raise RequestError(f"maskPrompts[{k}] must be an object")
        if e.get("objectId") is not None:
            oid = int(_num(e["objectId"], "objectId"))
        else:
            while next_id in explicit:
                next_id += 1
            oid = next_id
            explicit.add(oid)
        frame = resolve_image(e.get("image") or "", names)
        pts, labels = parse_points(e.get("points"), f"maskPrompts[{k}].points")
        boxes = parse_boxes(e.get("boxes"), f"maskPrompts[{k}].boxes")
        if not pts and not boxes:
            raise RequestError(f"maskPrompts[{k}] has neither points nor boxes")
        key = (frame, oid)
        if key in merged:
            fp = merged[key]
            fp.points += pts
            fp.labels += labels
            fp.boxes += boxes
        else:
            fp = FramePrompt(frame, oid, pts, labels, boxes, "user")
            merged[key] = fp
            out.append(fp)
    return out


def parse_classes(req: dict) -> List[str]:
    raw = req.get("maskClasses")
    if raw is None:
        return list(DEFAULT_DYNAMIC_CLASSES)
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        raise RequestError("maskClasses must be a list of strings")
    out = []
    for c in raw:
        if not isinstance(c, str):
            raise RequestError("maskClasses must be a list of strings")
        c = c.strip()
        if c and c not in out:
            out.append(c)
    if not out:
        raise RequestError("maskClasses is empty")
    return out


def _choice(req: dict, key: str, allowed, default):
    v = req.get(key) or default
    if v not in allowed:
        raise RequestError(f"{key} must be one of {' | '.join(allowed)}")
    return v


def mask_options(req: dict) -> dict:
    """Normalised mask options from a request (defaults applied, types checked)."""

    def num(key, default, lo, hi, cast=float):
        v = req.get(key)
        if v is None:
            return default
        f = cast(_num(v, key))
        if not lo <= f <= hi:
            raise RequestError(f"{key} must be in [{lo}, {hi}], got {v!r}")
        return f

    seq = req.get("maskSequence") or "auto"
    if seq not in ("auto", "video", "unordered"):
        raise RequestError("maskSequence must be auto | video | unordered")
    return {
        "threshold": num("maskThreshold", DEFAULT_THRESHOLD, 0.0, 1.0),
        "dilate": num("dilatePixels", DEFAULT_DILATE, 0, 512, int),
        "chunk": num("maskChunkSize", 64, 4, 100000, int),
        "overlap": num("maskChunkOverlap", 2, 1, 64, int),
        "coverage": num("maskCoverage", 0.5, 0.0, 1.0),
        "maxCond": num("maskMaxCondFrames", 4, 1, 64, int),
        "detectStride": num("maskDetectStride", 1, 1, 1000, int),
        "stageMax": num("maskTrackResolution", 1024, 256, 4096, int),
        "sequence": seq,
        "replace": bool(req.get("replaceMasks", False)),
        "invert": bool(req.get("invertMask", False)),
        "bidirectional": bool(req.get("maskBidirectional", True)),
        "saveDetections": bool(req.get("saveDetections", True)),
        "offloadState": req.get("maskOffloadState"),
        "precision": _choice(req, "maskPrecision", ("auto", "fp32", "bf16", "fp16"), "auto"),
    }
