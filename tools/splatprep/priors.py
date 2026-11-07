"""Prior camera poses (ARKit or known poses) — ``ProcessingRequest.priorPoses``.

Accepted files (auto-detected):

1. ``.splatcapture`` ``manifest.json`` (PROTOCOL §1): ``frames[].cameraToWorld`` is
   16 floats COLUMN-MAJOR in the ARKit camera convention.
2. ``dataset.json`` (PROTOCOL §2, ``"format": "splatdataset"``): row-major OpenCV.
3. ``priors.json``::

       {"format": "splatpriors", "convention": "opencv" | "arkit",
        "frames": [{"image": "000000.jpg",
                    "cameraToWorld": [[...4],[...4],[...4],[...4]]   // nested = ROW-major
                                   | [16 floats],                   // flat = COLUMN-major (simd)
                    "intrinsics": [fx, fy, cx, cy],                 // optional
                    "timestamp": 1.23}]}                            // optional

Frames are matched to images by basename (then by stem).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .geometry import arkit_to_opencv_c2w


@dataclass
class PriorFrame:
    image: str
    c2w: np.ndarray  # OpenCV convention, 4x4
    intrinsics: Optional[np.ndarray] = None  # fx fy cx cy (pixels of the original image)
    timestamp: Optional[float] = None


def _parse_matrix(m, column_major_if_flat: bool = True) -> np.ndarray:
    a = np.asarray(m, float)
    if a.shape == (4, 4):
        return a
    if a.size == 16:
        return a.reshape(4, 4).T if column_major_if_flat else a.reshape(4, 4)
    if a.shape == (3, 4):
        return np.vstack([a, [0, 0, 0, 1]])
    raise ValueError(f"bad cameraToWorld shape {a.shape}")


def load_priors(path: Path) -> List[PriorFrame]:
    data = json.loads(Path(path).read_text())
    frames: List[PriorFrame] = []
    fmt = data.get("format")
    if fmt == "splatdataset":
        for cam in data.get("cameras", []):
            K = np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], float)
            frames.append(PriorFrame(cam["image"], _parse_matrix(cam["cameraToWorld"], False), K))
        return frames
    convention = data.get("convention") or data.get("coordinateSystem") or ("arkit" if fmt == "splatcapture" else "opencv")
    for fr in data.get("frames", []):
        if fr.get("trackingState", "normal") == "notAvailable":
            continue
        c2w = _parse_matrix(fr["cameraToWorld"], True)
        if convention.lower() == "arkit":
            c2w = arkit_to_opencv_c2w(c2w)
        K = np.asarray(fr["intrinsics"], float) if fr.get("intrinsics") is not None else None
        frames.append(PriorFrame(fr["image"], c2w, K, fr.get("timestamp")))
    return frames


def match_priors(frames: List[PriorFrame], names: List[str]) -> Dict[str, PriorFrame]:
    """Map image names (as listed in imagesDir) to prior frames."""
    by_base = {Path(f.image).name: f for f in frames}
    by_stem = {Path(f.image).stem: f for f in frames}
    out = {}
    for n in names:
        f = by_base.get(Path(n).name) or by_stem.get(Path(n).stem)
        if f is not None:
            out[n] = f
    return out
