"""Multi-view rigs (360° video cut into perspective views): rig-aware pair selection and mapping.

A 360° frame cut into V views is one camera position with V known, fixed view rotations (all views share the
optical centre). Treated as V·F unrelated photos, matching pairs every view with every look-alike view and the
mapper solves V·F poses. With the rig:

* pairs are chosen per frame: neighbouring view directions in the next frame, the same direction a few frames
  further on, plus a few retrieval pairs between distant frames for loop closure — instead of top-K retrieval
  over all V·F views;
* the mapper registers whole frames (COLMAP ≥ 3.12 rigs: ``image_prefix`` sensors, frames grouped by the rest of
  the name), so it solves F poses with the view rotations held fixed, and with the exactly known intrinsics.

Request (from Splat Studio)::

    "rig": {"kind": "multiview",
            "groups": [{"000000_v00.jpg": R, "000000_v01.jpg": R, ...}, ...],   # one dict per frame, in time order
            "intrinsics": [fx, fy, cx, cy]}                                      # optional, shared by all views

``R`` is the 3×3 row-major rotation rig_from_cam (a direction in the view's camera frame — x right, y down,
z forward — times R gives the direction in the 360° frame).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Pair = Tuple[int, int]


@dataclass
class RigLayout:
    names: List[str]                       # original image names (ctx.names order)
    sensor: List[int]                      # per image: sensor index
    frame: List[int]                       # per image: frame index (time order)
    rotations: List[np.ndarray]            # per sensor: rig_from_cam 3×3
    intrinsics: Optional[List[float]] = None
    db_names: List[str] = field(default_factory=list)   # "sNN/FFFFFF.ext" names used in the database / image tree

    @property
    def num_sensors(self) -> int:
        return len(self.rotations)

    @property
    def num_frames(self) -> int:
        return max(self.frame) + 1 if self.frame else 0

    def directions(self) -> np.ndarray:
        """Optical axis of each sensor in the rig frame."""
        return np.stack([R[:, 2] for R in self.rotations])

    def original_name(self) -> Dict[str, str]:
        return {d: n for d, n in zip(self.db_names, self.names)}


def parse_rig(request: dict, names: Sequence[str]) -> Optional[RigLayout]:
    """RigLayout for a multiview rig request covering (most of) ``names``; None otherwise.

    Images outside the groups (or groups missing a view) are fine: they just form incomplete frames."""
    rig = request.get("rig") or {}
    if (rig.get("kind") or "").lower() != "multiview" or not rig.get("groups"):
        return None
    index = {Path(n).name: i for i, n in enumerate(names)}
    sensors: List[np.ndarray] = []
    sensor = [-1] * len(names)
    frame = [-1] * len(names)
    f = 0
    for g in rig["groups"]:
        used = False
        for img, R in g.items():
            i = index.get(Path(img).name)
            if i is None:
                continue
            R = np.asarray(R, np.float64).reshape(3, 3)
            s = next((k for k, S in enumerate(sensors) if np.abs(S - R).max() < 1e-3), None)
            if s is None:
                sensors.append(R)
                s = len(sensors) - 1
            sensor[i], frame[i] = s, f
            used = True
        f += used
    covered = sum(1 for s in sensor if s >= 0)
    if covered < 0.9 * len(names) or len(sensors) < 2 or f < 2:
        return None
    # Images not in any group (should not happen for 360° sets) become their own frame of sensor 0's camera
    # would break the rig, so a rig is only used when every image is covered.
    if covered != len(names):
        return None
    ext = {i: Path(n).suffix or ".jpg" for i, n in enumerate(names)}
    lay = RigLayout(list(names), sensor, frame, sensors,
                    [float(v) for v in rig["intrinsics"]] if rig.get("intrinsics") else None)
    lay.db_names = [f"s{sensor[i]:02d}/{frame[i]:06d}{ext[i]}" for i in range(len(names))]
    return lay


def rig_pairs(lay: RigLayout, descriptors: Optional[np.ndarray] = None, window: int = 3,
              neighbour_deg: float = 50.0, loop_k: int = 2) -> Tuple[List[Pair], dict]:
    """Pairs for a rig sequence.

    * frame t ↔ t+1: view pairs whose directions differ by ≤ ``neighbour_deg`` (same and adjacent directions);
    * frame t ↔ t+2…t+window: the same direction only;
    * loop closure: each view's ``loop_k`` most similar views (global descriptors) in frames more than
      ``window`` apart.
    Views of the same frame share the optical centre (no parallax), so they are never paired."""
    by_fs: Dict[Tuple[int, int], int] = {(lay.frame[i], lay.sensor[i]): i for i in range(len(lay.names))}
    D = lay.directions()
    ang = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1)))
    near = [np.nonzero(ang[s] <= neighbour_deg)[0] for s in range(lay.num_sensors)]
    pairs: Set[Pair] = set()
    F = lay.num_frames
    for t in range(F):
        for k in range(1, window + 1):
            if t + k >= F:
                break
            for a in range(lay.num_sensors):
                i = by_fs.get((t, a))
                if i is None:
                    continue
                for b in (near[a] if k == 1 else [a]):
                    j = by_fs.get((t + k, int(b)))
                    if j is not None:
                        pairs.add((min(i, j), max(i, j)))
    seq = len(pairs)
    loops = 0
    if descriptors is not None and loop_k > 0 and F > 2 * window + 1:
        fr = np.asarray(lay.frame)
        sim = descriptors @ descriptors.T
        far = np.abs(fr[:, None] - fr[None, :]) > window
        sim = np.where(far, sim, -np.inf)
        for i in range(len(lay.names)):
            row = sim[i]
            if not np.isfinite(row).any():
                continue
            for j in np.argpartition(-row, min(loop_k, len(row) - 1))[:loop_k]:
                if np.isfinite(row[j]):
                    p = (min(i, int(j)), max(i, int(j)))
                    if p not in pairs:
                        pairs.add(p)
                        loops += 1
    return sorted(pairs), {"strategy": "rig", "resolved": "rig", "sequentialPairs": seq, "loopPairs": loops,
                           "numPairs": len(pairs), "frames": F, "sensors": lay.num_sensors}


def link_tree(lay: RigLayout, image_dir: Path, out: Path) -> Path:
    """``out/sNN/FFFFFF.ext`` → original images (symlinks; copies if links are not allowed)."""
    import shutil

    for d, n in zip(lay.db_names, lay.names):
        dst = out / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src = (Path(image_dir) / n).resolve()
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return out


def _quat_wxyz(R: np.ndarray) -> List[float]:
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def write_rig_config(lay: RigLayout, path: Path) -> Path:
    """COLMAP rig config (``colmap rig_configurator`` / ``pycolmap.read_rig_config``).

    Sensor 0 is the reference; every sensor's pose relative to it is the known view rotation (zero baseline)."""
    R0 = lay.rotations[0]
    cams = []
    for s, R in enumerate(lay.rotations):
        cam = {"image_prefix": f"s{s:02d}/"}
        if s == 0:
            cam["ref_sensor"] = True
        else:
            # cam_from_ref = cam_from_rig360 · rig360_from_ref = Rᵀ · R0
            cam["cam_from_rig_rotation"] = _quat_wxyz(R.T @ R0)
            cam["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cams.append(cam)
    path.write_text(json.dumps([{"cameras": cams}], indent=1))
    return path
