"""Non-destructive editing: a deletion mask with undo / redo, a selection, a crop box and the threshold tools
(``edit.*`` settings). Applying edits produces a new :class:`GaussianCloud` only on export / save."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .splats import GaussianCloud

UNDO_LEVELS = 30


@dataclass
class CropBox:
    lo: np.ndarray
    hi: np.ndarray
    enabled: bool = False

    def inside(self, positions: np.ndarray) -> np.ndarray:
        return ((positions >= self.lo) & (positions <= self.hi)).all(1)


class EditState:
    def __init__(self, cloud: GaussianCloud):
        self.cloud = cloud
        n = cloud.count
        self.deleted = np.zeros(n, bool)
        self.selected = np.zeros(n, bool)
        lo, hi = cloud.robust_bounds(0.0) if n else (np.zeros(3), np.zeros(3))
        self.crop = CropBox(lo.astype(np.float32), hi.astype(np.float32))
        self._undo: List[np.ndarray] = []
        self._redo: List[np.ndarray] = []

    # ------------------------------------------------------------- history
    def _push(self) -> None:
        self._undo.append(np.packbits(self.deleted))
        if len(self._undo) > UNDO_LEVELS:
            self._undo.pop(0)
        self._redo.clear()

    def _unpack(self, packed) -> np.ndarray:
        return np.unpackbits(packed, count=self.cloud.count).astype(bool)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(np.packbits(self.deleted))
        self.deleted = self._unpack(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(np.packbits(self.deleted))
        self.deleted = self._unpack(self._redo.pop())
        return True

    def delete(self, mask: np.ndarray) -> int:
        """Delete splats where ``mask``; returns how many were newly deleted."""
        new = mask & ~self.deleted
        k = int(new.sum())
        if k:
            self._push()
            self.deleted = self.deleted | mask
        self.selected &= ~self.deleted
        return k

    def restore_all(self) -> None:
        if self.deleted.any():
            self._push()
            self.deleted = np.zeros(self.cloud.count, bool)

    # ------------------------------------------------------------- views
    def hidden(self) -> np.ndarray:
        """Not drawn: deleted, or outside an enabled crop box."""
        h = self.deleted.copy()
        if self.crop.enabled:
            h |= ~self.crop.inside(self.cloud.positions)
        return h

    @property
    def visible_count(self) -> int:
        return int(self.cloud.count - self.hidden().sum())

    def result(self, only_visible: bool = True) -> GaussianCloud:
        keep = ~self.hidden() if only_visible else ~self.deleted
        return self.cloud if keep.all() else self.cloud.subset(keep)

    # ------------------------------------------------------------- tools
    def select_brush(self, pixels: np.ndarray, depths: np.ndarray, x: float, y: float, radius: float,
                     deselect: bool = False) -> int:
        """Select (or deselect) the visible splats whose projected centre is within ``radius`` px of (x, y)."""
        d2 = (pixels[:, 0] - x) ** 2 + (pixels[:, 1] - y) ** 2
        hit = (d2 <= radius * radius) & (depths > 0) & ~self.hidden()
        if deselect:
            self.selected &= ~hit
        else:
            self.selected |= hit
        return int(hit.sum())

    def invert_selection(self) -> None:
        self.selected = ~self.selected & ~self.hidden()

    def clear_selection(self) -> None:
        self.selected[:] = False

    def delete_selected(self) -> int:
        k = self.delete(self.selected.copy())
        self.selected[:] = False
        return k

    def apply_crop(self) -> int:
        k = self.delete(~self.crop.inside(self.cloud.positions))
        self.crop.enabled = False
        return k

    def below_opacity(self, threshold: float) -> np.ndarray:
        return self.cloud.opacities < threshold

    def larger_than(self, world_size: float) -> np.ndarray:
        """Largest axis (3σ extent) above ``world_size``."""
        return np.exp(self.cloud.log_scales.max(1)) * 3 > world_size

    def isolated(self, neighbours: int = 4, radius: Optional[float] = None) -> np.ndarray:
        """Splats with fewer than ``neighbours`` other centres within ``radius`` (default: 1 % of the scene's
        robust size). Grid hashing: each splat is compared with the 27 cells around it."""
        pos = self.cloud.positions
        alive = ~self.deleted
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            return np.zeros(len(pos), bool)
        if radius is None:
            lo, hi = self.cloud.robust_bounds(0.02)
            radius = max(float(np.linalg.norm(hi - lo)) * 0.01, 1e-6)
        p = pos[idx]
        cell = np.floor(p / radius).astype(np.int64)
        cell -= cell.min(0)
        dims = cell.max(0) + 3
        key = lambda c: (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]  # noqa: E731
        k = key(cell + 1)
        order = np.argsort(k, kind="stable")
        ks = k[order]
        counts = np.zeros(len(p), np.int64)
        r2 = radius * radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = key(cell + 1 + np.array([dx, dy, dz]))
                    start = np.searchsorted(ks, nk, "left")
                    end = np.searchsorted(ks, nk, "right")
                    m = end - start
                    if not m.any():
                        continue
                    # cells are small: iterate over the j-th member of each neighbour cell
                    for j in range(int(m.max())):
                        has = m > j
                        if not has.any():
                            break
                        src = np.nonzero(has)[0]
                        other = order[start[src] + j]
                        d2 = ((p[src] - p[other]) ** 2).sum(1)
                        counts[src] += (d2 <= r2) & (other != src)
                        if j > 64:          # dense cell: plenty of neighbours anyway
                            counts[src] += m[src] - j - 1
                            break
        out = np.zeros(len(pos), bool)
        out[idx] = counts < neighbours
        return out


def estimate_size(count: int, fmt: str, sh_degree: int) -> int:
    """Bytes a file of ``count`` splats takes (``export`` size estimate)."""
    k = (sh_degree + 1) ** 2 - 1
    if fmt == "ply":
        return 300 + count * 4 * (17 + 3 * k)
    if fmt == "compressed.ply":
        return 1000 + count * 16 + (count + 255) // 256 * 72 + count * 3 * k
    if fmt == "splat":
        return count * 32
    if fmt == "spz":     # gzip on the quantised stream: roughly 0.75 of the raw size
        return int((16 + count * (9 + 1 + 3 + 3 + 3 + 3 * k)) * 0.75)
    return 0


def pick_point(pixels: np.ndarray, depths: np.ndarray, opacities: np.ndarray, hidden: np.ndarray,
               x: float, y: float, radius: float = 6.0) -> Optional[np.ndarray]:
    """Index of the nearest reasonably opaque splat under (x, y), for double-click → orbit centre."""
    d2 = (pixels[:, 0] - x) ** 2 + (pixels[:, 1] - y) ** 2
    cand = np.nonzero((d2 <= radius * radius) & (depths > 0) & ~hidden & (opacities > 0.3))[0]
    if len(cand) == 0:
        return None
    # the front-most 10 % of the hits, median: robust against a lone floater
    dz = depths[cand]
    front = cand[dz <= np.quantile(dz, 0.1)]
    return front


def brush_hits(pixels: np.ndarray, x: float, y: float, radius: float) -> Tuple[np.ndarray, int]:
    d2 = (pixels[:, 0] - x) ** 2 + (pixels[:, 1] - y) ** 2
    hit = d2 <= radius * radius
    return hit, int(hit.sum())
