"""Back-to-front depth order for the viewer, computed off the UI thread.

Keys are 16-bit logarithmic depths along the view direction (accurate to ~0.02 % of the distance, as the Mac's
fast sort), so ``np.argsort(kind="stable")`` runs numpy's radix sort: ~40 ms for 6 M splats. Splats behind
the camera (or deleted / cropped away: ``hidden``) go to the end and are not drawn.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import numpy as np


def depth_order(positions: np.ndarray, cam_pos, forward, near: float = 1e-3,
                hidden: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """(order uint32 far → near with invisible splats last, number of drawable splats)."""
    f = np.asarray(forward, np.float32)
    d = positions @ f - np.float32(np.asarray(cam_pos, np.float32) @ f)
    vis = d > near
    if hidden is not None:
        vis &= ~hidden
    n_vis = int(vis.sum())
    if n_vis == 0:
        return np.zeros(0, np.uint32), 0
    dv = d[vis]
    lo, hi = float(dv.min()), float(dv.max())
    llo, lhi = np.log(max(lo, near)), np.log(max(hi, near * 1.0001))
    span = max(lhi - llo, 1e-9)
    q = np.clip((np.log(np.maximum(d, near)) - llo) / span * 65534.0, 0, 65534).astype(np.uint16)
    keys = np.where(vis, np.uint16(65534) - q, np.uint16(65535)).astype(np.uint16)
    order = np.argsort(keys, kind="stable").astype(np.uint32)
    return order, n_vis


class BackgroundSorter:
    """Sorts on a worker thread; ``request`` coalesces (only the newest camera is sorted)."""

    def __init__(self, on_done: Callable[[int, np.ndarray, int], None]):
        self.on_done = on_done
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self._generation = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="splat-sort")
        self._thread.start()

    def request(self, positions, cam_pos, forward, near, hidden=None, generation: int = 0) -> None:
        with self._cv:
            self._pending = (positions, np.array(cam_pos, np.float32), np.array(forward, np.float32),
                             float(near), hidden, generation)
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._pending = self._pending, None
            positions, pos, fwd, near, hidden, gen = job
            try:
                order, n = depth_order(positions, pos, fwd, near, hidden)
            except Exception:  # pragma: no cover - never kill the thread
                continue
            self.on_done(gen, order, n)
