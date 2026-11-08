"""Training images in memory without exhausting it.

Large sets (a 360° walk cut into 9,824 views is ~56 GB of pixels and masks) used to be loaded whole into host RAM
and then copied again into pinned (unswappable) memory as each view was first used — the PC ran out of memory
and froze part-way through training. Now:

* ``gpu``   — everything fits a small budget: uint8 pixels resident on the GPU (as before);
* ``ram``   — fits comfortably in RAM: loaded once, kept as numpy (no pinned second copy);
* ``disk``  — larger: decoded from disk on demand into an LRU cache capped at a share of available RAM, with
              background threads decoding the views the shuffled order will use next.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

GPU_BUDGET = 3 * 1024 ** 3


def available_ram() -> int:
    """MemAvailable (Linux), else physical RAM / 2, else 16 GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (ValueError, OSError, AttributeError):
        return 16 * 1024 ** 3


def view_bytes(v) -> int:
    """Upper estimate for one view in memory: RGBA uint8 (+ float32 depth)."""
    return v.width * v.height * (4 + (4 if getattr(v.camera, "depth", None) else 0))


def choose_mode(estimated_total: int, ram: Optional[int] = None) -> str:
    forced = os.environ.get("SPLATWORKER_VIEW_MODE")          # gpu | ram | disk (testing)
    if forced in ("gpu", "ram", "disk"):
        return forced
    ram = available_ram() if ram is None else ram
    if estimated_total < GPU_BUDGET:
        return "gpu"
    if estimated_total < 0.5 * ram:
        return "ram"
    return "disk"


class ViewStore:
    """``get(view)`` → (image uint8 HxWx3, alpha uint8 HxW | None, depth float32 HxW | None, mask uint8 HxW | None)."""

    def __init__(self, loader: Callable, views: Sequence, mode: str, budget: int, threads: int = 4):
        self.loader = loader          # header view → fully loaded view
        self.mode = mode
        self.budget = max(budget, 256 * 1024 ** 2)
        self.cache: "OrderedDict[int, Tuple]" = OrderedDict()
        self.bytes = 0
        self.pending: Dict[int, Future] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="views") if mode == "disk" else None
        self.loads = 0

    def _load(self, v) -> Tuple:
        full = self.loader(v)
        return full.image, full.alpha, full.depth, getattr(full, "mask", None)

    @staticmethod
    def _size(t: Tuple) -> int:
        return sum(a.nbytes for a in t if a is not None)

    def _put(self, key: int, t: Tuple) -> None:
        with self.lock:
            if key in self.cache:
                return
            self.cache[key] = t
            self.bytes += self._size(t)
            while self.bytes > self.budget and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.bytes -= self._size(old)

    def get(self, v) -> Tuple:
        if v.image is not None:                       # gpu / ram modes: pixels live on the view
            return v.image, v.alpha, v.depth, getattr(v, "mask", None)
        key = id(v)
        with self.lock:
            t = self.cache.get(key)
            if t is not None:
                self.cache.move_to_end(key)
                return t
            fut = self.pending.get(key)
        if fut is not None:
            t = fut.result()
        else:
            t = self._load(v)
            self.loads += 1
        self._put(key, t)
        with self.lock:
            self.pending.pop(key, None)
        return t

    def prefetch(self, views: Sequence) -> None:
        if self.pool is None:
            return
        for v in views:
            key = id(v)
            with self.lock:
                if key in self.cache or key in self.pending:
                    continue
                self.pending[key] = self.pool.submit(self._load, v)
                self.loads += 1

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
