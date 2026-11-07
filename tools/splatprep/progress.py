"""Progress reporting (PROTOCOL.md §6).

Progress lines are ``{"stage": str, "progress": float, "message": str}`` where
``progress`` is the *overall* job progress in [0, 1] (the worker maps it to
``step/totalSteps = progress*1000``).  Stages are given weights so that the
overall value is monotonic.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Callable, Dict, IO, List, Optional, Sequence, Tuple

ProgressFn = Callable[[dict], None]


def null_progress(_: dict) -> None:
    pass


class JsonlWriter:
    """Writes one JSON object per line and flushes (thread safe)."""

    def __init__(self, stream: IO[str]):
        self.stream = stream
        self._lock = threading.Lock()

    def __call__(self, obj: dict) -> None:
        line = json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_json_default)
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)


class StagedProgress:
    """Maps per-stage progress to overall progress.

    >>> p = StagedProgress(cb, [("features", 0.4), ("matching", 0.4), ("export", 0.2)])
    >>> p.update("features", 0.5, "image 10/20")   # overall 0.2
    """

    def __init__(self, callback: Optional[ProgressFn], stages: Sequence[Tuple[str, float]]):
        self.callback = callback or null_progress
        total = sum(w for _, w in stages) or 1.0
        self.offsets: Dict[str, Tuple[float, float]] = {}
        acc = 0.0
        for name, w in stages:
            self.offsets[name] = (acc / total, w / total)
            acc += w
        self._last = 0.0
        self.messages: List[str] = []

    def update(self, stage: str, fraction: float = 0.0, message: str = "") -> None:
        off, w = self.offsets.get(stage, (self._last, 0.0))
        fraction = min(max(float(fraction), 0.0), 1.0)
        overall = max(self._last, min(1.0, off + w * fraction))
        self._last = overall
        self.callback({"stage": stage, "progress": round(overall, 4), "message": message})

    def message(self, stage: str, message: str) -> None:
        """Informational line that does not advance progress."""
        self.messages.append(message)
        self.callback({"stage": stage, "progress": round(self._last, 4), "message": message})

    def sub(self, stage: str) -> Callable[[float, str], None]:
        return lambda frac, msg="": self.update(stage, frac, msg)


def stderr_log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()
