"""Auto clean plan from a contribution census — the rules of the Mac's ``AutoClean.swift``.

* Invisible: never adds ``PEAK_THRESHOLD`` (2 %) to any photo. Removing these leaves every photo unchanged.
* Seen by too few photos: visible in fewer than ``FEW_VIEWS_FRACTION`` (25 %) of the photos a typical clearly
  visible splat (≥ 2 %, opacity > 0.2) is seen in — a blob hanging in front of a few cameras. Skipped when the
  typical count is below ``MIN_TYPICAL_VIEWS`` (12; small captures).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PEAK_THRESHOLD = 0.02
FEW_VIEWS_FRACTION = 0.25
MIN_TYPICAL_VIEWS = 12


@dataclass
class Plan:
    remove: np.ndarray
    invisible: int
    few_views: int
    typical_views: int      # 0 = the few-photos rule was not applied

    @property
    def total(self) -> int:
        return int(self.remove.sum())


def plan(peak: np.ndarray, views: np.ndarray, opacity_logits: np.ndarray, include_few_views: bool = True,
         fraction: float = FEW_VIEWS_FRACTION) -> Plan:
    n = min(len(peak), len(views))
    peak = np.asarray(peak[:n], np.float32)
    views = np.asarray(views[:n], np.float32)
    stride = max(1, n // 300_000)
    idx = np.arange(0, n, stride)
    opa = 1 / (1 + np.exp(-np.asarray(opacity_logits[:n], np.float32)[idx])) if len(opacity_logits) else np.ones(len(idx))
    sample = views[idx][(peak[idx] >= PEAK_THRESHOLD) & (opa > 0.2)]
    typical = float(np.sort(sample)[len(sample) // 2]) if len(sample) else 0.0
    cut = fraction * typical if include_few_views and typical >= MIN_TYPICAL_VIEWS and fraction > 0 else 0.0
    invisible = peak < PEAK_THRESHOLD
    few = ~invisible & (views < cut)
    return Plan(remove=invisible | few, invisible=int(invisible.sum()), few_views=int(few.sum()),
                typical_views=int(typical) if cut > 0 else 0)
