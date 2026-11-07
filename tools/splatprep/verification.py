"""Two-view geometric verification with OpenCV (fundamental matrix, MAGSAC++/RANSAC).

Produces the inlier matches written to ``two_view_geometries`` so the COLMAP /
GLOMAP mapper can run directly (no ``colmap matches_importer`` needed).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Hashable, Optional, Tuple

import numpy as np

MIN_INLIERS = 15


def verify_pair(
    kp0: np.ndarray,
    kp1: np.ndarray,
    matches: np.ndarray,
    threshold_px: float = 4.0,
    confidence: float = 0.9999,
    min_inliers: int = MIN_INLIERS,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Returns (F or None, boolean inlier mask over ``matches``)."""
    import cv2

    matches = np.asarray(matches).reshape(-1, 2)
    mask = np.zeros(len(matches), bool)
    if len(matches) < max(8, min_inliers):
        return None, mask
    p0 = np.asarray(kp0, np.float64)[matches[:, 0], :2]
    p1 = np.asarray(kp1, np.float64)[matches[:, 1], :2]
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    try:
        F, inl = cv2.findFundamentalMat(p0, p1, method, threshold_px, confidence, 10000)
    except cv2.error:
        F, inl = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, threshold_px, confidence)
    if F is None or inl is None or F.shape != (3, 3):
        return None, mask
    mask = inl.ravel().astype(bool)
    if mask.sum() < min_inliers:
        return None, np.zeros(len(matches), bool)
    return F, mask


def verify_pairs(
    kps,
    raw: Dict[Hashable, np.ndarray],
    threshold_px: float = 4.0,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[Hashable, np.ndarray], Dict[Hashable, np.ndarray]]:
    """``verify_pair`` over every pair, in parallel threads (OpenCV releases the GIL inside
    ``findFundamentalMat``, so this scales with cores — 10k+ pairs from big video / 360° sets took minutes on one).
    Returns (inlier masks for every pair, F for the pairs that passed); results don't depend on ``workers``."""
    items = list(raw.items())
    workers = max(1, workers or min(32, (os.cpu_count() or 4)))
    inliers: Dict[Hashable, np.ndarray] = {}
    Fs: Dict[Hashable, np.ndarray] = {}

    def one(item):
        (i, j), m = item
        return item[0], verify_pair(kps[i], kps[j], m, threshold_px=threshold_px)

    done = 0
    if workers == 1 or len(items) < 64:
        results = map(one, items)
    else:
        try:
            import cv2
            cv2.setNumThreads(1)   # parallel over pairs, not inside each solve
        except Exception:  # pragma: no cover
            pass
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(one, items, chunksize=16)
    try:
        for pair, (F, inl) in results:
            inliers[pair] = inl
            if F is not None:
                Fs[pair] = F
            done += 1
            if progress is not None and (done % 200 == 0 or done == len(items)):
                progress(done, len(items))
    finally:
        if workers > 1 and len(items) >= 64:
            pool.shutdown(wait=True)
    return inliers, Fs
