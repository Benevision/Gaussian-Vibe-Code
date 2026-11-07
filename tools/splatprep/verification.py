"""Two-view geometric verification with OpenCV (fundamental matrix, MAGSAC++/RANSAC).

Produces the inlier matches written to ``two_view_geometries`` so the COLMAP /
GLOMAP mapper can run directly (no ``colmap matches_importer`` needed).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
