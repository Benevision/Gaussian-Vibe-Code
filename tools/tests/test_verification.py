"""Parallel two-view verification gives the same answer as one thread."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from splatprep.verification import verify_pairs


def _pair(rng, n=200, outliers=0.3):
    # points on a plane seen by two cameras: an exact homography → consistent fundamental matrix
    p0 = rng.uniform(0, 1000, (n, 2))
    H = np.array([[1.02, 0.01, 12], [-0.015, 0.99, -7], [1e-5, 2e-5, 1]])
    q = np.c_[p0, np.ones(n)] @ H.T
    p1 = q[:, :2] / q[:, 2:3] + rng.normal(0, 0.3, (n, 2))
    bad = rng.random(n) < outliers
    p1[bad] = rng.uniform(0, 1000, (bad.sum(), 2))
    return p0, p1, np.c_[np.arange(n), np.arange(n)]


def test_parallel_matches_serial():
    rng = np.random.default_rng(3)
    kps, raw = [], {}
    for i in range(0, 160, 2):
        p0, p1, m = _pair(rng)
        kps += [p0, p1]
        raw[(i, i + 1)] = m
    raw[(0, 3)] = np.zeros((3, 2), int)          # too few matches → rejected
    a_inl, a_F = verify_pairs(kps, raw, workers=1)
    b_inl, b_F = verify_pairs(kps, raw, workers=8)
    assert set(a_inl) == set(b_inl) == set(raw)
    assert set(a_F) == set(b_F) and (0, 3) not in a_F and len(a_F) >= 70
    calls = []
    verify_pairs(kps, raw, workers=4, progress=lambda k, n: calls.append((k, n)))
    assert calls[-1] == (len(raw), len(raw))
