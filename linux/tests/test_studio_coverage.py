import json
import math

import numpy as np
import pytest

from splatworker.studio import coverage
from splatworker.studio.splats import GaussianCloud


def look_at(eye, target=(0, 0, 0), up=(0, -1, 0)):
    """Row-major OpenCV camera-to-world: +Z forward, +Y down."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, -np.asarray(up, float))
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, eye
    return m


def make_dataset(tmp_path, eyes, depth=None, w=160, h=120, fx=150.0):
    cams = []
    (tmp_path / "depth").mkdir(exist_ok=True)
    for i, e in enumerate(eyes):
        c = {"id": i, "image": f"images/{i:04d}.jpg", "width": w, "height": h, "fx": fx, "fy": fx,
             "cx": w / 2, "cy": h / 2, "cameraToWorld": look_at(e).tolist()}
        if depth is not None:
            dm = np.full((12, 16), depth, np.float16)
            (tmp_path / "depth" / f"{i}.bin").write_bytes(dm.astype("<f2").tobytes())
            c.update(depth=f"depth/{i}.bin", depthWidth=16, depthHeight=12)
        cams.append(c)
    (tmp_path / "dataset.json").write_text(json.dumps({"version": 1, "convention": "opencv", "cameras": cams}))
    return tmp_path


def cloud(pos, opacity=0.9, scale=0.01):
    n = len(pos)
    lo = math.log(opacity / (1 - opacity))
    return GaussianCloud(np.asarray(pos, np.float32), np.full((n, 3), math.log(scale)),
                         np.tile([1, 0, 0, 0], (n, 1)), np.full(n, lo), np.zeros((n, 1, 3)))


def ring(n, radius=4.0):
    return [(radius * math.sin(a), 0.0, -radius * math.cos(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def test_views_and_detail_counts(tmp_path):
    ds = make_dataset(tmp_path, ring(12))
    cams = coverage.load_cameras(ds)
    pts = np.array([[0, 0, 0], [0.2, 0.1, -0.1], [50, 0, 0], [0, 40, 0]], np.float32)
    cnt, best = coverage.raw_counts(pts, cams, chunk=3)            # chunk boundary inside the array
    assert cnt[0] == 12 and cnt[1] == 12                           # the middle is seen by every camera
    assert cnt[3] == 0                                             # far above every frame
    assert abs(best[0] - 150.0 / 4.0) < 1e-3                       # fx / distance
    assert cnt[2] < 12


def test_depth_maps_hide_what_is_behind_a_wall(tmp_path):
    ds = make_dataset(tmp_path, ring(10), depth=2.0)               # every camera saw a surface 2 m in front
    cams = coverage.load_cameras(ds)
    assert all(c.depth is not None for c in cams)
    cnt, _ = coverage.raw_counts(np.array([[0, 0, 0]], np.float32), cams)
    assert cnt[0] == 0                                             # 4 m away: behind the surface for all
    cnt2, _ = coverage.raw_counts(np.array([[0, 0, 0]], np.float32), coverage.load_cameras(ds, use_depth=False))
    assert cnt2[0] == 10
    r = coverage.compute(cloud(np.array([[0, 0, 0], [0, 0, -3.0]])), ds)
    assert r.used_depth and r.views[0] == 0 and r.views[1] > 0    # 1 m from the nearest camera: in front


def test_compute_scores_and_summary(tmp_path):
    ds = make_dataset(tmp_path, ring(16))
    rng = np.random.default_rng(0)
    centre = rng.normal(0, 0.2, (500, 3))
    seen_once = [(-3.0, 0.0, -0.5)]
    floaters = [(0, 30.0, 0), (0, -30.0, 0)]
    g = cloud(np.concatenate([centre, seen_once, floaters]))
    fractions = []
    r = coverage.compute(g, ds, chunk=128, progress=fractions.append)
    assert fractions and fractions[-1] == 1.0
    assert r.score.dtype == np.float32 and len(r.score) == g.count
    assert r.score[:500].mean() > 0.9                              # well seen
    assert r.views[-1] == 0 and r.views[-2] == 0 and r.score[-1] == 0
    assert 0 < r.unseen_fraction < 0.05
    assert r.median_views == 16
    assert not r.used_depth and "no depth maps" in r.summary()


def test_detail_ratio_scores_far_views_poorly():
    s, mb, mv = coverage.scores(np.array([10, 10, 10, 1, 10], np.float32),
                                np.array([100, 100, 40, 100, 20], np.float32), np.ones(5, np.float32))
    assert mb == 100 and mv == 10
    assert s[0] == 1 and s[3] == 0                                  # 1 view is poor whatever the detail
    assert 0 < s[2] < 0.5                                          # 2.5× further than typical
    assert s[4] == 0                                               # 5× further: poor
    assert coverage.ramp(0)[0] > 0.8 and coverage.ramp(1)[1] > 0.6


def test_cancel(tmp_path):
    ds = make_dataset(tmp_path, ring(4))
    with pytest.raises(coverage.CoverageCancelled):
        coverage.compute(cloud(np.zeros((10, 3))), ds, cancelled=lambda: True)


def test_no_cameras(tmp_path):
    (tmp_path / "dataset.json").write_text(json.dumps({"cameras": []}))
    with pytest.raises(ValueError):
        coverage.compute(cloud(np.zeros((3, 3))), tmp_path)
