"""Fitting a monocular (relative inverse-depth) prediction to the scene's 3D points."""
import numpy as np

from splatworker.depthprior import _project, _sample, fit_inverse_depth


def test_fit_recovers_scale_and_shift_despite_occluded_points():
    rng = np.random.default_rng(0)
    z = rng.uniform(1.0, 6.0, 500)
    a, b = 0.8, 0.05
    pred = (1.0 / z - b) / a                                  # what the model "sees" (affine in inverse depth)
    z_obs = z.copy()
    z_obs[:60] *= rng.uniform(1.5, 3.0, 60)                   # points hidden behind the surface: farther
    fit = fit_inverse_depth(pred, z_obs)
    assert fit is not None
    fa, fb, err = fit
    assert abs(fa - a) < 0.05 and abs(fb - b) < 0.02
    assert err < 0.02


def test_fit_rejects_too_few_points_and_inverted_scale():
    assert fit_inverse_depth(np.ones(5), np.ones(5)) is None
    z = np.linspace(1, 5, 100)
    assert fit_inverse_depth(-1.0 / z, z) is None             # nearer = smaller prediction: wrong sign


def test_project_and_sample():
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    pts = np.array([[0, 0, 2.0], [0.5, 0, 2.0], [0, 0, -1.0], [10, 0, 1.0]])
    u, v, z = _project(pts, np.eye(4), K, 100, 80)
    assert len(z) == 2 and np.allclose(u, [50, 75]) and np.allclose(z, [2, 2])
    img = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    assert np.allclose(_sample(img, np.array([10.5]), np.array([3.0])), img[3, 10] + 0.5)


def test_backproject_puts_points_on_the_surface_in_world_space():
    from types import SimpleNamespace

    from splatworker.depthprior import backproject

    c2w = np.eye(4); c2w[:3, 3] = [1.0, 2.0, 3.0]            # camera moved, looking down +z
    cam = SimpleNamespace(width=200, height=100, fx=100.0, fy=100.0, cx=100.0, cy=50.0, camera_to_world=c2w)
    depth = np.full((50, 100), 4.0, np.float32)               # half-resolution map of a wall 4 m away
    depth[:5] = 0                                             # no data at the top
    p, col = backproject(depth, None, cam, stride=1)
    assert len(p) == 45 * 100 and col.shape == p.shape
    assert np.allclose(p[:, 2], 3.0 + 4.0)
    assert abs(p[:, 0].mean() - 1.0) < 0.05


def test_cache_key_changes_with_pose():
    from types import SimpleNamespace

    from splatworker.depthprior import cache_key

    c2w = np.eye(4)
    a = SimpleNamespace(id=3, image="images/a.jpg", camera_to_world=c2w, fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=1, height=1)
    moved = np.eye(4); moved[0, 3] = 0.1
    b = SimpleNamespace(**{**a.__dict__, "camera_to_world": moved})
    assert cache_key(a) == cache_key(SimpleNamespace(**a.__dict__)) and cache_key(a) != cache_key(b)
    assert cache_key(a).startswith("3-")
