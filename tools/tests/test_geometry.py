import json

import numpy as np
import pytest

from splatprep.geometry import (
    Sim3,
    arkit_to_opencv_c2w,
    evaluate_poses,
    look_at,
    pose_auc,
    qvec_to_rotmat,
    robust_umeyama,
    rotation_angle_deg,
    rotmat_to_qvec,
    umeyama,
)
from splatprep.priors import load_priors, match_priors


def random_rotation(rng):
    q = rng.normal(size=4)
    return qvec_to_rotmat(q / np.linalg.norm(q))


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = random_rotation(rng)
        assert np.allclose(qvec_to_rotmat(rotmat_to_qvec(R)), R, atol=1e-9)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    T = Sim3(3.2, random_rotation(rng), rng.normal(size=3))
    est = umeyama(X, T.apply_points(X))
    assert est.s == pytest.approx(3.2)
    assert np.allclose(est.R, T.R, atol=1e-9) and np.allclose(est.t, T.t, atol=1e-9)


def test_robust_umeyama_rejects_outliers():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    T = Sim3(0.5, random_rotation(rng), rng.normal(size=3))
    Y = T.apply_points(X)
    Y[:5] += rng.normal(scale=5.0, size=(5, 3))
    est, inl = robust_umeyama(X, Y)
    assert not inl[:5].any() and inl[5:].all()
    assert est.s == pytest.approx(0.5, rel=1e-6)


def test_pose_auc():
    assert pose_auc([0, 0, 0], [5]) == [pytest.approx(1.0)]
    assert pose_auc([np.inf, np.inf], [5]) == [0.0]
    # half perfect, half failed
    assert pose_auc([0, np.inf], [10])[0] == pytest.approx(0.5)
    # uniform errors in [0,10] -> AUC@10 ~ 0.5
    a = pose_auc(np.linspace(0, 10, 1001), [10])[0]
    assert a == pytest.approx(0.5, abs=0.01)


def _orbit(n=12):
    return {f"{i}.png": look_at([4 * np.cos(2 * np.pi * i / n), 1.0, 4 * np.sin(2 * np.pi * i / n)], [0, 0, 0]) for i in range(n)}


def test_evaluate_poses_perfect_under_sim3():
    gt = _orbit()
    rng = np.random.default_rng(3)
    T = Sim3(0.3, random_rotation(rng), rng.normal(size=3))
    est = {k: T.apply_c2w(v) for k, v in gt.items()}
    r = evaluate_poses(est, gt)
    assert r["aucAt5"] == pytest.approx(1.0) and r["aucAt10"] == pytest.approx(1.0)
    assert r["ateRmse"] < 1e-9 and r["rotErrDegMean"] < 1e-6
    json.dumps(r)  # serialisable


def test_evaluate_poses_noise_and_missing():
    gt = _orbit(16)
    rng = np.random.default_rng(4)
    est = {}
    for k, v in list(gt.items())[:12]:  # 4 unregistered
        noisy = v.copy()
        noisy[:3, :3] = qvec_to_rotmat(np.r_[1.0, rng.normal(0, 0.02, 3)]) @ v[:3, :3]
        noisy[:3, 3] += rng.normal(0, 0.05, 3)
        est[k] = noisy
    r = evaluate_poses(est, gt)
    assert r["numEvaluated"] == 12
    assert 0 < r["aucAt10"] < 12 * 11 / (16 * 15) + 1e-6  # failed pairs cap the AUC
    assert 0 < r["ateRmse"] < 0.2
    assert 0 < r["rotErrDegMean"] < 6


def test_rotation_angle():
    R = look_at([0, 0, 0], [1, 0, 0])[:3, :3]
    R2 = look_at([0, 0, 0], [0, 0, 1])[:3, :3]
    assert rotation_angle_deg(R.T @ R2) == pytest.approx(90.0)


def test_arkit_manifest_priors(tmp_path):
    # ARKit camera at (1,2,3) looking down -Z (identity rotation), column-major 16 floats
    c2w_arkit = np.eye(4)
    c2w_arkit[:3, 3] = [1, 2, 3]
    manifest = {
        "format": "splatcapture",
        "version": 1,
        "coordinateSystem": "arkit",
        "frames": [
            {"index": 0, "image": "images/000000.jpg", "intrinsics": [100, 100, 50, 40], "cameraToWorld": c2w_arkit.T.reshape(-1).tolist(), "timestamp": 1.0}
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    frames = load_priors(p)
    m = match_priors(frames, ["000000.jpg"])
    c2w = m["000000.jpg"].c2w
    assert np.allclose(c2w[:3, 3], [1, 2, 3])
    # OpenCV forward (+Z cam) must be ARKit -Z world
    assert np.allclose(c2w[:3, 2], [0, 0, -1])
    assert np.allclose(c2w, arkit_to_opencv_c2w(c2w_arkit))
