import json

import numpy as np
import pytest

from splatprep.geometry import invert_pose
from splatprep.synth import dataset_hash, generate, make_textures, render, trajectory


def test_determinism(tmp_path):
    a = generate(tmp_path / "a", "sparse", num_views=4, width=96, height=72, seed=7, supersample=1)
    b = generate(tmp_path / "b", "sparse", num_views=4, width=96, height=72, seed=7, supersample=1)
    generate(tmp_path / "c", "sparse", num_views=4, width=96, height=72, seed=8, supersample=1)
    assert a == b
    assert dataset_hash(tmp_path / "a") == dataset_hash(tmp_path / "b")
    assert dataset_hash(tmp_path / "a") != dataset_hash(tmp_path / "c")


@pytest.mark.parametrize("kind,n", [("orbit", 60), ("walk", 60), ("sparse", 15)])
def test_trajectory_defaults(kind, n):
    poses, names = trajectory(kind, None, np.random.default_rng(0))
    assert len(poses) == n == len(set(names))
    for T in poses:
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0.999


def test_dataset_json_is_valid_ground_truth(synth_small):
    ds = json.loads((synth_small / "dataset.json").read_text())
    assert ds["format"] == "splatdataset" and ds["convention"] == "opencv"
    assert len(ds["cameras"]) == 8
    cam = ds["cameras"][0]
    assert (synth_small / cam["image"]).exists()
    M = np.asarray(cam["cameraToWorld"])
    assert M.shape == (4, 4) and np.allclose(M[3], [0, 0, 0, 1])
    # world up is +Y, OpenCV camera +Y is down, so upVector ~ +Y
    assert ds["upVector"][1] > 0.9


def test_depth_is_multiview_consistent():
    """A pixel back-projected with GT depth from view A must land on the same surface in view B."""
    tex = make_textures(np.random.default_rng(0), 128)
    poses, _ = trajectory("orbit", 24, np.random.default_rng(1))
    W, H, f = 120, 90, 100.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    _, dA = render(poses[0], K, W, H, tex)
    _, dB = render(poses[1], K, W, H, tex)
    ok = 0
    for v in range(10, H - 10, 7):
        for u in range(10, W - 10, 7):
            z = dA[v, u]
            if not np.isfinite(z):
                continue
            Xc = np.array([(u + 0.5 - K[0, 2]) / f * z, (v + 0.5 - K[1, 2]) / f * z, z])
            Xw = poses[0][:3, :3] @ Xc + poses[0][:3, 3]
            Xb = invert_pose(poses[1])[:3, :3] @ Xw + invert_pose(poses[1])[:3, 3]
            ub, vb = Xb[0] / Xb[2] * f + W / 2 - 0.5, Xb[1] / Xb[2] * f + H / 2 - 0.5
            if 0 <= round(ub) < W and 0 <= round(vb) < H:
                zb = dB[int(round(vb)), int(round(ub))]
                if np.isfinite(zb) and abs(zb - Xb[2]) < 0.05 * Xb[2]:
                    ok += 1
    assert ok > 20
