"""Skybox orientation and sampling (CPU torch when available)."""
import math

import numpy as np
import pytest

from splatworker.skybox import basis


def test_basis_is_orthonormal_and_up_first():
    for up in ([0, -1, 0], [0, 0, 1], [0.1, -0.9, 0.2]):
        u, f, r = basis(up)
        M = np.stack([r, u, f])
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-9)
        assert np.allclose(u, np.asarray(up) / np.linalg.norm(up))


def test_sky_samples_by_direction():
    torch = pytest.importorskip("torch")
    from splatworker.skybox import Skybox

    sb = Skybox(torch, [0, -1, 0], height=16, device="cpu")
    with torch.no_grad():
        sb.tex.zero_()
        sb.tex[0, 0, :8] = 10.0        # top half (towards up = −Y) red
    K = torch.tensor([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    w2c = torch.eye(4)
    d = sb.directions(K, w2c, 64, 64)
    assert torch.allclose(d.norm(dim=-1), torch.ones(64, 64), atol=1e-5)
    rgb = sb.sample(d)
    # camera looks along +Z with y down: the image's top rows look towards −Y (up) → redder than the bottom
    assert rgb[0, 32, 0] > rgb[63, 32, 0] + 0.2
    rgb.sum().backward()
    assert sb.tex.grad is not None and sb.tex.grad.abs().sum() > 0
