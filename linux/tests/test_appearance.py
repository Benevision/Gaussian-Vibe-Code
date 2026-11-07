"""Per-photo appearance correction (exposure seed + learning a colour gain)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from splatworker.trainer import AppearanceCorrection


def _view(exposure):
    return SimpleNamespace(camera=SimpleNamespace(exposure=exposure))


def test_seed_from_exposure_and_learning():
    cfg = SimpleNamespace(appearanceLR=0.05, appearanceReg=0.0)
    views = [_view(0.01), _view(0.02), _view(0.04)]
    ap = AppearanceCorrection(torch, cfg, views, "cpu")
    # median exposure 0.02 → gains (0.5, 1, 2)^(1/2.2)
    assert abs(ap.params[0][0].item() - 0.5 ** (1 / 2.2)) < 1e-6
    assert abs(ap.params[1][0].item() - 1.0) < 1e-6
    assert abs(ap.params[2][4].item() - 2.0 ** (1 / 2.2)) < 1e-6

    # learn a warm tint on view 1: target = render * (1.2, 1.0, 0.8)
    rgb = torch.rand(32, 32, 3)
    target = rgb * torch.tensor([1.2, 1.0, 0.8])
    for _ in range(300):
        loss = (ap.apply(1, rgb) - target).abs().mean()
        loss.backward()
        ap.step(1)
    M = ap.params[1][:9].view(3, 3).detach()
    assert abs(M[0, 0].item() - 1.2) < 0.05 and abs(M[2, 2].item() - 0.8) < 0.05
    # untouched views don't drift
    assert abs(ap.params[0][0].item() - 0.5 ** (1 / 2.2)) < 1e-6


def test_non_finite_gradient_is_skipped():
    cfg = SimpleNamespace(appearanceLR=0.05, appearanceReg=0.001)
    ap = AppearanceCorrection(torch, cfg, [_view(None)], "cpu")
    before = ap.params[0].detach().clone()
    out = ap.apply(0, torch.full((4, 4, 3), float("nan")))
    out.mean().backward()
    ap.step(0)
    assert torch.equal(ap.params[0].detach(), before)


def test_preview_splat_layout(tmp_path):
    import numpy as np
    from splatworker.trainer import export_preview_splat

    n = 50
    splats = {
        "means": torch.randn(n, 3), "scales": torch.randn(n, 3) * 0.1 - 3,
        "quats": torch.randn(n, 4), "opacities": torch.randn(n), "sh0": torch.randn(n, 1, 3) * 0.3,
    }
    out = tmp_path / "p.splat"
    assert export_preview_splat(splats, out, max_splats=20) == 20
    raw = out.read_bytes()
    assert len(raw) == 20 * 32
    rec = np.frombuffer(raw, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    score = torch.sigmoid(splats["opacities"]) * torch.exp(splats["scales"]).prod(1)
    best = int(torch.argmax(score))
    assert np.allclose(rec["p"][0], splats["means"][best].numpy(), atol=1e-6)
    assert np.allclose(rec["s"][0], torch.exp(splats["scales"][best]).numpy(), rtol=1e-5)


def test_update_version_compare():
    from splatworker.gui import updates
    assert updates.is_newer("1.1.1", "1.1.0") and updates.is_newer("1.10.0", "1.9.9")
    assert not updates.is_newer("1.1.0", "1.1.0") and not updates.is_newer("1.0.9", "1.1.0")
