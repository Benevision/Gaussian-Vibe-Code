"""Per-photo local colour correction (bilateral grid)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from splatworker.trainer import BilateralGrid


def _cfg(**k):
    return SimpleNamespace(bilateralGridLR=k.get("lr", 2e-2), bilateralGridTV=k.get("tv", 0.0))


def test_starts_as_identity():
    bg = BilateralGrid(torch, _cfg(), 2, "cpu")
    rgb = torch.rand(24, 32, 3)
    assert torch.allclose(bg.apply(0, rgb), rgb, atol=1e-5)


def test_learns_a_local_shadow_and_others_dont_drift():
    torch.manual_seed(0)
    bg = BilateralGrid(torch, _cfg(lr=5e-2), 2, "cpu")
    rgb = torch.rand(32, 32, 3) * 0.5 + 0.25
    target = rgb.clone()
    target[:, :16] *= 0.5                                 # left half of photo 0 in shadow
    for _ in range(400):
        loss = (bg.apply(0, rgb) - target).abs().mean()
        loss.backward()
        bg.step(0)
    out = bg.apply(0, rgb).detach()
    assert (out[:, :12] - target[:, :12]).abs().mean() < 0.03   # shadowed side darkened
    assert (out[:, 20:] - target[:, 20:]).abs().mean() < 0.03   # lit side unchanged
    assert torch.allclose(bg.apply(1, rgb), rgb, atol=1e-5)     # photo 1 untouched


def test_pin_mean_keeps_average_at_identity():
    bg = BilateralGrid(torch, _cfg(), 3, "cpu")
    with torch.no_grad():
        bg.grids[0].add_(0.3)
        bg.grids[1].sub_(0.1)
    bg.pin_mean()
    mean = torch.stack([g.detach() for g in bg.grids]).mean(0)
    assert torch.allclose(mean, bg.ident, atol=1e-6)
    # relative differences between photos survive
    assert torch.allclose(bg.grids[0] - bg.grids[1], torch.full_like(bg.ident, 0.4), atol=1e-6)


def test_tv_penalises_rough_grids():
    bg = BilateralGrid(torch, _cfg(tv=1.0), 1, "cpu")
    assert bg.tv_loss(0).item() == 0.0
    with torch.no_grad():
        bg.grids[0][:, :, :, ::2] += 0.5
    assert bg.tv_loss(0).item() > 0.0
