"""Coarse-to-fine resolution and decoupled SSIM helpers."""
import pytest

torch = pytest.importorskip("torch")

from splatworker.trainer import _ssim_fn, downscale_view, progressive_scale, ssim_split


def test_progressive_scale_schedule():
    assert progressive_scale(0, 24000) == 0.25
    assert progressive_scale(4000, 24000) == 0.5
    assert progressive_scale(8000, 24000) == 0.75
    assert progressive_scale(12000, 24000) == 1.0
    assert progressive_scale(29000, 24000) == 1.0


def test_downscale_view_keeps_projection():
    img = torch.rand(80, 120, 3)
    alpha = torch.ones(80, 120, 1)
    depth = torch.rand(80, 120)
    K = torch.tensor([[100.0, 0, 59.5], [0, 100.0, 39.5], [0, 0, 1]])
    i2, a2, d2, K2, w, h = downscale_view(torch, img, alpha, depth, K, 0.5)
    assert (w, h) == (60, 40) and i2.shape == (40, 60, 3) and a2.shape == (40, 60, 1) and d2.shape == (40, 60)
    # the image centre stays the principal point; focal halves
    assert abs(K2[0, 0].item() - 50) < 1e-5 and abs(K2[0, 2].item() - 29.5) < 1e-5 and abs(K2[1, 2].item() - 19.5) < 1e-5
    assert torch.allclose(K, torch.tensor([[100.0, 0, 59.5], [0, 100.0, 39.5], [0, 0, 1]]))   # input untouched


def test_ssim_split_matches_ssim_without_correction_and_ignores_gain_in_structure():
    torch.manual_seed(0)
    gt = torch.rand(1, 3, 40, 40)
    raw = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1)
    ssim, _ = _ssim_fn(torch)
    plain = ssim(raw, gt).item()
    # torch SSIM uses valid padding like ssim_split's conv
    assert abs(ssim_split(torch, raw, raw, gt).item() - plain) < 1e-5
    # a darker raw render (exposure fixed by the correction) keeps the structure score high
    dark = raw * 0.6
    assert ssim_split(torch, raw, dark, gt).item() > ssim(dark, gt).item()
