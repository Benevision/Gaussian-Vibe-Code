"""Depth-normal consistency (normalWeight) and the flat-disc regulariser: same maths as the Mac's kernels."""
import pytest

torch = pytest.importorskip("torch")

from splatworker.trainer import depth_normals, normal_consistency_loss  # noqa: E402


def _plane(H=60, W=80, tilt=0.3, dist=2.0):
    K = torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    ray = torch.stack(((xs - W / 2) / 50, (ys - H / 2) / 50, torch.ones_like(xs)), -1)
    n = torch.tensor([tilt, 0.0, -1.0]); n = n / n.norm()
    return K, (-dist) / (ray @ n), n


def test_sensor_normals_from_low_res_depth_match_the_plane():
    K, z, n = _plane()
    sensor = torch.nn.functional.interpolate(z[None, None][..., ::4, ::4], size=z.shape, mode="nearest")[0, 0]
    nt, vt = depth_normals(torch, sensor, K, k=4)
    assert vt.float().mean() > 0.5
    assert torch.allclose(nt[vt].mean(0).abs(), n.abs(), atol=0.01)


def test_flat_render_scores_near_zero_and_bumps_score_higher():
    K, z, _ = _plane()
    sensor = torch.nn.functional.interpolate(z[None, None][..., ::4, ::4], size=z.shape, mode="nearest")[0, 0]
    ones = torch.ones_like(z)
    flat = normal_consistency_loss(torch, z, ones, sensor, K, z.shape[1] // 4)
    torch.manual_seed(0)
    bumpy = (z + 0.02 * torch.randn_like(z)).requires_grad_()
    lb = normal_consistency_loss(torch, bumpy, ones, sensor, K, z.shape[1] // 4)
    lb.backward()
    assert float(flat) < 0.02 < float(lb)
    assert torch.isfinite(bumpy.grad).all() and bumpy.grad.abs().sum() > 0


def test_transparent_or_missing_depth_gives_no_loss():
    K, z, _ = _plane()
    assert normal_consistency_loss(torch, z, torch.zeros_like(z), z, K, z.shape[1]) is None
    assert normal_consistency_loss(torch, z, torch.ones_like(z), torch.zeros_like(z), K, z.shape[1]) is None
