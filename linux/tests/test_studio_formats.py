import gzip
import json

import numpy as np
import pytest

from splatworker.studio import formats
from splatworker.studio.splats import C0, GaussianCloud, Skybox, load_skybox, sky_paths


def cloud(n=1000, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    sh = rng.normal(scale=0.3, size=(n, k, 3)).astype(np.float32)
    return GaussianCloud(rng.uniform(-2, 2, (n, 3)), rng.uniform(-5, -1, (n, 3)), q,
                         rng.uniform(-3, 3, n), sh, antialiased=True, up=np.array([0, -1, 0], np.float32))


def same_rotation(a, b, tol):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.abs(np.abs((a * b).sum(1)) - 1).max() < tol


def test_ply_roundtrip_lossless(tmp_path):
    g = cloud()
    p = tmp_path / "a.ply"
    formats.save(g, p)
    h = formats.load(p)
    assert h.count == g.count and h.sh_degree == 3 and h.antialiased
    np.testing.assert_allclose(h.positions, g.positions)
    np.testing.assert_allclose(h.sh, g.sh)
    np.testing.assert_allclose(h.log_scales, g.log_scales)
    np.testing.assert_allclose(h.opacity_logits, g.opacity_logits)
    np.testing.assert_allclose(h.up, [0, -1, 0])
    assert same_rotation(h.rotations, g.rotations, 1e-5)


def test_ply_sh_degree_export_and_worker_ply(tmp_path):
    from splatworker.ply import write_gaussian_ply

    g = cloud(degree=3)
    formats.save(g, tmp_path / "d1.ply", sh_degree=1)
    assert formats.load(tmp_path / "d1.ply").sh_degree == 1
    # the trainer's own writer (45 f_rest columns, zero padded)
    write_gaussian_ply(tmp_path / "w.ply", g.positions, g.sh[:, 0], g.sh[:, 1:4], g.opacity_logits,
                       g.log_scales, g.rotations, up=[0, 0, 1])
    h = formats.load(tmp_path / "w.ply")
    assert h.sh_degree == 3
    np.testing.assert_allclose(h.sh[:, 1:4], g.sh[:, 1:4])
    np.testing.assert_allclose(h.sh[:, 4:], 0)
    np.testing.assert_allclose(h.up, [0, 0, 1])


def test_point_cloud_ply_opens_as_splats(tmp_path):
    p = tmp_path / "pts.ply"
    xyz = np.random.default_rng(1).uniform(-1, 1, (50, 3)).astype("<f4")
    rgb = np.full((50, 3), 200, np.uint8)
    rec = np.zeros(50, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["red"] = rec["green"] = rec["blue"] = 200
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex 50\n" + "".join(
        f"property {t} {n}\n" for n, t in [("x", "float"), ("y", "float"), ("z", "float"),
                                           ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]) + "end_header\n"
    p.write_bytes(hdr.encode() + rec.tobytes())
    g = formats.load(p)
    assert g.count == 50 and g.extra.get("pointCloud")
    np.testing.assert_allclose(g.base_colors, 200 / 255, atol=1e-5)


def test_compressed_ply_roundtrip(tmp_path):
    g = cloud(n=700)
    p = tmp_path / "a.compressed.ply"
    formats.save(g, p)
    assert formats.is_compressed_ply(p)
    h = formats.load(p)
    assert h.count == g.count and h.sh_degree == 3 and h.antialiased
    # Morton order: match splats by nearest position
    order = formats.morton_order(g.positions)
    gs = g.subset(order)
    ext = np.ptp(g.positions, 0).max()
    assert np.abs(h.positions - gs.positions).max() < ext / 1024 * 2
    assert np.abs(h.base_colors - gs.base_colors).max() < 2 / 255 + 1e-3
    assert np.abs(h.opacities - gs.opacities).max() < 1 / 255 + 1e-3
    assert np.abs(h.log_scales - gs.log_scales).max() < 4 / 1024 + 1e-3
    assert same_rotation(h.rotations, gs.rotations, 2e-3)
    assert np.abs(h.sh[:, 1:] - np.clip(gs.sh[:, 1:], -4, 4)).max() < 8 / 256 + 1e-3


def test_compressed_detected_without_compound_name(tmp_path):
    g = cloud(n=300, degree=0)
    p = tmp_path / "plain.ply"
    formats.write_compressed_ply(g, p)
    assert formats.load(p).count == 300


def test_dotsplat_roundtrip(tmp_path):
    g = cloud(n=400, degree=0)
    p = tmp_path / "a.splat"
    formats.save(g, p)
    h = formats.load(p)
    assert h.count == 400 and h.sh_degree == 0
    order = np.argsort(-(np.exp(g.log_scales.sum(1)) * g.opacities), kind="stable")
    gs = g.subset(order)
    np.testing.assert_allclose(h.positions, gs.positions)
    np.testing.assert_allclose(h.log_scales, gs.log_scales, atol=1e-5)
    assert np.abs(h.base_colors - gs.base_colors).max() <= 1 / 255 + 1e-4
    assert same_rotation(h.rotations, gs.rotations, 1e-3)


def test_spz_roundtrip_v2(tmp_path):
    g = cloud(n=500, degree=2)
    p = tmp_path / "a.spz"
    formats.save(g, p)
    raw = gzip.decompress(p.read_bytes())
    assert np.frombuffer(raw[:12], "<u4").tolist() == [formats.SPZ_MAGIC, 2, 500]
    h = formats.load(p)
    assert h.count == 500 and h.sh_degree == 2 and h.antialiased
    assert np.abs(h.positions - g.positions).max() < 1 / 4096 + 1e-6
    assert np.abs(h.log_scales - g.log_scales).max() < 1 / 32 + 1e-4
    assert np.abs(h.opacities - g.opacities).max() < 1 / 255 + 1e-3
    assert same_rotation(h.rotations, g.rotations, 2e-2)
    assert np.abs(h.base_colors - g.base_colors).max() < 0.02
    assert np.abs(h.sh[:, 1:] - np.clip(g.sh[:, 1:], -1, 127 / 128)).max() < 0.07


def _spz_v3_rotation_bytes(q_xyzw):
    """Reference smallest-three encoder (SPZ v3)."""
    out = []
    for q in q_xyzw:
        largest = int(np.argmax(np.abs(q)))
        if q[largest] < 0:
            q = -q
        comp = largest
        for i in range(4):
            if i == largest:
                continue
            neg = 1 if q[i] < 0 else 0
            mag = int(round(abs(q[i]) / np.sqrt(0.5) * 511))
            comp = (comp << 10) | (neg << 9) | mag
        out.append(comp)
    return np.array(out, "<u4").tobytes()


def test_spz_v3_rotations(tmp_path):
    rng = np.random.default_rng(3)
    n = 64
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)          # x y z w (RUB)
    head = np.zeros(16, np.uint8)
    head[:12] = np.array([formats.SPZ_MAGIC, 3, n], "<u4").view(np.uint8)
    head[13] = 12
    body = (np.zeros(n * 9, np.uint8).tobytes() + np.full(n, 128, np.uint8).tobytes()
            + np.full(n * 3, 128, np.uint8).tobytes() + np.full(n * 3, 100, np.uint8).tobytes()
            + _spz_v3_rotation_bytes(q))
    p = tmp_path / "v3.spz"
    p.write_bytes(gzip.compress(head.tobytes() + body))
    g = formats.load(p)
    expect = np.stack([q[:, 3], q[:, 0], -q[:, 1], -q[:, 2]], 1)     # RDF w x y z
    assert same_rotation(g.rotations, expect, 5e-3)


def test_skybox_sidecar(tmp_path):
    from PIL import Image

    g = cloud(n=50, degree=0)
    p = tmp_path / "s.ply"
    formats.save(g, p)
    img = np.zeros((8, 16, 3), np.uint8)
    img[0] = [10, 20, 250]          # top row: straight up
    img[-1] = [200, 100, 0]
    png, meta = sky_paths(p)
    Image.fromarray(img).save(png)
    meta.write_text(json.dumps({"up": [0, -1, 0], "forward": [0, 0, 1], "right": [-1, 0, 0]}))
    h = formats.load(p)
    assert isinstance(h.skybox, Skybox)
    assert h.skybox.sample(np.array([0, -1.0, 0])).tolist() == [10, 20, 250]
    assert h.skybox.sample(np.array([0, 1.0, 0])).tolist() == [200, 100, 0]
    # exported with the scene
    out = tmp_path / "out.spz"
    formats.save(h, out)
    assert load_skybox(out) is not None


def test_dc_convention():
    g = cloud(n=10, degree=0)
    np.testing.assert_allclose(g.base_colors, np.clip(g.sh[:, 0] * C0 + 0.5, 0, 1))


def test_unknown_format(tmp_path):
    (tmp_path / "x.obj").write_text("")
    with pytest.raises(formats.FormatError):
        formats.load(tmp_path / "x.obj")
