import numpy as np
import pytest

from splatworker.studio.camera import Camera, plane_basis
from splatworker.studio.edit import EditState, estimate_size
from splatworker.studio.sorting import BackgroundSorter, depth_order
from splatworker.studio.splats import C0, GaussianCloud


def test_camera_default_looks_along_plus_z_with_minus_y_up():
    cam = Camera(pitch=0.0, yaw=0.0, distance=2.0)
    R = cam.rotation()
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)        # OpenCV: right +x, down +y, forward +z
    np.testing.assert_allclose(cam.position(), [0, 0, -2])
    px, z = cam.project(np.array([[0.0, 0, 0], [0.0, -1, 0]]), 200, 100)
    np.testing.assert_allclose(px[0], [100, 50])
    assert px[1, 1] < 50                                         # up (−Y) is up on screen
    np.testing.assert_allclose(z, [2, 2])


def test_camera_rotation_orthonormal_any_up():
    for up in ([0, 0, 1], [1, 0, 0], [0.3, -0.8, 0.2]):
        cam = Camera(up=np.array(up, float), yaw=37, pitch=-20)
        R = cam.rotation()
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0.999
        # screen up (−down) has a positive component along the scene up
        assert -R[1] @ (np.array(up) / np.linalg.norm(up)) > 0


def test_orbit_fly_switch_keeps_view():
    cam = Camera(yaw=20, pitch=-10, distance=5, target=np.array([1.0, 2, 3]))
    pos, fwd = cam.position(), cam.forward()
    cam.set_mode("fly")
    np.testing.assert_allclose(cam.position(), pos)
    cam.move(forward=1.0)
    np.testing.assert_allclose(cam.position(), pos + fwd, atol=1e-9)
    cam.set_mode("orbit")
    np.testing.assert_allclose(cam.position(), pos + fwd, atol=1e-9)


def test_set_orbit_center_and_look_through():
    cam = Camera()
    cam.set_orbit_center([0.5, 0.2, 4.0])
    np.testing.assert_allclose(cam.target, [0.5, 0.2, 4.0])
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    cam.look_through(c2w)
    np.testing.assert_allclose(cam.position(), [1, 2, 3])
    np.testing.assert_allclose(cam.rotation(), np.eye(3), atol=1e-6)


def test_frame_and_flip_level():
    cam = Camera()
    cam.frame([-1, -1, -1], [1, 1, 1])
    px, _ = cam.project(np.array([[1.0, 1, 1], [-1.0, -1, -1]]), 400, 300)
    assert (px >= 0).all() and (px[:, 0] <= 400).all() and (px[:, 1] <= 300).all()
    cam.flip()
    np.testing.assert_allclose(cam.up, [0, 1, 0])
    cam.level_to_view()
    assert abs(np.linalg.norm(cam.up) - 1) < 1e-9
    a, b = plane_basis(cam.up)
    assert abs(a @ cam.up) < 1e-9 and abs(b @ cam.up) < 1e-9


def test_depth_order_back_to_front_hidden_last():
    pos = np.array([[0, 0, 1], [0, 0, 5], [0, 0, -1], [0, 0, 3], [0, 0, 2]], np.float32)
    hidden = np.array([False, False, False, False, True])
    order, n = depth_order(pos, [0, 0, 0], [0, 0, 1], hidden=hidden)
    assert n == 3
    assert order[:3].tolist() == [1, 3, 0]
    assert set(order[3:].tolist()) == {2, 4}


def test_background_sorter():
    import threading

    got = {}
    ev = threading.Event()

    def done(gen, order, n):
        got.update(gen=gen, order=order, n=n)
        ev.set()
    s = BackgroundSorter(done)
    pos = np.random.default_rng(0).uniform(1, 10, (10000, 3)).astype(np.float32)
    s.request(pos, [0, 0, 0], [0, 0, 1], 1e-3, generation=7)
    assert ev.wait(5)
    s.close()
    d = pos[got["order"][:got["n"]], 2]
    assert got["gen"] == 7 and got["n"] == 10000
    assert (np.diff(d) <= 1e-2 * d[1:]).all()          # far → near within 16-bit log precision


def cloud(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    rot = np.zeros((n, 4), np.float32)
    rot[:, 0] = 1
    return GaussianCloud(rng.uniform(-1, 1, (n, 3)), np.full((n, 3), -4.0), rot, rng.uniform(-4, 4, n),
                         np.zeros((n, 1, 3)))


def test_edit_undo_redo_and_thresholds():
    g = cloud()
    e = EditState(g)
    k = e.delete(e.below_opacity(0.5))
    assert k == int((g.opacities < 0.5).sum())
    assert e.visible_count == g.count - k
    e.undo()
    assert e.visible_count == g.count
    e.redo()
    assert e.visible_count == g.count - k
    e.restore_all()
    assert e.visible_count == g.count
    big = e.larger_than(0.01)
    assert big.all() == (np.exp(-4.0) * 3 > 0.01)


def test_crop_and_brush():
    g = cloud()
    e = EditState(g)
    e.crop.lo[:] = -0.5
    e.crop.hi[:] = 0.5
    e.crop.enabled = True
    inside = ((g.positions >= -0.5) & (g.positions <= 0.5)).all(1)
    assert e.visible_count == inside.sum()
    e.apply_crop()
    assert int((~e.deleted).sum()) == inside.sum() and not e.crop.enabled
    cam = Camera(pitch=0, distance=3)
    px, z = cam.project(g.positions, 200, 200)
    e.select_brush(px, z, 100, 100, 30)
    assert e.selected.any() and not (e.selected & e.deleted).any()
    n = int(e.selected.sum())
    assert e.delete_selected() == n and not e.selected.any()


def test_isolated():
    rng = np.random.default_rng(2)
    cluster = rng.normal(scale=0.01, size=(500, 3))
    lonely = np.array([[5.0, 5, 5], [-5.0, 4, 0]])
    pos = np.concatenate([cluster, lonely])
    n = len(pos)
    rot = np.zeros((n, 4))
    rot[:, 0] = 1
    g = GaussianCloud(pos, np.zeros((n, 3)), rot, np.zeros(n), np.zeros((n, 1, 3)))
    iso = EditState(g).isolated(neighbours=3, radius=0.05)
    assert iso[-2:].all() and iso[:500].mean() < 0.05


def test_estimate_size_matches_writers(tmp_path):
    from splatworker.studio import formats

    g = cloud(n=2048)
    for fmt in ("ply", "compressed.ply", "splat"):
        p = tmp_path / f"x.{fmt}"
        formats.save(g, p, fmt)
        est = estimate_size(g.count, fmt, g.sh_degree)
        assert abs(est - p.stat().st_size) / p.stat().st_size < 0.05, fmt


# ------------------------------------------------------------------------------------------------- GL render

def _gl():
    from splatworker.studio import gl

    try:
        ctx = gl.headless_context()
    except Exception as e:  # pragma: no cover - macOS / no EGL
        pytest.skip(f"no headless OpenGL: {e}")
    if gl.version() < (4, 3):  # pragma: no cover
        ctx.close()
        pytest.skip("OpenGL < 4.3")
    return ctx


def single(color, pos=(0, 0, 3), scale=0.3, opacity=0.99, sh_extra=None):
    rot = np.array([[1, 0, 0, 0]], np.float32)
    sh = np.zeros((1, 1 if sh_extra is None else 4, 3), np.float32)
    sh[0, 0] = (np.array(color) - 0.5) / C0
    if sh_extra is not None:
        sh[0, 1:] = sh_extra
    lo = np.log(opacity / (1 - opacity))
    return GaussianCloud(np.array([pos], np.float32), np.full((1, 3), np.log(scale)), rot, np.array([lo]), sh)


def render(g, cam=None, w=64, h=48, **kw):
    from splatworker.studio.renderer import SplatRenderer, ViewSettings, render_offscreen

    r = SplatRenderer()
    r.init_gl()
    r.set_cloud(g)
    cam = cam or Camera(pitch=0, distance=3)
    order, n = depth_order(g.positions, cam.position(), cam.forward(), cam.near)
    r.set_order(order, n)
    img = render_offscreen(r, cam, w, h, ViewSettings(background=(0, 0, 0), **kw))
    r.destroy()
    return img


def test_gl_single_splat_colour_and_falloff():
    ctx = _gl()
    try:
        img = render(single((0.9, 0.1, 0.1), pos=(0, 0, 0)))
        c = img[24, 32].astype(int)
        assert c[0] > 200 and c[1] < 50 and c[2] < 50, c
        assert img[0, 0, :3].max() == 0          # far corner: background
        assert img[24, 40, 0] < c[0]             # gaussian falloff
    finally:
        ctx.close()


def test_gl_back_to_front_blending():
    ctx = _gl()
    try:
        red = single((1, 0, 0), pos=(0, 0, 0))
        blue = single((0, 0, 1), pos=(0, 0, -1))        # nearer to the camera at z = −3
        g = GaussianCloud(np.concatenate([red.positions, blue.positions]),
                          np.concatenate([red.log_scales, blue.log_scales]),
                          np.concatenate([red.rotations, blue.rotations]),
                          np.concatenate([red.opacity_logits, blue.opacity_logits]),
                          np.concatenate([red.sh, blue.sh]))
        c = render(g)[24, 32].astype(int)
        assert c[2] > 200 and c[0] < 40, c       # the near (blue) splat covers the far one
    finally:
        ctx.close()


def test_gl_modes_and_selection_flags():
    ctx = _gl()
    try:
        g = single((0.5, 0.5, 0.5), pos=(0, 0, 0))
        depth = render(g, mode="depth")[24, 32]
        assert depth[:3].max() > 0
        centers = render(g, mode="centers")
        assert (centers[..., :3].max(-1) > 0).sum() < 40     # a dot, not a blob
    finally:
        ctx.close()


def test_gl_skybox():
    from splatworker.studio.splats import Skybox

    ctx = _gl()
    try:
        img = np.zeros((32, 64, 3), np.uint8)
        img[:16] = (0, 0, 255)       # upper hemisphere blue
        img[16:] = (0, 255, 0)       # lower hemisphere green
        g = single((1, 1, 1), pos=(100, 100, 100), opacity=0.01)
        g.skybox = Skybox(img, up=np.array([0, -1.0, 0]), forward=np.array([0, 0, 1.0]), right=np.array([-1.0, 0, 0]))
        cam = Camera(pitch=0, distance=3)
        out = render(g, cam)
        assert out[2, 32, 2] > 200 and out[2, 32, 1] < 50      # top of the screen looks up → blue
        assert out[45, 32, 1] > 200 and out[45, 32, 2] < 50    # bottom → green
        assert render(g, cam, mode="depth")[2, 32, :3].max() == 0   # no sky outside Colour mode
    finally:
        ctx.close()


def test_gl_view_dependent_sh():
    ctx = _gl()
    try:
        extra = np.zeros((3, 3), np.float32)
        extra[2] = [-1.5, 0, 0]          # coefficient 3 (−C1·x): red depends on the x of the view direction
        g = single((0.5, 0.5, 0.5), pos=(0, 0, 0), sh_extra=extra)
        left = render(g, Camera(pitch=0, yaw=30, distance=3))[24, 32, 0]
        right = render(g, Camera(pitch=0, yaw=-30, distance=3))[24, 32, 0]
        assert abs(int(left) - int(right)) > 40
        flat = render(g, Camera(pitch=0, yaw=30, distance=3), sh_degree=0)[24, 32, 0]
        assert abs(int(flat) - 127) < 20
    finally:
        ctx.close()
