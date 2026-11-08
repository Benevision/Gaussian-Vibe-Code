import threading
import time

import numpy as np
import pytest

from splatworker.studio import turntable
from splatworker.studio.camera import Camera
from splatworker.studio.turntable import ThreadedSink, TurntableOptions, orbit_camera


def test_orbit_camera_turns_around_the_orbit_centre():
    start = Camera(target=np.array([1.0, 0, 2]), distance=4, yaw=10, pitch=-20)
    o = TurntableOptions(seconds=2, fps=10)
    assert o.frames == 20
    c0 = orbit_camera(start, 0, o)
    np.testing.assert_allclose(c0.position(), start.position())
    half = orbit_camera(start, 10, o)
    assert abs(half.yaw - 190) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(half.position() - start.target), 4)
    np.testing.assert_allclose(half.position() - start.target, -(start.position() - start.target)
                               + 2 * ((start.position() - start.target) @ -start.up) * -start.up, atol=1e-9)
    assert start.yaw == 10                                   # the viewer's camera is untouched
    assert orbit_camera(start, 3, TurntableOptions(pitch=-45)).pitch == -45


def test_fly_camera_orbits_the_point_it_looks_at():
    start = Camera(distance=2)
    start.move(forward=1.0)                                   # now in fly mode
    c = orbit_camera(start, 0, TurntableOptions())
    assert c.mode == "orbit" and start.mode == "fly"
    np.testing.assert_allclose(c.position(), start.position(), atol=1e-9)


class SlowSink:
    path = None

    def __init__(self):
        self.frames = []
        self.closed = self.aborted = False
        self.gate = threading.Event()

    def write(self, rgb):
        self.gate.wait(5)
        self.frames.append(rgb.copy())

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


def test_threaded_sink_is_bounded_and_ordered():
    s = SlowSink()
    ts = ThreadedSink(s, depth=2)
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(5)]
    accepted = 0
    for f in frames:                         # the writer blocks on the gate: 1 in flight + 2 queued at most
        if ts.offer(f):
            accepted += 1
    assert accepted <= 3
    s.gate.set()
    for f in frames[accepted:]:
        ts.put(f)
    ts.finish()
    assert s.closed and [int(f[0, 0, 0]) for f in s.frames] == list(range(5))


def test_threaded_sink_reports_encoder_errors_and_aborts():
    class Bad(SlowSink):
        def write(self, rgb):
            raise RuntimeError("disk full")
    b = Bad()
    ts = ThreadedSink(b)
    ts.put(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(RuntimeError, match="disk full"):
        ts.finish()
    assert b.aborted and not b.closed


def test_png_fallback_names(tmp_path):
    sink = turntable.open_sink(tmp_path / "spin.mp4", 8, 8, 30, prefer="png")
    assert isinstance(sink, turntable.PngSequenceSink) and sink.path == tmp_path / "spin-frames"
    sink.write(np.zeros((8, 8, 3), np.uint8))
    assert (tmp_path / "spin-frames" / "frame-00000.png").is_file()


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


def _scene():
    from splatworker.studio.splats import C0, GaussianCloud

    # two coloured blobs either side of the centre: the view changes as the camera turns
    pos = np.array([[0.6, 0, 0], [-0.6, 0, 0]], np.float32)
    col = np.array([[1, 0, 0], [0, 0, 1]], np.float32)
    return GaussianCloud(pos, np.full((2, 3), np.log(0.25)), np.tile([1, 0, 0, 0], (2, 1)), np.full(2, 4.0),
                         ((col - 0.5) / C0)[:, None, :])


@pytest.mark.parametrize("prefer", ["png", None])
def test_render_turntable(tmp_path, prefer):
    from splatworker.studio.renderer import SplatRenderer, ViewSettings

    if prefer is None and not turntable.available_codecs():
        pytest.skip("no GStreamer video encoder here")
    ctx = _gl()
    try:
        g = _scene()
        r = SplatRenderer()
        r.init_gl()
        r.set_cloud(g)
        cam = Camera(pitch=0, yaw=0, distance=3)
        fr = []
        out = turntable.render_turntable(r, cam, ViewSettings(background=(0, 0, 0)),
                                         TurntableOptions(width=64, height=48, fps=8, seconds=1), tmp_path / "t.mp4",
                                         g.positions, progress=fr.append, prefer=prefer)
        r.destroy()
    finally:
        ctx.close()
    assert fr[-1] == 1.0 and len(fr) == 8
    if prefer == "png":
        files = sorted(out.iterdir())
        assert len(files) == 8
        from PIL import Image

        a = np.asarray(Image.open(files[0]))
        b = np.asarray(Image.open(files[4]))                  # half a turn: left and right swap
        assert a.shape == (48, 64, 3)
        left_a, left_b = a[:, :32].reshape(-1, 3).mean(0), b[:, :32].reshape(-1, 3).mean(0)
        assert (left_a[0] - left_a[2]) * (left_b[0] - left_b[2]) < 0
    else:
        assert out.suffix in (".mp4", ".webm") and out.stat().st_size > 100


def test_cancel_stops_and_cleans_up(tmp_path):
    from splatworker.studio.renderer import SplatRenderer, ViewSettings

    ctx = _gl()
    try:
        g = _scene()
        r = SplatRenderer()
        r.init_gl()
        r.set_cloud(g)
        calls = []
        with pytest.raises(turntable.TurntableCancelled):
            turntable.render_turntable(r, Camera(distance=3), ViewSettings(), TurntableOptions(64, 48, 8, 2),
                                       tmp_path / "c.mp4", g.positions, prefer="png",
                                       cancelled=lambda: calls.append(1) or len(calls) > 3)
        r.destroy()
    finally:
        ctx.close()
    time.sleep(0.01)
    assert len(list((tmp_path / "c-frames").iterdir())) <= 3
