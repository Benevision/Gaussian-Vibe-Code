"""Turntable video (Mac: ``VideoExporter.renderTurntable`` in ``Render/MediaExport.swift``).

The camera orbits the current orbit centre once (yaw + 360°) at the current distance and pitch; every frame is
depth-sorted from its own viewpoint, rendered offscreen and handed to an encoder **one at a time** — frames are
never collected, and the hand-over queue holds at most two, so memory stays at a few frames whatever the length.

Encoders (GTK-free, chosen by :func:`open_sink`, best first):

* GStreamer in this process (the GNOME runtime ships it): H.264 in MP4 when an H.264 encoder element exists
  (``x264enc``, ``openh264enc`` from the Flatpak openh264 extension, ``vah264enc`` / ``nvh264enc``), otherwise
  VP9 (then VP8) in WebM — ``vp9enc``/``vp8enc`` are in gst-plugins-good, always in the runtime;
* a numbered PNG sequence in a folder (no GStreamer at all), which any editor or ``ffmpeg`` turns into a video.
"""
from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .sorting import depth_order

RESOLUTIONS = {720: (1280, 720), 1080: (1920, 1080), 2160: (3840, 2160)}


class TurntableCancelled(Exception):
    pass


@dataclass
class TurntableOptions:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    seconds: float = 8.0
    revolutions: float = 1.0
    pitch: Optional[float] = None       # degrees; None keeps the current pitch

    @property
    def frames(self) -> int:
        return max(1, int(round(self.seconds * self.fps)))


def orbit_camera(start, index: int, options: TurntableOptions):
    """Camera of frame ``index``: ``start`` in orbit mode (a fly camera orbits the point it looks at), yaw advanced."""
    cam = copy.deepcopy(start)
    cam.set_mode("orbit")
    cam.yaw = start.yaw + 360.0 * options.revolutions * index / options.frames
    if options.pitch is not None:
        cam.pitch = float(options.pitch)
    return cam


def render_frames(renderer, start, settings, options: TurntableOptions, positions: np.ndarray,
                  hidden: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
    """Yield each frame (H, W, 3) uint8. Needs the renderer's GL context current while iterating; the caller should
    restore the viewer's own draw order afterwards (the renderer is left sorted for the last frame)."""
    from .renderer import render_offscreen

    lines = renderer._line_count               # no camera frustums / crop box in the video
    try:
        for i in range(options.frames):
            cam = orbit_camera(start, i, options)
            order, n = depth_order(positions, cam.position(), cam.forward(), cam.near, hidden)
            renderer.set_order(order, n)
            renderer._line_count = 0
            img = render_offscreen(renderer, cam, options.width, options.height, settings)
            renderer._line_count = lines
            yield np.ascontiguousarray(img[..., :3])
    finally:
        renderer._line_count = lines


# ---------------------------------------------------------------------------------------------------- sinks
class PngSequenceSink:
    """Frames as ``frame-00000.png`` … in a folder (the fallback when no video encoder is available)."""
    kind = "png"
    description = "PNG image sequence"

    def __init__(self, folder: Path):
        self.path = Path(folder)
        self.path.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, rgb: np.ndarray) -> None:
        from PIL import Image

        Image.fromarray(rgb).save(self.path / f"frame-{self.count:05d}.png", compress_level=3)
        self.count += 1

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


_H264 = ("x264enc", "openh264enc", "vah264enc", "vaapih264enc", "nvh264enc")


def _gst():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)
        return Gst
    except (ImportError, ValueError):
        return None


def available_codecs() -> List[Tuple[str, str]]:
    """[(codec, encoder element)] usable here, best first: ("h264", …), ("vp9", "vp9enc"), ("vp8", "vp8enc")."""
    Gst = _gst()
    if Gst is None:
        return []
    have = lambda *names: all(Gst.ElementFactory.find(n) for n in names)  # noqa: E731
    if not have("appsrc", "videoconvert", "filesink"):
        return []
    out = []
    if have("h264parse", "mp4mux"):
        out += [("h264", e) for e in _H264 if have(e)]
    if have("webmmux"):
        out += [(c, f"{c}enc") for c in ("vp9", "vp8") if have(f"{c}enc")]
    return out


def video_suffix() -> str:
    """The file suffix the best encoder here writes (".mp4", ".webm", or "" for a PNG folder)."""
    c = available_codecs()
    return "" if not c else (".mp4" if c[0][0] == "h264" else ".webm")


class GstSink:
    """appsrc → videoconvert → encoder → mux → filesink. ``appsrc`` blocks when its small queue is full, so the
    producer can never run ahead by more than a frame or two."""
    kind = "video"

    def __init__(self, path: Path, width: int, height: int, fps: int, codec: str, element: str):
        Gst = _gst()
        self.Gst = Gst
        self.path = Path(path)
        self.description = {"h264": "H.264 (MP4)", "vp9": "VP9 (WebM)", "vp8": "VP8 (WebM)"}[codec] + f" · {element}"
        self.fps = int(fps)
        self.count = 0
        kbps = max(1000, width * height * 8 // 1000)       # the Mac's bit rate: 8 bits per pixel per second
        enc = {
            "x264enc": f"x264enc bitrate={kbps} speed-preset=medium key-int-max={self.fps * 2}",
            "openh264enc": f"openh264enc bitrate={kbps * 1000} complexity=high",
            "vah264enc": f"vah264enc bitrate={kbps}",
            "vaapih264enc": f"vaapih264enc bitrate={kbps}",
            "nvh264enc": f"nvh264enc bitrate={kbps}",
            "vp9enc": f"vp9enc target-bitrate={kbps * 1000} deadline=1 cpu-used=4 row-mt=true "
                      f"keyframe-max-dist={self.fps * 2}",
            "vp8enc": f"vp8enc target-bitrate={kbps * 1000} deadline=1 cpu-used=4 keyframe-max-dist={self.fps * 2}",
        }[element]
        tail = "h264parse ! mp4mux" if codec == "h264" else "webmmux"
        caps = f"video/x-raw,format=RGB,width={width},height={height},framerate={self.fps}/1"
        desc = (f'appsrc name=src format=time is-live=false block=true max-bytes={width * height * 3 * 2} '
                f'caps="{caps}" ! videoconvert ! video/x-raw,format=I420 ! {enc} ! {tail} '
                f'! filesink name=out')
        self.pipeline = Gst.parse_launch(desc)
        self.pipeline.get_by_name("out").set_property("location", str(self.path))
        self.src = self.pipeline.get_by_name("src")
        self.bus = self.pipeline.get_bus()
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"GStreamer could not start the {element} encoder")

    def _check(self) -> None:
        msg = self.bus.pop_filtered(self.Gst.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            raise RuntimeError(f"video encoder: {err.message}")

    def write(self, rgb: np.ndarray) -> None:
        Gst = self.Gst
        self._check()
        buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(rgb, np.uint8).tobytes())
        buf.pts = self.count * Gst.SECOND // self.fps
        buf.duration = Gst.SECOND // self.fps
        if self.src.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            self._check()
            raise RuntimeError("the video encoder stopped accepting frames")
        self.count += 1

    def close(self) -> None:
        Gst = self.Gst
        self.src.emit("end-of-stream")
        msg = self.bus.timed_pop_filtered(120 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self.pipeline.set_state(Gst.State.NULL)
        if msg is None:
            raise RuntimeError("the video encoder did not finish")
        if msg.type == Gst.MessageType.ERROR:
            err, _dbg = msg.parse_error()
            raise RuntimeError(f"video encoder: {err.message}")

    def abort(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)
        try:
            self.path.unlink()
        except OSError:
            pass


def open_sink(path: Path, width: int, height: int, fps: int, prefer: Optional[str] = None):
    """The best sink for ``path`` (its suffix is corrected to the container actually written: .mp4 / .webm, or a
    ``<stem>-frames`` folder of PNGs when no encoder works). ``prefer`` = "png" forces the image sequence."""
    path = Path(path)
    stem = path.with_suffix("") if path.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov") else path
    errors = []
    if prefer != "png":
        for codec, element in available_codecs():
            target = stem.with_suffix(".mp4" if codec == "h264" else ".webm")
            try:
                return GstSink(target, width, height, fps, codec, element)
            except Exception as e:  # noqa: BLE001 — try the next encoder
                errors.append(f"{element}: {e}")
    sink = PngSequenceSink(stem.with_name(stem.name + "-frames"))
    sink.fallback_reason = "; ".join(errors) or ("" if prefer == "png" else "no GStreamer video encoder")
    return sink


class ThreadedSink:
    """Encodes on a worker thread, fed through a queue of at most ``depth`` frames: :meth:`offer` never blocks
    (the GTK main loop retries when it returns False), :meth:`put` does (CLI, tests)."""

    def __init__(self, sink, depth: int = 2):
        self.sink = sink
        self.error: Optional[str] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=depth)
        self._t = threading.Thread(target=self._run, daemon=True, name="turntable-encode")
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if self.error is None:
                try:
                    self.sink.write(item)
                except Exception as e:  # noqa: BLE001
                    self.error = str(e) or type(e).__name__

    def offer(self, rgb: np.ndarray) -> bool:
        if self.error:
            raise RuntimeError(self.error)
        try:
            self._q.put_nowait(rgb)
            return True
        except queue.Full:
            return False

    def put(self, rgb: np.ndarray) -> None:
        if self.error:
            raise RuntimeError(self.error)
        self._q.put(rgb)

    def finish(self) -> None:
        """Wait for the queued frames, close the file. Blocks: call it off the UI thread for long videos."""
        self._q.put(None)
        self._t.join()
        if self.error:
            self.sink.abort()
            raise RuntimeError(self.error)
        self.sink.close()

    def abort(self) -> None:
        self.error = self.error or "cancelled"
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(None)
        self._t.join()
        self.sink.abort()


def render_turntable(renderer, start, settings, options: TurntableOptions, path: Path, positions: np.ndarray,
                     hidden: Optional[np.ndarray] = None, progress: Optional[Callable[[float], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None, prefer: Optional[str] = None) -> Path:
    """Synchronous turntable (CLI, tests): renders on the calling thread (GL context current), encodes on a worker.
    Returns the file (or PNG folder) written."""
    sink = open_sink(path, options.width, options.height, options.fps, prefer)
    ts = ThreadedSink(sink)
    try:
        for i, frame in enumerate(render_frames(renderer, start, settings, options, positions, hidden)):
            if cancelled is not None and cancelled():
                raise TurntableCancelled()
            ts.put(frame)
            if progress is not None:
                progress((i + 1) / options.frames)
    except BaseException:
        ts.abort()
        raise
    ts.finish()
    return sink.path


def describe_encoder() -> str:
    c = available_codecs()
    if not c:
        return "No video encoder found: frames are saved as a PNG sequence"
    codec, element = c[0]
    return {"h264": "H.264 MP4", "vp9": "VP9 WebM", "vp8": "VP8 WebM"}[codec] + f" ({element})"
