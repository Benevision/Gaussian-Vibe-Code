"""The splat viewer widget: a GtkGLArea driving :class:`splatworker.studio.renderer.SplatRenderer`.

Input (same as the Mac, Ctrl for ⌘): drag orbit (Select tool: paint a selection, Alt erases), Shift-drag /
right / middle drag pan, scroll / pinch zoom, double-click sets the orbit centre, W A S D Q E fly (Shift ×4,
Alt ×¼), Tab orbit ↔ fly, F frame, R reset, 1–6 display modes, Delete removes the selection.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
from gi.repository import Gdk, GLib, Gtk

from ....studio.camera import Camera, adaptive_fly_speed
from ....studio.edit import EditState, pick_point
from ....studio.renderer import MODES, SplatRenderer, ViewSettings, box_lines, frustum_lines, pack_cloud
from ....studio.sorting import BackgroundSorter
from ....studio.splats import GaussianCloud

FLY_KEYS = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, -1, 0), "d": (0, 1, 0), "e": (0, 0, 1), "q": (0, 0, -1),
            "up": (1, 0, 0), "down": (-1, 0, 0), "left": (0, -1, 0), "right": (0, 1, 0)}


class ViewerArea(Gtk.GLArea):
    def __init__(self):
        super().__init__(hexpand=True, vexpand=True, focusable=True, can_focus=True)
        if hasattr(self, "set_allowed_apis") and hasattr(Gdk, "GLAPI"):
            self.set_allowed_apis(Gdk.GLAPI.GL)
        self.set_required_version(4, 3)
        self.set_has_depth_buffer(False)
        self.set_auto_render(False)
        self.renderer = SplatRenderer()
        self.camera = Camera()
        self.settings = ViewSettings()
        self.edit: Optional[EditState] = None
        self.cloud: Optional[GaussianCloud] = None
        self.gl_error: Optional[str] = None
        self.suspended = False            # a turntable video is using the renderer: don't draw
        self.tool = "navigate"            # navigate | select
        self.brush_radius = 30.0
        self.rotate_speed = self.pan_speed = self.zoom_speed = self.fly_speed = 1.0
        self.invert_vertical = False
        self.cameras_c2w: Optional[np.ndarray] = None
        self.show_cameras = True
        self.show_crop = False
        self.on_changed: Optional[Callable[[], None]] = None       # stats / selection changed
        self.on_gl_error: Optional[Callable[[str], None]] = None
        self.on_key_action: Optional[Callable[[str], None]] = None
        self._pending_cloud = None
        self._pending_coverage = None
        self._order_pending = None
        self._sort_gen = 0
        self._sorted_gen = -1
        self._keys = set()
        self._tick_id = None
        self._last_tick = 0.0
        self._drag_start = None
        self._drag_mode = None
        self._scene_radius = 1.0
        self._proj_cache = None
        self.fps = 0.0
        self._frames = []
        self.sorter = BackgroundSorter(self._sorted)
        self.connect("realize", self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render", self._on_render)
        self._install_input()

    # ------------------------------------------------------------------ GL lifecycle
    def _on_realize(self, *_):
        self.make_current()
        err = self.get_error()
        if err is not None:
            self._fail(f"OpenGL is not available: {err.message}")
            return
        try:
            self.renderer.init_gl()
        except Exception as e:  # noqa: BLE001
            self._fail(str(e))
            return
        if self._pending_cloud is not None:
            self._upload(*self._pending_cloud)
        self.update_lines()

    def _fail(self, msg: str) -> None:
        self.gl_error = msg
        if self.on_gl_error:
            GLib.idle_add(self.on_gl_error, msg)

    def _on_unrealize(self, *_):
        self.make_current()
        if self.get_error() is None:
            self.renderer.destroy()

    def _on_render(self, area, ctx) -> bool:
        if not self.renderer.ready or self.suspended:
            return True
        if self._order_pending is not None:
            order, n = self._order_pending
            self._order_pending = None
            self.renderer.set_order(order, n)
        sf = self.get_scale_factor()
        w, h = max(1, self.get_width() * sf), max(1, self.get_height() * sf)
        self._update_depth_range()
        try:
            self.renderer.render(self.camera, w, h, self.settings)
        except Exception as e:  # noqa: BLE001 — keep the UI alive, report once
            if self.gl_error is None:
                self._fail(f"render failed: {e}")
        now = time.monotonic()
        self._frames = [t for t in self._frames if now - t < 1.0] + [now]
        self.fps = len(self._frames)
        return True

    # ------------------------------------------------------------------ scene
    def set_cloud(self, cloud: Optional[GaussianCloud], edit: Optional[EditState] = None, reset_camera: bool = True,
                  packed: Optional[dict] = None) -> None:
        self.cloud = cloud
        self._pending_coverage = None
        self.edit = edit if edit is not None else (EditState(cloud) if cloud is not None else None)
        self._proj_cache = None
        if cloud is not None and cloud.count:
            lo, hi = cloud.robust_bounds(0.02)
            self._scene_radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
            if reset_camera:
                if cloud.up is not None:
                    self.camera.up = np.asarray(cloud.up, np.float64) / (np.linalg.norm(cloud.up) or 1)
                self.camera.frame(lo, hi)
                self.camera.pitch = -15.0
        if self.renderer.ready:
            self.make_current()
            self._upload(cloud, packed)
        else:
            self._pending_cloud = (cloud, packed)
        self.update_lines()
        self.request_sort()

    def _upload(self, cloud, packed) -> None:
        self._pending_cloud = None
        self.renderer.set_cloud(cloud, packed if packed is not None else (pack_cloud(cloud) if cloud else None))
        if self._pending_coverage is not None:
            self.renderer.set_coverage(self._pending_coverage)
            self._pending_coverage = None
        self.refresh_flags()
        self.queue_render()

    def refresh_flags(self) -> None:
        """Hidden / selected flags changed (edits, crop)."""
        self._proj_cache = None
        if self.cloud is None or self.edit is None or not self.renderer.ready:
            return
        self.make_current()
        self.renderer.set_flags(SplatRenderer.make_flags(self.edit.hidden(), self.edit.selected, self.cloud.count))
        self.request_sort()
        if self.on_changed:
            self.on_changed()

    def update_lines(self) -> None:
        parts = []
        if self.show_cameras and self.cameras_c2w is not None and len(self.cameras_c2w):
            parts.append(frustum_lines(self.cameras_c2w, self._scene_radius * 0.04))
        if self.show_crop and self.edit is not None:
            parts.append(box_lines(self.edit.crop.lo, self.edit.crop.hi))
        lines = np.concatenate(parts) if parts else None
        if self.renderer.ready:
            self.make_current()
            self.renderer.set_lines(lines)
        else:
            self._pending_lines = lines
        self.queue_render()

    def request_sort(self) -> None:
        if self.cloud is None:
            return
        self._sort_gen += 1
        hidden = self.edit.hidden() if self.edit is not None else None
        self.sorter.request(self.cloud.positions, self.camera.position(), self.camera.forward(), self.camera.near,
                            hidden, self._sort_gen)
        self.queue_render()

    def _sorted(self, gen: int, order, n: int) -> None:      # sorter thread
        def apply():
            if gen >= self._sorted_gen:
                self._sorted_gen = gen
                self._order_pending = (order, n)
                self.queue_render()
            return False
        GLib.idle_add(apply)

    def _update_depth_range(self) -> None:
        d = float(np.linalg.norm(self.camera.position() - self.camera.target)) if self.camera.mode == "orbit" else \
            self._scene_radius
        near = max(self.camera.near, d - self._scene_radius * 1.5, 1e-3)
        far = max(near * 1.01, d + self._scene_radius * 1.5)
        self.renderer.depth_range = (math.log(near), math.log(far))

    def camera_moved(self) -> None:
        self._proj_cache = None
        self.request_sort()

    # ------------------------------------------------------------------ projection helpers
    def _projected(self):
        if self._proj_cache is None and self.cloud is not None:
            px, z = self.camera.project(self.cloud.positions, self.get_width(), self.get_height())
            self._proj_cache = (px, z)
        return self._proj_cache

    def pick(self, x: float, y: float) -> Optional[np.ndarray]:
        if self.cloud is None:
            return None
        px, z = self._projected()
        hits = pick_point(px, z, self.cloud.opacities, self.edit.hidden(), x, y)
        if hits is None or len(hits) == 0:
            return None
        return np.median(self.cloud.positions[hits], axis=0)

    # ------------------------------------------------------------------ input
    def _install_input(self) -> None:
        drag = Gtk.GestureDrag(button=0)
        drag.connect("drag-begin", self._drag_begin)
        drag.connect("drag-update", self._drag_update)
        drag.connect("drag-end", self._drag_end)
        self.add_controller(drag)
        click = Gtk.GestureClick(button=1)
        click.connect("pressed", self._clicked)
        self.add_controller(click)
        scroll = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.VERTICAL
                                           | Gtk.EventControllerScrollFlags.KINETIC)
        scroll.connect("scroll", self._scroll)
        self.add_controller(scroll)
        zoom = Gtk.GestureZoom()
        zoom.connect("scale-changed", self._pinch)
        zoom.connect("begin", lambda *_: setattr(self, "_pinch_last", 1.0))
        self.add_controller(zoom)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._key_pressed)
        keys.connect("key-released", self._key_released)
        self.add_controller(keys)
        motion = Gtk.EventControllerMotion()
        motion.connect("enter", lambda *_: self.grab_focus())
        self.add_controller(motion)

    def _clicked(self, gesture, n_press, x, y):
        self.grab_focus()
        if n_press == 2 and self.tool == "navigate":
            p = self.pick(x, y)
            if p is not None:
                self.camera.set_orbit_center(p)
                self.camera_moved()

    def _drag_begin(self, gesture, x, y):
        self.grab_focus()
        button = gesture.get_current_button()
        mods = gesture.get_current_event_state()
        shift = bool(mods & Gdk.ModifierType.SHIFT_MASK)
        alt = bool(mods & Gdk.ModifierType.ALT_MASK)
        if button in (2, 3) or (button == 1 and shift and self.tool == "navigate"):
            self._drag_mode = "pan"
        elif button == 1 and self.tool == "select":
            self._drag_mode = "erase" if alt else "paint"
            self._paint(x, y)
        else:
            self._drag_mode = "orbit"
        self._drag_start = (x, y)
        self._drag_last = (0.0, 0.0)

    def _drag_update(self, gesture, dx, dy):
        ddx, ddy = dx - self._drag_last[0], dy - self._drag_last[1]
        self._drag_last = (dx, dy)
        if self._drag_mode == "orbit":
            k = 0.3 * self.rotate_speed
            self.camera.orbit(-ddx * k, (-ddy if self.invert_vertical else ddy) * k)
            self.camera_moved()
        elif self._drag_mode == "pan":
            self.camera.pan(ddx * self.pan_speed, ddy * self.pan_speed, self.get_height())
            self.camera_moved()
        elif self._drag_mode in ("paint", "erase"):
            self._paint(self._drag_start[0] + dx, self._drag_start[1] + dy)

    def _drag_end(self, *_):
        self._drag_mode = None

    def _paint(self, x, y) -> None:
        if self.cloud is None or self.edit is None:
            return
        px, z = self._projected()
        self.edit.select_brush(px, z, x, y, self.brush_radius, deselect=self._drag_mode == "erase")
        self.refresh_flags()

    def _scroll(self, ctl, dx, dy) -> bool:
        self.camera.zoom(math.pow(1.1, -dy * self.zoom_speed))
        self.camera_moved()
        return True

    def _pinch(self, gesture, scale):
        last = getattr(self, "_pinch_last", 1.0)
        if scale > 0 and last > 0:
            self.camera.zoom(scale / last)
            self._pinch_last = scale
            self.camera_moved()

    def _key_pressed(self, ctl, keyval, keycode, state) -> bool:
        name = (Gdk.keyval_name(keyval) or "").lower()
        if state & Gdk.ModifierType.CONTROL_MASK:
            return False
        if name in FLY_KEYS:
            self._keys.add(name)
            if self._tick_id is None:
                self._last_tick = time.monotonic()
                self._tick_id = self.add_tick_callback(self._fly_tick)
            return True
        actions = {"f": "frame", "r": "reset", "tab": "toggle-mode", "delete": "delete", "backspace": "delete",
                   "question": "shortcuts", "1": "mode-color", "2": "mode-depth", "3": "mode-centers",
                   "4": "mode-ellipses", "5": "mode-opacity", "6": "mode-coverage"}
        if name in actions and self.on_key_action:
            self.on_key_action(actions[name])
            return True
        return False

    def _key_released(self, ctl, keyval, keycode, state) -> None:
        self._keys.discard((Gdk.keyval_name(keyval) or "").lower())

    def _fly_tick(self, widget, clock) -> bool:
        now = time.monotonic()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now
        if not self._keys:
            self._tick_id = None
            return GLib.SOURCE_REMOVE
        display = self.get_display()
        seat = display.get_default_seat() if display else None
        mods = seat.get_keyboard().get_modifier_state() if seat and seat.get_keyboard() else 0
        boost = 4.0 if mods & Gdk.ModifierType.SHIFT_MASK else (0.25 if mods & Gdk.ModifierType.ALT_MASK else 1.0)
        v = adaptive_fly_speed(self._scene_radius, self.fly_speed) * boost * dt
        f = sum(FLY_KEYS[k][0] for k in self._keys)
        r = sum(FLY_KEYS[k][1] for k in self._keys)
        u = sum(FLY_KEYS[k][2] for k in self._keys)
        self.camera.move(forward=f * v, right=r * v, up=u * v)
        self.camera_moved()
        return GLib.SOURCE_CONTINUE

    # ------------------------------------------------------------------ misc
    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self.settings.mode = mode
            self.queue_render()

    def set_coverage(self, score: Optional[np.ndarray]) -> None:
        """Coverage scores for the open cloud (Coverage display mode); None clears them."""
        if self.renderer.ready and self._pending_cloud is None:
            self.make_current()
            self.renderer.set_coverage(score)
        else:
            self._pending_coverage = score
        self.queue_render()

    def frame(self, view: Optional[str] = None) -> None:
        if self.cloud is None or not self.cloud.count:
            return
        lo, hi = self.cloud.robust_bounds(0.02)
        self.camera.frame(lo, hi, view)
        self.camera_moved()

    def screenshot(self, width: int = 3840, height: int = 2160) -> Optional[np.ndarray]:
        if not self.renderer.ready:
            return None
        from ....studio.renderer import render_offscreen

        self.make_current()
        img = render_offscreen(self.renderer, self.camera, width, height, self.settings)
        self.queue_render()
        return img

    def close(self) -> None:
        self.sorter.close()
