"""Viewer page: header bar, home tiles, the GL viewer with a HUD, and the inspector (View / Edit / Export)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from ....studio import autoclean, formats
from ....studio.camera import UP_CHOICES
from ....studio.edit import EditState, estimate_size
from ....studio.projects import Project
from ....studio.renderer import pack_cloud
from ...formatting import megabytes
from ..common import alert, label
from .rows import button_row, combo_row, group, plain_expander, plain_row, scrolled, set_combo, spin_row, switch_row
from .viewer_area import ViewerArea

DISPLAY_MODES = [("color", "Colour"), ("depth", "Depth"), ("centers", "Centres"), ("ellipses", "Ellipses"),
                 ("opacity", "Opacity")]
EXPORT_FORMATS = [(k, v) for k, v in formats.FORMATS.items()]


def _count(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else (f"{n / 1e3:.0f} k" if n >= 1e4 else str(n))


def rotation_between(a, b) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating unit vector a onto b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(a @ b)
    if c < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    v = np.cross(a, b)
    q = np.array([1 + c, *v])
    return q / np.linalg.norm(q)


class ViewerPage(Gtk.Box):
    def __init__(self, window, controller, on_new_project: Callable[[Optional[str], Optional[str]], None]):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.window = window
        self.ctl = controller
        self.on_new_project = on_new_project
        self.path: Optional[Path] = None
        self.project: Optional[Project] = None
        self._live_job = None
        self._building = False

        view = Adw.ToolbarView(hexpand=True, vexpand=True)
        self.append(view)
        header = Adw.HeaderBar()
        self.title = Adw.WindowTitle(title="Viewer", subtitle="")
        header.set_title_widget(self.title)
        for icon, tip, cb in (("go-home-symbolic", "Home", self.go_home),
                              ("document-new-symbolic", "New project (Ctrl+N)", lambda: on_new_project(None, None)),
                              ("document-open-symbolic", "Open a splat or dataset (Ctrl+O)", self.open_dialog)):
            b = Gtk.Button(icon_name=icon, tooltip_text=tip)
            b.connect("clicked", lambda *_a, c=cb: c())
            header.pack_start(b)
        self.inspector_btn = Gtk.ToggleButton(icon_name="sidebar-show-right-symbolic", active=True,
                                              tooltip_text="Inspector")
        header.pack_end(self.inspector_btn)
        tools = Gtk.Box(css_classes=["linked"])
        self.nav_btn = Gtk.ToggleButton(icon_name="object-rotate-right-symbolic", active=True,
                                        tooltip_text="Navigate (drag to orbit)")
        self.sel_btn = Gtk.ToggleButton(icon_name="edit-select-symbolic", group=self.nav_btn,
                                        tooltip_text="Select (drag to paint, Alt to erase, Delete removes)")
        self.nav_btn.connect("toggled", lambda b: self._set_tool("navigate" if b.get_active() else "select"))
        tools.append(self.nav_btn)
        tools.append(self.sel_btn)
        header.pack_end(tools)
        orient = Gtk.MenuButton(icon_name="object-flip-vertical-symbolic", tooltip_text="Orientation")
        orient.set_popover(self._orientation_popover())
        header.pack_end(orient)
        reset = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Reset view (R)")
        reset.connect("clicked", lambda *_: self.area.frame())
        header.pack_end(reset)
        view.add_top_bar(header)

        self.split = Adw.OverlaySplitView(sidebar_position=Gtk.PackType.END, min_sidebar_width=300,
                                          max_sidebar_width=380)
        self.inspector_btn.bind_property("active", self.split, "show-sidebar", 1 | 2)   # bidirectional + sync
        view.set_content(self.split)

        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.stack.add_named(self._home(), "home")
        overlay = Gtk.Overlay()
        self.area = ViewerArea()
        self.area.on_changed = self._stats_changed
        self.area.on_gl_error = self._gl_error
        self.area.on_key_action = self._key_action
        overlay.set_child(self.area)
        self.hud = label("", ("caption", "osd"), wrap=False)
        self.hud.set_halign(Gtk.Align.START)
        self.hud.set_valign(Gtk.Align.END)
        self.hud.set_margin_start(8)
        self.hud.set_margin_bottom(8)
        overlay.add_overlay(self.hud)
        self.spinner_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, halign=Gtk.Align.CENTER,
                                   valign=Gtk.Align.CENTER, visible=False)
        self.spinner_box.add_css_class("osd")
        self.spinner_box.add_css_class("card")
        sp = Gtk.Spinner(spinning=True, width_request=32, height_request=32, margin_top=12)
        self.spinner_label = label("Loading…", wrap=False, xalign=0.5)
        self.spinner_label.set_margin_start(16)
        self.spinner_label.set_margin_end(16)
        self.spinner_label.set_margin_bottom(12)
        self.spinner_box.append(sp)
        self.spinner_box.append(self.spinner_label)
        overlay.add_overlay(self.spinner_box)
        self.stack.add_named(overlay, "viewer")
        self.error_page = Adw.StatusPage(icon_name="dialog-error-symbolic", title="The viewer can't start")
        self.stack.add_named(self.error_page, "error")
        self.split.set_content(self.stack)
        self.split.set_sidebar(self._inspector())
        self.split.set_show_sidebar(False)
        self.inspector_btn.set_sensitive(False)
        self._apply_prefs()
        GLib.timeout_add(500, self._hud_tick)
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

    # ================================================================== home
    def _home(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, halign=Gtk.Align.CENTER,
                      valign=Gtk.Align.CENTER, margin_top=24, margin_bottom=24, margin_start=24, margin_end=24)
        box.append(label("SplatWork", ("title-1",), xalign=0.5))
        box.append(label("View, create and edit 3D Gaussian splats — trained on this PC's GPU, which keeps working "
                         "for your Mac at the same time.", ("dim-label",), xalign=0.5))

        def tiles(title, items):
            box.append(label(title, ("heading",), xalign=0.0))
            flow = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE, max_children_per_line=4,
                               min_children_per_line=2, column_spacing=12, row_spacing=12, homogeneous=True)
            for icon, name, sub, cb in items:
                b = Gtk.Button()
                b.add_css_class("card")
                inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_top=14, margin_bottom=14,
                                margin_start=10, margin_end=10)
                inner.append(Gtk.Image(icon_name=icon, pixel_size=32))
                inner.append(label(name, ("heading",), xalign=0.5))
                inner.append(label(sub, ("caption", "dim-label"), xalign=0.5))
                b.set_child(inner)
                b.connect("clicked", lambda *_a, c=cb: c())
                flow.append(b)
            box.append(flow)
        np_ = self.on_new_project
        tiles("Create", [
            ("folder-pictures-symbolic", "Photos", "A folder of photos", lambda: np_("photos", None)),
            ("camera-video-symbolic", "Video", "Walk around the scene", lambda: np_("video", None)),
            ("preferences-system-network-symbolic", "360° video", "Equirectangular footage", lambda: np_("video360", None)),
            ("phone-symbolic", "iPhone capture", ".splatcapture with poses", lambda: np_("capture", None)),
            ("view-grid-symbolic", "COLMAP / Nerfstudio", "Posed images", lambda: np_("colmap", None)),
        ])
        tiles("View", [
            ("document-open-symbolic", "Open splat", "PLY, compressed PLY, .splat, .spz", self.open_dialog),
            ("folder-open-symbolic", "Open dataset", "A folder with dataset.json", self.open_dataset_dialog),
        ])
        box.append(label("Tip: drop files anywhere on this window. Press ? in the viewer for shortcuts.",
                         ("caption", "dim-label"), xalign=0.5))
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        clamp = Adw.Clamp(maximum_size=760)
        clamp.set_child(box)
        sw.set_child(clamp)
        return sw

    def go_home(self) -> None:
        self.stack.set_visible_child_name("home")
        self.split.set_show_sidebar(False)
        self.inspector_btn.set_sensitive(False)
        self.title.set_title("Viewer")
        self.title.set_subtitle("")

    # ================================================================== inspector
    def _inspector(self) -> Gtk.Widget:
        stack = Adw.ViewStack()
        stack.add_titled_with_icon(self._view_tab(), "view", "View", "view-reveal-symbolic")
        stack.add_titled_with_icon(self._edit_tab(), "edit", "Edit", "document-edit-symbolic")
        stack.add_titled_with_icon(self._export_tab(), "export", "Export", "document-save-symbolic")
        self.insp_stack = stack
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sw = Adw.ViewSwitcher(stack=stack, policy=Adw.ViewSwitcherPolicy.NARROW, margin_top=6, margin_bottom=6)
        box.append(sw)
        box.append(stack)
        box.add_css_class("background")
        return box

    def _view_tab(self) -> Gtk.Widget:
        a = self.area
        s = a.settings
        disp = group("Display")
        self.mode_row = combo_row("Display mode", DISPLAY_MODES, s.mode, a.set_mode, "viewer.displayMode")
        disp.add(self.mode_row)
        disp.add(spin_row("View-dependent colour", 0, 3, 1, s.sh_degree, lambda v: self._set("sh_degree", int(v)),
                          setting_id="viewer.maxSHDegree"))
        disp.add(spin_row("Splat scale", 0.1, 3.0, 0.05, s.splat_scale, lambda v: self._set("splat_scale", v), 2,
                          "viewer.splatScale"))
        disp.add(spin_row("Minimum opacity", 0.0, 0.5, 0.005, s.alpha_min, lambda v: self._set("alpha_min", v), 3,
                          "viewer.alphaCutoff"))
        disp.add(combo_row("Anti-aliasing filter", [("auto", "Auto (as trained)"), ("on", "On"), ("off", "Off")],
                           s.antialias, lambda v: self._set("antialias", v), "viewer.antialias"))
        bg_row = plain_row(title="Background colour")
        bg_row.set_tooltip_text("Colour behind the scene (display only).")
        self.bg_btn = Gtk.ColorDialogButton(dialog=Gtk.ColorDialog(with_alpha=False), valign=Gtk.Align.CENTER)
        c = Gdk.RGBA()
        c.red, c.green, c.blue, c.alpha = (*s.background, 1.0)
        self.bg_btn.set_rgba(c)
        self.bg_btn.connect("notify::rgba", self._bg_changed)
        bg_row.add_suffix(self.bg_btn)
        disp.add(bg_row)
        disp.add(switch_row("Show training cameras", a.show_cameras, self._show_cameras, "viewer.showCameras"))
        disp.add(switch_row("Learned skybox", s.show_sky, lambda v: self._set("show_sky", v),
                            subtitle="Draw the sky panorama trained with the scene (Colour mode)."))

        nav = group("Navigation")
        self.nav_mode_row = combo_row("Navigation mode", [("orbit", "Orbit"), ("fly", "Fly (W A S D)")], "orbit",
                                      self._nav_mode, "viewer.cameraMode")
        nav.add(self.nav_mode_row)
        self.fov_row = spin_row("Field of view", 20, 110, 1, a.camera.fov_y, self._fov, setting_id="viewer.fov")
        nav.add(self.fov_row)
        vb = Gtk.Box(css_classes=["linked"], halign=Gtk.Align.CENTER, margin_top=8, margin_bottom=4)
        for key in ("front", "back", "left", "right", "top"):
            b = Gtk.Button(label=key.title(), tooltip_text=f"Look at the scene from the {key}")
            b.connect("clicked", lambda *_x, k=key: self.area.frame(k))
            vb.append(b)
        nav.add(vb)
        nav.add(button_row("Frame scene", "Frame", lambda: self.area.frame(), subtitle="F"))
        p = self.ctl.prefs
        for title, attr, sid in (("Rotation speed", "rotate_speed", "nav.rotateSpeed"),
                                 ("Pan speed", "pan_speed", "nav.panSpeed"),
                                 ("Zoom speed", "zoom_speed", "nav.zoomSpeed"),
                                 ("Fly speed", "fly_speed", "nav.flySpeed")):
            nav.add(spin_row(title, 0.1, 5.0, 0.1, getattr(p, attr), lambda v, a_=attr: self._nav_pref(a_, v), 1, sid))
        nav.add(switch_row("Invert vertical rotation", p.invert_vertical,
                           lambda v: self._nav_pref("invert_vertical", v), "nav.invertVertical"))

        perf = group("Performance")
        perf.add(combo_row("Render resolution", [("1.0", "100 %"), ("0.75", "75 %"), ("0.5", "50 %"), ("0.33", "33 %")],
                           f"{s.render_scale:g}" if f"{s.render_scale:g}" in ("1", "0.75", "0.5", "0.33") else "1.0",
                           lambda v: self._set("render_scale", float(v)), "viewer.renderScale"))
        self.gl_row = plain_row(title="Renderer", subtitle="—")
        perf.add(self.gl_row)

        self.cams_group = group("Training cameras", "Click one to look through it.")
        self.cams_expander = plain_expander(title="Cameras", subtitle="No dataset")
        self.cams_group.add(self.cams_expander)
        self._cam_rows = []
        return scrolled(disp, nav, perf, self.cams_group)

    def _edit_tab(self) -> Gtk.Widget:
        hist = group("History")
        self.counts_row = plain_row(title="Splats", subtitle="—")
        hist.add(self.counts_row)
        hb = Gtk.Box(css_classes=["linked"], valign=Gtk.Align.CENTER)
        self.undo_btn = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Undo (Ctrl+Z)")
        self.redo_btn = Gtk.Button(icon_name="edit-redo-symbolic", tooltip_text="Redo (Ctrl+Shift+Z)")
        restore = Gtk.Button(label="Restore all")
        self.undo_btn.connect("clicked", lambda *_: self.undo())
        self.redo_btn.connect("clicked", lambda *_: self.redo())
        restore.connect("clicked", lambda *_: self._edit(lambda e: e.restore_all()))
        for b in (self.undo_btn, self.redo_btn, restore):
            hb.append(b)
        self.counts_row.add_suffix(hb)

        sel = group("Selection brush", "Choose the Select tool in the header bar, then paint over splats.")
        sel.add(spin_row("Brush radius", 4, 120, 1, self.area.brush_radius,
                         lambda v: setattr(self.area, "brush_radius", v), setting_id="edit.brush"))
        bb = plain_row(title="Selected")
        self.sel_label = bb
        box = Gtk.Box(css_classes=["linked"], valign=Gtk.Align.CENTER)
        for text, fn in (("Delete", lambda e: e.delete_selected()), ("Invert", lambda e: e.invert_selection()),
                         ("Clear", lambda e: e.clear_selection())):
            b = Gtk.Button(label=text)
            b.connect("clicked", lambda *_x, f=fn: self._edit(f))
            box.append(b)
        bb.add_suffix(box)
        sel.add(bb)

        crop = group("Crop box")
        self.crop_switch = switch_row("Crop box", False, self._crop_toggle, "edit.cropBox")
        crop.add(self.crop_switch)
        self.crop_rows = []
        for axis, name in enumerate("XYZ"):
            for side in ("lo", "hi"):
                r = spin_row(f"{name} {'min' if side == 'lo' else 'max'}", -1e4, 1e4, 0.01, 0.0,
                             lambda v, a_=axis, s_=side: self._crop_value(a_, s_, v), 3)
                r.set_sensitive(False)
                crop.add(r)
                self.crop_rows.append((axis, side, r))
        cb = plain_row(title="")
        box = Gtk.Box(css_classes=["linked"], valign=Gtk.Align.CENTER)
        fit = Gtk.Button(label="Fit to scene")
        fit.connect("clicked", lambda *_: self._crop_fit())
        apply_ = Gtk.Button(label="Apply crop")
        apply_.add_css_class("suggested-action")
        apply_.connect("clicked", lambda *_: self._edit(lambda e: e.apply_crop(), lines=True))
        box.append(fit)
        box.append(apply_)
        cb.add_suffix(box)
        crop.add(cb)

        clean = group("Clean up")
        self.few_views = switch_row("Include floaters", True, lambda v: None, "edit.autoCleanFewViews")
        self.auto_row = button_row("Auto clean", "Select", self.auto_clean, setting_id="edit.autoClean")
        clean.add(self.auto_row)
        clean.add(self.few_views)
        self.opa_thr = 0.05
        r = spin_row("Delete below opacity", 0.0, 0.5, 0.005, self.opa_thr, lambda v: setattr(self, "opa_thr", v), 3,
                     "edit.opacityThreshold")
        r.add_suffix(self._small_button("Delete", lambda: self._edit(lambda e: e.delete(e.below_opacity(self.opa_thr)))))
        clean.add(r)
        self.size_thr = 1.0
        r = spin_row("Delete larger than", 0.001, 1000, 0.01, self.size_thr, lambda v: setattr(self, "size_thr", v), 3,
                     "edit.sizeThreshold")
        r.add_suffix(self._small_button("Delete", lambda: self._edit(lambda e: e.delete(e.larger_than(self.size_thr)))))
        clean.add(r)
        self.iso_n = 4
        r = spin_row("Remove isolated splats", 1, 30, 1, self.iso_n, lambda v: setattr(self, "iso_n", int(v)),
                     setting_id="edit.outlierRadius")
        r.add_suffix(self._small_button("Remove", self._remove_isolated))
        clean.add(r)
        return scrolled(hist, sel, crop, clean)

    def _export_tab(self) -> Gtk.Widget:
        g = group("Export splat")
        self.exp_fmt = "ply"
        self.exp_sh = 3
        self.exp_visible = True
        self.exp_bake = False
        g.add(combo_row("File format", EXPORT_FORMATS, "ply", lambda v: self._exp("exp_fmt", v), "export.format"))
        g.add(spin_row("Export SH degree", 0, 3, 1, 3, lambda v: self._exp("exp_sh", int(v)),
                       setting_id="export.shDegree"))
        g.add(switch_row("Export only kept splats", True, lambda v: self._exp("exp_visible", v), "export.onlyVisible"))
        g.add(switch_row("Bake orientation", False, lambda v: self._exp("exp_bake", v), "export.bakeOrientation"))
        self.size_row = plain_row(title="Estimated size", subtitle="—")
        g.add(self.size_row)
        self.export_btn = button_row("Save a copy", "Export…", self.export_dialog)
        self.export_btn.button.add_css_class("suggested-action")
        g.add(self.export_btn)
        m = group("Media")
        m.add(button_row("Screenshot", "Save 4K PNG…", self.screenshot_dialog,
                         subtitle="3840 × 2160, the current view"))
        return scrolled(g, m)

    @staticmethod
    def _small_button(text, cb):
        b = Gtk.Button(label=text, valign=Gtk.Align.CENTER)
        b.connect("clicked", lambda *_: cb())
        return b

    # ------------------------------------------------------------------ inspector actions
    def _set(self, attr, value) -> None:
        setattr(self.area.settings, attr, value)
        self.area.queue_render()

    def _bg_changed(self, btn, _p) -> None:
        c = btn.get_rgba()
        self._set("background", (c.red, c.green, c.blue))
        self.ctl.prefs.background = [c.red, c.green, c.blue]
        self.ctl.save_prefs()

    def _show_cameras(self, v: bool) -> None:
        self.area.show_cameras = v
        self.area.update_lines()

    def _nav_mode(self, mode: str) -> None:
        self.area.camera.set_mode(mode)
        self.area.camera_moved()

    def _nav_pref(self, attr: str, v) -> None:
        setattr(self.ctl.prefs, attr, v)
        setattr(self.area, attr, v)
        self.ctl.save_prefs()

    def _fov(self, v: float) -> None:
        self.area.camera.fov_y = v
        self.area.camera_moved()

    def _set_tool(self, tool: str) -> None:
        self.area.tool = tool
        self.area.settings.highlight_selection = True

    def _exp(self, attr, v) -> None:
        setattr(self, attr, v)
        self._stats_changed()

    def _edit(self, fn, lines: bool = False) -> None:
        if self.area.edit is None:
            return
        fn(self.area.edit)
        if lines:
            self.area.show_crop = self.area.edit.crop.enabled
            self.crop_switch.set_active(self.area.edit.crop.enabled)
            self.area.update_lines()
        self.area.refresh_flags()

    def undo(self) -> None:
        self._edit(lambda e: e.undo())

    def redo(self) -> None:
        self._edit(lambda e: e.redo())

    def _crop_toggle(self, on: bool) -> None:
        if self.area.edit is None:
            return
        self.area.edit.crop.enabled = on
        self.area.show_crop = on
        for _a, _s, r in self.crop_rows:
            r.set_sensitive(on)
        self.area.update_lines()
        self.area.refresh_flags()

    def _crop_value(self, axis, side, v) -> None:
        if self.area.edit is None or self._building:
            return
        getattr(self.area.edit.crop, side)[axis] = v
        self.area.update_lines()
        if self.area.edit.crop.enabled:
            self.area.refresh_flags()

    def _crop_fit(self) -> None:
        e = self.area.edit
        if e is None:
            return
        lo, hi = e.cloud.robust_bounds(0.01)
        e.crop.lo[:], e.crop.hi[:] = lo, hi
        self._sync_crop_rows()
        self.area.update_lines()
        self.area.refresh_flags()

    def _sync_crop_rows(self) -> None:
        e = self.area.edit
        if e is None:
            return
        self._building = True
        ext = float(np.abs(np.concatenate([e.crop.lo, e.crop.hi])).max() or 1.0) * 4
        for axis, side, r in self.crop_rows:
            adj = r.get_adjustment()
            adj.set_lower(-ext)
            adj.set_upper(ext)
            adj.set_step_increment(ext / 400)
            adj.set_value(float(getattr(e.crop, side)[axis]))
        self._building = False

    def _remove_isolated(self) -> None:
        e = self.area.edit
        if e is None:
            return
        self._busy("Finding isolated splats…")

        def work():
            mask = e.isolated(self.iso_n)
            GLib.idle_add(lambda: (self._busy(None), self._edit(lambda ed: ed.delete(mask)), False)[-1])
        threading.Thread(target=work, daemon=True).start()

    def auto_clean(self) -> None:
        """Census job on the local worker (needs the project's dataset), then select what the plan removes."""
        if self.area.cloud is None or self.path is None:
            return
        proj = self.project
        if proj is None or not proj.has_dataset:
            alert(self.window, "Auto clean needs the photos",
                  "Open a splat from a project's outputs folder (or a project made on the Mac) so the photos and "
                  "their cameras are known.", (("ok", "OK", "default"),), lambda _r: None)
            return
        include = self.few_views.get_active()
        cloud = self.area.cloud
        self._busy("Auto clean: measuring every splat in every photo…")
        src = self.path

        def go(client):
            try:
                ds = client.local_dataset(proj.dataset_dir)["datasetId"]
                job = client.census(ds, src, f"Auto clean {proj.name}", self.ctl.priority)
                import time

                while True:
                    s = client.job(job)
                    if s["state"] in ("completed", "failed", "cancelled"):
                        break
                    GLib.idle_add(self.spinner_label.set_text,
                                  f"Auto clean: {s.get('message') or s['state']} ({s.get('step', 0)}/{s.get('totalSteps', 0)})")
                    time.sleep(1.0)
                client.delete_dataset(ds)
                if s["state"] != "completed":
                    raise RuntimeError(s.get("message") or s["state"])
                import io
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    p = client.download(f"/v1/jobs/{job}/census.npz", Path(td) / "c.npz")
                    z = np.load(io.BytesIO(p.read_bytes()))
                    peak, views = z["peak"], z["views"]
                client.delete(f"/v1/jobs/{job}")
                if len(peak) != cloud.count:
                    raise RuntimeError("the census does not match the open file (was it edited on disk?)")
                plan = autoclean.plan(peak, views, cloud.opacity_logits, include_few_views=include)
                GLib.idle_add(self._auto_done, plan)
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._auto_failed, str(e))
        self.ctl.with_server(go, self._auto_failed)

    def _auto_done(self, plan) -> bool:
        self._busy(None)
        e = self.area.edit
        e.selected = plan.remove & ~e.deleted
        self.area.refresh_flags()
        extra = f", {_count(plan.few_views)} seen by too few photos (typical {plan.typical_views})" \
            if plan.typical_views else ""
        self.ctl.toast(f"Auto clean selected {_count(plan.total)} splats: {_count(plan.invisible)} invisible{extra}. "
                       "Press Delete to remove them.")
        return False

    def _auto_failed(self, msg: str) -> bool:
        self._busy(None)
        self.ctl.toast(f"Auto clean failed: {msg}")
        return False

    # ------------------------------------------------------------------ orientation
    def _orientation_popover(self) -> Gtk.Popover:
        pop = Gtk.Popover()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, margin_top=6, margin_bottom=6,
                      margin_start=6, margin_end=6)
        for text, cb in (("Flip upside down", lambda: self.area.camera.flip()),
                         ("Level to current view", lambda: self.area.camera.level_to_view())):
            b = Gtk.Button(label=text, css_classes=["flat"])
            b.connect("clicked", lambda *_a, c=cb: (c(), self.area.camera_moved(), pop.popdown()))
            box.append(b)
        box.append(Gtk.Separator())
        for key, (vec, text) in UP_CHOICES.items():
            b = Gtk.Button(label=text, css_classes=["flat"])
            b.connect("clicked", lambda *_a, v=vec: (self.area.camera.set_up(v), self.area.camera_moved(),
                                                    pop.popdown()))
            box.append(b)
        pop.set_child(box)
        return pop

    # ------------------------------------------------------------------ keys
    def _key_action(self, action: str) -> None:
        if action == "frame":
            self.area.frame()
        elif action == "reset":
            self.area.frame()
        elif action == "toggle-mode":
            m = "fly" if self.area.camera.mode == "orbit" else "orbit"
            set_combo(self.nav_mode_row, m)
        elif action == "delete":
            self._edit(lambda e: e.delete_selected())
        elif action == "shortcuts":
            self.window.show_shortcuts()
        elif action.startswith("mode-"):
            set_combo(self.mode_row, action[5:])

    # ------------------------------------------------------------------ loading
    def _busy(self, text: Optional[str]) -> None:
        self.spinner_box.set_visible(text is not None)
        if text:
            self.spinner_label.set_text(text)

    def open_dialog(self) -> None:
        d = Gtk.FileDialog(title="Open a splat")
        f = Gtk.FileFilter(name="Splats (PLY, SPLAT, SPZ)")
        for pat in ("*.ply", "*.splat", "*.spz", "*.PLY", "*.SPLAT", "*.SPZ"):
            f.add_pattern(pat)
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(f)
        d.set_filters(store)
        d.open(self.window, None, lambda dlg, res: self._opened(dlg.open_finish, res))

    def open_dataset_dialog(self) -> None:
        d = Gtk.FileDialog(title="Open a dataset folder")
        d.select_folder(self.window, None, lambda dlg, res: self._opened(dlg.select_folder_finish, res))

    def _opened(self, finish, res) -> None:
        try:
            f = finish(res)
        except GLib.Error:
            return
        if f is not None and f.get_path():
            self.window.open_path(Path(f.get_path()))

    def _on_drop(self, target, value, x, y) -> bool:
        files = value.get_files() if hasattr(value, "get_files") else []
        paths = [Path(f.get_path()) for f in files if f.get_path()]
        if paths:
            self.window.open_path(paths[0])
            return True
        return False

    def load(self, path: Path, keep_camera: bool = False, live: bool = False) -> None:
        """Open a splat file (in a thread), then show it."""
        path = Path(path)
        if self.area.gl_error:
            self._gl_error(self.area.gl_error)
        self.stack.set_visible_child_name("viewer")
        if not live:
            self._busy(f"Opening {path.name}…")

        def work():
            try:
                g = formats.load(path)
                packed = pack_cloud(g)
                proj = Project.find(path)
                cams = None
                if proj is not None and proj.has_dataset:
                    try:
                        doc = json.loads((proj.dataset_dir / "dataset.json").read_text())
                        cams = [c for c in doc.get("cameras", [])]
                        if g.up is None and doc.get("upVector"):
                            g.up = np.asarray(doc["upVector"], np.float32)
                    except (OSError, ValueError):
                        cams = None
                GLib.idle_add(self._loaded, path, g, packed, proj, cams, keep_camera, live)
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._load_failed, path, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, path, g, packed, proj, cams, keep_camera, live) -> bool:
        self._busy(None)
        self.path = path if not live else self.path
        self.project = proj
        edit = EditState(g)
        self.area.cameras_c2w = np.array([np.asarray(c["cameraToWorld"], float) for c in cams]) if cams else None
        self.area.set_cloud(g, edit, reset_camera=not keep_camera, packed=packed)
        self.split.set_show_sidebar(self.split.get_show_sidebar() or not live)
        self.inspector_btn.set_sensitive(True)
        self.title.set_title(path.name if not live else f"{proj.name if proj else ''} — training preview")
        self.title.set_subtitle(f"{_count(g.count)} splats · SH {g.sh_degree}" + (" · skybox" if g.skybox else "")
                                + (" · point cloud" if g.extra.get("pointCloud") else ""))
        e = edit
        e.crop.lo[:], e.crop.hi[:] = g.robust_bounds(0.01)
        self._sync_crop_rows()
        self._fill_cameras(cams)
        self.gl_row.set_subtitle(f"{self.area.renderer.gl_renderer or 'OpenGL'} · {self.area.renderer.gl_version}")
        self._stats_changed()
        if not live:
            self.ctl.recents.add(path)
            self.window.recents_changed()
        return False

    def _load_failed(self, path, msg) -> bool:
        self._busy(None)
        if self.area.cloud is None:
            self.go_home()
        alert(self.window, f"Couldn't open {path.name}", msg, (("ok", "OK", "default"),), lambda _r: None)
        return False

    def _fill_cameras(self, cams) -> None:
        for r in self._cam_rows:
            self.cams_expander.remove(r)
        self._cam_rows = []
        if not cams:
            self.cams_expander.set_subtitle("No dataset")
            return
        held = sum(1 for c in cams if c.get("holdout"))
        self.cams_expander.set_subtitle(f"{len(cams)} cameras" + (f", {held} held out" if held else ""))
        for c in cams[:400]:
            tags = []
            if c.get("holdout"):
                tags.append("holdout")
            if c.get("depth"):
                tags.append("depth")
            if c.get("mask"):
                tags.append("mask")
            r = plain_row(title=Path(c["image"]).name, subtitle=" · ".join(tags), activatable=True)
            r.set_use_markup(False)
            import math

            fov = math.degrees(2 * math.atan(0.5 * c["height"] / c["fy"])) if c.get("fy") else None
            r.connect("activated", lambda _r, m=c["cameraToWorld"], f=fov: (self.area.camera.look_through(
                np.asarray(m, float), f), self.area.camera_moved()))
            self.cams_expander.add_row(r)
            self._cam_rows.append(r)

    def show_live_preview(self, path: Path, project: Project) -> None:
        """Training preview from a running job: reload, keeping the camera."""
        if self.area.cloud is not None and self.path is not None and Project.find(self.path) is not None and \
                self.project is not None and self.project.root != project.root:
            return      # the user is looking at another project's scene
        self.load(path, keep_camera=self.area.cloud is not None, live=True)

    def _gl_error(self, msg: str) -> bool:
        self.error_page.set_description(
            f"{msg}\n\nThe viewer needs OpenGL 4.3. On the PC this comes from the NVIDIA driver's Flatpak extension "
            "(org.freedesktop.Platform.GL.nvidia-…): run “flatpak update” and restart the app.")
        self.stack.set_visible_child_name("error")
        return False

    # ------------------------------------------------------------------ stats / prefs
    def _stats_changed(self) -> None:
        e = self.area.edit
        if e is None:
            return
        vis = e.visible_count
        deleted = int(e.deleted.sum())
        sel = int(e.selected.sum())
        self.counts_row.set_subtitle(f"{_count(vis)} visible of {_count(e.cloud.count)}"
                                     + (f" · {_count(deleted)} deleted" if deleted else ""))
        self.sel_label.set_subtitle(f"{_count(sel)} selected" if sel else "Nothing selected")
        self.undo_btn.set_sensitive(e.can_undo)
        self.redo_btn.set_sensitive(e.can_redo)
        n = vis if self.exp_visible else e.cloud.count - deleted
        deg = 0 if self.exp_fmt == "splat" else min(self.exp_sh, e.cloud.sh_degree)
        self.size_row.set_subtitle(f"≈ {megabytes(estimate_size(n, self.exp_fmt, deg) / 1e6)}")

    def _hud_tick(self) -> bool:
        e = self.area.edit
        if e is not None and self.stack.get_visible_child_name() == "viewer":
            sel = int(e.selected.sum())
            mode = self.area.camera.mode
            self.hud.set_text(f"{self.area.fps:.0f} fps · {_count(e.visible_count)} / {_count(e.cloud.count)} visible"
                              + (f" · {_count(sel)} selected" if sel else "") + f" · {mode}"
                              + (" · brush: drag to select, Alt deselects, Delete removes"
                                 if self.area.tool == "select" else ""))
        return True

    def _apply_prefs(self) -> None:
        p = self.ctl.prefs
        a = self.area
        a.rotate_speed, a.pan_speed, a.zoom_speed, a.fly_speed = p.rotate_speed, p.pan_speed, p.zoom_speed, p.fly_speed
        a.invert_vertical = p.invert_vertical
        a.camera.fov_y = p.fov
        a.settings.background = tuple(p.background)
        a.settings.render_scale = p.render_scale
        a.show_cameras = p.show_cameras

    # ------------------------------------------------------------------ export
    def _export_cloud(self):
        e = self.area.edit
        g = e.result(only_visible=self.exp_visible)
        if self.exp_bake:
            q = rotation_between(self.area.camera.up, [0, -1, 0])
            g = g.transformed(q)
            g.up = np.array([0, -1, 0], np.float32)
        return g

    def export_dialog(self) -> None:
        if self.area.edit is None:
            return
        stem = self.path.name.split(".")[0] if self.path else "scene"
        d = Gtk.FileDialog(title="Export splat", initial_name=f"{stem}-edited.{self.exp_fmt}")
        d.save(self.window, None, self._export_chosen)

    def _export_chosen(self, dlg, res) -> None:
        try:
            f = dlg.save_finish(res)
        except GLib.Error:
            return
        if f is None or not f.get_path():
            return
        dest = Path(f.get_path())
        fmt = self.exp_fmt
        if formats.format_for_path(dest) != fmt:
            dest = dest.with_name(dest.name + "." + fmt) if not dest.name.endswith(fmt) else dest
        g = self._export_cloud()
        sh = self.exp_sh
        self._busy(f"Saving {dest.name}…")

        def work():
            try:
                formats.save(g, dest, fmt, sh_degree=sh)
                GLib.idle_add(lambda: (self._busy(None), self.ctl.toast(f"Saved {dest.name}"), False)[-1])
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                GLib.idle_add(lambda: (self._busy(None), self.ctl.toast(f"Export failed: {msg}"), False)[-1])
        threading.Thread(target=work, daemon=True).start()

    def screenshot_dialog(self) -> None:
        stem = self.path.stem if self.path else "scene"
        d = Gtk.FileDialog(title="Save screenshot", initial_name=f"{stem}.png")
        d.save(self.window, None, self._screenshot_chosen)

    def _screenshot_chosen(self, dlg, res) -> None:
        try:
            f = dlg.save_finish(res)
        except GLib.Error:
            return
        if f is None or not f.get_path():
            return
        img = self.area.screenshot(3840, 2160)
        if img is None:
            return
        from PIL import Image

        Image.fromarray(img[..., :3]).save(f.get_path())
        self.ctl.toast(f"Saved {Path(f.get_path()).name}")
