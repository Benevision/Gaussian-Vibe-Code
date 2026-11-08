"""Mask editor window (the Mac's ``MaskEditorView``): a project's images with their masks overlaid (red =
ignored in training); click an object to select it with SAM 2 (left click = object, right click or Alt-click =
background), paint out / paint back with a brush, track the clicked objects through the sequence, mask fixed parts
of the frame, Save. The model and file layout are in ``studio/mask_edit.py``; edits are the dataset's
``splatprep-masks/manual`` layer, so they survive "Prepare again"."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from gi.repository import Adw, Gdk, GLib, Gtk

from ....studio import static_regions
from ....studio.mask_edit import MaskEditModel, run_job, scratch_dir
from ..common import run_in_thread

DISPLAY_SIDE = 1400
THUMB_SIDE = 96


def _texture(rgb: np.ndarray) -> Gdk.Texture:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    fmt = Gdk.MemoryFormat.R8G8B8A8 if rgb.shape[2] == 4 else Gdk.MemoryFormat.R8G8B8
    return Gdk.MemoryTexture.new(w, h, fmt, GLib.Bytes.new(rgb.tobytes()), rgb.strides[0])


class MaskEditorWindow(Adw.Window):
    def __init__(self, parent, controller, project):
        super().__init__(transient_for=parent, modal=False, default_width=1280, default_height=820,
                         title=f"Masks — {project.name}")
        self.ctl = controller
        self.project = project
        self.m = MaskEditModel(project.dataset_dir)
        self.tool = "click"
        self.brush = 40.0
        self.show_overlay = True
        self.busy = False
        self._thumbs: "OrderedDict[int, Gdk.Texture]" = OrderedDict()
        self._img: Optional[Gdk.Texture] = None
        self._img_size: Tuple[int, int] = (1, 1)
        self._overlay: Optional[Gdk.Texture] = None
        self._overlay_dirty = True
        self._fit = (1.0, 0.0, 0.0)          # scale (display px per image px), x0, y0
        self._painting = False

        view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self.title = Adw.WindowTitle(title="Masks", subtitle=project.name)
        header.set_title_widget(self.title)
        tools = Gtk.Box(css_classes=["linked"])
        self.tool_btns = {}
        first = None
        for key, label, tip in (("click", "Select object", "Click an object: SAM 2 selects it (right / Alt-click "
                                 "marks background)"), ("paint", "Paint out", "Brush: ignore these pixels"),
                                ("erase", "Paint back", "Brush: keep these pixels whatever the models said")):
            b = Gtk.ToggleButton(label=label, tooltip_text=tip, group=first)
            first = first or b
            b.connect("toggled", lambda btn, k=key: btn.get_active() and self._set_tool(k))
            tools.append(b)
            self.tool_btns[key] = b
        header.pack_start(tools)
        adj = Gtk.Adjustment(lower=2, upper=300, step_increment=2, page_increment=20, value=self.brush)
        self.brush_spin = Gtk.SpinButton(adjustment=adj, tooltip_text="Brush radius (image pixels)")
        adj.connect("value-changed", lambda a: setattr(self, "brush", a.get_value()))
        header.pack_start(self.brush_spin)
        self.save_btn = Gtk.Button(label="Save masks", css_classes=["suggested-action"], sensitive=False)
        self.save_btn.connect("clicked", lambda *_: self.save())
        header.pack_end(self.save_btn)
        self.spinner = Gtk.Spinner()
        header.pack_end(self.spinner)
        undo = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Undo (Ctrl+Z)")
        undo.connect("clicked", lambda *_: self.undo())
        header.pack_end(undo)
        ov = Gtk.ToggleButton(icon_name="view-reveal-symbolic", active=True, tooltip_text="Show the mask overlay")
        ov.connect("toggled", lambda b: (setattr(self, "show_overlay", b.get_active()), self.area.queue_draw()))
        header.pack_end(ov)
        view.add_top_bar(header)

        bar = Gtk.Box(spacing=6, margin_start=8, margin_end=8, margin_top=4, margin_bottom=4)
        self.track_btn = Gtk.Button(label="Track through all frames", sensitive=False,
                                    tooltip_text="Follow every selected object through the sequence (SAM 2 video)")
        self.track_btn.connect("clicked", lambda *_: self.track())
        self.next_btn = Gtk.Button(label="Next object", sensitive=False)
        self.next_btn.connect("clicked", lambda *_: (self.m.next_object(), self._sync()))
        self.reset_btn = Gtk.Button(label="Reset clicks", sensitive=False)
        self.reset_btn.connect("clicked", lambda *_: (self.m.reset_clicks(), self._changed()))
        clear = Gtk.Button(label="Clear frame", tooltip_text="Forget this image's manual edits")
        clear.connect("clicked", lambda *_: (self.m.clear_frame(), self._changed()))
        static = Gtk.Button(label="Mask fixed parts of the frame",
                            tooltip_text="Find logos, lens rims, propellers … that never move and ignore them")
        static.connect("clicked", lambda *_: self.mask_static())
        for w in (self.next_btn, self.reset_btn, self.track_btn, clear, static):
            bar.append(w)
        view.add_top_bar(bar)

        paned = Gtk.Paned(position=230, shrink_start_child=False)
        self.store = Gtk.StringList.new([f.name for f in self.m.frames])
        self.selection = Gtk.SingleSelection(model=self.store)
        self.selection.connect("notify::selected", lambda s, _p: self._select(s.get_selected()))
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._row_setup)
        factory.connect("bind", self._row_bind)
        self.list = Gtk.ListView(model=self.selection, factory=factory)
        self.list.add_css_class("navigation-sidebar")
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, width_request=200)
        sw.set_child(self.list)
        paned.set_start_child(sw)

        overlay = Gtk.Overlay()
        self.area = Gtk.DrawingArea(hexpand=True, vexpand=True)
        self.area.set_draw_func(self._draw)
        overlay.set_child(self.area)
        nav = Gtk.Box(spacing=8, halign=Gtk.Align.CENTER, valign=Gtk.Align.END, margin_bottom=10,
                      css_classes=["osd", "toolbar"])
        prev = Gtk.Button(icon_name="go-previous-symbolic", tooltip_text="Previous image (←)")
        prev.connect("clicked", lambda *_: self.go(-1))
        nxt = Gtk.Button(icon_name="go-next-symbolic", tooltip_text="Next image (→)")
        nxt.connect("clicked", lambda *_: self.go(1))
        self.pos_label = Gtk.Label()
        nav.append(prev)
        nav.append(self.pos_label)
        nav.append(nxt)
        nav.append(Gtk.Label(label="red = ignored in training", css_classes=["dim-label"]))
        overlay.add_overlay(nav)
        paned.set_end_child(overlay)

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        body.append(paned)
        paned.set_vexpand(True)
        foot = Gtk.Box(spacing=12, margin_start=10, margin_end=10, margin_top=6, margin_bottom=6)
        self.status = Gtk.Label(xalign=0, hexpand=True, ellipsize=3, css_classes=["dim-label"])
        self.unsaved = Gtk.Label()
        foot.append(self.status)
        foot.append(self.unsaved)
        body.append(foot)
        view.set_content(body)
        self.set_content(view)

        click = Gtk.GestureClick(button=0)
        click.connect("released", self._clicked)
        self.area.add_controller(click)
        drag = Gtk.GestureDrag(button=Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self._drag_begin)
        drag.connect("drag-update", self._drag_update)
        drag.connect("drag-end", lambda *_: setattr(self, "_painting", False))
        self.area.add_controller(drag)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._key)
        self.add_controller(keys)

        if not len(self.m):
            self.status.set_text("This project has no images yet: prepare it first.")
        else:
            self._load_current()
        self.tool_btns["click"].set_active(True)
        self._set_tool("click")
        self._sync()

    # ------------------------------------------------------------------ list
    def _row_setup(self, _f, item) -> None:
        box = Gtk.Box(spacing=8, margin_top=2, margin_bottom=2)
        pic = Gtk.Picture(width_request=64, height_request=48, content_fit=Gtk.ContentFit.CONTAIN)
        col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER)
        name = Gtk.Label(xalign=0, ellipsize=3, css_classes=["caption"])
        tag = Gtk.Label(xalign=0, css_classes=["caption", "dim-label"])
        col.append(name)
        col.append(tag)
        box.append(pic)
        box.append(col)
        item.set_child(box)

    def _row_bind(self, _f, item) -> None:
        i = item.get_position()
        pic = item.get_child().get_first_child()
        name = pic.get_next_sibling().get_first_child()
        tag = name.get_next_sibling()
        name.set_text(self.m.frames[i].name)
        tag.set_text("edited" if i in self.m.dirty else ("masked" if self.m.has_mask(i) else ""))
        pic.set_paintable(self._thumb(i))

    def _thumb(self, i: int) -> Optional[Gdk.Texture]:
        if i in self._thumbs:
            self._thumbs.move_to_end(i)
            return self._thumbs[i]
        try:
            t = _texture(self.m.thumbnail(i, THUMB_SIDE))
        except Exception:  # noqa: BLE001 — a broken image just has no thumbnail
            return None
        self._thumbs[i] = t
        while len(self._thumbs) > 300:
            self._thumbs.popitem(last=False)
        return t

    # ------------------------------------------------------------------ frames
    def _select(self, i: int) -> None:
        if i == Gtk.INVALID_LIST_POSITION or i == self.m.index:
            return
        self.m.select(i)
        self._load_current()
        self._sync()

    def go(self, d: int) -> None:
        if len(self.m):
            self.selection.set_selected(max(0, min(len(self.m) - 1, self.m.index + d)))

    def _load_current(self) -> None:
        f = self.m.current
        try:
            rgb = self.m.image(self.m.index, DISPLAY_SIDE)
        except Exception as e:  # noqa: BLE001
            self.status.set_text(f"Can't open {f.name}: {e}")
            self._img = None
            self.area.queue_draw()
            return
        self._img_size = (rgb.shape[1], rgb.shape[0])
        self._img = _texture(rgb)
        del rgb
        self._overlay_dirty = True
        self.area.queue_draw()

    def _refresh_rows(self) -> None:
        n = len(self.m)
        if n:
            self.store.items_changed(0, n, n)        # re-bind the visible rows (edited / masked tags)

    def _changed(self) -> None:
        self._overlay_dirty = True
        self.area.queue_draw()
        self._sync()

    def _sync(self) -> None:
        n = len(self.m)
        self.pos_label.set_text(f"Image {self.m.index + 1} of {n}" if n else "No images")
        self.unsaved.set_text(f"{len(self.m.dirty)} unsaved" if self.m.dirty else "")
        self.save_btn.set_sensitive(bool(self.m.dirty) and not self.busy)
        clicks = bool(self.m.points)
        self.next_btn.set_sensitive(clicks and not self.busy)
        self.reset_btn.set_sensitive(clicks and not self.busy)
        self.track_btn.set_sensitive((clicks or bool(self.m.tracked)) and not self.busy)
        self.track_btn.set_label(f"Track {len(self.m.tracked) + int(clicks)} object(s) through all frames"
                                 if (clicks or self.m.tracked) else "Track through all frames")
        self.brush_spin.set_sensitive(self.tool != "click")
        if self.busy:
            self.spinner.start()
        else:
            self.spinner.stop()

    def _set_tool(self, k: str) -> None:
        self.tool = k
        self.area.set_cursor(Gdk.Cursor.new_from_name("crosshair" if k != "click" else "pointer"))
        if k == "click":
            self.status.set_text("Click an object to select it · right-click or Alt-click marks background")
        else:
            self.status.set_text("Drag to paint" + (" out (ignored in training)" if k == "paint" else
                                                    " back (kept in training)"))
        self._sync()

    # ------------------------------------------------------------------ drawing
    def _overlay_texture(self) -> Optional[Gdk.Texture]:
        if self._overlay is not None and (not self._overlay_dirty or self.busy):
            return self._overlay           # (a background save / tracking owns the model meanwhile)
        f = self.m.current
        w, h = self._img_size
        keep = self.m.keep()
        ys = np.minimum(f.height - 1, (np.arange(h) * f.height) // max(h, 1))
        xs = np.minimum(f.width - 1, (np.arange(w) * f.width) // max(w, 1))
        ign = ~keep[ys][:, xs]
        px = np.zeros((h, w, 4), np.uint8)
        px[ign] = (255, 40, 40, 140)
        self._overlay = _texture(px)
        self._overlay_dirty = False
        return self._overlay

    def _draw(self, area, cr, width, height) -> None:
        cr.set_source_rgb(0.08, 0.08, 0.09)
        cr.paint()
        if self._img is None or not len(self.m):
            return
        f = self.m.current
        s = min(width / f.width, height / f.height)
        x0, y0 = (width - f.width * s) / 2, (height - f.height * s) / 2
        self._fit = (s, x0, y0)
        snap = Gtk.Snapshot()
        rect = _rect(x0, y0, f.width * s, f.height * s)
        snap.append_texture(self._img, rect)
        if self.show_overlay:
            ov = self._overlay_texture()
            if ov is not None:
                snap.append_texture(ov, rect)
        node = snap.to_node()
        if node is not None:
            node.draw(cr)
        for x, y, lab in self.m.points:
            cx, cy = x0 + x * s, y0 + y * s
            cr.arc(cx, cy, 6, 0, 6.2832)
            cr.set_source_rgb(0.2, 0.8, 0.3) if lab else cr.set_source_rgb(0.9, 0.2, 0.2)
            cr.fill_preserve()
            cr.set_source_rgb(1, 1, 1)
            cr.set_line_width(2)
            cr.stroke()

    def _to_image(self, x: float, y: float) -> Tuple[float, float]:
        s, x0, y0 = self._fit
        return (x - x0) / s, (y - y0) / s

    # ------------------------------------------------------------------ input
    def _key(self, _c, keyval, _code, state) -> bool:
        ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        if ctrl and keyval in (Gdk.KEY_z, Gdk.KEY_Z):
            self.undo()
            return True
        if ctrl and keyval in (Gdk.KEY_s, Gdk.KEY_S):
            self.save()
            return True
        if keyval == Gdk.KEY_Left:
            self.go(-1)
            return True
        if keyval == Gdk.KEY_Right:
            self.go(1)
            return True
        return False

    def _clicked(self, gesture, n, x, y) -> None:
        if self.tool != "click" or self.busy or not len(self.m):
            return
        ix, iy = self._to_image(x, y)
        f = self.m.current
        if not (0 <= ix < f.width and 0 <= iy < f.height):
            return
        btn = gesture.get_current_button()
        alt = bool(gesture.get_current_event_state() & Gdk.ModifierType.ALT_MASK)
        self.m.add_point(ix, iy, positive=(btn == Gdk.BUTTON_PRIMARY and not alt))
        self.area.queue_draw()
        self.segment()

    def _drag_begin(self, gesture, x, y) -> None:
        if self.tool == "click" or not len(self.m):
            return
        self._drag_origin = (x, y)
        self._painting = True
        ix, iy = self._to_image(x, y)
        self.m.paint(ix, iy, self.brush, remove=self.tool == "paint", begin=True)
        self._changed()

    def _drag_update(self, gesture, dx, dy) -> None:
        if not self._painting:
            return
        ox, oy = self._drag_origin
        ix, iy = self._to_image(ox + dx, oy + dy)
        self.m.paint(ix, iy, self.brush, remove=self.tool == "paint", begin=False)
        self._overlay_dirty = True
        self.area.queue_draw()

    def undo(self) -> None:
        if self.m.undo():
            if self.selection.get_selected() != self.m.index:
                self.selection.set_selected(self.m.index)
                self._load_current()
            self._changed()

    # ------------------------------------------------------------------ worker jobs
    def _job(self, request_fn, title: str, done) -> None:
        """Run a processing request of the dataset on the local worker; ``done(dest)`` on the main loop."""
        self.busy = True
        self._sync()
        self.status.set_text(f"{title}: starting the worker…")
        ds = self.project.dataset_dir
        dest = scratch_dir(ds)
        request = request_fn(f"_editor_{dest.name}")

        def status(s):
            msg = s.get("message") or s.get("state") or ""
            GLib.idle_add(lambda: (self.status.set_text(f"{title}: {msg}"), False)[1])

        def then(client):
            try:
                out = run_job(client, ds, request, f"{self.project.name}: {title}", dest, status)
                GLib.idle_add(lambda: (self._job_done(), done(out), False)[2])
            except Exception as e:  # noqa: BLE001 — shown in the status line
                GLib.idle_add(lambda: (self._job_done(), self.status.set_text(f"{title} failed: {e}"), False)[2])

        def fail(msg):
            self._job_done()
            self.status.set_text(f"{title}: {msg}")
            return False

        self.ctl.with_server(then, fail)

    def _job_done(self) -> None:
        self.busy = False
        self._sync()

    def segment(self) -> None:
        index = self.m.index
        self._job(self.m.preview_request, "Selecting the object", lambda out: self._segmented(out, index))

    def _segmented(self, out: Path, index: int) -> None:
        try:
            if index == self.m.index and self.m.points:
                area = self.m.apply_object(out / "preview.png")
                self.status.set_text(f"Object selected ({area:.1%} of the image). Click more, Next object, "
                                     "or Track through all frames.")
                self._changed()
        finally:
            _rmtree(out)

    def track(self) -> None:
        req = self.m.track_request("x")
        if req is None:
            return
        n = len(self.m.tracked)
        self._job(lambda o: dict(req, outputDir=o), f"Tracking {n} object(s)", self._tracked)

    def _tracked(self, out: Path) -> None:
        def work():
            return self.m.apply_tracked(out / "masks")

        def done(n, err):
            _rmtree(out)
            self.status.set_text(f"Tracking failed: {err}" if err else
                                 f"Tracked: {n} images changed. Review them, then Save.")
            self._refresh_rows()
            self._load_current()
            self._changed()

        run_in_thread(work, done)

    def mask_static(self) -> None:
        self.busy = True
        self._sync()
        self.status.set_text("Looking for fixed parts of the frame…")
        ds = self.project.dataset_dir

        def work():
            self.m.save() if self.m.dirty else None
            reps = static_regions.apply(ds)
            self.m.restack_all()
            return reps

        def done(reps, err):
            self.busy = False
            if err:
                self.status.set_text(f"Fixed parts: {err}")
            else:
                hit = [r for r in reps if r["applied"]]
                self.status.set_text(f"Fixed parts masked for {len(hit)} of {len(reps)} cameras" +
                                     (": " + ", ".join(f"{r['camera']} {r['staticFraction']:.1%}" for r in hit[:4])
                                      if hit else " (nothing fixed found)"))
            self._refresh_rows(); self._changed()

        run_in_thread(work, done)

    def save(self) -> None:
        if not self.m.dirty or self.busy:
            return
        self.busy = True
        self._sync()

        def done(st, err):
            self.busy = False
            self.status.set_text(f"Couldn't save: {err}" if err else
                                 f"Saved {st['saved']} images. Train again to use the masks.")
            self._refresh_rows(); self._changed()

        run_in_thread(self.m.save, done)


def _rect(x, y, w, h):
    from gi.repository import Graphene

    r = Graphene.Rect()
    r.init(x, y, w, h)
    return r


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)
    try:
        p.parent.rmdir()
    except OSError:
        pass


def open_mask_editor(parent, controller, project) -> MaskEditorWindow:
    w = MaskEditorWindow(parent, controller, project)
    w.present()
    return w
