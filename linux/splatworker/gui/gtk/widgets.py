"""Job rows, GPU rows, the QR dialog and the job log window."""

from __future__ import annotations

from typing import Callable, Optional

from gi.repository import Adw, Gdk, GLib, Gtk

from .. import formatting
from ..nvidia import GpuStats
from ..qr import qr_matrix, render_rgb
from .common import action_row, append_text, icon_button, label, run_in_thread, text_view

STATE_ICONS = {
    "queued": "content-loading-symbolic",
    "loading": "emblem-synchronizing-symbolic",
    "training": "emblem-synchronizing-symbolic",
    "completed": "emblem-ok-symbolic",
    "failed": "dialog-error-symbolic",
    "cancelled": "process-stop-symbolic",
    "paused": "media-playback-pause-symbolic",
}


class JobRow(Adw.ActionRow):
    def __init__(self, job_id: str, on_cancel: Callable[[str], None], on_delete: Callable[[str, str], None],
                 on_log: Callable[[str, str], None], on_export: Callable[[str, str, str], None],
                 on_prioritize: Optional[Callable[[str], None]] = None,
                 on_pause: Optional[Callable[[str, bool], None]] = None):
        super().__init__()
        self.set_use_markup(False)
        self.job_id = job_id
        self.view: Optional[formatting.JobView] = None
        self.icon = Gtk.Image(icon_name="content-loading-symbolic")
        self.add_prefix(self.icon)
        self.progress = Gtk.ProgressBar(valign=Gtk.Align.CENTER, show_text=True)
        self.progress.add_css_class("job-progress")
        self.add_suffix(self.progress)
        self.export_btn = icon_button("document-save-symbolic", "Save the result to your computer",
                                      lambda *_: on_export(self.job_id, self.view.export, self.view.title))
        self.cancel_btn = icon_button("process-stop-symbolic", "Cancel this job",
                                      lambda *_: on_cancel(self.job_id))
        self.log_btn = icon_button("text-x-generic-symbolic", "Show the job's log",
                                   lambda *_: on_log(self.job_id, self.view.title))
        self.delete_btn = icon_button("user-trash-symbolic", "Delete the job and its files",
                                      lambda *_: on_delete(self.job_id, self.view.title))
        self.next_btn = icon_button("go-top-symbolic", "Run next: move to the front of the queue",
                                    lambda *_: on_prioritize(self.job_id) if on_prioritize else None)
        # pause (False) / resume (True): a queued job is held back, a running training job saves its state and
        # frees the GPU (worker capability "pause")
        self.pause_btn = icon_button("media-playback-pause-symbolic", "Pause: frees the GPU until you resume",
                                     lambda *_: on_pause(self.job_id, False) if on_pause else None)
        self.resume_btn = icon_button("media-playback-start-symbolic", "Resume: back into the queue",
                                      lambda *_: on_pause(self.job_id, True) if on_pause else None)
        self.pause_btn.set_visible(False)
        self.resume_btn.set_visible(False)
        self._can_pause = on_pause is not None
        for b in (self.next_btn, self.pause_btn, self.resume_btn, self.export_btn, self.cancel_btn, self.log_btn,
                  self.delete_btn):
            self.add_suffix(b)

    def update(self, status: dict) -> None:
        v = formatting.job_view(status)
        if self.view == v:
            return
        self.view = v
        self.set_title(v.title)
        origin = status.get("origin")
        src = "" if not origin else (" · from this PC (Studio)" if origin == "local" else f" · from {origin}")
        msg = status.get("message") or ""
        paused = f" · {msg}" if v.state == "paused" and msg else ""
        self.set_subtitle(v.detail + paused + src)
        self.next_btn.set_visible(v.state == "queued")
        self.icon.set_from_icon_name(STATE_ICONS.get(v.state, "content-loading-symbolic"))
        self.progress.set_fraction(v.fraction)
        self.progress.set_text(f"{v.fraction * 100:.0f} %" if v.state != "queued" else "waiting")
        self.progress.set_visible(v.state in ("queued", "loading", "training", "paused"))
        if v.state == "paused":
            self.progress.set_text("paused")
        msg = status.get("message") or ""
        own = origin in (None, "local")      # other computers' jobs: only their own app pauses them
        self.pause_btn.set_visible(self._can_pause and own and (v.state == "queued" or (
            v.kind == "train" and v.state in ("loading", "training") and msg not in ("pausing", "cancelling"))))
        self.resume_btn.set_visible(self._can_pause and v.state == "paused")
        self.cancel_btn.set_visible(v.can_cancel or v.state == "paused")
        self.export_btn.set_visible(v.export is not None)
        if v.is_error:
            self.add_css_class("error")
        else:
            self.remove_css_class("error")


class GpuRow(Adw.ActionRow):
    """Live card for one GPU: VRAM bar, utilisation bar, temperature and power."""

    def __init__(self):
        super().__init__()
        self.set_use_markup(False)
        self.add_prefix(Gtk.Image(icon_name="video-display-symbolic"))
        bars = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, valign=Gtk.Align.CENTER,
                       width_request=170)
        self.vram = Gtk.LevelBar(min_value=0, max_value=1)
        self.vram.set_tooltip_text("Video memory in use")
        self.util = Gtk.LevelBar(min_value=0, max_value=1)
        self.util.set_tooltip_text("GPU utilisation")
        for bar in (self.vram, self.util):
            bar.remove_offset_value(Gtk.LEVEL_BAR_OFFSET_LOW)
            bar.remove_offset_value(Gtk.LEVEL_BAR_OFFSET_HIGH)
            bar.remove_offset_value(Gtk.LEVEL_BAR_OFFSET_FULL)
        bars.append(self.vram)
        bars.append(self.util)
        self.add_suffix(bars)

    def update(self, s: GpuStats) -> None:
        self.set_title(f"GPU {s.index}: {s.name}")
        parts = [f"{formatting.megabytes(s.memory_used_mb)} of {formatting.megabytes(s.memory_total_mb)} "
                 f"memory used ({formatting.megabytes(s.memory_free_mb)} free)"]
        if s.utilization_pct is not None:
            parts.append(f"{s.utilization_pct} % busy")
        if s.temperature_c is not None:
            parts.append(f"{s.temperature_c} °C")
        if s.power_w is not None:
            parts.append(f"{s.power_w:.0f} W")
        self.set_subtitle(" · ".join(parts))
        self.vram.set_value(s.memory_fraction)
        self.util.set_value((s.utilization_pct or 0) / 100.0)


def qr_available() -> bool:
    return qr_matrix("x") is not None


def show_qr(parent: Gtk.Window, payload: str, caption: str) -> None:
    matrix = qr_matrix(payload)
    win = Adw.Window(transient_for=parent, modal=True, title="Connection QR Code",
                     default_width=380, default_height=480)
    tv = Adw.ToolbarView()
    tv.add_top_bar(Adw.HeaderBar())
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_top=12, margin_bottom=18,
                  margin_start=18, margin_end=18)
    if matrix is None:
        box.append(label("QR codes need the \"qrcode\" Python package.", xalign=0.5))
    else:
        data, w, h = render_rgb(matrix, scale=8, border=4)
        texture = Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8, GLib.Bytes.new(data), w * 3)
        pic = Gtk.Picture.new_for_paintable(texture)
        pic.set_size_request(280, 280)
        pic.set_can_shrink(True)
        box.append(pic)
    box.append(label(caption, ("dim-label", "caption"), xalign=0.5))
    tv.set_content(box)
    win.set_content(tv)
    win.present()


def show_job_log(parent: Gtk.Window, client, job_id: str, title: str) -> None:
    win = Adw.Window(transient_for=parent, title=f"Log: {title}", default_width=820, default_height=560)
    tv = Adw.ToolbarView()
    header = Adw.HeaderBar()
    tv.add_top_bar(header)
    sw, view = text_view()
    tv.set_content(sw)
    win.set_content(tv)

    def load(*_):
        def done(text, error):
            view.get_buffer().set_text("")
            append_text(view, [f"Could not load the log: {error}"] if error else (text or "(empty)").splitlines(),
                        max_lines=20000)
        run_in_thread(lambda: client.log(job_id), done)

    header.pack_start(icon_button("view-refresh-symbolic", "Reload", load))
    load()
    win.present()


def empty_clients_row() -> Adw.ActionRow:
    return action_row("No Mac has connected yet",
                      "Requests from SplatWork will show up here.", icon="network-idle-symbolic")
