"""Small helpers shared by the windows (rows, dialogs, clipboard, threads)."""

from __future__ import annotations

import threading
from typing import Callable, Iterable, Optional, Sequence, Tuple

from gi.repository import Adw, Gdk, GLib, Gtk, Pango

CSS = """
.hero-title { font-size: 1.6em; font-weight: 800; }
switch.big-switch { border-radius: 9999px; }
switch.big-switch > slider { min-width: 34px; min-height: 34px; border-radius: 9999px; }
.status-dot { min-width: 12px; min-height: 12px; border-radius: 9999px; }
.status-running { background-color: @success_color; }
.status-starting { background-color: @warning_color; }
.status-stopped { background-color: alpha(@window_fg_color, 0.3); }
.status-failed { background-color: @error_color; }
.job-progress { min-width: 150px; }
textview.log-view { font-family: monospace; font-size: 0.9em; }
"""


def load_css() -> None:
    provider = Gtk.CssProvider()
    if hasattr(provider, "load_from_string"):  # GTK 4.12
        provider.load_from_string(CSS)
    else:  # pragma: no cover
        provider.load_from_data(CSS.encode(), -1)
    display = Gdk.Display.get_default()
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


def esc(text: Optional[str]) -> str:
    return GLib.markup_escape_text(text or "")


def action_row(title: str, subtitle: str = "", icon: Optional[str] = None,
               selectable: bool = False) -> Adw.ActionRow:
    row = Adw.ActionRow(title=title, subtitle=subtitle)
    row.set_use_markup(False)
    if selectable:
        row.set_subtitle_selectable(True)
    if icon:
        row.add_prefix(Gtk.Image(icon_name=icon))
    return row


def icon_button(icon: str, tooltip: str, callback: Callable[[Gtk.Button], None],
                classes: Sequence[str] = ("flat",)) -> Gtk.Button:
    b = Gtk.Button(icon_name=icon, tooltip_text=tooltip, valign=Gtk.Align.CENTER)
    for c in classes:
        b.add_css_class(c)
    b.connect("clicked", callback)
    return b


def label(text: str = "", classes: Iterable[str] = (), wrap: bool = True, xalign: float = 0.0,
          selectable: bool = False) -> Gtk.Label:
    lab = Gtk.Label(label=text, wrap=wrap, xalign=xalign, selectable=selectable)
    lab.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
    for c in classes:
        lab.add_css_class(c)
    return lab


def copy_text(widget: Gtk.Widget, text: str) -> None:
    clipboard = widget.get_clipboard()
    try:
        clipboard.set(text)
    except (TypeError, AttributeError):  # older PyGObject without the override
        clipboard.set_content(Gdk.ContentProvider.new_for_value(text))


def toast(overlay: Adw.ToastOverlay, title: str, button: Optional[str] = None,
          on_button: Optional[Callable[[], None]] = None, timeout: int = 4) -> None:
    t = Adw.Toast(title=title, timeout=timeout)
    if button and on_button:
        t.set_button_label(button)
        t.connect("button-clicked", lambda *_: on_button())
    overlay.add_toast(t)


def alert(parent: Gtk.Widget, heading: str, body: str,
          responses: Sequence[Tuple[str, str, str]] = (("ok", "OK", "default"),),
          callback: Optional[Callable[[str], None]] = None, default: Optional[str] = None,
          close: Optional[str] = None) -> None:
    """responses: (id, label, appearance) with appearance default|suggested|destructive."""
    appearance = {"default": Adw.ResponseAppearance.DEFAULT,
                  "suggested": Adw.ResponseAppearance.SUGGESTED,
                  "destructive": Adw.ResponseAppearance.DESTRUCTIVE}
    if hasattr(Adw, "AlertDialog"):  # libadwaita 1.5
        dlg = Adw.AlertDialog(heading=heading, body=body)
    else:  # pragma: no cover - libadwaita 1.4
        dlg = Adw.MessageDialog(transient_for=parent.get_root() if parent else None, heading=heading, body=body)
    for rid, text, kind in responses:
        dlg.add_response(rid, text)
        dlg.set_response_appearance(rid, appearance[kind])
    if default:
        dlg.set_default_response(default)
    dlg.set_close_response(close or responses[0][0])
    if callback:
        dlg.connect("response", lambda _d, rid: callback(rid))
    if hasattr(Adw, "AlertDialog") and isinstance(dlg, Adw.AlertDialog):
        dlg.present(parent)
    else:  # pragma: no cover
        dlg.present()


def run_in_thread(work: Callable[[], object], done: Optional[Callable[[object, Optional[BaseException]], None]] = None) -> None:
    """Run ``work`` off the main loop; call ``done(result, error)`` back on it."""
    def target():
        try:
            result, error = work(), None
        except BaseException as e:  # noqa: BLE001 - reported to the UI
            result, error = None, e
        if done is not None:
            GLib.idle_add(lambda: (done(result, error), False)[1])
    threading.Thread(target=target, daemon=True).start()


def scrolled_page(child: Gtk.Widget, max_width: int = 760) -> Gtk.ScrolledWindow:
    clamp = Adw.Clamp(maximum_size=max_width, child=child)
    clamp.set_margin_top(18)
    clamp.set_margin_bottom(24)
    clamp.set_margin_start(12)
    clamp.set_margin_end(12)
    sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
    sw.set_child(clamp)
    return sw


def text_view(monospace: bool = True) -> Tuple[Gtk.ScrolledWindow, Gtk.TextView]:
    tv = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR,
                      monospace=monospace)
    tv.add_css_class("log-view")
    tv.set_top_margin(8)
    tv.set_bottom_margin(8)
    tv.set_left_margin(10)
    tv.set_right_margin(10)
    sw = Gtk.ScrolledWindow(vexpand=True, hexpand=True, child=tv)
    return sw, tv


def append_text(tv: Gtk.TextView, lines: Sequence[str], max_lines: int = 5000, autoscroll: bool = True) -> None:
    if not lines:
        return
    buf = tv.get_buffer()
    buf.insert(buf.get_end_iter(), "\n".join(lines) + "\n")
    extra = buf.get_line_count() - max_lines
    if extra > 0:
        res = buf.get_iter_at_line(extra)  # GTK 4: (found, iter)
        it = res[1] if isinstance(res, tuple) else res
        buf.delete(buf.get_start_iter(), it)
    if autoscroll:
        mark = buf.get_mark("splatworker-end") or buf.create_mark("splatworker-end", buf.get_end_iter(), False)
        buf.move_mark(mark, buf.get_end_iter())
        tv.scroll_to_mark(mark, 0.0, False, 0.0, 1.0)
