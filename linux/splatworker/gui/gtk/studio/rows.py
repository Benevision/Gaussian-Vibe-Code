"""Preference rows wired to the settings catalog: every control shows its one-line summary as the subtitle and
the full documentation as a tooltip (the Mac's ⓘ popover)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from gi.repository import Adw, Gtk

from ....studio import catalog


def _doc(row, setting_id: Optional[str], subtitle: bool = True) -> None:
    if not setting_id:
        return
    d = catalog.doc(setting_id)
    if d is None:
        return
    if subtitle and not row.get_subtitle():
        row.set_subtitle(d["summary"])
    row.set_tooltip_text(catalog.tooltip(setting_id))


def spin_row(title: str, lo: float, hi: float, step: float, value: float, on_change: Callable[[float], None],
             digits: int = 0, setting_id: Optional[str] = None, subtitle: str = "") -> Adw.SpinRow:
    adj = Gtk.Adjustment(lower=lo, upper=hi, step_increment=step, page_increment=step * 10, value=value)
    row = Adw.SpinRow(adjustment=adj, digits=digits)
    row.set_use_markup(False)
    row.set_title(title)
    row.set_subtitle(subtitle)
    _doc(row, setting_id)
    adj.connect("value-changed", lambda a: on_change(a.get_value()))
    return row


def switch_row(title: str, value: bool, on_change: Callable[[bool], None], setting_id: Optional[str] = None,
               subtitle: str = "") -> Adw.SwitchRow:
    row = Adw.SwitchRow(active=value)
    row.set_use_markup(False)
    row.set_title(title)
    row.set_subtitle(subtitle)
    _doc(row, setting_id)
    row.connect("notify::active", lambda r, _p: on_change(r.get_active()))
    return row


def combo_row(title: str, choices: Sequence[Tuple[str, str]], value: str, on_change: Callable[[str], None],
              setting_id: Optional[str] = None, subtitle: str = "") -> Adw.ComboRow:
    keys = [k for k, _ in choices]
    model = Gtk.StringList.new([t for _, t in choices])
    row = Adw.ComboRow(model=model)
    row.set_use_markup(False)
    row.set_title(title)
    row.set_subtitle(subtitle)
    if value in keys:
        row.set_selected(keys.index(value))
    _doc(row, setting_id)
    row.connect("notify::selected", lambda r, _p: on_change(keys[r.get_selected()])
                if 0 <= r.get_selected() < len(keys) else None)
    row.keys = keys
    return row


def set_combo(row: Adw.ComboRow, value: str) -> None:
    if value in row.keys:
        row.set_selected(row.keys.index(value))


def button_row(title: str, label: str, on_click: Callable[[], None], subtitle: str = "",
               setting_id: Optional[str] = None, destructive: bool = False) -> Adw.ActionRow:
    row = plain_row(title, subtitle)
    b = Gtk.Button(label=label, valign=Gtk.Align.CENTER)
    if destructive:
        b.add_css_class("destructive-action")
    b.connect("clicked", lambda *_: on_click())
    row.add_suffix(b)
    row.button = b
    _doc(row, setting_id)
    return row


def group(title: str = "", description: str = "") -> Adw.PreferencesGroup:
    return Adw.PreferencesGroup(title=title, description=description)


def scrolled(*children: Gtk.Widget, margin: int = 12) -> Gtk.ScrolledWindow:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=margin, margin_bottom=margin,
                  margin_start=margin, margin_end=margin)
    for c in children:
        box.append(c)
    sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
    sw.set_child(box)
    sw.box = box
    return sw


def plain_row(title: str = "", subtitle: str = "", **kw) -> Adw.ActionRow:
    """ActionRow whose title / subtitle are plain text (never parsed as Pango markup)."""
    row = Adw.ActionRow(**kw)
    row.set_use_markup(False)
    row.set_title(title)
    row.set_subtitle(subtitle)
    return row


def plain_expander(title: str = "", subtitle: str = "", **kw) -> Adw.ExpanderRow:
    row = Adw.ExpanderRow(**kw)
    row.set_use_markup(False)
    row.set_title(title)
    row.set_subtitle(subtitle)
    return row
