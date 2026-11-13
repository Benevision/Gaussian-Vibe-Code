"""Compute Devices: other workers on the network (SplatWork Nodes, Macs with "Act as a worker"), found by
Bonjour or added by address, with their health; and the
"Train on" row of the New Project dialog. Logic lives in ``studio/devices.py`` and ``studio/compute.py``."""
from __future__ import annotations

import threading
from typing import Optional

from gi.repository import Adw, GLib, Gtk

from ....studio.devices import Device, parse_address
from .pages import _page
from .rows import combo_row, group, plain_row

STATUS_ICON = {"online": "emblem-ok-symbolic", "unauthorized": "dialog-password-symbolic",
               "offline": "network-offline-symbolic", "checking": "content-loading-symbolic",
               "unknown": "content-loading-symbolic"}
KIND_ICON = {"mac": "computer-symbolic", "iphone": "phone-symbolic", "linux": "network-server-symbolic"}


class DevicesPage(Gtk.Box):
    def __init__(self, window, controller):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.window = window
        self.ctl = controller
        self.reg = controller.devices
        self._pending = False
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=18, margin_bottom=18,
                      margin_start=18, margin_end=18)
        intro = group("How it works",
                      "Frames, masks and camera solving always run on this PC. Training can go to one of these "
                      "devices, or to all of them at once: big scenes are split into spatial blocks shared "
                      "out by speed and memory (a block only goes to a device it fits), trained in parallel and "
                      "merged. Choose it per project in New Project → Training → Train on.")
        box.append(intro)
        self.list_group = group("Workers on the network",
                                "SplatWork Nodes and Macs with “Act as a worker” appear here by themselves "
                                "(Bonjour _splatworker._tcp).")
        self._rows = []
        box.append(self.list_group)
        self.hidden_row = plain_row("Hidden workers", "")
        b = Gtk.Button(label="Show again", valign=Gtk.Align.CENTER)
        b.connect("clicked", lambda *_: self.reg.unhide_all())
        self.hidden_row.add_suffix(b)
        self.hidden_group = group()
        self.hidden_group.add(self.hidden_row)
        box.append(self.hidden_group)
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        clamp = Adw.Clamp(maximum_size=820)
        clamp.set_child(box)
        sw.set_child(clamp)
        add = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="Add worker…")
        add.connect("clicked", lambda *_: self.add_dialog())
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Check all workers now")
        refresh.connect("clicked", lambda *_: threading.Thread(target=self.reg.refresh, daemon=True).start())
        self.append(_page("Compute Devices", sw, "Other machines that can train", header_extra=[add, refresh]))
        self.reg.listeners.append(self._changed)
        self.refresh()
        GLib.timeout_add(1000, lambda: (self.ctl.start_devices(), False)[1])

    # ------------------------------------------------------------------ list
    def _changed(self) -> None:          # registry thread
        if not self._pending:
            self._pending = True
            GLib.timeout_add(250, self._flush)

    def _flush(self) -> bool:
        self._pending = False
        self.refresh()
        return False

    def online_count(self) -> int:
        return len(self.reg.online())

    def refresh(self) -> None:
        for r in self._rows:
            self.list_group.remove(r)
        self._rows = []
        devices = self.reg.devices()
        if not devices:
            hint = "No other workers found yet. Turn on “Act as a worker” in SplatWork on the Mac, open the " \
                   "Compute screen in SplatWork Capture, or add a worker by its address (+)."
            if self.reg.browse_error:
                hint += f" Bonjour isn't available here ({self.reg.browse_error})."
            r = plain_row("Nothing here yet", hint)
            r.set_subtitle_lines(0)
            self._rows.append(r)
            self.list_group.add(r)
        for d in devices:
            r = self._row(d)
            self._rows.append(r)
            self.list_group.add(r)
        n = len(self.reg.store.hidden)
        self.hidden_group.set_visible(n > 0)
        self.hidden_row.set_subtitle(f"{n} discovered worker{'s' if n != 1 else ''} removed from the list")

    def _row(self, d: Device) -> Adw.ActionRow:
        r = plain_row(d.name, d.summary())
        r.set_subtitle_lines(2)
        r.add_prefix(Gtk.Image(icon_name=KIND_ICON.get(d.kind, "network-server-symbolic")))
        st = Gtk.Image(icon_name=STATUS_ICON.get(d.status, "content-loading-symbolic"))
        st.set_tooltip_text(d.status + (f": {d.message}" if d.message else ""))
        r.add_suffix(st)
        if d.status == "unauthorized" or d.txt.get("auth") == "1" or d.token:
            kb = Gtk.Button(icon_name="dialog-password-symbolic", valign=Gtk.Align.CENTER,
                            tooltip_text="Access token…")
            kb.add_css_class("flat")
            kb.connect("clicked", lambda *_: self.token_dialog(d))
            r.add_suffix(kb)
        rm = Gtk.Button(icon_name="user-trash-symbolic" if d.manual else "view-conceal-symbolic",
                        valign=Gtk.Align.CENTER,
                        tooltip_text="Remove this worker" if d.manual else "Hide from the list (add it by address instead)")
        rm.add_css_class("flat")
        rm.connect("clicked", lambda *_: self.reg.remove(d.id))
        r.add_suffix(rm)
        r.device = d
        return r

    # ------------------------------------------------------------------ dialogs
    def add_dialog(self) -> None:
        dlg = Adw.AlertDialog(heading="Add worker",
                              body="The address of a SplatWork Node (port 7878) or a Mac with “Act as a worker” "
                                   "(port 7879). The access token is only needed if that worker has one (SplatWork "
                                   "Node → Preferences → Access token; on a Mac, Compute → Act as a worker).")
        lb = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        lb.add_css_class("boxed-list")
        addr = Adw.EntryRow(title="Address (host or host:port)")
        name = Adw.EntryRow(title="Name (optional)")
        tok = Adw.PasswordEntryRow(title="Access token (optional)")
        for w in (addr, name, tok):
            lb.append(w)
        err = Gtk.Label(xalign=0, wrap=True, visible=False)
        err.add_css_class("error")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.append(lb)
        box.append(err)
        dlg.set_extra_child(box)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("add", "Add")
        dlg.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("add")
        dlg.set_close_response("cancel")
        dlg.set_response_enabled("add", False)

        def changed(*_):
            try:
                parse_address(addr.get_text())
                err.set_visible(False)
                dlg.set_response_enabled("add", True)
            except ValueError as e:
                err.set_text(str(e))
                err.set_visible(bool(addr.get_text().strip()))
                dlg.set_response_enabled("add", False)
        addr.connect("changed", changed)

        def response(_d, rid):
            if rid != "add":
                return
            host, port = parse_address(addr.get_text())
            nm = name.get_text().strip() or None
            token = tok.get_text().strip() or None
            threading.Thread(target=self.reg.add_manual, args=(host, port, nm, token), daemon=True).start()
        dlg.connect("response", response)
        dlg.present(self.window)
        self.add_dialog_widget = dlg          # for the smoke test

    def token_dialog(self, d: Device) -> None:
        dlg = Adw.AlertDialog(heading=f"Access token for {d.name}",
                              body="The token set on that worker (empty = none).")
        tok = Adw.PasswordEntryRow(title="Access token")
        tok.set_text(d.token or "")
        lb = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        lb.add_css_class("boxed-list")
        lb.append(tok)
        dlg.set_extra_child(lb)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("save", "Save")
        dlg.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        dlg.connect("response", lambda _d, rid: rid == "save" and threading.Thread(
            target=self.reg.set_token, args=(d.id, tok.get_text().strip() or None), daemon=True).start())
        dlg.present(self.window)


# ============================================================================ New Project → Train on

def train_on_row(controller, options) -> Adw.ComboRow:
    """"Train on": This PC (default), one of the known devices, or all of them block-parallel."""
    reg = controller.devices
    choices = [("local", "This PC"), ("all", "All devices (block-parallel for big scenes)")]
    for d in reg.devices():
        if d.status in ("online", "unknown", "checking") or d.manual:
            state = "" if d.online else {"offline": " (offline)", "unauthorized": " (needs its token)"}.get(
                d.status, " (not checked yet)")
            choices.append((d.id, f"{d.name}{state}"))
    if options.train_on not in [k for k, _ in choices]:
        options.train_on = "local"
    row = combo_row("Train on", choices, options.train_on, lambda v: setattr(options, "train_on", v),
                    "net.placement",
                    subtitle="Preparation runs here; training can go to another worker or to all of them "
                             "(Compute Devices in the sidebar).")
    return row
