"""XDG Background portal (start at login, run without a window) via Gio D-Bus."""

from __future__ import annotations

from typing import Callable

from gi.repository import Gio, GLib

from ..autostart import (BACKGROUND_IFACE, PORTAL_BUS, PORTAL_PATH, REQUEST_IFACE, background_options,
                         handle_token, parse_background_response, request_path)

ResultCallback = Callable[[bool, bool, str], None]  # (background allowed, autostart, message)


def request_background(autostart: bool, callback: ResultCallback, parent_window: str = "") -> None:
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error as e:
        callback(False, False, f"No session bus: {e.message}")
        return
    token = handle_token()
    path = request_path(bus.get_unique_name(), token)
    sub = {"id": 0}

    def on_response(_conn, _sender, _path, _iface, _signal, params, *_user):
        if sub["id"]:
            bus.signal_unsubscribe(sub["id"])
            sub["id"] = 0
        code, results = params.unpack()
        callback(*parse_background_response(int(code), dict(results)))

    # Subscribe before calling, so a fast Response is not missed.
    sub["id"] = bus.signal_subscribe(PORTAL_BUS, REQUEST_IFACE, "Response", path, None,
                                     Gio.DBusSignalFlags.NONE, on_response)
    opts = {k: GLib.Variant(t, v) for k, (t, v) in background_options(autostart, token).items()}

    def on_called(conn, result):
        try:
            conn.call_finish(result)
        except GLib.Error as e:
            if sub["id"]:
                bus.signal_unsubscribe(sub["id"])
                sub["id"] = 0
            callback(False, False, f"The background portal is not available: {e.message}")

    bus.call(PORTAL_BUS, PORTAL_PATH, BACKGROUND_IFACE, "RequestBackground",
             GLib.Variant("(sa{sv})", (parent_window, opts)), GLib.VariantType.new("(o)"),
             Gio.DBusCallFlags.NONE, -1, None, on_called)
