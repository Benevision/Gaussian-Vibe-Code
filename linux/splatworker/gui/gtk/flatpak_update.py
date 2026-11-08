"""Self-update inside the Flatpak through the Flatpak portal (Gio D-Bus, no extra permissions).

* ``CreateUpdateMonitor`` → ``UpdateAvailable`` when the remote (the repo GitHub Pages publishes on every push to
  main) has a newer commit, or when someone already ran ``flatpak update`` (then only a restart is missing);
* ``Update`` downloads and deploys it, reporting ``Progress``;
* ``Spawn`` with ``LATEST_VERSION`` starts the new version (``splatworker-app --after-update``, which waits for this
  instance to exit), then this one quits.

The decisions (when to update, what the banner says) are in :mod:`splatworker.gui.updates`.
"""

from __future__ import annotations

import os
import secrets
from typing import Callable, Optional

from gi.repository import Gio, GLib

from .. import updates as U

BUS = "org.freedesktop.portal.Flatpak"
PATH = "/org/freedesktop/portal/Flatpak"
IFACE = "org.freedesktop.portal.Flatpak"
MONITOR_IFACE = "org.freedesktop.portal.Flatpak.UpdateMonitor"


def _bytestring(s: str) -> bytes:
    return s.encode("utf-8") + b"\0"


class SelfUpdater:
    """State machine for one app instance. ``on_change()`` is called on the main loop after every change."""

    def __init__(self, on_change: Callable[[], None]):
        self.on_change = on_change
        self.state = U.IDLE
        self.version = ""          # from version.json, when known
        self.fraction = 0.0
        self.error = ""
        self.bus: Optional[Gio.DBusConnection] = None
        self.monitor: Optional[str] = None
        self._subs = []
        self._nothing_for: Optional[str] = None   # version.json said newer but the portal found nothing yet

    @property
    def supported(self) -> bool:
        return self.monitor is not None

    # ---------------------------------------------------------------- setup
    def start(self) -> bool:
        try:
            self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            token = "splatwork" + secrets.token_hex(6)
            res = self.bus.call_sync(BUS, PATH, IFACE, "CreateUpdateMonitor",
                                     GLib.Variant("(a{sv})", ({"handle_token": GLib.Variant("s", token)},)),
                                     GLib.VariantType.new("(o)"), Gio.DBusCallFlags.NONE, 5000, None)
        except GLib.Error:
            self.monitor = None      # not a Flatpak, or a portal without update support
            return False
        self.monitor = res.unpack()[0]
        for signal, cb in (("UpdateAvailable", self._on_available), ("Progress", self._on_progress)):
            self._subs.append(self.bus.signal_subscribe(BUS, MONITOR_IFACE, signal, self.monitor, None,
                                                        Gio.DBusSignalFlags.NONE, cb))
        return True

    def close(self) -> None:
        if self.bus is None:
            return
        for s in self._subs:
            self.bus.signal_unsubscribe(s)
        self._subs = []
        if self.monitor:
            try:
                self.bus.call_sync(BUS, self.monitor, MONITOR_IFACE, "Close", None, None,
                                   Gio.DBusCallFlags.NONE, 2000, None)
            except GLib.Error:
                pass

    # ---------------------------------------------------------------- inputs
    def published(self, info: Optional[dict]) -> None:
        """Result of the version.json check (None = nothing newer)."""
        if not info:
            return
        self.version = str(info.get("version") or self.version)
        if self.state in (U.IDLE, U.FAILED) and self._nothing_for != self.version:
            self.state, self.error = U.AVAILABLE, ""
        self.on_change()

    def _on_available(self, _c, _s, _p, _i, _sig, params, *_u) -> None:
        state = U.state_from_monitor(dict(params.unpack()[0]))
        if state and self.state not in (U.DOWNLOADING, U.INSTALLED):
            self.state, self.error = state, ""
            self.on_change()

    def _on_progress(self, _c, _s, _p, _i, _sig, params, *_u) -> None:
        state, frac, err = U.parse_progress(dict(params.unpack()[0]))
        if state == U.IDLE:              # portal: nothing to update (the CDN may not have the new commit yet)
            self._nothing_for = self.version
        self.state, self.fraction, self.error = state, frac, err
        self.on_change()

    # ---------------------------------------------------------------- actions
    def update(self) -> None:
        if not self.supported or self.state == U.DOWNLOADING:
            return
        self.state, self.fraction = U.DOWNLOADING, 0.0
        self.on_change()

        def done(conn, result):
            try:
                conn.call_finish(result)
            except GLib.Error as e:
                self.state, self.error = U.FAILED, e.message
                self.on_change()
        self.bus.call(BUS, self.monitor, MONITOR_IFACE, "Update", GLib.Variant("(sa{sv})", ("", {})),
                      None, Gio.DBusCallFlags.NONE, -1, None, done)

    def restart(self, background: bool, quit_app: Callable[[], None]) -> bool:
        """Starts the new version (it waits for this instance to go), then calls ``quit_app``."""
        argv = U.restart_argv(background)
        try:
            self.bus.call_sync(
                BUS, PATH, IFACE, "Spawn",
                GLib.Variant("(ayaaya{uh}a{ss}ua{sv})",
                             (_bytestring(os.path.expanduser("~")), [_bytestring(a) for a in argv], {}, {},
                              U.SPAWN_LATEST_VERSION, {})),
                GLib.VariantType.new("(u)"), Gio.DBusCallFlags.NONE, 10000, None)
        except GLib.Error as e:
            self.state, self.error = U.FAILED, f"could not start the new version ({e.message})"
            self.on_change()
            return False
        quit_app()
        return True


def wait_for_previous_instance(app_id: str, timeout_s: float = 120.0) -> None:
    """``--after-update``: the old instance owns the app id on the session bus until it has fully shut down
    (server stopped). Starting before then would just activate it."""
    import time

    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error:
        time.sleep(5)
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            owned = bus.call_sync("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
                                  "NameHasOwner", GLib.Variant("(s)", (app_id,)), GLib.VariantType.new("(b)"),
                                  Gio.DBusCallFlags.NONE, 2000, None).unpack()[0]
        except GLib.Error:
            owned = False
        if not owned:
            return
        time.sleep(0.5)
