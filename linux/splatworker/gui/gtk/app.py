"""Adw.Application: actions, first-run flow, background mode, shutdown."""

from __future__ import annotations

import sys

from gi.repository import Adw, Gio, GLib, Gtk

from splatworker import __version__

from .. import APP_ID, APP_NAME
from ..app_state import AppState
from ..autostart import set_autostart_file
from ..paths import in_flatpak
from .common import alert, load_css, toast
from .portal import request_background
from .preferences import Preferences
from .setup_assistant import SetupAssistant
from .window import MainWindow

RESTART_SETTINGS = {"port", "name", "mdns", "require_token", "advertise_ip", "gpus", "max_upload_gb",
                    "max_extracted_gb", "data_dir", "log_level", "runtime"}


class SplatWorkerApp(Adw.Application):
    def __init__(self, background: bool = False):
        flags = getattr(Gio.ApplicationFlags, "DEFAULT_FLAGS", Gio.ApplicationFlags.FLAGS_NONE)
        super().__init__(application_id=APP_ID, flags=flags)
        GLib.set_application_name(APP_NAME)
        self.start_in_background = background
        self.state: AppState = None
        self.window: MainWindow = None
        self._held = False
        self._first_activate = True
        self._background_requested = False

    # ------------------------------------------------------------- startup
    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        load_css()
        self.state = AppState()
        for name, cb, accels in (("quit", self._quit, ["<primary>q"]),
                                 ("preferences", self._preferences, ["<primary>comma"]),
                                 ("setup", self._setup, []),
                                 ("help", self._help, ["F1"]),
                                 ("about", self._about, []),
                                 ("show", lambda *_: self._present(), [])):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", cb)
            self.add_action(action)
            if accels:
                self.set_accels_for_action(f"app.{name}", accels)

    def do_activate(self) -> None:
        first = self._first_activate
        self._first_activate = False
        if self.window is None:
            self.window = MainWindow(self, self.state)
            self.window.connect("close-request", self._on_close_request)
        if first and self.start_in_background:
            # Autostart at login: serve without a window.
            self._hold()
            self.state.start_server()
            self._request_background(self.state.settings.autostart)
            return
        self._present()
        if first:
            if not self.state.runtime_installed:
                self._setup()
            elif self.state.settings.start_server_on_launch and not self.state.server.is_active:
                self.window.start_server()

    def do_shutdown(self) -> None:
        if self.window is not None:
            self.window.updater.close()
            studio = getattr(self.window, "studio", None)
            if studio is not None:
                studio.shutdown()  # cancel local pipelines (while the server still answers), stop the receiver
        if self.state is not None:
            self.state.shutdown()  # stops the server (jobs are cancelled cleanly)
        Adw.Application.do_shutdown(self)

    # ------------------------------------------------------------- helpers
    def _present(self) -> None:
        if self.window is None:
            self.activate()
            return
        self.window.set_visible(True)
        self.window.present()

    def _hold(self) -> None:
        if not self._held:
            self.hold()
            self._held = True

    def _release(self) -> None:
        if self._held:
            self.release()
            self._held = False

    def _request_background(self, autostart: bool, notify_user: bool = False) -> None:
        if not in_flatpak():
            try:
                set_autostart_file(autostart, [sys.executable, "-m", "splatworker.gui.launcher", "--background"])
            except OSError as e:
                if self.window:
                    toast(self.window.toasts, f"Could not change autostart: {e}")
            return

        def done(allowed: bool, auto: bool, message: str) -> None:
            if not allowed and notify_user and self.window is not None:
                toast(self.window.toasts, message, timeout=8)
            if autostart and not auto and self.window is not None and notify_user:
                toast(self.window.toasts, "Starting at login was not allowed by the system.", timeout=8)
        request_background(autostart, done)
        self._background_requested = True

    def _on_close_request(self, win) -> bool:
        s = self.state.settings
        if self.state.server.is_active and s.run_in_background:
            win.set_visible(False)
            self._hold()
            if not self._background_requested:
                self._request_background(s.autostart)
            n = Gio.Notification.new(f"{APP_NAME} is still running")
            n.set_body("The server keeps working for your Mac. Open SplatWork again to see it, or choose "
                       "Quit in its menu to stop it.")
            n.set_default_action("app.show")
            self.send_notification("background", n)
            return True
        self._release()
        return False  # window closes; the app quits (do_shutdown stops the server)

    # ------------------------------------------------------------- actions
    def _quit(self, *_args) -> None:
        jobs = self.window._last_poll.jobs if self.window and self.window._last_poll else []
        running = [j for j in jobs if j.get("state") in ("loading", "training")]
        studio = getattr(self.window, "studio", None) if self.window else None
        local = studio.busy_names() if studio is not None else []
        if (running or local) and self.window is not None:
            self._present()

            def answer(rid):
                if rid == "quit":
                    self._really_quit()
            what = [f"{len(running)} job(s) are running"] if running else []
            if local:
                what.append("working on " + ", ".join(local))
            alert(self.window, "Quit SplatWork?",
                  " and ".join(what).capitalize() + "; this is stopped.",
                  (("cancel", "Cancel", "default"), ("quit", "Quit", "destructive")), answer)
        else:
            self._really_quit()

    def _really_quit(self) -> None:
        self._release()
        self.withdraw_notification("background")
        self.quit()

    def _preferences(self, *_args) -> None:
        self._present()
        Preferences(self.window, self.state, self._setting_changed, self._autostart_changed,
                    self._setup, studio=getattr(self.window, "studio", None)).present()

    def _setting_changed(self, name: str) -> None:
        self.window.refresh_static()
        if name in RESTART_SETTINGS and self.state.server.is_active:
            toast(self.window.toasts, "Restart the server to apply the change.", "Restart",
                  self.window.restart_server, timeout=8)

    def _autostart_changed(self, enabled: bool) -> None:
        self._request_background(enabled, notify_user=True)

    def _setup(self, *_args) -> None:
        self._present()

        def finished(start_server: bool) -> None:
            self.state.reload_runtime()
            self.window.refresh_static()
            if start_server:
                if self.state.server.is_active:
                    self.window.restart_server()
                else:
                    self.window.start_server()
        SetupAssistant(self.window, self.state, finished).present()

    def _help(self, *_args) -> None:
        self._present()
        self.window.stack.set_visible_child_name("help")

    def _about(self, *_args) -> None:
        kwargs = dict(application_name=APP_NAME, application_icon=APP_ID, version=__version__,
                      developer_name="SplatWork", license_type=Gtk.License.MIT_X11,
                      comments="View, create and edit Gaussian splats on this PC, and train them for "
                               "SplatWork on your Mac with this computer's NVIDIA GPU.")
        if hasattr(Adw, "AboutDialog"):  # libadwaita 1.5
            Adw.AboutDialog(**kwargs).present(self.window)
        else:  # pragma: no cover
            Adw.AboutWindow(transient_for=self.window, **kwargs).present()


def run_app(background: bool = False) -> int:
    app = SplatWorkerApp(background=background)
    return app.run([sys.argv[0]])
