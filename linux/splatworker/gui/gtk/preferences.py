"""Settings page. Every setting has its explanation right under it."""

from __future__ import annotations

from typing import Callable, List, Optional

from gi.repository import Adw, Gio, GLib, Gtk

from ..installer import remove_runtime
from ..nvidia import GpuInfo, detect
from ..paths import in_flatpak
from .common import action_row, alert, run_in_thread

LOG_LEVELS = ["critical", "error", "warning", "info", "debug"]


class Preferences:
    """Builds an Adw.PreferencesDialog (libadwaita >= 1.5) or Adw.PreferencesWindow."""

    def __init__(self, window, state, on_changed: Callable[[str], None],
                 on_autostart: Callable[[bool], None], on_setup: Callable[[], None]):
        self.window = window
        self.state = state
        self.on_changed = on_changed          # called with the setting name after saving
        self.on_autostart = on_autostart
        self.on_setup = on_setup
        self._gpu_checks: List[Gtk.CheckButton] = []
        self._gpus: List[GpuInfo] = []

        if hasattr(Adw, "PreferencesDialog"):
            self.dialog = Adw.PreferencesDialog(title="Settings")
        else:  # pragma: no cover - libadwaita 1.4
            self.dialog = Adw.PreferencesWindow(title="Settings", transient_for=window, modal=True)
        page = Adw.PreferencesPage(title="Settings", icon_name="emblem-system-symbolic")
        for grp in (self._network(), self._storage(), self._gpus_group(), self._startup(),
                    self._runtime(), self._advanced()):
            page.add(grp)
        self.dialog.add(page)

    def present(self) -> None:
        if isinstance(self.dialog, getattr(Adw, "PreferencesDialog", ())):
            self.dialog.present(self.window)
        else:  # pragma: no cover
            self.dialog.present()

    # ------------------------------------------------------------- helpers
    @property
    def s(self):
        return self.state.settings

    def _set(self, name: str, value) -> bool:
        old = getattr(self.s, name)
        setattr(self.s, name, value)
        err = self.s.field_errors(only=name)
        if err:
            setattr(self.s, name, old)
            self._toast(err[name])
            return False
        self.state.save_settings()
        self.on_changed(name)
        return True

    def _toast(self, text: str) -> None:
        t = Adw.Toast(title=text, timeout=4)
        if hasattr(self.dialog, "add_toast"):
            self.dialog.add_toast(t)

    def _switch(self, grp, name: str, title: str, subtitle: str,
                after: Optional[Callable[[bool], None]] = None) -> Adw.SwitchRow:
        row = Adw.SwitchRow(title=title, subtitle=subtitle, active=bool(getattr(self.s, name)))
        row.set_use_markup(False)

        def changed(r, _p):
            if self._set(name, r.get_active()) and after:
                after(r.get_active())
        row.connect("notify::active", changed)
        grp.add(row)
        return row

    def _spin(self, grp, name: str, title: str, subtitle: str, lo: float, hi: float, step: float,
              digits: int = 0) -> Adw.SpinRow:
        row = Adw.SpinRow.new_with_range(lo, hi, step)
        row.set_title(title)
        row.set_subtitle(subtitle)
        row.set_use_markup(False)
        row.set_digits(digits)
        row.set_value(float(getattr(self.s, name)))
        cast = int if digits == 0 else float
        row.connect("notify::value", lambda r, _p: self._set(name, cast(r.get_value())))
        grp.add(row)
        return row

    # -------------------------------------------------------------- groups
    def _network(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="Network",
                                 description="How the Mac finds and talks to this computer. Changes apply when "
                                             "the server restarts.")
        self._spin(g, "port", "Port",
                   "The TCP port the server listens on. Splat Studio uses 7878 by default; change it only if "
                   "another program already uses that port. Allow it in your firewall.", 1024, 65535, 1)
        name = Adw.EntryRow(title="Name shown on the Mac (empty = this computer's name)",
                            show_apply_button=True)
        name.set_text(self.s.name)
        name.connect("apply", lambda r: self._set("name", r.get_text().strip()))
        g.add(name)
        self._switch(g, "mdns", "Announce on the local network (Bonjour)",
                     "Lets Splat Studio find this computer automatically. Needs multicast (UDP port 5353) "
                     "between the two machines. Turn off to add the worker on the Mac by address instead.")
        self._switch(g, "require_token", "Require an access token",
                     "Strongly recommended. Without a token anyone on your network could upload data and use "
                     "your GPU.")
        ip = Adw.EntryRow(title="Advertised address (empty = automatic)", show_apply_button=True)
        ip.set_text(self.s.advertise_ip)
        ip.connect("apply", lambda r: self._set("advertise_ip", r.get_text().strip()))
        g.add(ip)
        g.add(action_row("Only needed when the Mac sees the wrong address, e.g. with a VPN or Docker network. "
                         "Enter this computer's LAN IPv4 address."))
        return g

    def _storage(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="Storage",
                                 description="Uploaded datasets and job results are kept here until you delete "
                                             "them.")
        self.dir_row = action_row("Data folder", str(self.s.effective_data_dir(self.state.paths)),
                                  icon="folder-symbolic")
        choose = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose.connect("clicked", lambda *_: self._choose_folder())
        reset = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Use the default folder",
                           valign=Gtk.Align.CENTER)
        reset.add_css_class("flat")
        reset.connect("clicked", lambda *_: self._set_dir(""))
        self.dir_row.add_suffix(reset)
        self.dir_row.add_suffix(choose)
        g.add(self.dir_row)
        if in_flatpak():
            g.add(action_row("Inside the Flatpak, a folder you choose here is shared with the app through the "
                             "document portal. The default folder needs no extra permission."))
        self._spin(g, "max_upload_gb", "Maximum upload size (GB)",
                   "Largest dataset (zip) the Mac may send. Raise it for very large captures; it protects the "
                   "disk from filling up.", 1, 1000, 1)
        self._spin(g, "max_extracted_gb", "Maximum unpacked size (GB)",
                   "Largest size a dataset may have once unzipped (protects against zip bombs).", 1, 2000, 1)
        return g

    def _gpus_group(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="GPUs",
                                 description="Each selected GPU trains one job at a time; more GPUs = more jobs "
                                             "in parallel.")
        self.all_gpus = Adw.SwitchRow(title="Use all GPUs", subtitle="Recommended. Turn off to choose GPUs, "
                                      "e.g. to keep one free for gaming or display.",
                                      active=self.s.gpus is None)
        self.all_gpus.connect("notify::active", lambda *_: self._gpus_changed())
        g.add(self.all_gpus)
        self.gpu_group = g
        self._gpu_rows: List[Adw.ActionRow] = []

        def done(info, error):
            if error is None and info is not None:
                self._gpus = info.gpus
                for gpu in info.gpus:
                    row = action_row(f"GPU {gpu.index}: {gpu.name}",
                                     f"{gpu.memory_total_mb / 1024:.0f} GB" if gpu.memory_total_mb else "")
                    check = Gtk.CheckButton(active=self.s.gpus is None or gpu.index in (self.s.gpus or []),
                                            valign=Gtk.Align.CENTER)
                    check.connect("toggled", lambda *_: self._gpus_changed())
                    row.add_prefix(check)
                    row.set_activatable_widget(check)
                    row.set_sensitive(self.s.gpus is not None)
                    self._gpu_checks.append(check)
                    self._gpu_rows.append(row)
                    g.add(row)
                if not info.gpus:
                    g.add(action_row("No NVIDIA GPU detected.", icon="dialog-warning-symbolic"))
        run_in_thread(detect, done)
        return g

    def _gpus_changed(self) -> None:
        use_all = self.all_gpus.get_active()
        for r in self._gpu_rows:
            r.set_sensitive(not use_all)
        if use_all:
            self._set("gpus", None)
            return
        chosen = [gpu.index for gpu, c in zip(self._gpus, self._gpu_checks) if c.get_active()]
        if not chosen and self._gpus:
            self._toast("Select at least one GPU.")
            return
        self._set("gpus", chosen or None)

    def _startup(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="Startup and background")
        self._switch(g, "start_server_on_launch", "Start the server when the app opens",
                     "So you never forget to turn it on before training from the Mac.")
        self._switch(g, "run_in_background", "Keep running when the window is closed",
                     "The server keeps serving the Mac in the background. Open Splat Worker again from the app "
                     "grid to see it, or choose Quit in the menu to stop it.")
        self._switch(g, "autostart", "Start automatically when you log in",
                     "Starts Splat Worker in the background at login (the system may ask for permission). "
                     "Useful for a dedicated GPU box.", after=self.on_autostart)
        return g

    def _runtime(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="GPU runtime",
                                 description="PyTorch, gsplat and the other libraries downloaded by the setup "
                                             "assistant.")
        rt = self.state.reload_runtime()
        row = action_row("Installed", rt.describe() if rt.is_installed(self.state.paths) else "Not installed",
                         icon="application-x-addon-symbolic")
        g.add(row)
        repair = action_row("Reinstall or change", "Runs the setup assistant again (keeps your jobs and "
                            "settings). Use it after a big driver update or if training stops working.")
        b = Gtk.Button(label="Open Setup…", valign=Gtk.Align.CENTER)
        b.connect("clicked", lambda *_: self._open_setup())
        repair.add_suffix(b)
        g.add(repair)
        remove = action_row("Remove the GPU runtime", f"Frees several GB in {self.state.paths.data_root}. "
                            "Jobs, datasets and settings are kept.")
        rb = Gtk.Button(label="Remove…", valign=Gtk.Align.CENTER)
        rb.add_css_class("destructive-action")
        rb.connect("clicked", lambda *_: self._remove_runtime(row))
        remove.add_suffix(rb)
        g.add(remove)
        return g

    def _advanced(self) -> Adw.PreferencesGroup:
        g = Adw.PreferencesGroup(title="Advanced")
        combo = Adw.ComboRow(title="Log detail", subtitle="\"debug\" shows every step; use it when reporting "
                             "a problem.", model=Gtk.StringList.new(LOG_LEVELS))
        combo.set_selected(LOG_LEVELS.index(self.s.log_level))
        combo.connect("notify::selected", lambda r, _p: self._set("log_level", LOG_LEVELS[r.get_selected()]))
        g.add(combo)
        return g

    # -------------------------------------------------------------- actions
    def _choose_folder(self) -> None:
        dlg = Gtk.FileDialog(title="Choose the data folder", modal=True)
        dlg.set_initial_folder(Gio.File.new_for_path(str(self.s.effective_data_dir(self.state.paths).parent)))

        def done(d, result):
            try:
                f = d.select_folder_finish(result)
            except GLib.Error:
                return  # cancelled
            if f is not None and f.get_path():
                self._set_dir(f.get_path())
        dlg.select_folder(self.window, None, done)

    def _set_dir(self, path: str) -> None:
        if self._set("data_dir", path):
            self.dir_row.set_subtitle(str(self.s.effective_data_dir(self.state.paths)))

    def _open_setup(self) -> None:
        if hasattr(self.dialog, "close"):
            self.dialog.close()
        self.on_setup()

    def _remove_runtime(self, row) -> None:
        def answer(rid):
            if rid != "remove":
                return
            if self.state.server.is_active:
                self.state.stop_server(wait=True)
            remove_runtime(self.state.paths)
            self.state.reload_runtime()
            row.set_subtitle("Not installed")
            self.on_changed("runtime")
        alert(self.window, "Remove the GPU runtime?",
              "The server is stopped. You will have to run the setup assistant (a large download) before "
              "this computer can train again.",
              (("cancel", "Cancel", "default"), ("remove", "Remove", "destructive")), answer)
