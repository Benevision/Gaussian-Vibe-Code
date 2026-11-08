"""Main window: Status (server switch, address, token, GPUs, clients), Jobs, Log, Connect."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from gi.repository import Adw, Gio, GLib, Gtk

from .. import APP_NAME, formatting
from ..app_state import Poll
from ..server_process import FAILED, RUNNING, STARTING, STOPPED, STOPPING
from .common import (action_row, alert, append_text, copy_text, icon_button, label, run_in_thread,
                     scrolled_page, text_view, toast)
from .help_page import HelpPage
from .widgets import GpuRow, JobRow, empty_clients_row, qr_available, show_job_log, show_qr

POLL_MS = 2000
STATE_TEXT = {
    STOPPED: ("Server is off", "Turn it on so your Mac can send training jobs to this computer."),
    STARTING: ("Starting…", "This takes a few seconds."),
    RUNNING: ("Server is running", "Your Mac can now find and use this computer."),
    STOPPING: ("Stopping…", "Running jobs are being cancelled."),
    FAILED: ("Server stopped with a problem", ""),
}


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app, state):
        super().__init__(application=app, title=APP_NAME, default_width=920, default_height=760)
        self.app = app
        self.state = state
        self._updating = False
        self._polling = False
        self._token_visible = False
        self._log_seen = 0
        self._job_rows: Dict[str, JobRow] = {}
        self._gpu_rows: List[GpuRow] = []
        self._last_poll: Optional[Poll] = None

        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)
        view = Adw.ToolbarView()
        self.toasts.set_child(view)

        header = Adw.HeaderBar()
        self.stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)
        menu = Gio.Menu()
        section = Gio.Menu()
        section.append("Settings", "app.preferences")
        section.append("Set Up GPU Runtime…", "app.setup")
        section.append("How to Connect", "app.help")
        menu.append_section(None, section)
        section2 = Gio.Menu()
        section2.append(f"About {APP_NAME}", "app.about")
        section2.append("Quit", "app.quit")
        menu.append_section(None, section2)
        header.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu,
                                       tooltip_text="Main menu", primary=True))
        view.add_top_bar(header)

        self.banner = Adw.Banner(title="The GPU runtime is not installed yet, so this computer cannot train.",
                                 button_label="Set Up…")
        self.banner.connect("button-clicked", lambda *_: self.app.activate_action("setup", None))
        view.add_top_bar(self.banner)
        self.update_banner = Adw.Banner(button_label="How to Update")
        self.update_banner.connect("button-clicked", lambda *_: self._update_clicked())
        view.add_top_bar(self.update_banner)
        self._update_info = None
        self._update_busy = False
        from .flatpak_update import SelfUpdater
        self.updater = SelfUpdater(lambda: GLib.idle_add(self._render_update))
        self.updater.start()
        GLib.timeout_add_seconds(5, self._check_updates_once)
        GLib.timeout_add_seconds(6 * 3600, self._check_updates)
        GLib.timeout_add_seconds(30, self._auto_update_tick)

        view.set_content(self.stack)
        bottom = Adw.ViewSwitcherBar(stack=self.stack)
        view.add_bottom_bar(bottom)
        self._narrow_bar = bottom

        self.stack.add_titled_with_icon(self._build_status(), "status", "Status", "network-server-symbolic")
        self.stack.add_titled_with_icon(self._build_jobs(), "jobs", "Jobs", "view-list-symbolic")
        self.stack.add_titled_with_icon(self._build_log(), "log", "Log", "utilities-terminal-symbolic")
        self.help = HelpPage(self.toasts)
        self.stack.add_titled_with_icon(self.help, "help", "Connect", "help-browser-symbolic")

        # Show the bottom switcher on narrow windows (phones / Steam Deck portrait).
        if hasattr(Adw, "Breakpoint"):
            bp = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 560sp"))
            bp.add_setter(switcher, "visible", False)
            bp.add_setter(bottom, "reveal", True)
            self.add_breakpoint(bp)

        self.refresh_static()
        self._render(self.state.poll(with_gpus=True))
        GLib.timeout_add(POLL_MS, self._tick)

    # ================================================================ status
    def _build_status(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)

        # --- hero: big switch
        hero = Gtk.Box(spacing=16, margin_top=6)
        hero.add_css_class("card")
        inner = Gtk.Box(spacing=16, margin_top=18, margin_bottom=18, margin_start=18, margin_end=18,
                        hexpand=True)
        self.dot = Gtk.Box(valign=Gtk.Align.CENTER)
        self.dot.add_css_class("status-dot")
        self.dot.add_css_class("status-stopped")
        inner.append(self.dot)
        texts = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, hexpand=True, valign=Gtk.Align.CENTER)
        self.hero_title = label("Server is off", ("hero-title",))
        self.hero_sub = label("", ("dim-label",))
        texts.append(self.hero_title)
        texts.append(self.hero_sub)
        inner.append(texts)
        self.switch = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="Start or stop the server")
        self.switch.add_css_class("big-switch")
        self.switch.connect("notify::active", self._on_switch)
        inner.append(self.switch)
        hero.append(inner)
        box.append(hero)

        # --- connection
        conn = Adw.PreferencesGroup(title="Connection",
                                    description="What the Mac needs. Usually it finds this computer by itself.")
        self.addr_row = action_row("Address", "", icon="network-wired-symbolic", selectable=True)
        self.addr_row.add_suffix(icon_button("edit-copy-symbolic", "Copy the address", self._copy_address))
        conn.add(self.addr_row)
        self.name_row = action_row("Bonjour name", "", icon="network-workgroup-symbolic")
        conn.add(self.name_row)
        self.token_row = action_row("Access token", "", icon="dialog-password-symbolic")
        self.reveal_btn = icon_button("view-reveal-symbolic", "Show the token", self._toggle_token)
        self.token_row.add_suffix(self.reveal_btn)
        self.token_row.add_suffix(icon_button("edit-copy-symbolic", "Copy the token", self._copy_token))
        self.token_row.add_suffix(icon_button("view-refresh-symbolic", "Make a new token", self._regen_token))
        self.qr_btn = icon_button("view-grid-symbolic",
                                  "Show a QR code with the connection details", self._show_qr)
        self.qr_btn.set_visible(qr_available())
        self.token_row.add_suffix(self.qr_btn)
        conn.add(self.token_row)
        self.caps_row = action_row("Can do", "—", icon="emblem-system-symbolic")
        conn.add(self.caps_row)
        box.append(conn)

        # --- GPUs
        self.gpu_group = Adw.PreferencesGroup(title="Graphics card",
                                              description="Live usage. Memory fills up while a job trains.")
        self.gpu_placeholder = action_row("No GPU statistics", "", icon="dialog-information-symbolic")
        self.gpu_group.add(self.gpu_placeholder)
        box.append(self.gpu_group)

        # --- clients
        self.clients_group = Adw.PreferencesGroup(title="Connected Macs",
                                                  description="Computers that talked to this worker in the "
                                                              "last 5 minutes.")
        self._client_rows: List[Gtk.Widget] = []
        box.append(self.clients_group)
        self.requests_row = Adw.ExpanderRow(title="Recent requests", subtitle="")
        self._request_rows: List[Gtk.Widget] = []
        req_group = Adw.PreferencesGroup()
        req_group.add(self.requests_row)
        box.append(req_group)
        return scrolled_page(box)

    # ================================================================== jobs
    def _build_jobs(self) -> Gtk.Widget:
        self.jobs_stack = Gtk.Stack()
        empty = Adw.StatusPage(icon_name="view-list-symbolic", title="No jobs yet",
                               description="Jobs sent from SplatWork appear here with their progress.")
        self.jobs_stack.add_named(empty, "empty")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.jobs_group = Adw.PreferencesGroup(title="Jobs",
                                               description="Newest last. Results stay here until you delete "
                                                           "them (the Mac downloads them automatically).")
        box.append(self.jobs_group)
        self.jobs_stack.add_named(scrolled_page(box, 900), "list")
        self.jobs_off = Adw.StatusPage(icon_name="network-offline-symbolic", title="The server is off",
                                       description="Start the server to see its jobs.")
        self.jobs_stack.add_named(self.jobs_off, "off")
        return self.jobs_stack

    # =================================================================== log
    def _build_log(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        bar = Gtk.Box(spacing=6, margin_top=6, margin_bottom=6, margin_start=12, margin_end=12)
        bar.append(label("Server output", ("heading",), wrap=False))
        spacer = Gtk.Box(hexpand=True)
        bar.append(spacer)
        self.autoscroll = Gtk.CheckButton(label="Follow", active=True)
        bar.append(self.autoscroll)
        bar.append(icon_button("edit-copy-symbolic", "Copy the whole log", self._copy_log))
        bar.append(icon_button("folder-open-symbolic", "Open the data folder", self._open_data_dir))
        box.append(bar)
        sw, self.log_view = text_view()
        box.append(sw)
        return box

    # ============================================================== actions
    def _on_switch(self, sw, _pspec) -> None:
        if self._updating:
            return
        if sw.get_active():
            self.start_server()
        else:
            self.stop_server()

    def start_server(self) -> None:
        if not self.state.runtime_installed:
            toast(self.toasts, "Starting without the GPU runtime: this computer cannot train yet.",
                  "Set Up…", lambda: self.app.activate_action("setup", None), timeout=8)
        if not self.state.start_server():
            snap = self.state.server.snapshot()
            alert(self, "The server could not start", snap.message)
        self._poll_now()

    def stop_server(self) -> None:
        busy = [j for j in (self._last_poll.jobs if self._last_poll else [])
                if j.get("state") in ("loading", "training")]
        if busy:
            def answer(rid):
                if rid == "stop":
                    self._do_stop()
                else:
                    self._set_switch(True)
            alert(self, "Stop the server?",
                  f"{len(busy)} job(s) are running and will be stopped. The Mac will see them as failed.",
                  (("cancel", "Keep Running", "default"), ("stop", "Stop Server", "destructive")), answer)
        else:
            self._do_stop()

    def _do_stop(self) -> None:
        threading.Thread(target=self.state.stop_server, kwargs={"wait": True}, daemon=True).start()
        self._poll_now()

    def restart_server(self) -> None:
        def work():
            self.state.restart_server()
        run_in_thread(work, lambda *_: self._poll_now())

    def _copy_address(self, *_):
        text = self.addr_row.get_subtitle().split("\n")[0]
        copy_text(self, text)
        toast(self.toasts, "Address copied")

    def _copy_token(self, *_):
        tok = self.state.token()
        if tok:
            copy_text(self, tok)
            toast(self.toasts, "Token copied. Paste it in SplatWork on the Mac.")

    def _toggle_token(self, *_):
        self._token_visible = not self._token_visible
        self.reveal_btn.set_icon_name("view-conceal-symbolic" if self._token_visible else "view-reveal-symbolic")
        self.refresh_static()

    def _regen_token(self, *_):
        def answer(rid):
            if rid != "new":
                return
            self.state.regenerate_token()
            self.refresh_static()
            if self.state.server.is_active:
                self.restart_server()
            toast(self.toasts, "New token created. Enter it on the Mac again.")
        alert(self, "Make a new access token?",
              "Macs that use the current token will be locked out until you enter the new one there. "
              "The server restarts (running jobs are stopped).",
              (("cancel", "Cancel", "default"), ("new", "New Token", "destructive")), answer)

    def _show_qr(self, *_):
        addrs = self.state.lan_addresses()
        ip = addrs[0] if addrs else "127.0.0.1"
        s = self.state.settings
        show_qr(self, formatting.connection_uri(ip, s.port, s.effective_name(), self.state.token()),
                "Contains the address and the access token. Only show it to people you trust.")

    def _copy_log(self, *_):
        copy_text(self, "\n".join(self.state.server.log.tail()))
        toast(self.toasts, "Log copied")

    def _open_data_dir(self, *_):
        path = self.state.settings.effective_data_dir(self.state.paths)
        path.mkdir(parents=True, exist_ok=True)
        f = Gio.File.new_for_path(str(path))
        if hasattr(Gtk, "FileLauncher"):
            Gtk.FileLauncher.new(f).launch(self, None, None)
        else:  # pragma: no cover
            Gio.AppInfo.launch_default_for_uri(f.get_uri(), None)

    # ---------------------------------------------------------- job actions
    def _client(self):
        return self.state.client()

    def _job_cancel(self, job_id: str) -> None:
        run_in_thread(lambda: self._client().cancel(job_id), self._after_job_action)

    def _job_delete(self, job_id: str, title: str) -> None:
        def answer(rid):
            if rid == "delete":
                run_in_thread(lambda: self._client().delete(job_id), self._after_job_action)
        alert(self, f"Delete “{title}”?",
              "The job and its files (result, previews, log) are deleted from this computer. A running job is "
              "cancelled first. Results the Mac already downloaded are not affected.",
              (("cancel", "Cancel", "default"), ("delete", "Delete", "destructive")), answer)

    def _after_job_action(self, _result, error) -> None:
        if error is not None:
            toast(self.toasts, f"Failed: {getattr(error, 'message', error)}")
        self._poll_now()

    def _job_log(self, job_id: str, title: str) -> None:
        client = self._client()
        if client is not None:
            show_job_log(self, client, job_id, title)

    def _job_export(self, job_id: str, artifact: str, title: str) -> None:
        ext = ".ply" if artifact.endswith(".ply") else ".zip"
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip() or job_id
        dlg = Gtk.FileDialog(title="Save the result", modal=True, initial_name=safe + ext)
        downloads = GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_DOWNLOAD)
        if downloads:
            dlg.set_initial_folder(Gio.File.new_for_path(downloads))

        def chosen(d, result):
            try:
                f = d.save_finish(result)
            except GLib.Error:
                return
            path = f.get_path() if f is not None else None
            if not path:
                return
            toast(self.toasts, "Saving…")
            client = self._client()
            run_in_thread(lambda: client.download(job_id, artifact, path),
                          lambda _r, e: toast(self.toasts, f"Could not save: {getattr(e, 'message', e)}"
                                              if e else f"Saved to {path}"))
        dlg.save(self, None, chosen)

    # ============================================================== polling
    def _tick(self) -> bool:
        if self.get_visible():
            self._poll_now()
        return True

    def _poll_now(self) -> None:
        if self._polling:
            return
        self._polling = True

        def work():
            return self.state.poll(with_gpus=True)

        def done(poll, error):
            self._polling = False
            if error is None and poll is not None:
                self._render(poll)
        run_in_thread(work, done)

    def _set_switch(self, active: bool) -> None:
        self._updating = True
        try:
            self.switch.set_active(active)
        finally:
            self._updating = False

    def _check_updates_once(self) -> bool:
        self._check_updates()
        return False

    def _check_updates(self) -> bool:
        from .. import updates

        def work():
            info = updates.check()
            GLib.idle_add(self._set_update_info, info)
        threading.Thread(target=work, daemon=True).start()
        return True  # keep the periodic timer

    def _set_update_info(self, info) -> bool:
        self._update_info = info
        if self.updater.supported:
            self.updater.published(info)
        else:
            self._render_update()
        return False

    def _render_update(self) -> bool:
        from .. import updates as U
        auto = self.state.settings.auto_update
        if self.updater.supported:
            u = self.updater
            title, button = U.banner_text(u.state, u.version or (self._update_info or {}).get("version", ""),
                                          auto, self._update_busy, u.fraction, u.error)
        elif self._update_info:      # not a Flatpak install: tell, don't act
            title, button = f"SplatWork Node {self._update_info.get('version')} is available.", "How to Update"
        else:
            title, button = "", None
        self.update_banner.set_title(title)
        if button:
            self.update_banner.set_button_label(button)
        self.update_banner.set_revealed(bool(title))
        return False

    def _update_clicked(self) -> None:
        from .. import updates as U
        u = self.updater
        if not u.supported or u.state == U.FAILED:
            self._show_update_help()
        elif u.state == U.AVAILABLE:
            u.update()               # downloading + installing doesn't touch running jobs; only the restart does
        elif u.state == U.INSTALLED:
            self._confirm_if_busy("Restart now?", "restart into the new version", self._restart_for_update)

    def _confirm_if_busy(self, heading: str, what: str, action) -> None:
        jobs = self._last_poll.jobs if self._last_poll else []
        from .. import updates as U
        busy = U.busy_jobs(jobs)
        if not busy:
            action()
            return
        alert(self, heading, f"{len(busy)} job(s) are running or queued. To {what} now, they are stopped "
                             "(the Mac can start them again).",
              (("cancel", "Cancel", "default"), ("go", "Stop Jobs and Continue", "destructive")),
              lambda r: action() if r == "go" else None)

    def _restart_for_update(self) -> None:
        background = not self.get_visible()
        self.updater.restart(background, self.app._really_quit)

    def _auto_update_tick(self) -> bool:
        """Every 30 s: with automatic updates on, install / restart once no job is running or queued."""
        from .. import updates as U
        if not self.updater.supported or self.updater.state not in (U.AVAILABLE, U.INSTALLED):
            return True
        client = self.state.client() if self.state.server.is_active else None

        def work():
            try:
                busy = bool(U.busy_jobs(client.jobs())) if client is not None else False
            except Exception:
                busy = True          # can't tell: don't restart under a job
            GLib.idle_add(self._auto_update_decide, busy)
        threading.Thread(target=work, daemon=True).start()
        return True

    def _auto_update_decide(self, busy: bool) -> bool:
        from .. import updates as U
        self._update_busy = busy
        action = U.next_action(self.updater.state, self.state.settings.auto_update, busy)
        if action == "update":
            self.updater.update()
        elif action == "restart":
            self._restart_for_update()
        self._render_update()
        return False

    def _show_update_help(self) -> None:
        from .. import updates
        v = (self._update_info or {}).get("version", "")
        err = self.updater.error if self.updater.supported else ""
        alert(self, f"Update to {v}" if v else "Update",
              (f"The automatic update failed: {err}\n\n" if err else "")
              + "Updates arrive like any other Flatpak: GNOME Software or Discover offer them automatically, "
              "or run this in a terminal (then restart SplatWork Node):\n\n" + updates.UPDATE_COMMAND,
              responses=(("close", "Close", "default"), ("copy", "Copy Command", "suggested")),
              callback=lambda r: copy_text(self, updates.UPDATE_COMMAND) if r == "copy" else None,
              default="copy", close="close")

    def refresh_static(self) -> None:
        """Parts that depend on settings / runtime rather than on the live poll."""
        s = self.state.settings
        self.banner.set_revealed(not self.state.runtime_installed)
        port = self.state.running_port if self.state.server.is_active and self.state.running_port else s.port
        addrs = self.state.lan_addresses()
        self.addr_row.set_subtitle("\n".join(formatting.addresses_text(addrs, port)))
        self.name_row.set_subtitle(f"{s.effective_name()}" + ("" if s.mdns else " (announcing is off)"))
        if s.require_token:
            tok = self.state.token() or ""
            self.token_row.set_subtitle(tok if self._token_visible else "•" * 12 + "  (hidden)")
        else:
            self.token_row.set_subtitle("Off: anyone on your network can use this worker")
        self.help.update(addrs, port, s.effective_name(), s.require_token, s.mdns)

    def _render(self, p: Poll) -> None:
        self._last_poll = p
        st = p.server.state
        title, sub = STATE_TEXT.get(st, (st, ""))
        if st == FAILED:
            sub = p.server.message
        elif st == RUNNING and p.api_error:
            sub = f"Running, but not answering: {p.api_error}"
        elif st == RUNNING and self.state.needs_restart():
            sub = "Settings changed: turn the server off and on to apply them."
        self.hero_title.set_text(title)
        self.hero_sub.set_text(sub)
        for c in ("status-running", "status-starting", "status-stopped", "status-failed"):
            self.dot.remove_css_class(c)
        self.dot.add_css_class({RUNNING: "status-running", STARTING: "status-starting",
                                STOPPING: "status-starting", FAILED: "status-failed"}.get(st, "status-stopped"))
        self._set_switch(st in (RUNNING, STARTING))
        self.switch.set_sensitive(st != STOPPING)
        self.caps_row.set_subtitle(formatting.capabilities_text(p.info) if p.info else
                                   ("—" if st != RUNNING else "…"))
        self._render_gpus(p)
        self._render_clients()
        self._render_jobs(p)
        self._render_log()

    def _render_gpus(self, p: Poll) -> None:
        if not p.gpus:
            self.gpu_placeholder.set_visible(True)
            self.gpu_placeholder.set_title("No GPU statistics")
            self.gpu_placeholder.set_subtitle(p.gpu_error or "No NVIDIA GPU detected.")
        else:
            self.gpu_placeholder.set_visible(False)
        while len(self._gpu_rows) < len(p.gpus):
            r = GpuRow()
            self.gpu_group.add(r)
            self._gpu_rows.append(r)
        while len(self._gpu_rows) > len(p.gpus):
            self.gpu_group.remove(self._gpu_rows.pop())
        for row, s in zip(self._gpu_rows, p.gpus):
            row.update(s)

    def _render_clients(self) -> None:
        for r in self._client_rows:
            self.clients_group.remove(r)
        self._client_rows = []
        clients = self.state.server.requests.active_clients()
        if not clients:
            r = empty_clients_row()
            self.clients_group.add(r)
            self._client_rows.append(r)
        for c in clients[:8]:
            sub = f"{c.requests} request{'s' if c.requests != 1 else ''} · last {c.last_path} · " \
                  f"{formatting.ago(c.last_seen)}"
            if c.unauthorized:
                sub += f" · {c.unauthorized} rejected (wrong or missing token)"
            r = action_row(c.address, sub, icon="computer-symbolic")
            self.clients_group.add(r)
            self._client_rows.append(r)
        for r in self._request_rows:
            self.requests_row.remove(r)
        self._request_rows = []
        recent = self.state.server.requests.recent_requests(15)
        self.requests_row.set_subtitle(f"{len(recent)} shown" if recent else "None yet")
        for rec in recent:
            r = action_row(f"{rec.method} {rec.path}", f"{rec.client} · HTTP {rec.status} · {formatting.ago(rec.time)}")
            self.requests_row.add_row(r)
            self._request_rows.append(r)

    def _render_jobs(self, p: Poll) -> None:
        if p.server.state != RUNNING:
            self.jobs_stack.set_visible_child_name("off")
            return
        jobs = p.jobs
        self.jobs_stack.set_visible_child_name("list" if jobs else "empty")
        seen = set()
        for j in jobs:
            jid = j.get("jobId")
            seen.add(jid)
            row = self._job_rows.get(jid)
            if row is None:
                row = JobRow(jid, self._job_cancel, self._job_delete, self._job_log, self._job_export)
                self._job_rows[jid] = row
                self.jobs_group.add(row)
            row.update(j)
        for jid in list(self._job_rows):
            if jid not in seen:
                self.jobs_group.remove(self._job_rows.pop(jid))
        active = sum(1 for j in jobs if j.get("state") in ("loading", "training"))
        queued = sum(1 for j in jobs if j.get("state") == "queued")
        self.jobs_group.set_description(
            f"{active} running, {queued} waiting. Results stay here until you delete them (the Mac downloads "
            "them automatically).")

    def _render_log(self) -> None:
        buf = self.state.server.log
        new = buf.since(self._log_seen)
        self._log_seen = buf.total
        append_text(self.log_view, new, autoscroll=self.autoscroll.get_active())
