"""First-run setup assistant: detect the GPU, explain the download, install the runtime."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from gi.repository import Adw, GLib, Gtk

from ..installer import InstallError, Installer
from ..nvidia import DriverInfo, detect
from ..paths import splatprep_dir
from ..pip_progress import human_bytes
from ..runtime_plan import CPU_PROFILE_ID, GPU_PROFILES, Plan, build_plan, choose_profile, unpublished_profiles
from .common import action_row, alert, append_text, esc, label, scrolled_page, text_view

PROFILE_CHOICES = [(None, "Automatic (recommended)")] + [(p.id, p.label) for p in GPU_PROFILES] + \
    [(CPU_PROFILE_ID, "Processing only (no training)")]


class SetupAssistant(Adw.Window):
    def __init__(self, parent: Gtk.Window, state, on_finished: Optional[Callable[[bool], None]] = None):
        super().__init__(transient_for=parent, modal=True, title="Set Up SplatWork",
                         default_width=680, default_height=720)
        self.state = state
        self.on_finished = on_finished
        self.driver: Optional[DriverInfo] = None
        self.plan: Optional[Plan] = None
        self.installer: Optional[Installer] = None
        self.failed_step: Optional[int] = None
        self.override: Optional[str] = None
        self.splatprep_src = splatprep_dir()
        self.with_splatprep = state.settings.with_splatprep and self.splatprep_src is not None
        self.with_lightglue = state.settings.with_lightglue
        self.succeeded = False
        self._running = False
        self._last_result = "plan"

        self.toolbar = Adw.ToolbarView()
        self.header = Adw.HeaderBar()
        self.toolbar.add_top_bar(self.header)
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.toolbar.set_content(self.stack)
        self.set_content(self.toolbar)

        self._build_detect_page()
        self._build_progress_page()
        self.connect("close-request", self._on_close_request)
        self.start_detection()

    # --------------------------------------------------------------- detect
    def _build_detect_page(self) -> None:
        page = Adw.StatusPage(icon_name="system-search-symbolic", title="Looking for your NVIDIA GPU…",
                              description="This only takes a moment.")
        spinner = Gtk.Spinner(spinning=True, width_request=32, height_request=32, halign=Gtk.Align.CENTER)
        page.set_child(spinner)
        self.stack.add_named(page, "detect")

    def start_detection(self) -> None:
        self.stack.set_visible_child_name("detect")

        def work():
            info = detect()
            # which runtimes can actually be downloaded right now (our own gsplat builds may lag behind)
            self.unavailable = unpublished_profiles(local_dir=self.state.paths.local_wheels_dir)
            GLib.idle_add(self._show_plan, info)

        threading.Thread(target=work, daemon=True).start()

    # ----------------------------------------------------------------- plan
    def _show_plan(self, driver: DriverInfo) -> bool:
        self.driver = driver
        old = self.stack.get_child_by_name("plan")
        if old is not None:
            self.stack.remove(old)
        self.stack.add_named(self._build_plan_page(), "plan")
        self.stack.set_visible_child_name("plan")
        return False

    def _build_plan_page(self) -> Gtk.Widget:
        driver = self.driver
        choice = choose_profile(driver, self.override, getattr(self, "unavailable", ()))
        self.choice = choice
        self.plan = build_plan(choice, self.state.paths, with_splatprep=self.with_splatprep,
                               with_lightglue=self.with_lightglue, splatprep_dir=self.splatprep_src)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)

        # Header: what we found
        head = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        if driver.has_gpu:
            gpu = driver.gpus[0]
            icon, title = ("emblem-ok-symbolic", gpu.name if len(driver.gpus) == 1
                           else f"{len(driver.gpus)} GPUs: {', '.join(g.name for g in driver.gpus)}")
            facts = [f"{gpu.memory_total_mb / 1024:.0f} GB video memory" if gpu.memory_total_mb else None,
                     f"driver {driver.driver_version}" if driver.driver_version else None,
                     (f"supports CUDA {driver.cuda_version[0]}.{driver.cuda_version[1]}"
                      if driver.cuda_version else None)]
            subtitle = " · ".join(f for f in facts if f)
        else:
            problem = driver.problem()
            icon, title = "dialog-warning-symbolic", problem[0] if problem else "No NVIDIA GPU found"
            subtitle = ""
        img = Gtk.Image(icon_name=icon, pixel_size=64)
        img.add_css_class("success" if driver.has_gpu and choice.can_train else "warning")
        head.append(img)
        head.append(label(title, ("title-1",), xalign=0.5))
        if subtitle:
            head.append(label(subtitle, ("dim-label",), xalign=0.5))
        head.append(label(choice.summary, xalign=0.5))
        box.append(head)

        if choice.blocking:
            grp = Adw.PreferencesGroup(title="Why training is not available")
            grp.add(action_row(choice.blocking, icon="dialog-information-symbolic"))
            grp.add(action_row("You can still install the processing-only runtime: the Mac can send dataset "
                               "processing jobs (feature matching, SfM, masks) to this computer.",
                               icon="emblem-default-symbolic"))
            box.append(grp)
        if choice.warnings:
            grp = Adw.PreferencesGroup(title="Things to know")
            for w in choice.warnings:
                grp.add(action_row(w, icon="dialog-warning-symbolic"))
            box.append(grp)

        # What will be downloaded
        steps = Adw.PreferencesGroup(
            title="What will be installed",
            description=esc(f"About {self.plan.download_mb / 1000:.1f} GB will be downloaded from the internet "
                            f"and about {self.plan.disk_mb / 1000:.1f} GB of disk space used, in "
                            f"{self.state.paths.data_root}. Nothing is installed outside SplatWork's own "
                            "folder; removing the app removes it all."))
        for s in self.plan.steps:
            r = action_row(s.title.replace("Downloading ", "").replace("Installing ", ""), s.description)
            if s.download_mb:
                r.add_suffix(label(human_bytes(s.download_mb * 1_000_000), ("dim-label",), wrap=False))
            steps.add(r)
        box.append(steps)

        # Options
        opts = Adw.PreferencesGroup(title="Options")
        sp = Adw.SwitchRow(title="Dataset processing (splatprep)",
                           subtitle="Lets the Mac run feature matching, SfM and masking on this computer. "
                                    "Adds about 60 MB." if self.splatprep_src else
                                    "Not included in this build of SplatWork.",
                           active=self.with_splatprep, sensitive=self.splatprep_src is not None)
        sp.connect("notify::active", self._on_option, "splatprep")
        opts.add(sp)
        lg = Adw.SwitchRow(title="Processing models (recommended)",
                           subtitle="Camera solving (COLMAP, LightGlue matchers) and moving-object masking "
                                    "(OWLv2 + SAM 2), so the Mac can hand these jobs to this GPU. "
                                    "Adds about 300 MB now; model weights (≈1–2 GB) download on first use.",
                           active=self.with_lightglue, sensitive=self.with_splatprep)
        lg.connect("notify::active", self._on_option, "lightglue")
        opts.add(lg)
        box.append(opts)

        adv = Adw.PreferencesGroup(title="Advanced")
        combo = Adw.ComboRow(title="Runtime",
                             subtitle="Leave on Automatic unless the support team asked you to change it. "
                                      "Pick an older CUDA build if training fails with a driver error.",
                             model=Gtk.StringList.new([t for _, t in PROFILE_CHOICES]))
        combo.set_selected([pid for pid, _ in PROFILE_CHOICES].index(self.override))
        combo.connect("notify::selected", self._on_profile)
        adv.add(combo)
        box.append(adv)

        # Buttons
        buttons = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER, margin_top=6)
        later = Gtk.Button(label="Not Now")
        later.add_css_class("pill")
        later.connect("clicked", lambda *_: self.close())
        install = Gtk.Button(label="Install" if self.plan.can_train else "Install Processing Runtime")
        install.add_css_class("pill")
        install.add_css_class("suggested-action")
        install.connect("clicked", lambda *_: self.start_install())
        buttons.append(later)
        buttons.append(install)
        box.append(buttons)
        return scrolled_page(box, 640)

    def _on_option(self, row, _pspec, which: str) -> None:
        if which == "splatprep":
            self.with_splatprep = row.get_active()
            if not self.with_splatprep:
                self.with_lightglue = False
        else:
            self.with_lightglue = row.get_active()
        GLib.idle_add(self._show_plan, self.driver)

    def _on_profile(self, row, _pspec) -> None:
        self.override = PROFILE_CHOICES[row.get_selected()][0]
        GLib.idle_add(self._show_plan, self.driver)

    # ------------------------------------------------------------- progress
    def _build_progress_page(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        self.p_title = label("Installing the GPU runtime", ("title-2",), xalign=0.5)
        box.append(self.p_title)
        box.append(label("You can keep using the computer. Large downloads may take a while on slow "
                         "connections.", ("dim-label",), xalign=0.5))
        self.p_bar = Gtk.ProgressBar(show_text=True)
        box.append(self.p_bar)
        self.p_step = label("", ("heading",))
        self.p_msg = label("", ("dim-label", "caption"))
        box.append(self.p_step)
        box.append(self.p_msg)
        self.p_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.p_list.add_css_class("boxed-list")
        box.append(self.p_list)
        sw, self.p_log = text_view()
        sw.set_min_content_height(200)
        self.p_expander = Gtk.Expander(label="Details", child=sw)
        box.append(self.p_expander)
        self.p_cancel = Gtk.Button(label="Cancel", halign=Gtk.Align.CENTER)
        self.p_cancel.add_css_class("pill")
        self.p_cancel.add_css_class("destructive-action")
        self.p_cancel.connect("clicked", lambda *_: self._on_cancel_clicked())
        box.append(self.p_cancel)
        self.stack.add_named(scrolled_page(box, 640), "progress")

    def start_install(self, start_at: int = 0) -> None:
        problem = self.plan.free_space_problem(self.state.paths.data_root)
        if problem:
            alert(self, "Not enough disk space", problem)
            return
        self.state.settings.with_splatprep = self.plan.with_splatprep
        self.state.settings.with_lightglue = self.plan.with_lightglue
        self.state.save_settings()
        if self.state.server.is_active:
            self.state.stop_server(wait=True)  # the runtime is about to change under it
        self._step_icons: List[Gtk.Image] = []
        child = self.p_list.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self.p_list.remove(child)
            child = nxt
        for i, s in enumerate(self.plan.steps):
            r = action_row(s.title)
            icon = Gtk.Image(icon_name="emblem-ok-symbolic" if i < start_at else "content-loading-symbolic")
            icon.set_opacity(1.0 if i < start_at else 0.3)
            r.add_suffix(icon)
            self.p_list.append(r)
            self._step_icons.append(icon)
        if start_at == 0:
            self.p_log.get_buffer().set_text("")
        self._running = True
        self.p_cancel.set_label("Cancel")
        self.p_cancel.add_css_class("destructive-action")
        self.stack.set_visible_child_name("progress")
        self.installer = Installer(self.plan, self.state.paths,
                                   on_event=lambda kind, data: GLib.idle_add(self._on_event, kind, data))
        inst = self.installer
        threading.Thread(target=lambda: inst.run(start_at), daemon=True, name="splatworker-setup").start()

    def _on_event(self, kind: str, data: dict) -> bool:
        if kind == "step":
            i = data["index"]
            self.p_step.set_text(f"Step {i + 1} of {data['count']}: {data['step'].title}")
            for j, icon in enumerate(self._step_icons):
                if j < i:
                    icon.set_from_icon_name("emblem-ok-symbolic")
                    icon.set_opacity(1.0)
                elif j == i:
                    icon.set_from_icon_name("emblem-synchronizing-symbolic")
                    icon.set_opacity(1.0)
        elif kind == "progress":
            self.p_bar.set_fraction(data["overall"])
            self.p_bar.set_text(f"{data['overall'] * 100:.0f} %")
            self.p_msg.set_text(data["message"])
        elif kind == "log":
            append_text(self.p_log, [data["line"]], max_lines=3000)
        elif kind == "done":
            self.succeeded = True
            self._show_done(data.get("verify") or {})
        elif kind == "failed":
            self._show_failed(data["error"])
        elif kind == "cancelled":
            self._show_result("cancelled", "process-stop-symbolic", "Setup cancelled",
                              "Nothing was broken. Already downloaded files are kept, so trying again is faster.",
                              retry=True)
        return False

    def _on_cancel_clicked(self) -> None:
        if self._running:
            self._confirm_cancel()
        else:
            self.stack.set_visible_child_name(self._last_result)

    def _confirm_cancel(self) -> None:
        alert(self, "Cancel the installation?", "You can resume it later; finished downloads are kept.",
              (("continue", "Keep Installing", "default"), ("cancel", "Cancel Installation", "destructive")),
              lambda rid: self.installer.cancel() if rid == "cancel" and self.installer else None,
              close="continue")

    # -------------------------------------------------------------- results
    def _show_done(self, verify: dict) -> None:
        self.state.reload_runtime()
        if self.plan.can_train:
            devices = ", ".join(verify.get("devices") or []) or "your GPU"
            desc = (f"PyTorch {verify.get('torch')} and gsplat {verify.get('gsplat')} are installed and can use "
                    f"{devices}. Start the server, then open SplatWork on your Mac.")
        else:
            desc = ("The processing runtime is installed. This computer can run dataset processing jobs from "
                    "the Mac, but not training.")
        self._show_result("done", "emblem-ok-symbolic", "All set!", desc, retry=False, start=True)

    def _show_failed(self, err: InstallError) -> None:
        self.failed_step = err.step_index if err.step_index >= 0 else 0
        self._show_result("failed", "dialog-error-symbolic", err.title,
                          f"{err.detail}\n\n{err.hint}".strip(), retry=True, log=True)

    def _show_result(self, name: str, icon: str, title: str, description: str, retry: bool = False,
                     start: bool = False, log: bool = False) -> None:
        old = self.stack.get_child_by_name(name)
        if old is not None:
            self.stack.remove(old)
        page = Adw.StatusPage(icon_name=icon, title=title, description=esc(description))
        buttons = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER)
        if retry:
            b = Gtk.Button(label="Retry")
            b.add_css_class("pill")
            b.add_css_class("suggested-action")
            b.connect("clicked", lambda *_: self.start_install(self.failed_step or 0 if name == "failed" else 0))
            buttons.append(b)
        if log:
            b = Gtk.Button(label="Show Log")
            b.add_css_class("pill")
            b.connect("clicked", lambda *_: self._show_log())
            buttons.append(b)
        if start:
            b = Gtk.Button(label="Start the Server")
            b.add_css_class("pill")
            b.add_css_class("suggested-action")
            b.connect("clicked", lambda *_: self._finish(start_server=True))
            buttons.append(b)
        close = Gtk.Button(label="Close")
        close.add_css_class("pill")
        close.connect("clicked", lambda *_: self._finish(start_server=False))
        buttons.append(close)
        page.set_child(buttons)
        self._running = False
        self._last_result = name
        self.p_cancel.set_label("Back")
        self.p_cancel.remove_css_class("destructive-action")
        self.stack.add_named(page, name)
        self.stack.set_visible_child_name(name)

    def _show_log(self) -> None:
        self.p_expander.set_expanded(True)
        self.stack.set_visible_child_name("progress")

    def _finish(self, start_server: bool) -> None:
        cb = self.on_finished
        self.on_finished = None
        self.destroy()
        if cb:
            cb(start_server)

    def _on_close_request(self, *_args) -> bool:
        if self._running:
            self._confirm_cancel()
            return True
        if self.on_finished:
            cb, self.on_finished = self.on_finished, None
            cb(False)
        return False
