"""Projects page: every project in the projects folder, and for the selected one its preparation stages,
live training dashboard (step, steps/s, ETA, splats, PSNR / holdout PSNR charts), log, outputs and actions."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from gi.repository import Adw, Gio, GLib, Gtk

from ....studio.pipeline import stage_titles
from ....studio.presets import PRESETS, ProjectOptions
from ....studio.projects import Project, list_projects
from ...formatting import count, duration
from ..common import alert, append_text, label, run_in_thread, text_view
from .rows import combo_row, group, plain_row


class Chart(Gtk.DrawingArea):
    """PSNR (train line, holdout dots) and splat count over steps."""

    def __init__(self):
        super().__init__(content_height=150, hexpand=True)
        self.history = []
        self.set_draw_func(self._draw)

    def set_history(self, h) -> None:
        self.history = list(h)
        self.queue_draw()

    def _draw(self, area, cr, w, h) -> None:
        fg = self.get_color()
        cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.08)
        cr.rectangle(0, 0, w, h)
        cr.fill()
        pts = [(s, p, e, n) for s, p, e, n in self.history if s]
        if len(pts) < 2:
            cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.5)
            cr.move_to(8, h / 2)
            cr.show_text("Charts appear once training reports progress.")
            return
        smax = max(p[0] for p in pts)
        nmax = max((p[3] or 0) for p in pts) or 1
        ps = [p[1] for p in pts if p[1]] + [p[2] for p in pts if p[2]]
        lo, hi = (min(ps) - 0.5, max(ps) + 0.5) if ps else (0, 1)
        x = lambda s: 6 + (w - 12) * s / smax  # noqa: E731
        # splats: filled area
        cr.set_source_rgba(0.35, 0.6, 1.0, 0.25)
        cr.move_to(x(pts[0][0]), h)
        for s, _p, _e, n in pts:
            cr.line_to(x(s), h - (h - 10) * (n or 0) / nmax)
        cr.line_to(x(pts[-1][0]), h)
        cr.close_path()
        cr.fill()
        y = lambda v: h - 6 - (h - 16) * (v - lo) / max(hi - lo, 1e-6)  # noqa: E731
        cr.set_source_rgba(0.95, 0.55, 0.15, 0.95)
        cr.set_line_width(2)
        first = True
        for s, p, _e, _n in pts:
            if p:
                (cr.move_to if first else cr.line_to)(x(s), y(p))
                first = False
        cr.stroke()
        cr.set_source_rgba(0.2, 0.75, 0.35, 1)
        for s, _p, e, _n in pts:
            if e:
                cr.arc(x(s), y(e), 3.5, 0, 6.3)
                cr.fill()
        cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.7)
        cr.move_to(8, 14)
        cr.show_text(f"PSNR {lo + 0.5:.1f}–{hi - 0.5:.1f} dB (orange: training, green: holdout) · "
                     f"splats up to {count(nmax)}")


class TrainingPanel(Gtk.Box):
    """Live dashboard of the project's training job, with Pause / Resume and Save checkpoint (as on the Mac)."""

    def __init__(self, on_pause: Optional[Callable[[dict], None]] = None,
                 on_checkpoint: Optional[Callable[[dict], None]] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.job: Optional[dict] = None
        self.busy = False                   # a pause / resume / checkpoint request is on its way
        top = Gtk.Box(spacing=6)
        self.status = label("", ("heading",))
        self.status.set_hexpand(True)
        top.append(self.status)
        self.pause_btn = Gtk.Button(label="Pause", tooltip_text="Pause training: the state is saved and the GPU "
                                    "is free for other jobs until you resume")
        self.pause_btn.connect("clicked", lambda *_: on_pause and self.job and on_pause(self.job))
        self.ckpt_btn = Gtk.Button(label="Save checkpoint", tooltip_text="Save the splats as they are now to the "
                                   "project's results (training carries on)")
        self.ckpt_btn.connect("clicked", lambda *_: on_checkpoint and self.job and on_checkpoint(self.job))
        top.append(self.pause_btn)
        top.append(self.ckpt_btn)
        self.append(top)
        self.bar = Gtk.ProgressBar(show_text=True)
        self.append(self.bar)
        grid = Gtk.Grid(column_spacing=24, row_spacing=4)
        self.cells = {}
        for i, (key, title) in enumerate((("step", "Step"), ("rate", "Steps/s"), ("elapsed", "Elapsed"),
                                          ("eta", "Remaining"), ("splats", "Splats"), ("psnr", "PSNR (train)"),
                                          ("eval", "PSNR (holdout)"), ("gpu", "GPU"))):
            t = label(title, ("caption", "dim-label"), wrap=False)
            v = label("—", ("numeric",), wrap=False)
            grid.attach(t, i % 4, (i // 4) * 2, 1, 1)
            grid.attach(v, i % 4, (i // 4) * 2 + 1, 1, 1)
            self.cells[key] = v
        self.append(grid)
        self.chart = Chart()
        self.append(self.chart)
        self._rate = (None, None)

    def update(self, s: Optional[dict], history) -> None:
        if not s:
            return
        self.job = s
        total = s.get("totalSteps") or 0
        step = s.get("step") or 0
        state = s.get("state", "")
        msg = s.get("message") or ""
        pausing = msg == "pausing" or msg == "saving the training state"
        self.status.set_text({"queued": "Waiting for the GPU (other jobs first)", "loading": "Loading the photos…",
                              "training": "Pausing…" if pausing else "Training", "paused": "Paused",
                              "completed": "Finished", "failed": "Failed",
                              "cancelled": "Stopped"}.get(state, state) + (f" — {msg}"
                                                                           if state in ("queued", "loading", "paused")
                                                                           and msg else ""))
        if state == "paused" and msg.lower().startswith("paused"):
            self.status.set_text(msg[:1].upper() + msg[1:])      # "Paused at step 9,000"
        running = state in ("queued", "loading", "training")
        self.pause_btn.set_visible(running or state == "paused")
        self.pause_btn.set_label("Resume" if state == "paused" else "Pause")
        if state == "paused":
            self.pause_btn.add_css_class("suggested-action")
        else:
            self.pause_btn.remove_css_class("suggested-action")
        self.pause_btn.set_sensitive(not self.busy and not pausing)
        self.ckpt_btn.set_visible(state in ("training", "paused"))
        self.ckpt_btn.set_sensitive(not self.busy and not pausing and (state == "training" or
                                                                       s.get("checkpointStep") is not None))
        self.bar.set_fraction(step / total if total else 0)
        self.bar.set_text(f"{100 * step / total:.0f} %" if total else "")
        now = time.monotonic()
        last_step, last_t = self._rate
        if last_step is not None and step > last_step and now > last_t:
            self.cells["rate"].set_text(f"{(step - last_step) / (now - last_t):.1f}")
        if last_step is None or step != last_step:
            self._rate = (step, now)
        self.cells["step"].set_text(f"{step:,} / {total:,}")
        self.cells["elapsed"].set_text(duration(s.get("elapsedSeconds")))
        self.cells["eta"].set_text(duration(s.get("etaSeconds")))
        self.cells["splats"].set_text(count(s.get("numGaussians")))
        self.cells["psnr"].set_text(f"{s['psnr']:.2f} dB" if s.get("psnr") else "—")
        self.cells["eval"].set_text(f"{s['evalPsnr']:.2f} dB" if s.get("evalPsnr") else "—")
        self.cells["gpu"].set_text(s.get("gpu") or "—")
        self.chart.set_history(history)


class ProjectsPage(Gtk.Box):
    def __init__(self, window, controller, on_new_project: Callable, on_open: Callable[[Path], None]):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.window = window
        self.ctl = controller
        self.on_open = on_open
        self.current: Optional[Project] = None
        self._log_seen = 0
        self._listening = None

        view = Adw.ToolbarView(vexpand=True)
        self.append(view)
        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Projects", subtitle=str(controller.projects_root)))
        new = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="New project (Ctrl+N)")
        new.connect("clicked", lambda *_: on_new_project(None, None))
        header.pack_start(new)
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Refresh")
        refresh.connect("clicked", lambda *_: self.refresh())
        header.pack_end(refresh)
        folder = Gtk.Button(icon_name="folder-open-symbolic", tooltip_text="Open the projects folder")
        folder.connect("clicked", lambda *_: self._show_folder(controller.projects_root))
        header.pack_end(folder)
        view.add_top_bar(header)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, position=300, shrink_start_child=False)
        self.listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.listbox.add_css_class("navigation-sidebar")
        self.listbox.connect("row-selected", self._selected)
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, width_request=240)
        sw.set_child(self.listbox)
        paned.set_start_child(sw)
        self.detail_stack = Gtk.Stack()
        self.detail_stack.add_named(Adw.StatusPage(icon_name="folder-symbolic", title="No project selected",
                                                   description="Make one with New, or pick one on the left."),
                                    "empty")
        self.detail_stack.add_named(self._detail(), "detail")
        paned.set_end_child(self.detail_stack)
        view.set_content(paned)
        controller.listeners.append(self._changed)
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _detail(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=18, margin_bottom=18,
                      margin_start=18, margin_end=18)
        self.d_title = label("", ("title-2",))
        self.d_sub = label("", ("dim-label",))
        box.append(self.d_title)
        box.append(self.d_sub)
        actions = Gtk.Box(spacing=6)
        self.open_btn = Gtk.Button(label="Open latest result")
        self.open_btn.add_css_class("suggested-action")
        self.open_btn.connect("clicked", lambda *_: self._open_latest())
        self.train_btn = Gtk.Button(label="Train again")
        self.train_btn.connect("clicked", lambda *_: self._train_again())
        self.prepare_btn = Gtk.Button(label="Prepare again")
        self.prepare_btn.connect("clicked", lambda *_: self._prepare_again())
        self.stop_btn = Gtk.Button(label="Stop")
        self.stop_btn.add_css_class("destructive-action")
        self.stop_btn.connect("clicked", lambda *_: self._stop())
        folder = Gtk.Button(icon_name="folder-open-symbolic", tooltip_text="Show the project folder")
        folder.connect("clicked", lambda *_: self.current and self._show_folder(self.current.root))
        trash = Gtk.Button(icon_name="user-trash-symbolic", tooltip_text="Move the project to the trash")
        trash.connect("clicked", lambda *_: self._trash())
        for b in (self.open_btn, self.train_btn, self.prepare_btn, self.stop_btn, folder, trash):
            actions.append(b)
        box.append(actions)
        self.preset = "high"
        train_opts = group("Training settings")
        train_opts.add(combo_row("Quality preset", [(k, f"{v['title']} — {v['subtitle']}") for k, v in PRESETS.items()],
                                 "high", lambda v: setattr(self, "preset", v), "train.preset"))
        box.append(train_opts)
        self.stages = group("Preparation")
        self._stage_rows = []
        box.append(self.stages)
        self.train_group = group("Live training")
        self.panel = TrainingPanel(self._pause_resume, self._save_checkpoint)
        self.panel.set_margin_top(6)
        self.train_group.add(self.panel)
        box.append(self.train_group)
        self.outputs = group("Results", "Newest first. Click to open in the viewer.")
        self._out_rows = []
        box.append(self.outputs)
        logs = group("Log")
        sw, self.log_view = text_view()
        sw.set_size_request(-1, 220)
        logs.add(sw)
        box.append(logs)
        outer = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        clamp = Adw.Clamp(maximum_size=900)
        clamp.set_child(box)
        outer.set_child(clamp)
        return outer

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        sel = self.current.root if self.current else None
        while (row := self.listbox.get_row_at_index(0)) is not None:
            self.listbox.remove(row)
        projects = list_projects(self.ctl.projects_root)
        for p in projects:
            run = self.ctl.run_for(p.root)
            sub = ""
            if run and run.active:
                sub = f"{run.stage} {int(run.fraction * 100)} %"
            elif p.outputs():
                sub = f"{len(p.outputs())} result{'s' if len(p.outputs()) != 1 else ''}"
            elif p.has_dataset:
                sub = "prepared"
            row = plain_row(title=p.name, subtitle=sub, activatable=True)
            row.set_use_markup(False)
            row.project = p
            icon = "media-playback-start-symbolic" if run and run.active else (
                "object-select-symbolic" if p.outputs() else "folder-symbolic")
            row.add_prefix(Gtk.Image(icon_name=icon))
            self.listbox.append(row)
            if sel is not None and p.root == sel:
                self.listbox.select_row(row)
        if not projects:
            self.detail_stack.set_visible_child_name("empty")

    def select(self, root: Path) -> None:
        self.refresh()
        i = 0
        while (row := self.listbox.get_row_at_index(i)) is not None:
            if getattr(row, "project", None) is not None and row.project.root == Path(root):
                self.listbox.select_row(row)
                return
            i += 1

    def _selected(self, lb, row) -> None:
        if row is None:
            return
        self.current = row.project
        self._log_seen = 0
        self.log_view.get_buffer().set_text("")
        self._attach()
        self._render()
        self.detail_stack.set_visible_child_name("detail")

    def _attach(self) -> None:
        if self._listening is not None:
            try:
                self._listening.listeners.remove(self._event)
            except ValueError:
                pass
        self._listening = None
        run = self.ctl.run_for(self.current.root) if self.current else None
        if run is not None:
            run.listeners.append(self._event)
            self._listening = run
            lines = [d for k, d in run.events if k == "log"]
            append_text(self.log_view, lines)
            self._log_seen = len(run.events)
        elif self.current is not None and self.current.log_file.is_file():
            try:
                append_text(self.log_view, self.current.log_file.read_text(encoding="utf-8").splitlines()[-400:])
            except OSError:
                pass

    def _event(self, kind, data) -> None:
        if kind == "log":
            append_text(self.log_view, [data])
        elif kind == "preview":
            self.window.live_preview(Path(data), self.current)
        elif kind == "done" and data.get("output"):
            self.ctl.toast(f"{self.current.name}: training finished")
            self.window.notify_done(self.current, Path(data["output"]))
        elif kind == "failed":
            self.ctl.toast(f"{self.current.name}: {data}")
        self._render()

    def _changed(self) -> None:
        self.refresh()
        if self.current is not None:
            self._attach()
            self._render()

    def _render(self) -> None:
        p = self.current
        if p is None:
            return
        meta = p.meta
        run = self.ctl.run_for(p.root)
        opts = ProjectOptions.from_dict(meta["options"]) if isinstance(meta.get("options"), dict) and \
            meta["options"].get("source_kind") else None
        self.d_title.set_text(p.name)
        src = f"{opts.source_kind}: {opts.source}" if opts else "made on another machine"
        stats = meta.get("datasetStats") or {}
        cams = f" · {stats.get('cameras')} cameras" if stats.get("cameras") else ""
        self.d_sub.set_text(f"{p.root}\n{src}{cams}")
        active = bool(run and run.active)
        self.stop_btn.set_visible(active)
        self.train_btn.set_sensitive(not active and p.has_dataset)
        self.prepare_btn.set_sensitive(not active and opts is not None)
        self.open_btn.set_sensitive(bool(p.outputs()))
        # stages
        for r in self._stage_rows:
            self.stages.remove(r)
        self._stage_rows = []
        titles = run.pipeline._titles if run else (stage_titles(opts, False) if opts else [])
        durations = (run.pipeline.durations if run else meta.get("stageDurations")) or {}
        cur = run.stage if run else None
        for t in titles:
            if t == "Training":
                continue
            done = t in durations
            if run and run.active and t == cur:
                icon, sub = "media-playback-start-symbolic", f"{int(run.fraction * 100)} % · {run.message}"
            elif done:
                icon, sub = "object-select-symbolic", duration(durations[t])
            elif run and run.state == "failed" and t == cur:
                icon, sub = "dialog-error-symbolic", run.message
            else:
                icon, sub = "content-loading-symbolic", ""
            row = plain_row(title=t, subtitle=sub)
            row.set_use_markup(False)
            row.add_prefix(Gtk.Image(icon_name=icon))
            self.stages.add(row)
            self._stage_rows.append(row)
        self.stages.set_visible(bool(titles))
        # training
        job = run.last_job if run and run.last_job and run.last_job.get("kind") == "train" else None
        self.train_group.set_visible(job is not None)
        if job:
            self.panel.update(job, run.history)
        # outputs
        for r in self._out_rows:
            self.outputs.remove(r)
        self._out_rows = []
        for f in p.outputs()[:20]:
            row = plain_row(title=f.name, subtitle=time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
                                + f" · {f.stat().st_size / 1e6:.0f} MB"
                                + (" · skybox" if f.with_suffix(".sky.png").is_file() else ""), activatable=True)
            row.set_use_markup(False)
            row.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
            row.connect("activated", lambda _r, path=f: self.on_open(path))
            self.outputs.add(row)
            self._out_rows.append(row)
        self.outputs.set_visible(bool(self._out_rows))

    # ------------------------------------------------------------------ actions
    def _open_latest(self) -> None:
        if self.current and self.current.outputs():
            self.on_open(self.current.outputs()[0])

    def _options(self) -> Optional[ProjectOptions]:
        o = self.current.meta.get("options") if self.current else None
        if isinstance(o, dict) and o.get("source_kind"):
            return ProjectOptions.from_dict(o)
        return ProjectOptions(name=self.current.name, source_kind="dataset", source=str(self.current.dataset_dir))

    def _train_again(self) -> None:
        o = self._options()
        o.preset = self.preset
        self.ctl.start_pipeline(self.current, o, train_only=True, config=o.train_config())
        self._attach_later()

    def _prepare_again(self) -> None:
        o = self._options()
        o.preset = self.preset
        self.ctl.start_pipeline(self.current, o)
        self._attach_later()

    def _attach_later(self) -> None:
        def later():
            if self.current is not None and self.ctl.run_for(self.current.root):
                self._attach()
                self._render()
                return False
            return True
        GLib.timeout_add(300, later)

    def _job_client(self):
        run = self.ctl.run_for(self.current.root) if self.current else None
        return run.pipeline.client if run is not None else self.ctl.client()

    def _pause_resume(self, job: dict) -> None:
        client, job_id, resume = self._job_client(), job.get("jobId"), job.get("state") == "paused"
        if client is None or not job_id:
            return
        self.panel.busy = True
        self.panel.pause_btn.set_sensitive(False)

        def done(_r, error):
            self.panel.busy = False
            if error is not None:
                self.ctl.toast(f"Couldn't {'resume' if resume else 'pause'}: {getattr(error, 'message', error)}")
            else:
                self.ctl.toast("Resumed: back in the queue" if resume else
                               "Pausing… the training state is saved and the GPU freed")
            self.panel.update(self.panel.job, self.panel.chart.history)
        run_in_thread(lambda: client.resume(job_id) if resume else client.pause(job_id), done)

    def _save_checkpoint(self, job: dict) -> None:
        """Current splats → outputs/<name>-<time>-step<N>.ply (listed under Results, opens in the viewer)."""
        client, job_id, project = self._job_client(), job.get("jobId"), self.current
        if client is None or not job_id or project is None:
            return
        self.panel.busy = True
        self.panel.ckpt_btn.set_sensitive(False)
        self.ctl.toast("Saving a checkpoint…")

        def dest(step: int) -> Path:
            p = project.new_output_path()
            return p.with_name(f"{p.stem}-step{step}{p.suffix}")

        def done(result, error):
            self.panel.busy = False
            if error is not None:
                self.ctl.toast(f"Couldn't save a checkpoint: {getattr(error, 'message', error)}")
            else:
                path, step = result
                self.ctl.toast(f"Checkpoint at step {step:,} saved: {path.name}")
            self.panel.update(self.panel.job, self.panel.chart.history)
            if self.current is project:
                self._render()
        run_in_thread(lambda: client.checkpoint(job_id, dest), done)

    def _stop(self) -> None:
        run = self.ctl.run_for(self.current.root) if self.current else None
        if run and run.active:
            run.pipeline.cancel()
            self.ctl.toast("Stopping…")

    def _trash(self) -> None:
        p = self.current
        if p is None:
            return

        def answer(rid):
            if rid != "trash":
                return
            try:
                Gio.File.new_for_path(str(p.root)).trash(None)
            except GLib.Error as e:
                self.ctl.toast(f"Couldn't move it to the trash: {e.message}")
                return
            self.current = None
            self.refresh()
            self.detail_stack.set_visible_child_name("empty")
        alert(self.window, f"Move “{p.name}” to the trash?", "The photos, dataset and every result in it go to the "
              "trash (you can restore them from there).",
              (("cancel", "Cancel", "default"), ("trash", "Move to Trash", "destructive")), answer)

    def _show_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(path))).launch(self.window, None, None, None)
