"""Settings Reference, iPhone Inbox and Models & Benchmarks pages."""
from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from typing import Callable

from gi.repository import Adw, Gio, GLib, Gtk

from ....studio import benchmark, catalog
from ....studio.inbox import format_code, list_captures, new_code
from ...paths import splatprep_dir
from ..common import action_row, alert, label
from .rows import group, plain_expander, plain_row, switch_row


def _page(title: str, content: Gtk.Widget, subtitle: str = "", header_extra=()) -> Adw.ToolbarView:
    view = Adw.ToolbarView(vexpand=True, hexpand=True)
    header = Adw.HeaderBar()
    header.set_title_widget(Adw.WindowTitle(title=title, subtitle=subtitle))
    for w in header_extra:
        header.pack_end(w)
    view.add_top_bar(header)
    view.set_content(content)
    return view


# ============================================================================ settings reference

class ReferencePage(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.search = Gtk.SearchEntry(placeholder_text="Search settings", hexpand=True)
        self.search.connect("search-changed", lambda *_: self._fill())
        paned = Gtk.Paned(position=340, shrink_start_child=False)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_top=6, margin_start=6, margin_end=6)
        left.append(self.search)
        self.listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.listbox.add_css_class("navigation-sidebar")
        self.listbox.set_header_func(self._header)
        self.listbox.connect("row-selected", self._selected)
        sw = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER, width_request=280)
        sw.set_child(self.listbox)
        left.append(sw)
        paned.set_start_child(left)
        detail = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_top=24, margin_bottom=24,
                         margin_start=24, margin_end=24)
        self.d_title = label("", ("title-2",), selectable=True)
        self.d_badge = label("", ("caption", "dim-label"))
        self.d_summary = label("", ("heading",), selectable=True)
        self.d_facts = label("", ("monospace",), selectable=True)
        self.d_detail = label("", (), selectable=True)
        self.d_effects = label("", ("dim-label",), selectable=True)
        for w in (self.d_title, self.d_badge, self.d_summary, self.d_facts, self.d_detail, self.d_effects):
            detail.append(w)
        dsw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        clamp = Adw.Clamp(maximum_size=760)
        clamp.set_child(detail)
        dsw.set_child(clamp)
        paned.set_end_child(dsw)
        self.append(_page("Settings Reference", paned, f"{len(catalog.settings())} settings, same as the Mac"))
        self._fill()

    def _fill(self) -> None:
        while (row := self.listbox.get_row_at_index(0)) is not None:
            self.listbox.remove(row)
        for s in catalog.search(self.search.get_text()):
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, margin_top=4, margin_bottom=4, margin_start=6)
            box.append(label(s["title"], wrap=False))
            box.append(label(s["summary"], ("caption", "dim-label")))
            row.set_child(box)
            row.setting = s
            self.listbox.append(row)
        first = self.listbox.get_row_at_index(0)
        if first is not None:
            self.listbox.select_row(first)

    def _header(self, row, before) -> None:
        g = row.setting["group"]
        if before is None or before.setting["group"] != g:
            h = label(g, ("heading",))
            h.set_margin_top(10)
            h.set_margin_start(6)
            row.set_header(h)
        else:
            row.set_header(None)

    def _selected(self, lb, row) -> None:
        if row is None:
            return
        s = row.setting
        self.d_title.set_text(s["title"])
        self.d_badge.set_text(f"{s['group']} · {'Advanced' if s['advanced'] else 'Easy'} · {s['id']}")
        self.d_summary.set_text(s["summary"])
        facts = [f"Default  {s['defaultValue']}"] if s.get("defaultValue") else []
        if s.get("range"):
            facts.append(f"Range    {s['range']}")
        self.d_facts.set_text("\n".join(facts))
        self.d_detail.set_text(s["detail"])
        self.d_effects.set_text(f"Effect: {s['effects']}" if s.get("effects") else "")


# ============================================================================ iPhone inbox

class InboxPage(Gtk.Box):
    def __init__(self, window, controller, on_new_project: Callable[[str, str], None]):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.window = window
        self.ctl = controller
        self.on_new_project = on_new_project
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=18, margin_bottom=18,
                      margin_start=18, margin_end=18)
        recv = group("Receive from iPhone", "SplatWork Capture finds this PC on the network (Bonjour "
                     "_splatcapture._tcp) and sends captures here.")
        self.recv_switch = switch_row("Receive captures", controller.prefs.receive_captures, self._toggle,
                                      "net.receive")
        recv.add(self.recv_switch)
        self.status_row = action_row("Status", "Off", icon="network-wireless-symbolic", selectable=True)
        recv.add(self.status_row)
        self.code_row = action_row("Pairing code", "", icon="dialog-password-symbolic")
        b = Gtk.Button(label="New code", valign=Gtk.Align.CENTER)
        b.connect("clicked", lambda *_: self._new_code())
        self.code_row.add_suffix(b)
        recv.add(self.code_row)
        recv.add(switch_row("Require pairing code", controller.prefs.require_pairing, self._pairing, "net.pairing"))
        self.progress = Gtk.ProgressBar(show_text=True, visible=False)
        recv.add(self.progress)
        box.append(recv)
        self.list_group = group("Captures", "Newest first.")
        self._rows = []
        box.append(self.list_group)
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        clamp = Adw.Clamp(maximum_size=820)
        clamp.set_child(box)
        sw.set_child(clamp)
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Refresh")
        refresh.connect("clicked", lambda *_: self.refresh())
        self.append(_page("iPhone Inbox", sw, header_extra=[refresh]))
        controller.receiver_listeners.append(self._event)
        controller.listeners.append(self.refresh)
        self.refresh()
        self._render_status()

    def _toggle(self, on: bool) -> None:
        self.ctl.prefs.receive_captures = on
        self.ctl.save_prefs()
        self.ctl.start_receiver() if on else self.ctl.stop_receiver()

    def _pairing(self, on: bool) -> None:
        self.ctl.prefs.require_pairing = on
        self.ctl.save_prefs()
        if self.ctl.receiver:
            self.ctl.receiver.require_code = on
        self._render_status()

    def _new_code(self) -> None:
        if self.ctl.receiver:
            self.ctl.receiver.new_code()
        self._render_status()

    def _event(self, kind, data) -> None:
        if kind == "progress":
            name, done, total = data
            self.progress.set_visible(done < total)
            self.progress.set_fraction(done / total if total else 0)
            self.progress.set_text(f"Receiving “{name}”: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        else:
            self.progress.set_visible(False)
        self._render_status()

    def _render_status(self) -> None:
        r = self.ctl.receiver
        if r is not None and r.running:
            addrs = self.ctl.state.lan_addresses()
            self.status_row.set_subtitle("Listening on " + ", ".join(f"{a}:{r.port}" for a in addrs or ["this PC"])
                                         + (f"\n{r.error}" if r.error else ""))
            self.code_row.set_subtitle(format_code(r.code) if r.require_code else "Not required")
        else:
            self.status_row.set_subtitle("Off" if r is None or r.error is None else f"Off: {r.error}")
            self.code_row.set_subtitle("—")

    def refresh(self) -> None:
        for row in self._rows:
            self.list_group.remove(row)
        self._rows = []
        caps = list_captures(self.ctl.inbox_dir)
        if not caps:
            row = action_row("No captures yet", "Record one with SplatWork Capture on the iPhone and tap Send.",
                             icon="phone-symbolic")
            self.list_group.add(row)
            self._rows.append(row)
        for c in caps:
            sub = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(c.received))} · {c.frames} frames · " \
                  f"{c.size / 1e6:.0f} MB" + (f" · from {c.sender}" if c.sender else "")
            row = plain_row(title=c.name, subtitle=sub)
            row.set_use_markup(False)
            if c.thumbnail:
                try:
                    pic = Gtk.Picture.new_for_filename(str(c.thumbnail))
                    pic.set_size_request(64, 48)
                    pic.set_content_fit(Gtk.ContentFit.COVER)
                    row.add_prefix(pic)
                except Exception:
                    pass
            make = Gtk.Button(label="New splat…", valign=Gtk.Align.CENTER)
            make.add_css_class("suggested-action")
            make.connect("clicked", lambda *_b, p=c.path: self.on_new_project("capture", str(p)))
            show = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Show")
            show.connect("clicked", lambda *_b, p=c.path: Gtk.FileLauncher.new(Gio.File.new_for_path(str(p))).launch(
                self.window, None, None, None))
            trash = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Move to Trash")
            trash.connect("clicked", lambda *_b, cap=c: self._trash(cap))
            for b in (make, show, trash):
                row.add_suffix(b)
            self.list_group.add(row)
            self._rows.append(row)

    def _trash(self, cap) -> None:
        def answer(rid):
            if rid == "trash":
                try:
                    Gio.File.new_for_path(str(cap.path)).trash(None)
                except GLib.Error:
                    shutil.rmtree(cap.path, ignore_errors=True)
                self.refresh()
        alert(self.window, f"Move “{cap.name}” to the trash?", "", (("cancel", "Cancel", "default"),
                                                                    ("trash", "Move to Trash", "destructive")), answer)


# ============================================================================ models

class ModelsPage(Gtk.Box):
    def __init__(self, window, app_state, studio=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.state = app_state
        self.studio = studio
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=18, margin_bottom=18,
                      margin_start=18, margin_end=18)
        rt = group("GPU runtime", "PyTorch, gsplat and the processing models live in this app's data folder.")
        self.rt_row = action_row("Status", "", icon="application-x-addon-symbolic")
        setup = Gtk.Button(label="Set Up / Repair…", valign=Gtk.Align.CENTER)
        setup.connect("clicked", lambda *_: window.get_application().activate_action("setup", None))
        self.rt_row.add_suffix(setup)
        rt.add(self.rt_row)
        self.models_row = action_row("Installed processing models", "Start the server to list them.",
                                     icon="preferences-system-symbolic")
        rt.add(self.models_row)
        box.append(rt)
        cards = group("Models", "Camera solving and masking models splatprep can use (licence, size and what each "
                                "is good at).")
        for card in self._cards():
            row = plain_expander(title=card.get("name") or card.get("id", "?"),
                                  subtitle=" · ".join(x for x in (card.get("kind"), card.get("license"),
                                                                  str(card.get("installSize") or "")) if x))
            row.set_use_markup(False)
            for key, title in (("summary", "Summary"), ("bestFor", "Best for"), ("strengths", "Strengths"),
                               ("weaknesses", "Weaknesses"), ("speed", "Speed"), ("quality", "Quality"),
                               ("url", "Project page")):
                v = card.get(key)
                if v:
                    r = plain_row(title=title, subtitle=v if isinstance(v, str) else ", ".join(map(str, v)))
                    r.set_use_markup(False)
                    r.set_subtitle_selectable(True)
                    row.add_row(r)
            cards.add(row)
        box.append(self._benchmark_group())
        box.append(cards)
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        clamp = Adw.Clamp(maximum_size=820)
        clamp.set_child(box)
        sw.set_child(clamp)
        self.append(_page("Models & Benchmarks", sw))
        self.update(None)

    # ------------------------------------------------------------------ benchmark
    def _benchmark_group(self) -> Gtk.Widget:
        g = group("Benchmark", "Training speed on the standard SplatWork scene (1 M splats, 1600 × 1200 photos, "
                               "2,000 steps): the same test the Mac's Compute → Benchmark runs, so the numbers "
                               "compare. Takes a few minutes; the GPU's queue runs it like any other job.")
        self.bench_run = plain_row(title="This PC", subtitle="Not measured yet")
        self.bench_btn = Gtk.Button(label="Run benchmark", valign=Gtk.Align.CENTER)
        self.bench_btn.add_css_class("suggested-action")
        self.bench_btn.connect("clicked", lambda *_: self.run_benchmark())
        self.bench_cancel = Gtk.Button(label="Cancel", valign=Gtk.Align.CENTER, visible=False)
        self.bench_cancel.connect("clicked", lambda *_: self._bench_flag.set())
        self.bench_run.add_suffix(self.bench_cancel)
        self.bench_run.add_suffix(self.bench_btn)
        self.bench_btn.set_visible(self.studio is not None)
        g.add(self.bench_run)
        self.bench_progress = Gtk.ProgressBar(show_text=True, visible=False, margin_top=8)
        g.add(self.bench_progress)
        frame = Gtk.Frame(margin_top=8)
        self.bench_grid = Gtk.Grid(column_spacing=18, row_spacing=6, margin_top=10, margin_bottom=10,
                                   margin_start=12, margin_end=12)
        frame.set_child(self.bench_grid)
        g.add(frame)
        self._bench_flag = threading.Event()
        self._bench_running = False
        self.refresh_benchmarks()
        return g

    BENCH_COLUMNS = ("Device", "GPU", "Steps/s", "Splats", "Photos", "Upload", "Download", "Measured")

    @staticmethod
    def benchmark_rows(results):
        """Table cells (strings) for saved results, fastest first."""
        def mbps(v):
            return f"{v:.0f} MB/s" if isinstance(v, (int, float)) and v > 0 else "—"

        def when(d):
            try:
                return datetime.fromisoformat(str(d).replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
            except ValueError:
                return str(d or "—")
        out = []
        for r in results:
            out.append((str(r.get("name") or r.get("deviceId")), str(r.get("gpu") or "—"),
                        f"{float(r['trainStepsPerSecond']):.1f}", f"{int(r.get('splats') or 0) / 1e6:.1f} M",
                        f"{float(r.get('megapixels') or 0):.1f} MP", mbps(r.get("uploadMBps")),
                        mbps(r.get("downloadMBps")), when(r.get("date"))))
        return out

    def refresh_benchmarks(self) -> None:
        while (c := self.bench_grid.get_first_child()) is not None:
            self.bench_grid.remove(c)
        results = benchmark.sorted_results(self.state.paths.config_dir)
        rows = self.benchmark_rows(results)
        if not rows:
            self.bench_grid.attach(label("No results yet: run the benchmark (or `splatworker benchmark`).",
                                         ("dim-label",)), 0, 0, 1, 1)
            return
        for col, title in enumerate(self.BENCH_COLUMNS):
            self.bench_grid.attach(label(title, ("heading",), wrap=False), col, 0, 1, 1)
        for r, cells in enumerate(rows, start=1):
            for col, text in enumerate(cells):
                lab = label(text, ("numeric",) if col in (2, 3, 4, 5, 6) else (), wrap=col == 1,
                            xalign=1.0 if col in (2, 3, 4, 5, 6) else 0.0)
                self.bench_grid.attach(lab, col, r, 1, 1)
        local = next((x for x in results if x.get("deviceId") == "local"), None)
        if local is not None and not self._bench_running:
            self.bench_run.set_subtitle(f"{float(local['trainStepsPerSecond']):.1f} steps/s on "
                                        f"{local.get('gpu') or 'this PC'}")

    def run_benchmark(self) -> None:
        if self.studio is None or self._bench_running:
            return
        self._bench_running = True
        self._bench_flag.clear()
        self.bench_btn.set_sensitive(False)
        self.bench_cancel.set_visible(True)
        self.bench_progress.set_visible(True)
        self.bench_progress.set_fraction(0.0)
        self.bench_progress.set_text("Starting the worker server…")
        cfg = self.state.paths.config_dir
        scene = self.state.paths.cache_dir / benchmark.SCENE_DIR
        name = self.state.settings.effective_name()

        def progress(msg, frac):
            GLib.idle_add(lambda: (self.bench_progress.set_fraction(min(max(frac, 0.0), 1.0)),
                                   self.bench_progress.set_text(msg), False)[-1])

        def go(client):
            try:
                rec = benchmark.run(client, scene, local=True, progress=progress, cancelled=self._bench_flag.is_set,
                                    priority=self.studio.priority, device_id="local", name=name)
                benchmark.save_benchmark(cfg, rec)
                GLib.idle_add(self._bench_done, f"{rec['trainStepsPerSecond']:.1f} steps/s")
            except benchmark.BenchmarkCancelled:
                GLib.idle_add(self._bench_done, "Cancelled")
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._bench_done, f"Failed: {getattr(e, 'message', None) or e}")
        self.studio.with_server(go, lambda msg: self._bench_done(f"Failed: {msg}"))

    def _bench_done(self, msg: str) -> bool:
        self._bench_running = False
        self.bench_btn.set_sensitive(True)
        self.bench_cancel.set_visible(False)
        self.bench_progress.set_visible(False)
        self.bench_run.set_subtitle(msg)
        self.refresh_benchmarks()
        if self.studio is not None:
            self.studio.toast(f"Benchmark: {msg}")
        return False

    @staticmethod
    def _cards():
        d = splatprep_dir()
        if d is None:
            return []
        try:
            data = json.loads((d / "splatprep" / "model_cards.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        models = data.get("models", data) if isinstance(data, dict) else data
        if isinstance(models, dict):
            models = [dict(v, id=k) for k, v in models.items() if isinstance(v, dict)]
        out = []
        for c in models if isinstance(models, list) else []:
            if not isinstance(c, dict):
                continue
            c = dict(c)
            c.setdefault("name", c.get("displayName") or c.get("id"))
            sp = c.get("speed")
            if isinstance(sp, dict):
                c["speed"] = sp.get("cuda") or sp.get("notes") or ""
            lic = c.get("license") or c.get("licence")
            if isinstance(lic, dict):
                c["license"] = lic.get("weights") or lic.get("code") or lic.get("name") or ""
            out.append(c)
        return out

    def update(self, info) -> None:
        self.rt_row.set_subtitle("Installed" if self.state.runtime_installed else "Not installed yet")
        if info:
            models = info.get("models") or []
            self.models_row.set_subtitle(", ".join(models) if models else "None (install the processing models)")
